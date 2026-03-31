"""
models/nn_model.py — GKX (2020) MLP architecture and NN1–NN5 training runner.

Architecture  : Linear → BN → ReLU → Dropout(0.5) per hidden layer, then Linear output
HP search     : grid over lr × L1 on val balanced_accuracy (single seed)
Ensemble      : n_seeds independent runs on train+val → soft-vote
"""

import logging
import os
from itertools import product
from time import perf_counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from sklearn.metrics import balanced_accuracy_score

from config import NN_HIDDEN, NN_DROPOUT, NN_EPOCHS, NN_SEEDS, NN_LR_GRID, NN_WD_GRID
from metrics import compute_metrics, aggregate_metrics, print_single
from portfolio import compute_portfolio_metrics

log = logging.getLogger(__name__)


# ── Device helper ─────────────────────────────────────────────────────────

def get_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    if torch.cuda.is_available():
        log.info("Device: CUDA GPU"); return torch.device("cuda")
    if torch.backends.mps.is_available():
        log.info("Device: Apple MPS"); return torch.device("mps")
    log.info("Device: CPU"); return torch.device("cpu")


# ── Architecture ──────────────────────────────────────────────────────────

class _MLPNet(nn.Module):
    """
    GKX (2020) MLP architecture.

    Per-layer order (Table 1 / Appendix A):
        Linear → BatchNorm1d → ReLU → Dropout(p)
    Output layer: Linear only (no BN / activation / dropout).
    """
    def __init__(self, input_dim: int, hidden: int,
                 n_layers: int, n_classes: int, dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers += [nn.Linear(in_dim, hidden),
                       nn.BatchNorm1d(hidden),
                       nn.ReLU(),
                       nn.Dropout(dropout)]
            in_dim = hidden
        layers.append(nn.Linear(hidden, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── DataLoader factory ────────────────────────────────────────────────────

def _make_loaders(x_tr, y_tr, x_va, y_va, batch_size: int, device):
    pin = device.type == "cuda"
    nw  = min(4, os.cpu_count() or 1) if device.type != "mps" else 0

    def _ldr(X, y, shuffle):
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=nw, pin_memory=pin,
                          persistent_workers=(nw > 0))
    return _ldr(x_tr, y_tr, True), _ldr(x_va, y_va, False)


# ── Training helpers ──────────────────────────────────────────────────────

def _l1_loss(model: nn.Module) -> torch.Tensor:
    """Sum of absolute values of all trainable parameters (L1 penalty)."""
    return sum(p.abs().sum() for p in model.parameters() if p.requires_grad)


def _train_one_epoch(model, loader, optimizer, loss_fn, l1_lambda, device):
    """One training epoch with CE loss + L1 penalty."""
    model.train()
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(xb), yb) + l1_lambda * _l1_loss(model)
        loss.backward()
        optimizer.step()


@torch.no_grad()
def _eval_balanced_acc(model, X_t, y_t, device, chunk: int = 200_000) -> float:
    model.eval()
    preds = []
    for i in range(0, len(X_t), chunk):
        preds.append(model(X_t[i:i + chunk].to(device)).argmax(1).cpu())
    return balanced_accuracy_score(y_t, torch.cat(preds).numpy())


@torch.no_grad()
def _predict_proba(model, X_t, device, n_classes: int,
                   chunk: int = 200_000) -> np.ndarray:
    model.eval()
    sm    = nn.Softmax(dim=1)
    probs = []
    for i in range(0, len(X_t), chunk):
        probs.append(sm(model(X_t[i:i + chunk].to(device))).cpu())
    return torch.cat(probs).numpy()


# ── NN1–NN5 runner ────────────────────────────────────────────────────────

def run_nn(n_layers: int,
           x_train, y_train, x_val, y_val, x_test, y_test,
           test_meta=None, device=None,
           lr_grid=None, l1_grid=None,
           hidden: int = NN_HIDDEN, dropout: float = NN_DROPOUT,
           epochs: int = NN_EPOCHS, n_seeds: int = NN_SEEDS,
           batch_size: int = 512, universe_label: str = "") -> tuple:
    """
    GKX (2020) NN with ``n_layers`` hidden layers.

    Phase 1 — HP search on val balanced_accuracy (single seed=0).
    Phase 2 — 10-seed ensemble on train+val; soft-vote probabilities → metrics.

    Returns
    -------
    (metrics_dict, mean_overall, std_overall)
    """
    if device is None:
        device = get_device()
    if lr_grid is None:
        lr_grid = NN_LR_GRID
    if l1_grid is None:
        l1_grid = NN_WD_GRID

    t0      = perf_counter()
    tag     = f"[{universe_label}] NN{n_layers}" if universe_label else f"NN{n_layers}"
    n_feats = x_train.shape[1]
    n_cls   = 3
    loss_fn = nn.CrossEntropyLoss()
    x_val_t = torch.from_numpy(x_val)
    x_tst_t = torch.from_numpy(x_test)

    # ── Phase 1: HP search ─────────────────────────────────────────────────
    n_combos = len(lr_grid) * len(l1_grid)
    log.info(f"{tag} HP search: {n_combos} combos (lr×L1), seed=0, {epochs} epochs…")
    best_hp, best_hp_score = (lr_grid[0], l1_grid[0]), -1.0

    for lr, l1 in product(lr_grid, l1_grid):
        torch.manual_seed(0); np.random.seed(0)
        model = _MLPNet(n_feats, hidden, n_layers, n_cls, dropout).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=lr)
        tr_ld, _ = _make_loaders(x_train, y_train, x_val, y_val,
                                   batch_size, device)
        best_acc = -1.0
        for _ in range(epochs):
            _train_one_epoch(model, tr_ld, opt, loss_fn, l1, device)
            acc = _eval_balanced_acc(model, x_val_t, y_val, device)
            if acc > best_acc:
                best_acc = acc
        if best_acc > best_hp_score:
            best_hp_score = best_acc
            best_hp       = (lr, l1)

    best_lr, best_l1 = best_hp
    log.info(f"{tag} best lr={best_lr}  L1={best_l1}  "
             f"val_balanced_acc={best_hp_score:.4f}")

    # ── Phase 2: ensemble over n_seeds ─────────────────────────────────────
    x_full   = np.vstack([x_train, x_val])
    y_full   = np.concatenate([y_train, y_val])
    x_full_t = torch.from_numpy(x_full)
    all_probs = []
    log.info(f"{tag} ensemble: {n_seeds} seeds on train+val, "
             f"lr={best_lr} L1={best_l1} epochs={epochs}…")

    for seed in tqdm(range(n_seeds), desc=f"{tag} seeds"):
        torch.manual_seed(seed); np.random.seed(seed)
        model = _MLPNet(n_feats, hidden, n_layers, n_cls, dropout).to(device)
        if (hasattr(torch, "compile") and device.type == "cuda"
                and torch.__version__ >= "2.0"):
            try:
                model = torch.compile(model)
            except Exception:
                pass
        opt   = torch.optim.Adam(model.parameters(), lr=best_lr)
        fl_ld, _ = _make_loaders(x_full, y_full, x_full, y_full,
                                   batch_size, device)
        for _ in tqdm(range(epochs), desc=f"  seed {seed}", leave=False):
            _train_one_epoch(model, fl_ld, opt, loss_fn, best_l1, device)
        all_probs.append(_predict_proba(model, x_tst_t, device, n_cls))

    # soft vote → single ensemble prediction
    ensemble_prob = np.stack(all_probs, axis=0).mean(axis=0)
    ensemble_pred = ensemble_prob.argmax(axis=1)

    m = compute_metrics(y_test, ensemble_pred, ensemble_prob)
    if test_meta is not None:
        m["portfolio"] = compute_portfolio_metrics(
            ensemble_pred, test_meta["fwd_ret"],
            test_meta["mktcap"], test_meta["eom"],
            rf=test_meta.get("rf", 0.0),
            y_prob=ensemble_prob)    # ← enables decile sub-selection from tercile
    else:
        m["portfolio"] = None

    per_seed_metrics = [compute_metrics(y_test, p.argmax(1), p) for p in all_probs]
    _, std_ov        = aggregate_metrics(per_seed_metrics)
    m["_per_seed_std"] = std_ov

    log.info(f"{tag} done in {perf_counter() - t0:.1f}s")
    print_single(tag, m, mean_std=(m["overall"], std_ov))
    return m, m["overall"], std_ov
