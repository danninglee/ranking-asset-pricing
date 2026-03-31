"""
metrics.py — Classification metric computation and console reporting.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    matthews_corrcoef, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score,
)

from config import CLASS_NAMES


# ── Core computation ──────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, y_prob=None, n_classes: int = 3) -> dict:
    """
    Compute overall and per-class classification metrics.

    Returns
    -------
    dict with keys: "overall", "per_class", "confusion_matrix"
    """
    labels = list(range(n_classes))
    try:
        auc_macro    = roc_auc_score(y_true, y_prob, multi_class="ovr",
                                     average="macro", labels=labels)
        auc_weighted = roc_auc_score(y_true, y_prob, multi_class="ovr",
                                     average="weighted", labels=labels)
    except Exception:
        auc_macro = auc_weighted = float("nan")

    overall = {
        "acc":          accuracy_score(y_true, y_pred),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "f1_macro":     f1_score(y_true, y_pred, average="macro",
                                 labels=labels, zero_division=0),
        "f1_weighted":  f1_score(y_true, y_pred, average="weighted",
                                 labels=labels, zero_division=0),
        "mcc":          matthews_corrcoef(y_true, y_pred),
        "auc_macro":    auc_macro,
        "auc_weighted": auc_weighted,
    }

    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    pred_counts = np.bincount(y_pred, minlength=n_classes)
    total       = len(y_true)

    per_class = {}
    for i in labels:
        y_bin = (y_true == i).astype(int)
        try:
            auc_i = roc_auc_score(y_bin, y_prob[:, i]) if y_prob is not None \
                    else float("nan")
        except Exception:
            auc_i = float("nan")
        per_class[i] = {
            "count":           int(support[i]),
            "proportion":      float(support[i] / total),
            "precision":       float(prec[i]),
            "recall":          float(rec[i]),
            "f1":              float(f1[i]),
            "auc":             auc_i,
            "predicted_count": int(pred_counts[i]),
            "predicted_prop":  float(pred_counts[i] / total),
        }

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {"overall": overall, "per_class": per_class, "confusion_matrix": cm}


def aggregate_metrics(metrics_list: list[dict]) -> tuple[dict, dict]:
    """Average overall dicts across seeds; return (mean_dict, std_dict)."""
    keys = metrics_list[0]["overall"].keys()
    mean = {k: float(np.mean([m["overall"][k] for m in metrics_list])) for k in keys}
    std  = {k: float(np.std( [m["overall"][k] for m in metrics_list], ddof=1))
            for k in keys}
    return mean, std


# ── Console printing ──────────────────────────────────────────────────────

def print_single(title: str, metrics: dict,
                 mean_std: tuple | None = None) -> None:
    """Print classification + optional portfolio metrics for one condition."""
    sep = "=" * 62
    print(f"\n{sep}\n  {title}\n{sep}")

    ov  = metrics["overall"] if mean_std is None else mean_std[0]
    std = {} if mean_std is None else mean_std[1]

    def _fmt(k):
        v = ov[k]
        s = f"  (±{std[k]:.6f})" if k in std else ""
        return f"  {k:<20} {v:.6f}{s}"

    print("\n  -- Overall metrics --")
    for k in ov:
        print(_fmt(k))

    if mean_std is None:
        print("\n  -- Per-class breakdown --")
        hdr = (f"  {'Class':<22} {'True N':>7} {'True%':>6} "
               f"{'Prec':>7} {'Recall':>7} {'F1':>7} {'AUC':>7} "
               f"{'Pred N':>7} {'Pred%':>6}")
        print(hdr)
        print("  " + "-" * 83)
        for i, cname in CLASS_NAMES.items():
            p = metrics["per_class"][i]
            print(f"  {cname:<22} "
                  f"{p['count']:>7d} {p['proportion']*100:>5.1f}% "
                  f"{p['precision']:>7.4f} {p['recall']:>7.4f} {p['f1']:>7.4f} "
                  f"{p['auc']:>7.4f} "
                  f"{p['predicted_count']:>7d} {p['predicted_prop']*100:>5.1f}%")

        print("\n  -- Confusion matrix (row=true, col=pred) --")
        cm = metrics["confusion_matrix"]
        print("  " + " " * 22 + "".join(f"  Pred{i}" for i in CLASS_NAMES))
        for i, cname in CLASS_NAMES.items():
            print("  " + f"{cname:<22}"
                  + "".join(f"  {cm[i, j]:>5d}" for j in CLASS_NAMES))

    port = metrics.get("portfolio")
    if port is None:
        return

    print("\n  -- Portfolio (Long=Winner, Short=Loser, L/S=Long−Short) --")
    print("     (Sharpe & t-stat on excess returns; rf subtracted per GKX 2020)")
    col_labels = ["EW-Long", "EW-Short", "EW-L/S", "VW-Long", "VW-Short", "VW-L/S"]
    col_keys   = [("ew", "long"), ("ew", "short"), ("ew", "long_short"),
                  ("vw", "long"), ("vw", "short"), ("vw", "long_short")]
    cw = 10
    print(f"  {'':22}" + "".join(f"{c:>{cw}}" for c in col_labels))
    print("  " + "-" * (22 + cw * len(col_labels)))
    for stat, label in [("annual_ret",        "Annual return"),
                         ("annual_excess_ret", "Annual excess ret"),
                         ("sharpe",            "Sharpe ratio"),
                         ("t_stat",            "t-statistic"),
                         ("max_drawdown",      "Max drawdown"),
                         ("hit_rate",          "Hit rate")]:
        row = f"  {label:<22}"
        for w, leg in col_keys:
            v = port[w][leg].get(stat, np.nan)
            row += f"{v:>{cw}.4f}" if not np.isnan(v) else f"{'nan':>{cw}}"
        print(row)
