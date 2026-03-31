"""
sampling.py — Class-imbalance resampling strategies.

NOTE: With the shift to balanced tercile labels (33/33/33 split in
data_pipeline.py), resampling is no longer needed for the main experiment
pipeline and ``--method none`` is the new default.  These utilities are
retained for ablation studies or backwards compatibility.

Sampling (--method)
───────────────────
  none  No resampling (default)
  under Undersample Middle → ratio × minority_count  (default ratio=1.0)
  over  Oversample Loser/Winner → ratio × middle_count  (default ratio=1.0)
"""

import logging

import numpy as np

log = logging.getLogger(__name__)


def undersample_middle(x, y, ratio: float = 1.0, seed: int = 0):
    """Remove Middle rows so count = ratio × avg_minority_count."""
    rng            = np.random.default_rng(seed)
    minority_count = int(np.sum(y != 1) / 2)
    target_middle  = int(minority_count * ratio)
    mid_idx        = np.where(y == 1)[0]
    non_mid_idx    = np.where(y != 1)[0]
    if target_middle >= len(mid_idx):
        return x, y
    kept = rng.choice(mid_idx, size=target_middle, replace=False)
    idx  = rng.permutation(np.concatenate([non_mid_idx, kept]))
    log.info(f"  undersample  before={np.bincount(y)}  "
             f"after={np.bincount(y[idx])}  ratio={ratio}")
    return x[idx], y[idx]


def oversample_minority(x, y, ratio: float = 1.0, seed: int = 0):
    """Duplicate Loser/Winner rows so each reaches ratio × middle_count."""
    rng          = np.random.default_rng(seed)
    middle_count = int(np.sum(y == 1))
    target_min   = int(middle_count * ratio)
    parts_x, parts_y = [x], [y]
    for cls in [0, 2]:
        cls_idx  = np.where(y == cls)[0]
        n_to_add = target_min - len(cls_idx)
        if n_to_add <= 0:
            continue
        dup = rng.choice(cls_idx, size=n_to_add, replace=True)
        parts_x.append(x[dup])
        parts_y.append(y[dup])
    x_new = np.vstack(parts_x)
    y_new = np.concatenate(parts_y)
    sh    = rng.permutation(len(y_new))
    x_new, y_new = x_new[sh], y_new[sh]
    log.info(f"  oversample   before={np.bincount(y)}  "
             f"after={np.bincount(y_new)}  ratio={ratio}")
    return x_new, y_new


def apply_sampling(x_tr, y_tr, x_va, y_va, method: str, ratio: float):
    """Dispatch to the correct sampling function for both train and val splits.

    Parameters
    ----------
    method : "none" | "under" | "over"
    ratio  : sampling ratio
    """
    if method == "under":
        x_tr, y_tr = undersample_middle(x_tr, y_tr, ratio=ratio, seed=0)
        x_va, y_va = undersample_middle(x_va, y_va, ratio=ratio, seed=1)
    elif method == "over":
        x_tr, y_tr = oversample_minority(x_tr, y_tr, ratio=ratio, seed=0)
        x_va, y_va = oversample_minority(x_va, y_va, ratio=ratio, seed=1)
    # "none" → passthrough
    return x_tr, y_tr, x_va, y_va
