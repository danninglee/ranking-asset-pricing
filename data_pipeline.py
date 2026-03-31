"""
data_pipeline.py — Data loading, label creation, imputation and train/val/test splits.

Mirrors cl3_sas.py exactly:
  Source  : data/stock_sample_r1000.sas7bdat + r2000.sas7bdat
  Time    : eom (SAS date → datetime)
  Target  : forward_return → cross-sectional rank label 0/1/2
  Impute  : median per feature (fit on train+val only)
  Train   : 1985–2004   Val : 2005–2012   Test : 2013–2024
"""

import logging

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from config import NON_FEATURE_COLS, OLS3_FEATURES

log = logging.getLogger(__name__)


# ── Label generation ──────────────────────────────────────────────────────

def add_rank_label(df: pd.DataFrame,
                   target_col: str = "forward_return") -> pd.DataFrame:
    """
    Assign cross-sectional tercile labels each month.

    Label scheme (equal-sized terciles, ~33 % each):
      0 → bottom 33 %  (Loser tercile)
      1 → middle 33 %
      2 → top    33 %  (Winner tercile)

    Balanced classes eliminate the need for resampling.  The extreme-decile
    long/short portfolio is constructed at backtest time by ranking predicted
    probabilities within each tercile (see portfolio.py).
    """
    df = df.copy()

    def _rank_one_month(x):
        q33 = x.quantile(1 / 3)
        q67 = x.quantile(2 / 3)
        return pd.cut(x, bins=[x.min() - 1e-8, q33, q67, x.max() + 1e-8],
                      labels=[0, 1, 2])

    df["ylabel"] = (
        df.groupby("eom")[target_col]
        .transform(_rank_one_month)
        .astype(np.int64)
    )
    return df


# ── SAS file reading ──────────────────────────────────────────────────────

def _read_sas(path: str) -> pd.DataFrame:
    try:
        return pd.read_sas(path, format="sas7bdat", encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_sas(path, format="sas7bdat", encoding="latin-1")


def _parse_eom(col: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(col):
        return pd.to_datetime(col, unit="D", origin="1960-01-01").dt.normalize()
    return pd.to_datetime(col).dt.normalize()


def _prepare_df(df: pd.DataFrame, label: str) -> pd.DataFrame:
    df = df.copy()
    df["eom"] = _parse_eom(df["eom"])
    before = len(df)
    df = df.dropna(subset=["forward_return"])
    if (d := before - len(df)):
        log.info(f"[{label}] Dropped {d:,} rows with missing forward_return")
    df = add_rank_label(df, target_col="forward_return")
    return df


# ── Splits + imputation ───────────────────────────────────────────────────

def make_splits(df: pd.DataFrame, feature_cols: list,
                label: str) -> tuple[dict, dict]:
    """
    Returns
    -------
    splits    : {split_name: (X, y_int)}
    test_meta : dict{"fwd_ret", "mktcap", "eom", "rf"}  (for portfolio construction)
    """
    year = df["eom"].dt.year
    split_masks = {
        "train": (year >= 1985) & (year <= 2004),
        "val":   (year >= 2005) & (year <= 2012),
        "test":  (year >= 2013) & (year <= 2024),
    }

    tv_mask = split_masks["train"] | split_masks["val"]
    imputer = SimpleImputer(strategy="median")
    imputer.fit(df.loc[tv_mask, feature_cols])

    X_all   = imputer.transform(df[feature_cols].to_numpy(dtype=np.float32))
    df_feat = pd.DataFrame(X_all, index=df.index, columns=feature_cols)

    splits = {}
    for name, mask in split_masks.items():
        X = df_feat.loc[mask].to_numpy(dtype=np.float32)
        y = df.loc[mask, "ylabel"].to_numpy(dtype=np.int64)
        splits[name] = (X, y)
        dist = np.bincount(y)
        log.info(f"[{label}] {name:5s}  n={len(y):,}  dist={dist}"
                 f"  ({dist / len(y) * 100}%)")

    # test metadata for portfolio construction (never modified by sampling)
    tm = split_masks["test"]
    test_meta = {
        "fwd_ret": df.loc[tm, "forward_return"].to_numpy(dtype=np.float64),
        "mktcap":  df.loc[tm, "MarketCap"].to_numpy(dtype=np.float64),
        "eom":     df.loc[tm, "eom"].to_numpy(),
        "rf":      (df.loc[tm].groupby("eom")["rf"].first().to_numpy(dtype=np.float64)
                    if "rf" in df.columns else 0.0),
    }
    return splits, test_meta


# ── Universe loader ───────────────────────────────────────────────────────

def load_universe(r1000_path: str, r2000_path: str):
    """
    Load and prepare r1000, r2000, and combined universes.

    Returns
    -------
    universes    : {"r1000": df, "r2000": df, "combined": df}
    feature_cols : list[str]
    """
    log.info("Loading r1000 …")
    raw_r1000 = _read_sas(r1000_path)
    log.info(f"  r1000 raw shape: {raw_r1000.shape}")
    log.info("Loading r2000 …")
    raw_r2000 = _read_sas(r2000_path)
    log.info(f"  r2000 raw shape: {raw_r2000.shape}")

    df_r1000    = _prepare_df(raw_r1000, "r1000")
    df_r2000    = _prepare_df(raw_r2000, "r2000")
    df_combined = _prepare_df(
        pd.concat([raw_r1000, raw_r2000], ignore_index=True), "combined")

    universes    = {"r1000": df_r1000, "r2000": df_r2000, "combined": df_combined}
    feature_cols = [c for c in df_r1000.columns if c not in NON_FEATURE_COLS]
    log.info(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    for f in OLS3_FEATURES:
        if f not in feature_cols:
            raise ValueError(f"OLS3 feature '{f}' not found. "
                             f"Available: {feature_cols}")
    return universes, feature_cols