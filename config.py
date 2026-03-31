"""
config.py — Global constants and GKX (2020) hyperparameter grids.
"""

# ── Class labels ───────────────────────────────────────────────────────────
# Labels now represent equal terciles (33/33/33).
# The extreme-decile long/short portfolio is carved out of these terciles
# at backtest time using predicted probabilities (see portfolio.py).
CLASS_NAMES = {0: "Loser (bottom 33%)", 1: "Middle (33%)", 2: "Winner (top 33%)"}

# ── Column sets ───────────────────────────────────────────────────────────
NON_FEATURE_COLS = {
    "permno", "eom", "MarketCap",
    "forward_return", "forward_return_demean", "forward_return_rank",
    "ylabel",
}

OLS3_FEATURES = ["ME", "B2M", "R12_2"]

# ── GKX 2020 hyperparameter grids (Appendix Table A.7) ────────────────────
RF_GRID = {
    "n_estimators":     [300],
    "max_depth":        [1, 2, 4, 6],
    "min_samples_leaf": [5],
    "max_features":     [3, 5, 10, 20, 30, 40],
}

GBRT_GRID = {
    "n_estimators":  [100, 300],
    "max_depth":     [1, 2, 4],
    "learning_rate": [0.01, 0.1],
    "subsample":     [0.5],
}

SVM_C_GRID  = [0.01, 0.1, 1.0, 10.0]
KNN_K_GRID  = [5, 10, 20, 50, 100]
NN_LR_GRID  = [1e-2, 1e-3, 1e-4]
NN_WD_GRID  = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]   # L1 penalty values
NN_HIDDEN   = 32
NN_DROPOUT  = 0.5
NN_EPOCHS   = 100
NN_SEEDS    = 10
