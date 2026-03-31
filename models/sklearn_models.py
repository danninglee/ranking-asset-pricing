"""
models/sklearn_models.py — Sklearn baseline runners.

All models follow the same protocol:
  1. Grid search on val balanced_accuracy to select hyperparameters.
  2. Refit on train+val with the best parameters.
  3. Predict test set → compute metrics + portfolio.

Models
──────
  Logistic-FULL  : all features, no regularisation (≈ OLS baseline)
  Logistic-OLS3  : ME / B2M / R12_2 only
  LinearSVC      : L2 penalty; C grid search
  KNN            : k grid search (balanced_accuracy on val)
  RF             : GKX Table A.7 grid; n_seeds ensemble (mean ± std)
  GBRT           : GKX Table A.7 grid; n_seeds ensemble (mean ± std)
"""

import logging
from itertools import product
from time import perf_counter

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from tqdm import tqdm

from config import (
    OLS3_FEATURES,
    SVM_C_GRID, KNN_K_GRID,
    RF_GRID, GBRT_GRID,
)
from metrics import compute_metrics, aggregate_metrics, print_single
from portfolio import compute_portfolio_metrics

log = logging.getLogger(__name__)


# ── Shared helper ─────────────────────────────────────────────────────────

def _wrap(y_pred, y_prob, y_test, test_meta) -> dict:
    """Compute classification metrics + portfolio, return unified dict."""
    m = compute_metrics(y_test, y_pred, y_prob)
    if test_meta is not None:
        m["portfolio"] = compute_portfolio_metrics(
            y_pred, test_meta["fwd_ret"], test_meta["mktcap"], test_meta["eom"],
            rf=test_meta.get("rf", 0.0),
            y_prob=y_prob)           # ← enables decile sub-selection from tercile
    else:
        m["portfolio"] = None
    return m


# ── Logistic regression ───────────────────────────────────────────────────

def run_logistic(x_train, y_train, x_val, y_val, x_test, y_test,
                 test_meta=None, universe_label: str = "") -> dict:
    """Logistic regression on all features (≈ OLS baseline)."""
    tag = f"[{universe_label}] Logistic-FULL" if universe_label else "Logistic-FULL"
    t0  = perf_counter()
    x_full = np.vstack([x_train, x_val])
    y_full = np.concatenate([y_train, y_val])
    clf = LogisticRegression(solver="lbfgs", max_iter=300, n_jobs=-1
                             ).fit(x_full, y_full)
    m = _wrap(clf.predict(x_test), clf.predict_proba(x_test), y_test, test_meta)
    print_single(tag, m)
    log.info(f"{tag} done in {perf_counter() - t0:.1f}s")
    return m


def run_logistic_ols3(x_train, y_train, x_val, y_val, x_test, y_test,
                      feature_cols, test_meta=None, universe_label: str = "") -> dict:
    """Logistic regression on the three OLS features: ME, B2M, R12_2."""
    tag = f"[{universe_label}] Logistic-OLS3" if universe_label else "Logistic-OLS3"
    t0  = perf_counter()
    idx    = [feature_cols.index(c) for c in OLS3_FEATURES]
    x_full = np.vstack([x_train[:, idx], x_val[:, idx]])
    y_full = np.concatenate([y_train, y_val])
    clf = LogisticRegression(solver="lbfgs", max_iter=300).fit(x_full, y_full)
    m = _wrap(clf.predict(x_test[:, idx]),
              clf.predict_proba(x_test[:, idx]), y_test, test_meta)
    print_single(tag, m)
    log.info(f"{tag} done in {perf_counter() - t0:.1f}s")
    return m


# ── LinearSVC ─────────────────────────────────────────────────────────────

def run_svm(x_train, y_train, x_val, y_val, x_test, y_test,
            test_meta=None, universe_label: str = "",
            c_grid=None) -> dict:
    """
    LinearSVC with C grid search on val balanced_accuracy.

    Probabilities are obtained via CalibratedClassifierCV (Platt scaling),
    consistent with GKX requiring probability estimates for AUC.
    """
    if c_grid is None:
        c_grid = SVM_C_GRID
    tag = f"[{universe_label}] LinearSVC" if universe_label else "LinearSVC"
    t0  = perf_counter()

    best_c, best_score = c_grid[0], -1.0
    for c in c_grid:
        clf   = LinearSVC(C=c, max_iter=5000).fit(x_train, y_train)
        score = balanced_accuracy_score(y_val, clf.predict(x_val))
        if score > best_score:
            best_score, best_c = score, c

    log.info(f"{tag} best C={best_c}  val_bal_acc={best_score:.4f}")
    x_full = np.vstack([x_train, x_val])
    y_full = np.concatenate([y_train, y_val])

    cal_clf = CalibratedClassifierCV(
        LinearSVC(C=best_c, max_iter=5000), cv=5, method="sigmoid"
    ).fit(x_full, y_full)
    m = _wrap(cal_clf.predict(x_test), cal_clf.predict_proba(x_test),
              y_test, test_meta)
    print_single(tag, m)
    log.info(f"{tag} done in {perf_counter() - t0:.1f}s")
    return m


# ── KNN ───────────────────────────────────────────────────────────────────

def run_knn(x_train, y_train, x_val, y_val, x_test, y_test,
            test_meta=None, universe_label: str = "",
            k_grid=None) -> dict:
    """KNN with k grid search on val balanced_accuracy."""
    if k_grid is None:
        k_grid = KNN_K_GRID
    tag = f"[{universe_label}] KNN" if universe_label else "KNN"
    t0  = perf_counter()

    best_k, best_score = k_grid[0], -1.0
    for k in k_grid:
        clf   = KNeighborsClassifier(n_neighbors=k, weights="distance",
                                     metric="euclidean", n_jobs=-1
                                     ).fit(x_train, y_train)
        score = balanced_accuracy_score(y_val, clf.predict(x_val))
        if score > best_score:
            best_score, best_k = score, k

    log.info(f"{tag} best k={best_k}  val_bal_acc={best_score:.4f}")
    x_full = np.vstack([x_train, x_val])
    y_full = np.concatenate([y_train, y_val])
    clf    = KNeighborsClassifier(n_neighbors=best_k, weights="distance",
                                   metric="euclidean", n_jobs=-1
                                   ).fit(x_full, y_full)
    m = _wrap(clf.predict(x_test), clf.predict_proba(x_test), y_test, test_meta)
    print_single(tag, m)
    log.info(f"{tag} done in {perf_counter() - t0:.1f}s")
    return m


# ── Random Forest ─────────────────────────────────────────────────────────

def run_rf(x_train, y_train, x_val, y_val, x_test, y_test,
           test_meta=None, universe_label: str = "",
           param_grid=None, n_seeds: int = 10) -> tuple:
    """
    Random Forest with GKX 2020 grid search on val balanced_accuracy.
    Best params → refit on train+val across n_seeds → report mean ± std.

    Returns
    -------
    (metrics_dict, mean_overall, std_overall)
    """
    if param_grid is None:
        param_grid = RF_GRID
    tag = f"[{universe_label}] RF" if universe_label else "RF"
    t0  = perf_counter()

    combos = list(product(
        param_grid["n_estimators"],
        param_grid["max_depth"],
        param_grid["min_samples_leaf"],
        param_grid["max_features"],
    ))
    log.info(f"{tag} grid search over {len(combos)} combos…")

    best_params, best_score = combos[0], -1.0
    for n_est, md, msl, mf in tqdm(combos, desc=f"{tag} val search"):
        clf   = RandomForestClassifier(n_estimators=n_est, max_depth=md,
                                        min_samples_leaf=msl, max_features=mf,
                                        random_state=0, n_jobs=-1
                                        ).fit(x_train, y_train)
        score = balanced_accuracy_score(y_val, clf.predict(x_val))
        if score > best_score:
            best_score  = score
            best_params = (n_est, md, msl, mf)

    n_est, md, msl, mf = best_params
    log.info(f"{tag} best params={best_params}  val_bal_acc={best_score:.4f}")

    x_full = np.vstack([x_train, x_val])
    y_full = np.concatenate([y_train, y_val])
    metrics_list = []
    for seed in tqdm(range(n_seeds), desc=f"{tag} seeds"):
        clf    = RandomForestClassifier(n_estimators=n_est, max_depth=md,
                                         min_samples_leaf=msl, max_features=mf,
                                         random_state=seed, n_jobs=-1
                                         ).fit(x_full, y_full)
        m = _wrap(clf.predict(x_test), clf.predict_proba(x_test), y_test, test_meta)
        metrics_list.append(m)

    mean_ov, std_ov = aggregate_metrics(metrics_list)
    rep = metrics_list[-1]
    rep["overall"] = mean_ov
    print_single(tag, rep, mean_std=(mean_ov, std_ov))
    log.info(f"{tag} done in {perf_counter() - t0:.1f}s")
    return rep, mean_ov, std_ov


# ── Gradient Boosted Regression Trees ────────────────────────────────────

def run_gbrt(x_train, y_train, x_val, y_val, x_test, y_test,
             test_meta=None, universe_label: str = "",
             param_grid=None, n_seeds: int = 10) -> tuple:
    """
    Gradient Boosting with GKX 2020 grid search on val balanced_accuracy.

    Returns
    -------
    (metrics_dict, mean_overall, std_overall)
    """
    if param_grid is None:
        param_grid = GBRT_GRID
    tag = f"[{universe_label}] GBRT" if universe_label else "GBRT"
    t0  = perf_counter()

    combos = list(product(
        param_grid["n_estimators"],
        param_grid["max_depth"],
        param_grid["learning_rate"],
        param_grid["subsample"],
    ))
    log.info(f"{tag} grid search over {len(combos)} combos…")

    best_params, best_score = combos[0], -1.0
    for n_est, md, lr, ss in tqdm(combos, desc=f"{tag} val search"):
        clf   = GradientBoostingClassifier(n_estimators=n_est, max_depth=md,
                                            learning_rate=lr, subsample=ss,
                                            random_state=0
                                            ).fit(x_train, y_train)
        score = balanced_accuracy_score(y_val, clf.predict(x_val))
        if score > best_score:
            best_score  = score
            best_params = (n_est, md, lr, ss)

    n_est, md, lr, ss = best_params
    log.info(f"{tag} best params={best_params}  val_bal_acc={best_score:.4f}")

    x_full = np.vstack([x_train, x_val])
    y_full = np.concatenate([y_train, y_val])
    metrics_list = []
    for seed in tqdm(range(n_seeds), desc=f"{tag} seeds"):
        clf    = GradientBoostingClassifier(n_estimators=n_est, max_depth=md,
                                             learning_rate=lr, subsample=ss,
                                             random_state=seed
                                             ).fit(x_full, y_full)
        m = _wrap(clf.predict(x_test), clf.predict_proba(x_test), y_test, test_meta)
        metrics_list.append(m)

    mean_ov, std_ov = aggregate_metrics(metrics_list)
    rep = metrics_list[-1]
    rep["overall"] = mean_ov
    print_single(tag, rep, mean_std=(mean_ov, std_ov))
    log.info(f"{tag} done in {perf_counter() - t0:.1f}s")
    return rep, mean_ov, std_ov
