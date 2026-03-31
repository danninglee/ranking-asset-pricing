"""
experiment.py — Per-universe experiment orchestrator.

``run_universe`` ties together data splits, sampling, and all model runners
for a single universe (r1000 / r2000 / combined) under one sampling condition.
"""

import logging

from config import OLS3_FEATURES, NN_EPOCHS, NN_SEEDS
from data_pipeline import make_splits
from sampling import apply_sampling
from models.sklearn_models import (
    run_logistic, run_logistic_ols3,
    run_svm, run_knn, run_rf, run_gbrt,
)
from models.nn_model import run_nn

log = logging.getLogger(__name__)


def run_universe(universe_label: str, df, feature_cols: list,
                 models: list, method: str = "none", ratio: float = 1.0,
                 device=None, n_seeds: int = NN_SEEDS,
                 nn_epochs: int = NN_EPOCHS) -> dict:
    """
    Run all requested models for one universe under one sampling condition.

    Parameters
    ----------
    universe_label : "r1000" | "r2000" | "combined"
    df             : prepared DataFrame for this universe
    feature_cols   : list of feature column names
    models         : list of model keys, e.g. ["logistic","rf","nn1","nn3"]
    method         : "none" | "under" | "over"
                     Default is "none" — balanced tercile labels make
                     resampling unnecessary.
    ratio          : sampling ratio (only used when method != "none")
    device         : torch.device (None → auto-detect)
    n_seeds        : number of random seeds for stochastic models
    nn_epochs      : training epochs for NN models

    Returns
    -------
    {model_name: {condition_name: metrics_dict}}
    """
    banner = (f"{'='*18}  UNIVERSE: {universe_label.upper()}  "
              f"METHOD: {method.upper()}  {'='*18}")
    print(f"\n\n{banner}")
    log.info(f"Universe={universe_label}  method={method}  ratio={ratio}")

    splits, test_meta = make_splits(df, feature_cols, universe_label)
    x_train, y_train  = splits["train"]
    x_val,   y_val    = splits["val"]
    x_test,  y_test   = splits["test"]

    cond_name = ("No sampling" if method == "none"
                 else f"{'Oversample' if method == 'over' else 'Undersample'}"
                      f" ratio={ratio}")

    x_tr_s, y_tr_s, x_va_s, y_va_s = apply_sampling(
        x_train, y_train, x_val, y_val, method, ratio)

    all_res  = {}
    kw_base  = dict(test_meta=test_meta, universe_label=universe_label)

    def _store(key, m):
        all_res.setdefault(key, {})[cond_name] = (
            m if isinstance(m, dict) else m[0])

    if "logistic" in models:
        _store("Logistic-FULL",
               run_logistic(x_tr_s, y_tr_s, x_va_s, y_va_s,
                            x_test, y_test, **kw_base))

    if "logistic3" in models:
        _store("Logistic-OLS3",
               run_logistic_ols3(x_tr_s, y_tr_s, x_va_s, y_va_s,
                                  x_test, y_test, feature_cols, **kw_base))

    if "svm" in models:
        _store("LinearSVC",
               run_svm(x_tr_s, y_tr_s, x_va_s, y_va_s,
                       x_test, y_test, **kw_base))

    if "knn" in models:
        _store("KNN",
               run_knn(x_tr_s, y_tr_s, x_va_s, y_va_s,
                       x_test, y_test, **kw_base))

    if "rf" in models:
        rep, _, _ = run_rf(x_tr_s, y_tr_s, x_va_s, y_va_s,
                            x_test, y_test, n_seeds=n_seeds, **kw_base)
        _store("RF", rep)

    if "gbrt" in models:
        rep, _, _ = run_gbrt(x_tr_s, y_tr_s, x_va_s, y_va_s,
                              x_test, y_test, n_seeds=n_seeds, **kw_base)
        _store("GBRT", rep)

    for n_lay in range(1, 6):
        key = f"nn{n_lay}"
        if key in models:
            rep, _, _ = run_nn(n_lay, x_tr_s, y_tr_s, x_va_s, y_va_s,
                                x_test, y_test,
                                test_meta=test_meta, device=device,
                                n_seeds=n_seeds, epochs=nn_epochs,
                                universe_label=universe_label)
            _store(f"NN{n_lay}", rep)

    return all_res
