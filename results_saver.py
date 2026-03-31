"""
results_saver.py — Save experiment results as CSV files.

Output structure
────────────────
  <save_dir>/<universe>/overall_metrics.csv
  <save_dir>/<universe>/per_class_metrics.csv
  <save_dir>/<universe>/portfolio_metrics.csv
  <save_dir>/all_universes_overall.csv
  <save_dir>/all_universes_per_class.csv
  <save_dir>/all_universes_portfolio.csv
"""

import logging
import os
from os.path import join

import pandas as pd

from config import CLASS_NAMES

log = logging.getLogger(__name__)


def save_results(all_universe_results: dict, save_dir: str) -> None:
    """
    Persist all experiment results to CSV.

    Parameters
    ----------
    all_universe_results : {universe: {model_name: {condition: metrics_dict}}}
    save_dir             : root output directory
    """
    os.makedirs(save_dir, exist_ok=True)
    all_overall, all_pc, all_port = [], [], []

    for uni, all_results in all_universe_results.items():
        uni_dir = join(save_dir, uni)
        os.makedirs(uni_dir, exist_ok=True)
        rows_ov, rows_pc, rows_port = [], [], []

        for model_name, conditions in all_results.items():
            for cond_name, metrics in conditions.items():
                base = {"universe": uni, "model": model_name,
                        "condition": cond_name}

                # overall
                row = {**base}
                row.update({k: round(v, 6) for k, v in metrics["overall"].items()})
                rows_ov.append(row)
                all_overall.append(row)

                # per-class
                for cls_id, cls_name in CLASS_NAMES.items():
                    row_pc = {**base, "class": cls_id, "class_name": cls_name}
                    row_pc.update({
                        k: (round(v, 6) if isinstance(v, float) else v)
                        for k, v in metrics["per_class"][cls_id].items()
                    })
                    rows_pc.append(row_pc)
                    all_pc.append(row_pc)

                # portfolio
                port = metrics.get("portfolio")
                if port is not None:
                    for w in ["ew", "vw"]:
                        for leg in ["long", "short", "long_short"]:
                            row_p = {**base, "weighting": w, "portfolio": leg}
                            row_p.update({
                                k: round(v, 6) if isinstance(v, float) else v
                                for k, v in port[w][leg].items()
                            })
                            rows_port.append(row_p)
                            all_port.append(row_p)

        pd.DataFrame(rows_ov).to_csv(
            join(uni_dir, "overall_metrics.csv"), index=False)
        pd.DataFrame(rows_pc).to_csv(
            join(uni_dir, "per_class_metrics.csv"), index=False)
        log.info(f"Saved → {uni_dir}/overall_metrics.csv")

        if rows_port:
            pd.DataFrame(rows_port).to_csv(
                join(uni_dir, "portfolio_metrics.csv"), index=False)
            log.info(f"Saved → {uni_dir}/portfolio_metrics.csv")

    for fname, rows in [
        ("all_universes_overall.csv",   all_overall),
        ("all_universes_per_class.csv", all_pc),
        ("all_universes_portfolio.csv", all_port),
    ]:
        if rows:
            pd.DataFrame(rows).to_csv(join(save_dir, fname), index=False)
            log.info(f"Saved → {join(save_dir, fname)}")
