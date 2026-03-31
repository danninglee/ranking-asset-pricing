"""
run_experiment.py — CLI entry point for GKX (2020) classification baselines.

Usage examples
──────────────
  # All models, all universes, undersample (default)
  python run_experiment.py

  # Only RF and NN3 on r1000, no resampling
  python run_experiment.py --models rf nn3 --universes r1000 --method none

  # Oversample with ratio 0.5, 5 seeds, 50 NN epochs, save to custom dir
  python run_experiment.py --method over --ratio 0.5 --seeds 5 \\
      --nn_epochs 50 --save_dir results/my_run

See module-level docstring in cl3_sas_baselines1.py for full specification.
"""

import argparse
import logging
import warnings

from config import NN_SEEDS, NN_EPOCHS
from data_pipeline import load_universe
from models.nn_model import get_device
from experiment import run_universe
from results_saver import save_results

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
log = logging.getLogger(__name__)


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="GKX 2020 classification baselines on SAS dataset")
    p.add_argument("--r1000",     default="data/stock_sample_r1000.sas7bdat")
    p.add_argument("--r2000",     default="data/stock_sample_r2000.sas7bdat")
    p.add_argument("--models",    nargs="+",
                   default=["logistic", "logistic3", "svm", "knn",
                            "rf", "gbrt", "nn1", "nn2", "nn3", "nn4", "nn5"],
                   help="Models to run (default: all)")
    p.add_argument("--method",    default="none",
                   choices=["none", "under", "over"],
                   help="Sampling strategy: none | under | over  (default: none — "
                        "balanced tercile labels make resampling unnecessary)")
    p.add_argument("--ratio",     type=float, default=1.0,
                   help="Sampling ratio (default: 1.0)")
    p.add_argument("--universes", nargs="+",
                   default=["r1000", "r2000", "combined"],
                   choices=["r1000", "r2000", "combined"])
    p.add_argument("--seeds",     type=int, default=NN_SEEDS,
                   help=f"Random seeds for stochastic models (default: {NN_SEEDS})")
    p.add_argument("--nn_epochs", type=int, default=NN_EPOCHS,
                   help=f"Training epochs for NN models (default: {NN_EPOCHS})")
    p.add_argument("--device",    default="auto",
                   help="Device: auto | cpu | cuda | mps")
    p.add_argument("--save_dir",  default="results/classification/baselines_sas")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = get_device(args.device)
    models = [m.lower() for m in args.models]

    universes, feature_cols = load_universe(args.r1000, args.r2000)

    all_universe_results = {}
    for uni_label in args.universes:
        all_universe_results[uni_label] = run_universe(
            universe_label=uni_label,
            df=universes[uni_label],
            feature_cols=feature_cols,
            models=models,
            method=args.method,
            ratio=args.ratio,
            device=device,
            n_seeds=args.seeds,
            nn_epochs=args.nn_epochs,
        )

    save_results(all_universe_results, args.save_dir)


if __name__ == "__main__":
    main()
