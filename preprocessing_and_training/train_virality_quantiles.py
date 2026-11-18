#!/usr/bin/env python3
"""Run the paper's four virality quantiles on EVONS and FakeNewsNet."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "preprocessing_and_training" / "training"))

from fakenewsnet_prefix import DEFAULT_WINDOWS, validate_windows
from baseline_backend import resolve_backend, require_gpu
Q_VALUES = (0.5, 0.75, 0.9, 0.95)


def validate_quantiles(values):
    quantiles = [float(v) for v in values.split(",")]
    if not quantiles or any(not 0 < q < 1 for q in quantiles):
        raise ValueError("Quantiles must all lie strictly within (0,1)")
    if len(set(quantiles)) != len(quantiles):
        raise ValueError("Duplicate quantiles are not allowed")
    return quantiles


def run(dataset="both", quantiles=Q_VALUES, fnn_windows=DEFAULT_WINDOWS,
        baseline_backend=None):
    validate_windows(fnn_windows)
    backend = resolve_backend(baseline_backend)
    if backend == 'gpu' and dataset in ('both', 'fakenewsnet'):
        require_gpu()
    for q in quantiles:
        env = dict(os.environ, VIRALITY_QUANTILE=str(q), BASELINE_BACKEND=backend,
                   FNN_WINDOWS=fnn_windows,
                   PAPER_METRICS_DIR=str(ROOT / "results" / "metrics"))
        label = str(round(q * 100))
        if dataset in ("evons", "both"):
            print(f"Running EVONS q={q}", flush=True)
            subprocess.run(
                [sys.executable, str(ROOT / "preprocessing_and_training/training/train_evons_virality.py")],
                cwd=ROOT, env=env, check=True
            )
        if dataset in ("fakenewsnet", "both"):
            print(f"Running FakeNewsNet q={q}", flush=True)
            subprocess.run(
                [sys.executable, str(ROOT / "preprocessing_and_training/training/train_fakenewsnet_virality.py")],
                cwd=ROOT, env=env, check=True,
            )
            target = ROOT / "results/metrics" / f"fakenewsnet_virality_prefix_q{label}"
            result_file = target / "fakenewsnet_virality_merged_cv_results.csv"
            if not result_file.is_file():
                raise RuntimeError(f"Missing expected output: {result_file}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("evons", "fakenewsnet", "both"), default="both")
    parser.add_argument("--quantiles", default="0.5,0.75,0.9,0.95")
    parser.add_argument("--fnn-windows", default=DEFAULT_WINDOWS,
                        help="FakeNewsNet text windows selected from 1,3,5,10,full; cohort always >=11 tweets")
    parser.add_argument("--baseline-backend", choices=("cpu", "gpu"),
                        default=os.environ.get("BASELINE_BACKEND", "cpu"))
    args = parser.parse_args()
    run(args.dataset, validate_quantiles(args.quantiles), args.fnn_windows,
        args.baseline_backend)


if __name__ == "__main__":
    main()
