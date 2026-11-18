#!/usr/bin/env python3
"""Run the current paper pipeline from local source data to tables and figures.

Stages can also be run separately: preprocess, check, audit, train, paper.
FakeNewsNet inputs are recovered tweet JSON files supplied by the data owner;
this repository does not collect or rehydrate tweets.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "preprocessing_and_training" / "training"))
from baseline_backend import require_gpu, resolve_backend
from fakenewsnet_prefix import DEFAULT_WINDOWS, validate_windows


OUT = ROOT / "results"
METRICS = OUT / "metrics"
AUDITS = OUT / "audits"
EVONS = ROOT / "evons" / "data"
FNN = ROOT / "FakeNewsNet" / "data"
QUANTILES = ("0.5", "0.75", "0.9", "0.95")
REQUIRED = {
    "EVONS": [EVONS / path for path in (
        "evons.csv", "title_embeddings.pt", "desc_embeddings.pt",
        "title_embeddings_mistral.pt", "desc_embeddings_mistral.pt",
    )],
    "FakeNewsNet": [FNN / path for path in (
        "ordered_fake_propagation_paths.jsonl",
        "ordered_real_propagation_paths.jsonl",
        "fake_news_net/ordered_fake_propagation_paths_emb.pt",
        "fake_news_net/ordered_real_propagation_paths_emb.pt",
        "fake_news_net/ordered_fake_propagation_paths_emb_mistral.pt",
        "fake_news_net/ordered_real_propagation_paths_emb_mistral.pt",
    )],
}


def run(command, *, env=None) -> None:
    args = [str(item) for item in command]
    print("\n$ " + " ".join(args), flush=True)
    subprocess.run(args, check=True, cwd=ROOT, env=env)


def check() -> bool:
    complete = True
    for paths in REQUIRED.values():
        for path in paths:
            exists = path.is_file()
            print(("OK      " if exists else "MISSING ") + str(path.relative_to(ROOT)))
            complete &= exists
    return complete


def preprocess(encoders: str) -> None:
    run([sys.executable, "preprocessing_and_training/preprocess_datasets.py", "--dataset", "both",
         "--encoders", encoders])


def audit() -> None:
    AUDITS.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "metrics_and_audits/dataset_audit.py", "--dataset", "both",
         "--evons", EVONS / "evons.csv",
         "--fake-jsonl", FNN / "ordered_fake_propagation_paths.jsonl",
         "--real-jsonl", FNN / "ordered_real_propagation_paths.jsonl",
         "--out", AUDITS / "dataset_audit.json"])
    for category in ("fake", "real"):
        run([sys.executable, "metrics_and_audits/audit_fakenewsnet_future_engagement.py",
             "--input-jsonl", FNN / f"ordered_{category}_propagation_paths.jsonl",
             "--out", AUDITS / f"future_{category}.csv"])


def train(fnn_windows: str, baseline_backend: str) -> None:
    backend = resolve_backend(baseline_backend)
    if backend == "gpu":
        require_gpu()
    METRICS.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, BASELINE_BACKEND=backend,
               PAPER_METRICS_DIR=str(METRICS), FNN_WINDOWS=fnn_windows)
    run([sys.executable, "preprocessing_and_training/training/train_evons_veracity.py"], env=env)
    run([sys.executable, "preprocessing_and_training/training/train_fakenewsnet_veracity.py"], env=env)
    run([sys.executable, "preprocessing_and_training/train_virality_quantiles.py",
         "--dataset", "both", "--quantiles", ",".join(QUANTILES),
         "--baseline-backend", backend, "--fnn-windows", fnn_windows], env=env)
    print("All four paper tasks completed", flush=True)


def paper_outputs() -> None:
    run([sys.executable, "metrics_and_audits/paired_resampling_inference.py",
         "--metrics-dir", METRICS,
         "--output-dir", OUT / "inference"])
    run([sys.executable, "paper/generate_tables.py",
         "--metrics-dir", METRICS, "--audit-dir", AUDITS,
         "--output-dir", OUT / "paper" / "tables"])
    run([sys.executable, "paper/generate_figures.py",
         "--metrics-dir", METRICS, "--audit-dir", AUDITS,
         "--output-dir", OUT / "paper" / "figures"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preprocess", "check", "audit", "train", "paper", "all"),
                        default="check")
    parser.add_argument("--encoders", default="roberta,mistral",
                        help="Encoders to create during preprocessing")
    parser.add_argument("--fnn-windows", default=DEFAULT_WINDOWS,
                        help="FakeNewsNet windows among 1,3,5,10,full")
    parser.add_argument("--baseline-backend", choices=("cpu", "gpu"),
                        default=os.environ.get("BASELINE_BACKEND", "gpu"))
    args = parser.parse_args()
    windows = validate_windows(args.fnn_windows)
    if args.stage == "all" and set(windows) != set(DEFAULT_WINDOWS.split(",")):
        parser.error("--stage all requires the complete FakeNewsNet window set: " + DEFAULT_WINDOWS)
    failure_marker = OUT / "RUN_FAILED.txt"
    if failure_marker.exists():
        failure_marker.unlink()
    try:
        if args.stage in ("preprocess", "all"):
            preprocess(args.encoders)
        if args.stage == "check":
            if not check():
                raise FileNotFoundError("Required source data or embeddings are missing.")
            return
        if args.stage in ("audit", "all"):
            audit_inputs = (EVONS / "evons.csv", FNN / "ordered_fake_propagation_paths.jsonl",
                            FNN / "ordered_real_propagation_paths.jsonl")
            missing = [path for path in audit_inputs if not path.is_file()]
            if missing:
                raise FileNotFoundError("Audit inputs are missing: " + ", ".join(map(str, missing)))
            audit()
        if args.stage in ("train", "all"):
            if not check():
                raise FileNotFoundError("Required source data or embeddings are missing; run --stage preprocess first.")
            train(args.fnn_windows, args.baseline_backend)
        if args.stage == "paper":
            required_audits = [AUDITS / "dataset_audit.json",
                               AUDITS / "future_fake.csv", AUDITS / "future_real.csv"]
            missing_audits = [path for path in required_audits if not path.is_file()]
            if missing_audits:
                raise FileNotFoundError(
                    "Paper generation requires current audits: "
                    + ", ".join(map(str, missing_audits))
                )
            paper_outputs()
        elif args.stage == "all":
            paper_outputs()
    except Exception:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "RUN_FAILED.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"Failed: consult {OUT / 'RUN_FAILED.txt'}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
