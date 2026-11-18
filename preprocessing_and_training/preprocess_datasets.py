#!/usr/bin/env python3
"""Prepare EVONS and FakeNewsNet inputs for the current paper pipeline.

EVONS embeddings can be generated from its CSV or supplied as precomputed
files. FakeNewsNet can start from recovered raw tweet files or anonymized
propagation paths that this script rehydrates through the X API.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(*args: str | Path) -> None:
    command = [sys.executable, *(str(arg) for arg in args)]
    print("$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def preprocess_evons(csv: Path, output_dir: Path, encoders: tuple[str, ...],
                     embedding_source: str) -> None:
    if embedding_source == "precomputed":
        required = {
            "roberta": ("title_embeddings.pt", "desc_embeddings.pt"),
            "mistral": ("title_embeddings_mistral.pt", "desc_embeddings_mistral.pt"),
        }
        missing = [output_dir / name for encoder in encoders
                   for name in required[encoder] if not (output_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing precomputed EVONS embeddings:\n" + "\n".join(map(str, missing))
            )
        if not csv.is_file():
            raise FileNotFoundError(
                f"EVONS CSV is still required for labels, metadata, and audits: {csv}"
            )
        print(f"Using precomputed EVONS embeddings from {output_dir}", flush=True)
        return
    if "roberta" in encoders:
        run("preprocessing_and_training/evons/generate_roberta_embeddings.py", "--csv", csv,
            "--output-dir", output_dir)
    if "mistral" in encoders:
        run("preprocessing_and_training/evons/generate_mistral_embeddings.py", "--csv", csv,
            "--output-dir", output_dir)


def preprocess_fakenewsnet(raw_root: Path, data_dir: Path,
                            output_dir: Path, encoders: tuple[str, ...],
                            categories: tuple[str, ...], source: str,
                            anonymized_dir: Path) -> None:
    if source == "raw":
        run("preprocessing_and_training/fakenewsnet/extract_propagation_paths.py", "--raw-root", raw_root,
            "--output-dir", data_dir, "--categories", *categories)
    else:
        run("preprocessing_and_training/fakenewsnet/rehydrate_anonymized.py",
            "--input-dir", anonymized_dir, "--output-dir", data_dir,
            "--categories", *categories)
    run("preprocessing_and_training/fakenewsnet/sort_tweet_sequences.py", "--data-dir", data_dir,
        "--categories", *categories)
    if "roberta" in encoders:
        run("preprocessing_and_training/fakenewsnet/generate_roberta_embeddings.py", "--data-dir", data_dir,
            "--output-dir", output_dir, "--categories", *categories)
    if "mistral" in encoders:
        run("preprocessing_and_training/fakenewsnet/generate_mistral_embeddings.py", "--data-dir", data_dir,
            "--output-dir", output_dir, "--categories", *categories)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("evons", "fakenewsnet", "both"), default="both")
    parser.add_argument("--encoders", default="roberta,mistral",
                        help="Comma-separated subset of roberta,mistral")
    parser.add_argument("--evons-embeddings", choices=("generate", "precomputed"),
                        default="generate",
                        help="Generate embeddings from the EVONS CSV or reuse existing .pt files")
    parser.add_argument("--fnn-source", choices=("raw", "anonymized"), default="raw",
                        help="Use recovered raw tweet files or rehydrate anonymized tweet IDs")
    parser.add_argument("--evons-csv", type=Path, default=ROOT / "evons/data/evons.csv")
    parser.add_argument("--evons-embedding-dir", "--evons-output-dir", dest="evons_embedding_dir",
                        type=Path, default=ROOT / "evons/data")
    parser.add_argument("--fnn-raw-root", type=Path,
                        default=ROOT / "FakeNewsNet/data/raw_data/politifact")
    parser.add_argument("--fnn-data-dir", type=Path, default=ROOT / "FakeNewsNet/data")
    parser.add_argument("--fnn-anonymized-dir", type=Path, default=ROOT / "FakeNewsNet/data")
    parser.add_argument("--fnn-output-dir", type=Path,
                        default=ROOT / "FakeNewsNet/data/fake_news_net")
    parser.add_argument("--categories", nargs="+", choices=("fake", "real"),
                        default=("fake", "real"))
    args = parser.parse_args()
    encoders = tuple(part.strip().lower() for part in args.encoders.split(",") if part.strip())
    if not encoders or set(encoders) - {"roberta", "mistral"} or len(set(encoders)) != len(encoders):
        parser.error("--encoders must be a duplicate-free subset of roberta,mistral")
    if args.dataset in ("evons", "both"):
        preprocess_evons(args.evons_csv.resolve(), args.evons_embedding_dir.resolve(),
                         encoders, args.evons_embeddings)
    if args.dataset in ("fakenewsnet", "both"):
        preprocess_fakenewsnet(args.fnn_raw_root.resolve(), args.fnn_data_dir.resolve(),
                               args.fnn_output_dir.resolve(), encoders,
                               tuple(args.categories), args.fnn_source,
                               args.fnn_anonymized_dir.resolve())


if __name__ == "__main__":
    main()
