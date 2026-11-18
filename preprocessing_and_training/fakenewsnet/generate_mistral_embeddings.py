#!/usr/bin/env python3
"""Create per-tweet Mistral embeddings for ordered FakeNewsNet JSONL files."""
from __future__ import annotations

import argparse
from pathlib import Path

from embedding_utils import generate


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "FakeNewsNet" / "data"
DEFAULT_OUTPUT = DEFAULT_DATA / "fake_news_net"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--categories", nargs="+", choices=("fake", "real"),
                        default=("fake", "real"))
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    data_dir, output_dir = args.data_dir.resolve(), args.output_dir.resolve()
    for category in args.categories:
        manifest = generate(
            data_dir / f"ordered_{category}_propagation_paths.jsonl",
            output_dir, "mistral", batch_size=args.batch_size,
        )
        print(manifest)


if __name__ == "__main__":
    main()
