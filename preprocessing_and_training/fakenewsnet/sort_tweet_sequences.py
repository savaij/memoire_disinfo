#!/usr/bin/env python3
"""Order each recovered tweet sequence chronologically, with a stable tie-break."""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "FakeNewsNet" / "data"


def parse_created_at(value: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"created_at must be a string, got {type(value).__name__}")
    try:
        return datetime.strptime(value, "%a %b %d %H:%M:%S %z %Y")
    except ValueError as exc:
        raise ValueError(f"Unsupported FakeNewsNet created_at value: {value!r}") from exc


def order_file(source: Path, destination: Path) -> dict:
    n_stories = n_tweets = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open(encoding="utf-8") as inp, destination.open("w", encoding="utf-8") as out:
        for line_number, line in enumerate(inp, 1):
            if not line.strip():
                continue
            story = json.loads(line)
            if not isinstance(story, list):
                raise ValueError(f"{source}:{line_number}: expected a JSON array")
            story.sort(key=lambda tweet: (parse_created_at(tweet["created_at"]),
                                          str(tweet.get("id_str", ""))))
            out.write(json.dumps(story, ensure_ascii=False) + "\n")
            n_stories += 1
            n_tweets += len(story)
    return {"input": str(source), "output": str(destination),
            "stories": n_stories, "tweets": n_tweets,
            "tweet_order": "created_at ascending, then id_str ascending"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--categories", nargs="+", choices=("fake", "real"),
                        default=("fake", "real"))
    args = parser.parse_args()
    data_dir = args.data_dir.resolve()
    for category in args.categories:
        result = order_file(
            data_dir / f"{category}_propagation_paths.jsonl",
            data_dir / f"ordered_{category}_propagation_paths.jsonl",
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
