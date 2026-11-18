#!/usr/bin/env python3
"""Build one compact JSONL propagation per recovered PolitiFact story.

Input is the already recovered FakeNewsNet directory tree, not a Twitter/X
collection or rehydration step. Story and tweet files are traversed in stable
lexicographic order so the resulting row order can be recorded and reused.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "FakeNewsNet" / "data"
TWEET_FIELDS = ("created_at", "id_str", "text", "favorite_count")
USER_FIELDS = {
    "id_str": "user_id",
    "screen_name": "user_screen_name",
    "followers_count": "followers_count",
    "friends_count": "following_count",
    "verified": "verified",
}


def compact_tweet(tweet: dict, source: Path) -> dict:
    result = {key: tweet[key] for key in TWEET_FIELDS if key in tweet}
    user = tweet.get("user")
    if isinstance(user, dict):
        result.update({new: user[old] for old, new in USER_FIELDS.items() if old in user})
    missing = {"created_at", "id_str", "text"} - result.keys()
    if missing:
        raise ValueError(f"{source}: missing required tweet fields {sorted(missing)}")
    return result


def extract_category(input_root: Path, category: str, output_path: Path) -> dict:
    category_root = input_root / category
    if not category_root.is_dir():
        raise FileNotFoundError(category_root)
    story_dirs = sorted((path for path in category_root.iterdir() if path.is_dir()),
                        key=lambda path: path.name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_stories = n_tweets = 0
    with output_path.open("w", encoding="utf-8") as out:
        for story_dir in tqdm(story_dirs, desc=f"Extract {category}"):
            tweets_dir = story_dir / "tweets"
            if not tweets_dir.is_dir():
                continue
            tweet_files = sorted(tweets_dir.glob("*.json"), key=lambda path: path.name)
            if not tweet_files:
                continue
            tweets = []
            for tweet_file in tweet_files:
                with tweet_file.open(encoding="utf-8") as stream:
                    tweets.append(compact_tweet(json.load(stream), tweet_file))
            out.write(json.dumps(tweets, ensure_ascii=False) + "\n")
            n_stories += 1
            n_tweets += len(tweets)
    return {"category": category, "story_directories": len(story_dirs),
            "stories_written": n_stories, "tweets_written": n_tweets,
            "story_order": "lexicographic directory name",
            "tweet_file_order": "lexicographic filename", "output": str(output_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path,
                        default=DEFAULT_DATA / "raw_data" / "politifact")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--categories", nargs="+", choices=("fake", "real"),
                        default=("fake", "real"))
    args = parser.parse_args()
    for category in args.categories:
        result = extract_category(
            args.raw_root.resolve(), category,
            args.output_dir.resolve() / f"{category}_propagation_paths.jsonl",
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
