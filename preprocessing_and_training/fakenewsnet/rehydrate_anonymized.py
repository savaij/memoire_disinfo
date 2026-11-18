#!/usr/bin/env python3
"""Rehydrate anonymized FakeNewsNet propagation paths with the X API v2.

The input contains one story per JSONL line and tweet IDs in ``id_str``.
Available tweets are written in the same story and tweet order to the compact
JSONL format used by the rest of this repository. Unavailable tweets are
omitted; stories with no recovered tweets are omitted as well.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "FakeNewsNet" / "data"
API_URL = "https://api.twitter.com/2/tweets"
TWEET_FIELDS = "created_at,public_metrics,author_id"
EXPANSIONS = "author_id"
USER_FIELDS = "username,public_metrics,verified"
MAX_RETRIES = 8


def chunked(values: list[str], size: int) -> list[list[str]]:
    return [values[start:start + size] for start in range(0, len(values), size)]


def fetch_tweets_by_ids(ids: list[str], bearer_token: str) -> dict[str, Any]:
    """Fetch one X API v2 lookup batch, retrying rate limits and server errors."""
    params = {
        "ids": ",".join(ids),
        "tweet.fields": TWEET_FIELDS,
        "expansions": EXPANSIONS,
        "user.fields": USER_FIELDS,
    }
    headers = {"Authorization": f"Bearer {bearer_token}"}
    for retry in range(MAX_RETRIES + 1):
        response = requests.get(API_URL, headers=headers, params=params, timeout=30)
        if response.status_code == 200:
            return response.json()
        if response.status_code == 429 or response.status_code >= 500:
            if retry == MAX_RETRIES:
                break
            retry_after = response.headers.get("Retry-After")
            try:
                delay = float(retry_after) if retry_after else min(60, 2 ** (retry + 1))
            except ValueError:
                delay = min(60, 2 ** (retry + 1))
            time.sleep(delay)
            continue
        try:
            detail = response.json()
        except ValueError:
            detail = response.text
        raise RuntimeError(f"X API error {response.status_code}: {detail}")
    raise RuntimeError(
        f"X API lookup failed after {MAX_RETRIES + 1} attempts "
        f"(last HTTP status: {response.status_code})"
    )


def map_tweet(tweet: dict[str, Any], user_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    author_id = tweet.get("author_id")
    user = user_by_id.get(str(author_id), {}) if author_id is not None else {}
    public_metrics = tweet.get("public_metrics") or {}
    user_metrics = user.get("public_metrics") or {}
    return {
        "created_at": tweet.get("created_at"),
        "id_str": str(tweet["id"]),
        "text": tweet.get("text"),
        "favorite_count": public_metrics.get("like_count"),
        "user_id": str(author_id) if author_id is not None else None,
        "user_screen_name": user.get("username"),
        "followers_count": user_metrics.get("followers_count"),
        "following_count": user_metrics.get("following_count"),
        "verified": user.get("verified"),
    }


def rehydrate_sequence(sequence: list[dict[str, Any]], bearer_token: str) -> list[dict[str, Any]]:
    ids = [str(item["id_str"]) for item in sequence
           if isinstance(item, dict) and item.get("id_str") is not None]
    if not ids:
        return []

    tweet_by_id: dict[str, dict[str, Any]] = {}
    user_by_id: dict[str, dict[str, Any]] = {}
    for batch in chunked(ids, 100):
        raw = fetch_tweets_by_ids(batch, bearer_token)
        includes = raw.get("includes") or {}
        for user in includes.get("users") or []:
            if "id" in user:
                user_by_id[str(user["id"])] = user
        for tweet in raw.get("data") or []:
            if "id" in tweet:
                tweet_by_id[str(tweet["id"])] = tweet

    # Preserve the anonymized sequence order and discard IDs that the API no
    # longer returns (deleted, protected, or otherwise unavailable tweets).
    return [map_tweet(tweet_by_id[tid], user_by_id) for tid in ids if tid in tweet_by_id]


def rehydrate_category(input_dir: Path, output_dir: Path, category: str,
                       bearer_token: str) -> dict[str, Any]:
    source = input_dir / f"anonymized_{category}_propagation_paths.jsonl"
    destination = output_dir / f"{category}_propagation_paths.jsonl"
    temporary = destination.with_name(destination.name + ".tmp")
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists() or temporary.exists():
        raise FileExistsError(
            f"Refusing to overwrite {destination}; move or remove the existing output first"
        )

    stories_read = stories_written = tweets_written = 0
    try:
        with source.open(encoding="utf-8") as inp, temporary.open("x", encoding="utf-8") as out:
            for line_number, line in enumerate(inp, 1):
                if not line.strip():
                    continue
                sequence = json.loads(line)
                if not isinstance(sequence, list):
                    raise ValueError(f"Expected a JSON array at {source}:{line_number}")
                if any(not isinstance(item, dict) for item in sequence):
                    raise ValueError(f"Expected tweet objects at {source}:{line_number}")
                stories_read += 1
                recovered = rehydrate_sequence(sequence, bearer_token)
                if recovered:
                    out.write(json.dumps(recovered, ensure_ascii=False) + "\n")
                    stories_written += 1
                    tweets_written += len(recovered)
                if stories_read % 100 == 0:
                    print(f"{category}: processed {stories_read} stories", flush=True)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    os.replace(temporary, destination)
    return {
        "category": category,
        "stories_read": stories_read,
        "stories_written": stories_written,
        "tweets_written": tweets_written,
        "stories_without_recovered_tweets": stories_read - stories_written,
        "output": str(destination),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_DATA,
                        help="Directory containing anonymized_{fake,real}_propagation_paths.jsonl")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--categories", nargs="+", choices=("fake", "real"),
                        default=("fake", "real"))
    args = parser.parse_args()
    token = os.environ.get("TWITTER_BEARER_TOKEN")
    if not token:
        parser.error("Set TWITTER_BEARER_TOKEN to a valid X API bearer token")
    input_dir, output_dir = args.input_dir.resolve(), args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = [input_dir / f"anonymized_{category}_propagation_paths.jsonl"
               for category in args.categories]
    destinations = [output_dir / f"{category}_propagation_paths.jsonl"
                    for category in args.categories]
    missing = [path for path in sources if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing anonymized inputs:\n" + "\n".join(map(str, missing)))
    possible_outputs = destinations + [path.with_name(path.name + ".tmp")
                                       for path in destinations]
    existing = [path for path in possible_outputs if path.exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing outputs:\n" +
                              "\n".join(map(str, existing)))
    try:
        for category in args.categories:
            print(json.dumps(rehydrate_category(input_dir, output_dir, category, token), indent=2))
    except Exception:
        # Keep fake/real outputs as a consistent pair if the second category
        # cannot be rehydrated. The preflight above ensures these are ours.
        for path in destinations:
            path.unlink(missing_ok=True)
            path.with_name(path.name + ".tmp").unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    main()
