#!/usr/bin/env python3
"""Create EVONS Mistral title and description embeddings in CSV row order."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm
try:
    from mistralai import Mistral
except ImportError:  # mistralai >=3 exposes the client in this submodule
    from mistralai.client import Mistral


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = ROOT / "evons" / "data" / "evons.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compute_embeddings(client, texts: list[str], batch_size: int) -> torch.Tensor:
    vectors = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Mistral embeddings"):
        response = client.embeddings.create(model="mistral-embed", inputs=texts[start:start + batch_size])
        batch = list(response.data)
        expected = min(batch_size, len(texts) - start)
        if len(batch) != expected:
            raise ValueError(f"Mistral returned {len(batch)} vectors; expected {expected}")
        if batch and all(getattr(item, "index", None) is not None for item in batch):
            batch.sort(key=lambda item: item.index)
        vectors.extend(item.embedding for item in batch)
    return torch.tensor(vectors, dtype=torch.float32).contiguous()


def generate(csv_path: Path = DEFAULT_CSV, output_dir: Path | None = None,
             batch_size: int = 256) -> dict:
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("Set MISTRAL_API_KEY to generate Mistral embeddings")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    csv_path = csv_path.resolve()
    output_dir = (output_dir or csv_path.parent).resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)
    frame = pd.read_csv(csv_path)
    required = {"title", "description"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing EVONS columns: {sorted(missing)}")
    titles = frame["title"].fillna("").astype(str).tolist()
    descriptions = frame["description"].fillna("").astype(str).tolist()
    client = Mistral(api_key=api_key)
    output_dir.mkdir(parents=True, exist_ok=True)
    title_path = output_dir / "title_embeddings_mistral.pt"
    description_path = output_dir / "desc_embeddings_mistral.pt"
    torch.save(compute_embeddings(client, titles, batch_size), title_path)
    torch.save(compute_embeddings(client, descriptions, batch_size), description_path)
    manifest = {
        "dataset": "EVONS", "encoder": "mistral-embed",
        "pooling": "Mistral embedding API", "batch_size": batch_size,
        "n_rows": len(frame),
        "row_order": "CSV order; title and description encoded separately",
        "source_csv": str(csv_path), "source_csv_sha256": sha256(csv_path),
        "api_model_is_versioned": False,
        "outputs": {
            title_path.name: {"sha256": sha256(title_path), "shape": list(torch.load(title_path, map_location="cpu", weights_only=True).shape)},
            description_path.name: {"sha256": sha256(description_path), "shape": list(torch.load(description_path, map_location="cpu", weights_only=True).shape)},
        },
    }
    (output_dir / "mistral_embeddings_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(generate(args.csv, args.output_dir, args.batch_size), indent=2))


if __name__ == "__main__":
    main()
