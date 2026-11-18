#!/usr/bin/env python3
"""Create EVONS RoBERTa title and description embeddings in CSV row order."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = ROOT / "evons" / "data" / "evons.csv"
DEFAULT_MODEL = "FacebookAI/roberta-base"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def embed_texts(texts: list[str], model_name: str = DEFAULT_MODEL,
                batch_size: int = 64, device: str | None = None) -> torch.Tensor:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not texts:
        raise ValueError("Cannot embed an empty text column")
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(target_device)
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            tokens = tokenizer(
                texts[start:start + batch_size], padding=True, truncation=True,
                return_tensors="pt",
            ).to(target_device)
            chunks.append(model(**tokens).last_hidden_state[:, 0, :].cpu())
    return torch.cat(chunks, dim=0).contiguous()


def generate(csv_path: Path = DEFAULT_CSV, output_dir: Path | None = None,
             model_name: str = DEFAULT_MODEL, batch_size: int = 64,
             device: str | None = None) -> dict:
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
    output_dir.mkdir(parents=True, exist_ok=True)
    title_path = output_dir / "title_embeddings.pt"
    description_path = output_dir / "desc_embeddings.pt"
    torch.save(embed_texts(titles, model_name, batch_size, device), title_path)
    torch.save(embed_texts(descriptions, model_name, batch_size, device), description_path)
    manifest = {
        "dataset": "EVONS", "encoder": model_name,
        "pooling": "first token of final hidden state",
        "batch_size": batch_size, "n_rows": len(frame),
        "row_order": "CSV order; title and description encoded separately",
        "source_csv": str(csv_path), "source_csv_sha256": sha256(csv_path),
        "outputs": {
            title_path.name: {"sha256": sha256(title_path), "shape": list(torch.load(title_path, map_location="cpu", weights_only=True).shape)},
            description_path.name: {"sha256": sha256(description_path), "shape": list(torch.load(description_path, map_location="cpu", weights_only=True).shape)},
        },
    }
    (output_dir / "roberta_embeddings_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    args = parser.parse_args()
    print(json.dumps(generate(args.csv, args.output_dir, args.model,
                             args.batch_size, args.device), indent=2))


if __name__ == "__main__":
    main()
