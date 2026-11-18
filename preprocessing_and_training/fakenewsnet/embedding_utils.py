"""Shared per-tweet embedding generation for the paper's two encoders."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import torch
from tqdm.auto import tqdm


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[list[dict]]:
    with path.open(encoding="utf-8") as stream:
        sequences = [json.loads(line) for line in stream if line.strip()]
    if any(not isinstance(sequence, list) for sequence in sequences):
        raise ValueError(f"Expected a JSON array on each line of {path}")
    return sequences


def flatten_texts(sequences: list[list[dict]]) -> tuple[list[str], list[int]]:
    texts, lengths = [], []
    for story_index, sequence in enumerate(sequences):
        if not sequence:
            raise ValueError(f"Story {story_index} is empty; remove it during extraction")
        lengths.append(len(sequence))
        for tweet_index, tweet in enumerate(sequence):
            text = tweet.get("text")
            if not isinstance(text, str):
                raise ValueError(f"Story {story_index}, tweet {tweet_index}: text is not a string")
            texts.append(text)
    return texts, lengths


def roberta_embeddings(texts: list[str], model_name: str, batch_size: int,
                       device: str | None) -> torch.Tensor:
    from transformers import AutoModel, AutoTokenizer

    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(target_device)
    model.eval()
    chunks = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(texts), batch_size), desc="RoBERTa embeddings"):
            tokens = tokenizer(texts[start:start + batch_size], padding=True,
                               truncation=True, return_tensors="pt").to(target_device)
            chunks.append(model(**tokens).last_hidden_state[:, 0, :].cpu())
    if not chunks:
        raise ValueError("No tweets found in input sequences")
    return torch.cat(chunks, dim=0).contiguous()


def mistral_embeddings(texts: list[str], batch_size: int) -> torch.Tensor:
    try:
        from mistralai import Mistral
    except ImportError:  # mistralai >=3 exposes the client in this submodule
        from mistralai.client import Mistral

    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("Set MISTRAL_API_KEY to generate Mistral embeddings")
    client = Mistral(api_key=api_key)
    vectors = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Mistral embeddings"):
        response = client.embeddings.create(model="mistral-embed", inputs=texts[start:start + batch_size])
        batch = list(response.data)
        if len(batch) != min(batch_size, len(texts) - start):
            raise ValueError("Mistral returned an unexpected number of embeddings")
        # API objects include input indices in current SDKs. Use them when
        # available; otherwise preserve the response's documented input order.
        if batch and all(getattr(item, "index", None) is not None for item in batch):
            batch.sort(key=lambda item: item.index)
        vectors.extend(item.embedding for item in batch)
    if not vectors:
        raise ValueError("No tweets found in input sequences")
    return torch.tensor(vectors, dtype=torch.float32).contiguous()


def generate(input_path: Path, output_dir: Path, encoder: str, *,
             batch_size: int | None = None, model_name: str | None = None,
             device: str | None = None) -> dict:
    if encoder not in {"roberta", "mistral"}:
        raise ValueError("encoder must be roberta or mistral")
    default_batch = 32 if encoder == "roberta" else 256
    batch_size = default_batch if batch_size is None else batch_size
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    input_path, output_dir = input_path.resolve(), output_dir.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    sequences = read_jsonl(input_path)
    texts, lengths = flatten_texts(sequences)
    model = model_name or ("FacebookAI/roberta-base" if encoder == "roberta" else "mistral-embed")
    if encoder == "roberta":
        flat = roberta_embeddings(texts, model, batch_size, device)
    else:
        flat = mistral_embeddings(texts, batch_size)
    if len(flat) != len(texts):
        raise ValueError(f"Embedded {len(flat)} tweets, expected {len(texts)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    category = input_path.name.removeprefix("ordered_").removesuffix("_propagation_paths.jsonl")
    suffix = "_mistral" if encoder == "mistral" else ""
    output_path = output_dir / f"ordered_{category}_propagation_paths_emb{suffix}.pt"
    nested, start = [], 0
    for length in lengths:
        nested.append([row.cpu() for row in flat[start:start + length]])
        start += length
    torch.save(nested, output_path)
    manifest = {
        "dataset": "FakeNewsNet PolitiFact", "category": category,
        "encoder": model, "pooling": "first token of final hidden state" if encoder == "roberta" else "API sentence embedding",
        "batch_size": batch_size, "n_stories": len(sequences), "n_tweets": len(texts),
        "row_order": "story JSONL order and tweet order within each story",
        "source_jsonl": str(input_path), "source_jsonl_sha256": sha256(input_path),
        "embedding_file": str(output_path), "embedding_sha256": sha256(output_path),
        "embedding_dimension": int(flat.shape[1]),
    }
    manifest_path = output_dir / f"{category}_{encoder}_embeddings_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
