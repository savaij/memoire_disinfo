"""Shared EVONS CSV and embedding loading for the current paper tasks."""
from __future__ import annotations

from pathlib import Path

import torch


EMBEDDING_FILES = {
    "roberta": ("title_embeddings.pt", "desc_embeddings.pt", 768),
    "mistral": ("title_embeddings_mistral.pt", "desc_embeddings_mistral.pt", 1024),
}


def load_embeddings(data_dir: Path, encoder: str, *, expected_rows: int | None = None):
    if encoder not in EMBEDDING_FILES:
        raise ValueError(f"Unknown EVONS encoder: {encoder}")
    title_name, description_name, expected_dimension = EMBEDDING_FILES[encoder]
    title = torch.load(data_dir / title_name, map_location="cpu")
    description = torch.load(data_dir / description_name, map_location="cpu")
    title = torch.as_tensor(title, dtype=torch.float32)
    description = torch.as_tensor(description, dtype=torch.float32)
    if title.ndim != 2 or description.ndim != 2:
        raise ValueError(f"{encoder} embeddings must both be 2D matrices")
    if title.shape != description.shape:
        raise ValueError(f"{encoder} title/description embedding shapes differ: "
                         f"{tuple(title.shape)} != {tuple(description.shape)}")
    if title.shape[1] != expected_dimension:
        raise ValueError(f"{encoder} embeddings have width {title.shape[1]}; "
                         f"expected {expected_dimension}")
    if expected_rows is not None and title.shape[0] != expected_rows:
        raise ValueError(f"{encoder} embeddings have {title.shape[0]} rows; "
                         f"CSV has {expected_rows}")
    if not torch.isfinite(title).all() or not torch.isfinite(description).all():
        raise ValueError(f"{encoder} embeddings contain non-finite values")
    return title, description
