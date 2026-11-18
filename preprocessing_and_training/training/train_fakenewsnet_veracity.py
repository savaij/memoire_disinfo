"""Train and evaluate the FakeNewsNet paper classifiers.

Both tasks use the same training code and read their text inputs from
precomputed tweet embeddings. Run this script for veracity and
train_fakenewsnet_virality.py for total-engagement classification.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "FakeNewsNet" / "data"
EMBEDDING_DIR = DATA_DIR / "fake_news_net"
OUTPUT_DIR = Path(os.environ.get("PAPER_METRICS_DIR", ROOT / "results" / "metrics"))
sys.path.insert(0, str(ROOT / "metrics_and_audits"))

from baseline_backend import build_baselines, fit_evaluate, require_gpu, resolve_backend
from fakenewsnet_prefix import (
    DEFAULT_WINDOWS,
    make_splits,
    prepare_features,
    split_fingerprint,
    validate_windows,
)
from imbalanced_evaluation import evaluate_with_calibration


CATEGORIES = ("fake", "real")
ENCODERS = {"bert": "", "mistral": "_mistral"}
def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def prepare_inputs(windows, task, quantile):
    path_files = {
        category: DATA_DIR / f"ordered_{category}_propagation_paths.jsonl"
        for category in CATEGORIES
    }
    embedding_files = {
        encoder: {
            category: EMBEDDING_DIR / f"ordered_{category}_propagation_paths_emb{suffix}.pt"
            for category in CATEGORIES
        }
        for encoder, suffix in ENCODERS.items()
    }
    required = list(path_files.values())
    for files in embedding_files.values():
        required.extend(files.values())
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing FakeNewsNet inputs:\n" + "\n".join(missing))

    paths = {category: read_jsonl(path) for category, path in path_files.items()}
    prepared = {}
    reference_y, reference_metadata = None, None
    for encoder, files in embedding_files.items():
        embeddings = {
            category: torch.load(path, map_location="cpu")
            for category, path in files.items()
        }
        features, y, metadata = prepare_features(
            paths, embeddings, windows, task=task, quantile=quantile,
        )
        if reference_y is None:
            reference_y, reference_metadata = y, metadata
        elif not np.array_equal(y, reference_y) or metadata != reference_metadata:
            raise ValueError("Encoder cohorts or targets differ; comparison aborted")
        prepared[encoder] = features
        del embeddings
    return prepared, reference_y, reference_metadata


class FeatureDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.X[index], self.y[index]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def predict_probabilities(model, loader, device):
    batches = []
    with torch.no_grad():
        for features, _ in loader:
            batches.append(torch.sigmoid(model(features.to(device))).cpu().numpy())
    return np.concatenate(batches)


def run_mlp_cv(X, y, splits, epochs=12, batch_size=64, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []

    for fold, (fit, cal, test) in enumerate(splits, 1):
        np.random.seed(2000 + fold)
        torch.manual_seed(2000 + fold)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(2000 + fold)
        model = MLP(input_dim=X.shape[1]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()

        fit_loader = DataLoader(
            FeatureDataset(X[fit], y[fit]), batch_size=batch_size, shuffle=True,
        )
        cal_loader = DataLoader(FeatureDataset(X[cal], y[cal]), batch_size=batch_size)
        test_loader = DataLoader(FeatureDataset(X[test], y[test]), batch_size=batch_size)

        model.train()
        for _ in range(epochs):
            for features, labels in fit_loader:
                features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = loss_fn(model(features), labels)
                loss.backward()
                optimizer.step()

        model.eval()
        metrics = evaluate_with_calibration(
            y[test], predict_probabilities(model, test_loader, device),
            y[cal], predict_probabilities(model, cal_loader, device),
        )
        metrics["n_fit"] = len(fit)
        metrics["n_calibration"] = len(cal)
        metrics["fold"] = fold
        metrics["split_fingerprint"] = split_fingerprint(fit, cal, test)
        rows.append(metrics)

    return pd.DataFrame(rows)


def run_baselines_cv(X, y, splits, backend, label):
    X_numpy = X.numpy()
    models = build_baselines(backend=backend, logreg_random_state=42)
    rows = []
    for name, model in models.items():
        for fold, (fit, cal, test) in enumerate(splits, 1):
            metrics = fit_evaluate(
                model, X_numpy, y, fit, cal, test, backend=backend,
                label=f"{label} {name} fold={fold}",
            )
            metrics["n_fit"] = len(fit)
            metrics["n_calibration"] = len(cal)
            metrics["model"] = name
            metrics["fold"] = fold
            metrics["split_fingerprint"] = split_fingerprint(fit, cal, test)
            rows.append(metrics)

    return pd.DataFrame(rows)


def write_results(all_results, task, output_name):
    out_dir = OUTPUT_DIR / output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results.to_csv(out_dir / f"fakenewsnet_{task}_merged_cv_results.csv", index=False)
    print("Saved results to", out_dir)


def run_experiment(task):
    backend = resolve_backend()
    if backend == "gpu":
        require_gpu()
    windows = validate_windows(os.environ.get("FNN_WINDOWS", DEFAULT_WINDOWS))
    quantile = float(os.environ.get("VIRALITY_QUANTILE", "0.5")) if task == "virality" else 0.5

    # Validate both encoders and make one set of splits before fitting any model.
    prepared, y, metadata = prepare_inputs(windows, task, quantile)
    splits = make_splits(y)
    results = []
    for encoder, features in prepared.items():
        print(encoder, metadata)
        for window in windows:
            X = torch.from_numpy(features[window])
            print(f"Running {encoder}, window={window}, shape={tuple(X.shape)}", flush=True)
            mlp_results = run_mlp_cv(X, y, splits)
            mlp_results["model"] = f"mlp_{encoder}"
            baseline_results = run_baselines_cv(
                X, y, splits, backend, label=f"FakeNewsNet {encoder} window={window}",
            )
            baseline_results["model"] += f"_{encoder}"
            for frame in (mlp_results, baseline_results):
                frame["observation_window"] = "full" if window == "full" else f"k{window}"
                for key, value in metadata.items():
                    frame[key] = value
                results.append(frame)

    output_name = f"fakenewsnet_{task}_prefix"
    if task == "virality":
        output_name += f"_q{round(quantile * 100)}"
    write_results(pd.concat(results, ignore_index=True), task, output_name)


if __name__ == "__main__":
    run_experiment("disinformation")
