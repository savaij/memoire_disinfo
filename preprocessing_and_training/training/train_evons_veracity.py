"""Train the six EVONS veracity classifiers used in the current paper."""

from pathlib import Path
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'metrics_and_audits'))
from imbalanced_evaluation import evaluate_with_calibration
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from baseline_backend import build_baselines, fit_evaluate, resolve_backend, require_gpu
from evons_inputs import load_embeddings

DATA_DIR = ROOT / 'evons' / 'data'
CSV_PATH = DATA_DIR / 'evons.csv'


# %%

def load_disinfo_data(csv_path: Path):
    df = pd.read_csv(csv_path)
    df['title'] = df['title'].fillna('')
    df['description'] = df['description'].fillna('')
    df = df[['title', 'description', 'is_fake']]
    y = df['is_fake'].astype(int).to_numpy()
    return df, y


class PairDataset(Dataset):
    def __init__(self, title_emb, desc_emb, y):
        self.t = title_emb
        self.d = desc_emb
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.cat([self.t[idx], self.d[idx]], dim=0), self.y[idx]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


# %%

def train_eval_mlp(title_emb, desc_emb, y, epochs=12, batch_size=64, lr=1e-3, n_splits=10):
    X = torch.cat([title_emb, desc_emb], dim=1)
    skf = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fold_metrics = []
    for fold, (tr, va) in enumerate(skf.split(X.numpy(), y), 1):
        fit, cal = train_test_split(tr, test_size=0.25, stratify=y[tr], random_state=1000 + fold)
        model = MLP(input_dim=X.shape[1]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.BCEWithLogitsLoss()

        train_ds = PairDataset(title_emb[fit], desc_emb[fit], y[fit])
        val_ds = PairDataset(title_emb[va], desc_emb[va], y[va])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        cal_loader = DataLoader(PairDataset(title_emb[cal], desc_emb[cal], y[cal]), batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        model.train()
        for _ in tqdm(range(epochs), desc=f'Fold {fold}/{n_splits}'):
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()

        model.eval()
        def predict_prob(loader):
            values = []
            with torch.no_grad():
                for xb, _ in loader:
                    values.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
            return np.concatenate(values)
        m = evaluate_with_calibration(y[va], predict_prob(val_loader),
                                      y[cal], predict_prob(cal_loader))
        m['n_fit'] = len(fit)
        m['n_calibration'] = len(cal)
        m['fold'] = fold
        fold_metrics.append(m)

    return pd.DataFrame(fold_metrics)


def run_baselines(title_emb, desc_emb, y, n_splits=10, label="EVONS"):
    X = torch.cat([title_emb, desc_emb], dim=1).numpy()
    skf = StratifiedShuffleSplit(n_splits=n_splits, test_size=0.2, random_state=42)

    models = build_baselines()

    rows = []
    for name, model in models.items():
        for fold, (tr, va) in enumerate(skf.split(X, y), 1):
            fit, cal = train_test_split(tr, test_size=0.25, stratify=y[tr], random_state=1000 + fold)
            m = fit_evaluate(model, X, y, fit, cal, va,
                             backend=resolve_backend(),
                             label=f'{label} {name} fold={fold}')
            m['n_fit'] = len(fit)
            m['n_calibration'] = len(cal)
            m['model'] = name
            m['fold'] = fold
            rows.append(m)
    return pd.DataFrame(rows)


# %%

def main():
    if resolve_backend() == "gpu":
        require_gpu()

    required = [
        CSV_PATH,
        DATA_DIR / 'title_embeddings.pt',
        DATA_DIR / 'desc_embeddings.pt',
        DATA_DIR / 'title_embeddings_mistral.pt',
        DATA_DIR / 'desc_embeddings_mistral.pt',
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError('Fichiers manquants\n' + '\n'.join(missing))
    print('All required files found.')

    df, y = load_disinfo_data(CSV_PATH)
    print('Rows:', len(df), '| Positive rate:', y.mean().round(4))

    results = []
    for mode in ['bert', 'mistral']:
        print(f'\n=== Running MLP ({mode}) ===')
        encoder = 'roberta' if mode == 'bert' else mode
        t, d = load_embeddings(DATA_DIR, encoder, expected_rows=len(y))
        mlp_df = train_eval_mlp(t, d, y)
        mlp_df['model'] = f'mlp_{mode}'
        results.append(mlp_df)

        base_df = run_baselines(t, d, y, label=f"EVONS {mode}")
        base_df['model'] = base_df['model'] + f'_{mode}'
        results.append(base_df)

    all_results = pd.concat(results, ignore_index=True)
    output_dir = Path(os.environ.get('PAPER_METRICS_DIR', ROOT / 'results' / 'metrics'))
    output_dir.mkdir(parents=True, exist_ok=True)
    all_results.to_csv(output_dir / 'evons_disinformation_merged_cv_results.csv', index=False)
    print('Saved results to', output_dir)


if __name__ == "__main__":
    main()



