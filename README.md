# M2 Mémoire – Disinformation Detection & Virality Prediction

This repository contains the codebase for the master thesis (M2 « Humanités Numériques », École nationale des chartes) studying automatic classification of (a) fake vs. real news and (b) viral vs. non‑viral items / propagations across two different news datasets. 

## 1. Tasks

| Task | Definition | Label Source |
|------|------------|--------------|
| Disinformation Detection | Binary classification: fake vs real news item / propagation | Dataset ground‑truth labels (Evons & FakeNewsNet-Politifact) |
| Virality Prediction | Binary classification: will an item / propagation be “viral” | Evons: ≥ 95th percentile Facebook engagement. FakeNewsNet: ≥ median total likes across propagations |

## 2. Datasets

### Evons Dataset
Static news articles with metadata and engagement statistics (e.g. Facebook). Modeling treats each article independently (non‑sequential). Text = title + caption/description. Virality label derived from high‑end engagement threshold (95th percentile). See `evons/data/readme.md` for download & embedding links.

### FakeNewsNet (Politifact subset)
Twitter propagation trees. Each propagation becomes a sequence of per‑tweet features: text embedding + scalar metadata (verification flag, follower/following counts, favorites, elapsed time, etc.). Virality defined via median total likes threshold. Raw data access subject to Twitter/X restrictions — see `FakeNewsNet/data/readme.md`.

## 3. Repository Overview

```
memoire_disinfo/
├── evons/
│   ├── data/                        # Evons raw data & precomputed embeddings (external download)
│   ├── disinformation_detection/    # MLP variants (RoBERTa & Mistral)
│   └── virality_prediction/         # MLP baseline + source / engagement feature variants
│
└── FakeNewsNet/
    ├── data/                        # Politifact sequences & embeddings (external download)
    ├── data_preprocessing/          # Scripts: ordering, path creation, embedding generation
    ├── disinformation_detection/    # Sequence model notebooks (CNN/RNN/GRU/LSTM/Transformer)
    └── virality_prediction/         # Same architectures for virality label
```

Both dataset folders intentionally mirror a two‑task layout for clarity and comparability.


## 4. Data Access & Privacy

Data are **not** committed because of size & licensing:
* Evons: follow upstream instructions; precomputed embeddings via provided Drive folder.
* FakeNewsNet Politifact: raw data requires permission / contact; processed sequences & embeddings via protected Drive link. Respect Twitter/X terms for any redistribution.

See each dataset’s `data/readme.md` for authoritative links and any required credentials or requests.


## 5. How To Reproduce

1. Obtain & place data as per `evons/data/readme.md` and `FakeNewsNet/data/readme.md`.
2. (Optional) For FakeNewsNet: regenerate using scripts in `FakeNewsNet/data_preprocessing/` (`path_creation.py`, `ordering_data.py`, `create_embeddings.py`, `create_embeddings_mistral.py`). <br> Evons notebook already provide code for embedding texts on-the-fly if not available in `evons/data` folder. If you download already processed data, you can skip this step.
3. Open the relevant notebook (e.g., `evons/disinformation_detection/MLP.ipynb`) and execute cells top‑to‑bottom. Notebooks are self‑contained (data paths assume relative placement inside each dataset’s `data/`).
4. Compare output metrics across variants.

### Suggested Python Environment 
The following packages are required:
`torch`, `transformers`, `scikit-learn`, `pandas`, `numpy`, `tqdm`, `matplotlib`, `seaborn`, `wandb`. 

`mistralai` required for Mistral embedding generation.

## 6. Design Rationale (Dataset‑Specific Modeling)
* Evons articles lack temporal structure → simple MLP suffices; experimentation focuses on integrating minimal, interpretable source/engagement signals.
* FakeNewsNet propagations capture temporal diffusion dynamics → sequence models (CNN for local n‑gram‑like temporal patterns, RNN/GRU/LSTM for recurrent dependencies, Transformer for global self‑attention) allow exploring how representation choice influences prediction under short sequences.

## 7. Folder Quick Reference

| Path | Purpose |
|------|---------|
| `evons/disinformation_detection/` | Article fake vs real (MLP variants) |
| `evons/virality_prediction/` | Article virality (MLP + feature fusion variants) |
| `FakeNewsNet/data_preprocessing/` | Build tweet sequences & embeddings |
| `FakeNewsNet/disinformation_detection/` | Propagation fake vs real (sequence models) |
| `FakeNewsNet/virality_prediction/` | Propagation virality (sequence models) |


---
For questions or access issues (e.g., processed data links), contact the author.
