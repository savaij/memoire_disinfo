# EVONS inputs

EVONS training uses article labels and metadata from the dataset CSV, plus title and description embeddings. The CSV is required for either input route because embeddings do not contain the target labels or audit fields.

## Get the dataset CSV

Follow the curator's instructions in the [EVONS repository](https://github.com/krstovski/evons) and place `evons.csv` here:

```text
evons/data/evons.csv
```

## Option 1: generate embeddings from the CSV

This is the default. Run the pipeline with the encoders you want to generate:

```bash
python preprocessing_and_training/preprocess_datasets.py \
  --dataset evons --evons-embeddings generate
```

RoBERTa embeddings are generated locally. Mistral embeddings use the Mistral API and require `MISTRAL_API_KEY`.

## Option 2: reuse precomputed embeddings

Download the embeddings from [the shared folder](https://drive.google.com/drive/folders/1X27WjPEKzAcC5jXai8cuI8FEmWkjJb6l?usp=sharing) and put all four files in `evons/data/embeddings/`:

- `title_embeddings.pt` and `desc_embeddings.pt` (RoBERTa)
- `title_embeddings_mistral.pt` and `desc_embeddings_mistral.pt` (Mistral)

Then run:

```bash
python preprocessing_and_training/preprocess_datasets.py \
  --dataset evons --evons-embeddings precomputed \
  --evons-embedding-dir evons/data/embeddings
```

The `.pt` files must have one row per CSV record, in the same row order. For the complete paper training run, provide both encoders. The runner's `--evons-embeddings precomputed` and `--evons-embedding-dir evons/data/embeddings` options select this route when running the full pipeline.
