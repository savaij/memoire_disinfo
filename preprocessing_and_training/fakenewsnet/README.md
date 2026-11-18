# FakeNewsNet PolitiFact inputs

The pipeline accepts either the recovered per-tweet files used by the current workflow or anonymized propagation paths that can be rehydrated through the X API. The resulting ordered JSONL files and embeddings are local working data and are not included in this repository.

## Option 1: recovered raw tweets

Follow the collection instructions in the [FakeNewsNet repository](https://github.com/KaiDMML/FakeNewsNet), then place the PolitiFact tweet files in this structure:

```text
FakeNewsNet/data/raw_data/politifact/
├── fake/<story>/tweets/*.json
└── real/<story>/tweets/*.json
```

This is the default input route. The extractor compacts the tweet fields and the next step orders each story chronologically:

```bash
python preprocessing_and_training/preprocess_datasets.py \
  --dataset fakenewsnet --fnn-source raw
```

## Option 2: anonymized propagation paths

Download the two JSONL files from [the shared data folder](https://drive.proton.me/urls/6ZDQPJY178#XMn79ZYqEcxt) and put them in `FakeNewsNet/data/`:

- `anonymized_fake_propagation_paths.jsonl`
- `anonymized_real_propagation_paths.jsonl`

Each line must contain a story's list of tweet objects, with each object containing an `id_str`. Provide an X API bearer token in `TWITTER_BEARER_TOKEN`, then run:

```bash
export TWITTER_BEARER_TOKEN="..."
python preprocessing_and_training/preprocess_datasets.py \
  --dataset fakenewsnet --fnn-source anonymized
```

The script looks up IDs in batches of up to 100. IDs that the API no longer returns are omitted, and stories with no recovered tweets are excluded. The output is sorted by creation time and can be used by the same embedding and training steps as the raw route. Rehydration is subject to X API access, rate limits, and the current availability and metrics of each tweet.

## Outputs

Both routes write `fake_propagation_paths.jsonl`, `real_propagation_paths.jsonl`, and their chronologically ordered counterparts to `FakeNewsNet/data/`. RoBERTa and Mistral embeddings are written under `FakeNewsNet/data/fake_news_net/`. Set `--encoders roberta` or `--encoders mistral` to create one encoder's embeddings; the complete paper training run needs both.
