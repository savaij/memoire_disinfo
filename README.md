# What Do Misinformation Benchmarks Measure?

This repository contains the code and saved results for the paper “What Do Misinformation Benchmarks Measure? Construct Validity in Veracity and Engagement Prediction.”

The study uses EVONS and the PolitiFact part of FakeNewsNet as datasets. It examines whether models can predict the fake or real labels recorded in each dataset, and whether they can identify items with many social media interactions, such as likes, comments, shares, or tweet favorites. For FakeNewsNet, it also compares predictions based on the first few tweets with predictions based on all recovered tweets. A separate analysis checks whether favorites on early tweets are related to favorites on later tweets.

**The repository includes both the code and the saved training results used to create the paper’s tables.** It also includes the audit summaries, statistical comparisons, tables, and figures produced from those results. The original data and trained model files are not included.

## Repository contents

- `preprocessing_and_training/` prepares the data, creates text representations, and trains the models.
- `metrics_and_audits/` checks the data and calculates model scores and comparisons.
- `paper/` contains the scripts that create the paper’s tables and figures.
- `run_pipeline/run.py` runs the steps above.
- `results/` contains the saved training scores, audit summaries, comparisons, tables, and figures.

## Rebuild the paper tables and figures

The saved results are enough to recreate the tables and generated figures. You do not need the original datasets or a Mistral API key for this step.

```bash
python -m pip install -r requirements.txt
python run_pipeline/run.py --stage paper
```

The new files are written to `results/inference/` and `results/paper/`.

## Run individual steps

Install the packages in `requirements.txt` first. Each step can then be run on its own with `--stage`:

- **Check required files:** `python run_pipeline/run.py --stage check` lists missing data or embeddings. This is also the default if you run the script without `--stage`.
- **Prepare data and create text inputs:** `python run_pipeline/run.py --stage preprocess` reads the source files, creates the FakeNewsNet tweet sequences, and generates model inputs. By default it uses both RoBERTa and Mistral, so it needs `MISTRAL_API_KEY`. You can select one with `--encoders roberta` or `--encoders mistral`; the full training run needs both.
- **Create data summaries:** `python run_pipeline/run.py --stage audit` reads the EVONS CSV and the prepared FakeNewsNet sequences, then writes summaries to `results/audits/`.
- **Train the models:** `python run_pipeline/run.py --stage train --baseline-backend gpu` needs the source data and both sets of model inputs. It writes scores to `results/metrics/`. Use `--baseline-backend cpu` to run the logistic regression and random forest models on the CPU. You can limit the FakeNewsNet tweet windows with `--fnn-windows 1,3,5,10,full`; this is the default and is required for the complete paper results.
- **Rebuild comparisons, tables, and figures:** `python run_pipeline/run.py --stage paper` uses the saved scores and data summaries. It does not train models or need the source datasets.
- **Run every step in sequence:** `python run_pipeline/run.py --stage all --baseline-backend gpu` prepares data, creates summaries, trains models, and rebuilds the paper outputs. This requires all five FakeNewsNet tweet windows.

When running the steps separately from source data, use this order: `preprocess`, `check`, `audit`, `train`, then `paper`. The full run may take several hours.

## Inputs for runs from source data

The source datasets are not included. Obtain them separately and follow their access and use rules. Place the files here:

- EVONS CSV: `evons/data/evons.csv`
- Already recovered FakeNewsNet PolitiFact tweet files: `FakeNewsNet/data/raw_data/politifact/{fake,real}/<story>/tweets/`

The pipeline does not download or recover FakeNewsNet tweets. It starts from the tweet files you provide. It turns text into numeric inputs for the models using RoBERTa and Mistral. Mistral requires the `MISTRAL_API_KEY` environment variable, and the model provided by the service can change over time.

The training commands above use the GPU backend by default. It requires a compatible NVIDIA GPU and CUDA setup; install its package with `python -m pip install -r requirements-gpu.txt`. To use the CPU versions of the logistic regression and random forest models, set `--baseline-backend cpu` instead. Results may differ from the saved results.

The full run covers engagement cutoffs at 50%, 75%, 90%, and 95%. For FakeNewsNet it compares text from the first 1, 3, 5, or 10 tweets with the full recovered sequence. These comparisons use stories with at least 11 recovered tweets. Training can take several hours. A new run may not exactly match the saved scores because some training steps use random initialization and the Mistral service can change.

## Results

- `results/metrics/` contains the saved model scores from each evaluation run. These files are used to build the paper tables.
- `results/audits/` contains summaries of data checks and the early-to-later tweet engagement analysis.
- `results/inference/` contains the model comparisons used in the paper.
- `results/paper/tables/` contains the generated LaTeX tables and supporting CSV files.
- `results/paper/figures/` contains the generated figures in PDF and PNG formats.

Keep source data, generated embeddings, raw text, API keys, and local training logs out of version control.
