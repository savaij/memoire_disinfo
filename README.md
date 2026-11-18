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

## Run the full analysis

The source datasets are not included. Obtain them separately and follow their access and use rules. Place the files here:

- EVONS CSV: `evons/data/evons.csv`
- Already recovered FakeNewsNet PolitiFact tweet files: `FakeNewsNet/data/raw_data/politifact/{fake,real}/<story>/tweets/`

The pipeline does not download or recover FakeNewsNet tweets. It starts from the tweet files you provide. It turns text into numeric inputs for the models using RoBERTa and Mistral. Mistral requires the `MISTRAL_API_KEY` environment variable. The Mistral model provided by the service can change over time.

To run all steps, including data preparation, training, and table and figure generation:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-gpu.txt
python run_pipeline/run.py --stage all --baseline-backend gpu
```

The GPU option requires a compatible NVIDIA GPU and CUDA setup. To use the CPU versions of the logistic regression and random forest models instead, replace `gpu` with `cpu`; results may differ from the saved results.

The full run covers engagement cutoffs at 50%, 75%, 90%, and 95%. For FakeNewsNet it compares text from the first 1, 3, 5, or 10 tweets with the full recovered sequence. These comparisons use stories with at least 11 recovered tweets. Training can take several hours. A new run may not exactly match the saved scores because some training steps use random initialization and the Mistral service can change.

## Results

- `results/metrics/` contains the saved model scores from each evaluation run. These files are used to build the paper tables.
- `results/audits/` contains summaries of data checks and the early-to-later tweet engagement analysis.
- `results/inference/` contains the model comparisons used in the paper.
- `results/paper/tables/` contains the generated LaTeX tables and supporting CSV files.
- `results/paper/figures/` contains the generated figures in PDF and PNG formats.

Keep source data, generated embeddings, raw text, API keys, and local training logs out of version control.
