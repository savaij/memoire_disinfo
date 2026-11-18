"""Train FakeNewsNet classifiers for total engagement above a quantile."""

from train_fakenewsnet_veracity import run_experiment


if __name__ == "__main__":
    run_experiment("virality")
