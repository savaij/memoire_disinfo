#!/usr/bin/env python3
"""Generate the current paper's data-derived figures from aggregate CSVs."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from generate_tables import summarize_experiments


ROOT = Path(__file__).resolve().parents[1]
QUANTILES = (0.5, 0.75, 0.9, 0.95)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def save_figure(fig, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    fig.savefig(output.with_suffix(".pdf"))


def future_figures(audit_dir: Path, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = {name: read_rows(audit_dir / f"future_{name}.csv") for name in ("real", "fake")}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True, constrained_layout=True)
    for axis, (name, rows) in zip(axes, datasets.items()):
        n_values = {int(row["n_sequences"]) for row in rows}
        if len(n_values) != 1:
            raise ValueError(f"{name}: the fixed future-audit cohort has inconsistent size")
        for q in QUANTILES:
            selected = sorted((row for row in rows if float(row["quantile"]) == q),
                              key=lambda row: int(row["k_prefix_tweets"]))
            if len(selected) != 4:
                raise ValueError(f"{name}: expected four prefix lengths at q={q}")
            axis.plot([int(row["k_prefix_tweets"]) for row in selected],
                      [float(row["auc_prefix_for_future_label"]) for row in selected],
                      marker="o", label=f"q{round(q * 100)}")
        axis.axhline(.5, color="grey", alpha=.5, linestyle=":", linewidth=1)
        axis.set(xlabel="First k recovered tweets",
                 title=f"{name.capitalize()}-labelled stories (n={n_values.pop()})",
                 xticks=[1, 3, 5, 10], ylim=(.45, 1.01))
        axis.grid(axis="y", alpha=.15)
        axis.legend(title="Future-label quantile", loc="upper left", fontsize=8)
    axes[0].set_ylabel("ROC-AUC: prefix favorites vs later-tweet favorites")
    fig.suptitle("Retrospective future-engagement audit on fixed cohorts", fontsize=12)
    save_figure(fig, output_dir / "Fig2.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True, constrained_layout=True)
    for axis, (name, rows) in zip(axes, datasets.items()):
        for q in (.5, .95):
            selected = sorted((row for row in rows if float(row["quantile"]) == q),
                              key=lambda row: int(row["k_prefix_tweets"]))
            x = [int(row["k_prefix_tweets"]) for row in selected]
            axis.plot(x, [float(row["auc_prefix_for_future_label"]) for row in selected],
                      marker="o", label=f"q{round(q * 100)} future-only")
            axis.plot(x, [float(row["auc_prefix_for_total_label_matched_cohort"]) for row in selected],
                      marker="x", linestyle="--", label=f"q{round(q * 100)} total")
        axis.axhline(.5, color="grey", alpha=.5, linestyle=":", linewidth=1)
        axis.set(xlabel="First k recovered tweets", title=f"{name.capitalize()}-labelled",
                 xticks=[1, 3, 5, 10], ylim=(.45, 1.01))
        axis.grid(axis="y", alpha=.15)
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Rank ROC-AUC")
    fig.suptitle("Matched cohort: future-only versus total engagement", fontsize=12)
    save_figure(fig, output_dir / "FigS1.png")
    plt.close(fig)


def load_experiment_summaries(metrics_dir: Path) -> dict:
    summaries, _, _ = summarize_experiments(metrics_dir)
    grouped = {}
    for (dataset, task, quantile, window), rows in summaries.items():
        key = f"{dataset}_{task}_{quantile}"
        bucket = grouped.setdefault(key, {})
        for row in rows:
            bucket[window, row["model"]] = row["f1_tuned"]
    return grouped


def virality_figure(metrics_dir: Path, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summaries = load_experiment_summaries(metrics_dir)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    qs = (50, 75, 90, 95)
    evons_best = []
    for q in qs:
        rows = summaries[f"evons_virality_{q}"]
        evons_best.append(max(rows.values()))
    axes[0].plot(qs, evons_best, marker="o", color="#285f9e")
    axes[0].set(title="EVONS", xlabel="Engagement percentile", ylabel="Best mean calibrated F1",
                xticks=qs, ylim=(0, 1))
    axes[0].grid(axis="y", alpha=.15)

    styles = {"k1": ("First tweet", "o"), "k10": ("First 10 tweets", "s"),
              "full": ("Full sequence", "^")}
    for window, (label, marker) in styles.items():
        scores = [max(value for (candidate, _), value in summaries[f"fakenewsnet_virality_{q}"].items()
                      if candidate == window) for q in qs]
        axes[1].plot(qs, scores, marker=marker, label=label)
    axes[1].set(title="FakeNewsNet–PolitiFact", xlabel="Engagement percentile",
                xticks=qs, ylim=(0, 1))
    axes[1].grid(axis="y", alpha=.15)
    axes[1].legend()
    fig.suptitle("Calibrated F1 across engagement cutoffs and text windows")
    save_figure(fig, output_dir / "Fig3.png")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-dir", type=Path, default=ROOT / "results/metrics")
    parser.add_argument("--audit-dir", type=Path, default=ROOT / "results/audits")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/paper/figures")
    args = parser.parse_args()
    future_figures(args.audit_dir, args.output_dir)
    virality_figure(args.metrics_dir, args.output_dir)
    print(f"Wrote Fig2, Fig3 and FigS1 to {args.output_dir}")


if __name__ == "__main__":
    main()
