import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RECALL_RATIO = 1.0


def make_recall_plot(
    progress: pd.DataFrame,
    naive_recall: float,
    ratio_target: float,
    output_path: Path,
    random: pd.DataFrame | None = None,
    naive: pd.DataFrame | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        progress["cumulative_annotations"],
        progress["recall"],
        marker="o",
        label="Active learning",
    )
    # Do not plot a naïve progression curve; we only use its endpoint as a target.
    if random is not None:
        ax.plot(
            random["cumulative_annotations"],
            random["recall"],
            marker="^",
            linestyle="--",
            color="tab:orange",
            label="Random sampling",
        )
    # No reference line; keep the plot minimal (AL vs Random only)
    ax.set_xlabel("Cumulative annotations")
    ax.set_ylabel("Recall")
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def make_precision_plot(
    progress: pd.DataFrame,
    naive_precision: float,
    output_path: Path,
    random: pd.DataFrame | None = None,
    naive: pd.DataFrame | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(
        progress["cumulative_annotations"],
        progress["precision"],
        marker="o",
        color="tab:purple",
        label="Active learning",
    )
    # Do not plot a naïve progression curve; we only use its endpoint as a target.
    if random is not None:
        ax.plot(
            random["cumulative_annotations"],
            random["precision"],
            marker="^",
            linestyle="--",
            color="tab:orange",
            label="Random sampling",
        )
    ax.set_xlabel("Cumulative annotations")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def make_clicks_bar(
    naive_clicks: int,
    ratio_clicks: int | None,
    final_clicks: int,
    output_path: Path,
    random_final: int | None = None,
    random_ratio: int | None = None,
) -> None:
    # Simplified comparison: labels needed to match the naive full-review recall
    labels = []
    values = []
    colors = []

    if ratio_clicks is not None:
        labels.append("AL labels to naive")
        values.append(ratio_clicks)
        colors.append("tab:blue")

    if random_ratio is not None:
        labels.append("Random labels to naive")
        values.append(random_ratio)
        colors.append("tab:orange")

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(labels, values, color=colors)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.02,
            f"{value}",
            ha="center",
            va="bottom",
        )
    ax.set_ylabel("Annotations (labels)")
    ax.set_ylim(0, max(values) * 1.25 if values else 1)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate plots summarising AL runs")
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing summary.json and al_progression.csv",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    summary_path = results_dir / "summary.json"
    progression_path = results_dir / "al_progression.csv"

    if not summary_path.exists() or not progression_path.exists():
        raise FileNotFoundError("Expected summary.json and al_progression.csv in results directory")

    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    progress = pd.read_csv(progression_path)

    naive_metrics = summary["naive"]["metrics"]
    naive_hist = summary["naive"].get("history")
    al_summary = summary["active_learning"]
    random_summary = summary.get("random")

    naive_recall = float(naive_metrics["recall"])
    naive_precision = float(naive_metrics["precision"])
    ratio_clicks = al_summary.get("clicks_to_recall_ratio")
    final_clicks = int(al_summary.get("final_annotations", 0))
    naive_clicks = int(summary["naive"]["annotations"])
    ratio_recall = naive_recall * summary.get("recall_ratio_target", RECALL_RATIO)

    random_df = None
    random_ratio = None
    random_final = None
    if random_summary and random_summary.get("history"):
        random_df = pd.DataFrame(random_summary["history"])
        random_ratio = random_summary.get("clicks_to_recall_ratio")
        random_final = random_summary.get("final_annotations")

    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    naive_df = pd.DataFrame(naive_hist) if naive_hist else None

    make_recall_plot(
        progress,
        naive_recall,
        ratio_recall,
        plots_dir / "recall_vs_annotations.png",
        random=random_df,
        naive=naive_df,
    )
    make_precision_plot(
        progress,
        naive_precision,
        plots_dir / "precision_vs_annotations.png",
        random=random_df,
        naive=naive_df,
    )
    make_clicks_bar(
        naive_clicks,
        ratio_clicks,
        final_clicks,
        plots_dir / "annotation_comparison.png",
        random_final=random_final,
        random_ratio=random_ratio,
    )


if __name__ == "__main__":
    main()
