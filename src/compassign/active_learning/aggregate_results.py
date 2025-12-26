"""Aggregate active learning run results.

This helper scans one or more base directories for `summary.json` files (as written by
active learning experiments) and writes a flat CSV/JSON table with key metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _get_nested_value(payload: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
    return default if current is None else current


def extract_row(path: Path) -> dict[str, Any]:
    """Extract a single aggregate row from a `summary.json` file."""
    with path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    config = summary.get("config", {})
    return {
        "path": str(path.parent),
        "seed": config.get("seed"),
        "threshold": config.get("threshold"),
        "batch_size": config.get("batch_size"),
        "rounds": config.get("rounds"),
        "acquisition": config.get("acquisition"),
        "lambda_fp": config.get("lambda_fp"),
        "mass_error_ppm": config.get("mass_error_ppm"),
        "decoy_fraction": config.get("decoy_fraction"),
        "initial_labeled_fraction": config.get("initial_labeled_fraction"),
        # dataset
        "n_peaks": _get_nested_value(summary, "dataset", "n_peaks"),
        "mean_candidates": _get_nested_value(summary, "dataset", "mean_candidates"),
        # naive
        "naive_precision": _get_nested_value(summary, "naive", "metrics", "precision"),
        "naive_recall": _get_nested_value(summary, "naive", "metrics", "recall"),
        # Active learning
        "al_clicks_to_target": _get_nested_value(
            summary, "active_learning", "clicks_to_target_recall"
        ),
        "al_clicks_to_ratio": _get_nested_value(
            summary, "active_learning", "clicks_to_recall_ratio"
        ),
        "al_final_annotations": _get_nested_value(summary, "active_learning", "final_annotations"),
        "al_precision": _get_nested_value(summary, "active_learning", "metrics", "precision"),
        "al_recall": _get_nested_value(summary, "active_learning", "metrics", "recall"),
        "al_f1": _get_nested_value(summary, "active_learning", "metrics", "f1"),
        "al_auc": _get_nested_value(summary, "active_learning", "recall_auc"),
        # Random
        "rand_clicks_to_ratio": _get_nested_value(summary, "random", "clicks_to_recall_ratio"),
        "rand_final_annotations": _get_nested_value(summary, "random", "final_annotations"),
        "rand_precision": _get_nested_value(summary, "random", "metrics", "precision"),
        "rand_recall": _get_nested_value(summary, "random", "metrics", "recall"),
        "rand_f1": _get_nested_value(summary, "random", "metrics", "f1"),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate active learning run results")
    parser.add_argument(
        "--base-dirs",
        nargs="+",
        default=["output"],
        help="Directories to search for summary.json files",
    )
    parser.add_argument(
        "--outdir",
        default="output/aggregate",
        help="Where to write aggregate.csv/json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    summary_paths: list[Path] = []
    for base_dir in args.base_dirs:
        summary_paths.extend(Path(base_dir).rglob("summary.json"))

    rows = [extract_row(path) for path in summary_paths]
    df = pd.DataFrame(rows)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df.to_csv(outdir / "aggregate.csv", index=False)
    (outdir / "aggregate.json").write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
    print(f"Wrote {len(df)} rows to {outdir}/aggregate.csv")


if __name__ == "__main__":
    main()
