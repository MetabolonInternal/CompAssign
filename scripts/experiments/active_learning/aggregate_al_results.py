#!/usr/bin/env python3
"""Aggregate AL experiment results across folders.

Scans for summary.json files and produces a flat CSV/JSON with key metrics:
- clicks to match naive recall / 95% of naive
- final recall/precision/F1 for AL and Random
- naive metrics
- AUCs when present
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def extract_row(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        s = json.load(fh)

    def g(d: Dict, *keys, default=None):
        cur = d
        for k in keys:
            if cur is None:
                return default
            cur = cur.get(k)
        return cur if cur is not None else default

    cfg = s.get("config", {})
    row = {
        "path": str(path.parent),
        "seed": cfg.get("seed"),
        "threshold": cfg.get("threshold"),
        "batch_size": cfg.get("batch_size"),
        "rounds": cfg.get("rounds"),
        "acquisition": cfg.get("acquisition"),
        "lambda_fp": cfg.get("lambda_fp"),
        "mass_error_ppm": cfg.get("mass_error_ppm"),
        "decoy_fraction": cfg.get("decoy_fraction"),
        "initial_labeled_fraction": cfg.get("initial_labeled_fraction"),
        # dataset
        "n_peaks": g(s, "dataset", "n_peaks"),
        "mean_candidates": g(s, "dataset", "mean_candidates"),
        # naive
        "naive_precision": g(s, "naive", "metrics", "precision"),
        "naive_recall": g(s, "naive", "metrics", "recall"),
        # AL
        "al_clicks_to_target": g(s, "active_learning", "clicks_to_target_recall"),
        "al_clicks_to_ratio": g(s, "active_learning", "clicks_to_recall_ratio"),
        "al_final_annotations": g(s, "active_learning", "final_annotations"),
        "al_precision": g(s, "active_learning", "metrics", "precision"),
        "al_recall": g(s, "active_learning", "metrics", "recall"),
        "al_f1": g(s, "active_learning", "metrics", "f1"),
        "al_auc": g(s, "active_learning", "recall_auc"),
        # Random
        "rand_clicks_to_ratio": g(s, "random", "clicks_to_recall_ratio"),
        "rand_final_annotations": g(s, "random", "final_annotations"),
        "rand_precision": g(s, "random", "metrics", "precision"),
        "rand_recall": g(s, "random", "metrics", "recall"),
        "rand_f1": g(s, "random", "metrics", "f1"),
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate AL run results")
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
    args = parser.parse_args()

    paths: List[Path] = []
    for root in args.base_dirs:
        root_path = Path(root)
        paths.extend(root_path.rglob("summary.json"))

    rows = [extract_row(p) for p in paths]
    df = pd.DataFrame(rows)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "aggregate.csv", index=False)
    (outdir / "aggregate.json").write_text(df.to_json(orient="records", indent=2))
    print(f"Wrote {len(df)} rows to {outdir}/aggregate.csv")


if __name__ == "__main__":
    main()
