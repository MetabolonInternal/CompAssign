#!/usr/bin/env python3
"""
Train a fast ridge RT model on a production RT CSV (no MCMC).

Model form (per supercategory × comp_id):
    rt = intercept + dot(coefs, x_run)
where x_run are run covariates (IS*/RS*/optional ES_*).

This CLI always stores Bayesian ridge posterior summaries in the model artifact so
`eval_rt_ridge_prod_streaming.py --use-posterior` can compute Student-t predictive
intervals (point predictions are unchanged).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.compassign.rt.fast_ridge_prod import (  # noqa: E402
    train_ridge_prod_model,
    write_ridge_prod_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a fast ridge RT model on a production RT CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-csv", type=Path, required=True, help="Production RT CSV to train on."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/rt_ridge_prod"),
        help="Output directory (will contain config.json and models/rt_ridge_model.npz).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for determinism.")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200_000,
        help="Rows per chunk when streaming training CSV.",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=0,
        help="Maximum number of training rows to process (0 = all).",
    )
    parser.add_argument(
        "--lambda-ridge",
        type=float,
        default=1e-3,
        help="L2 regularization strength for ridge regression (slopes only; intercept unpenalized).",
    )
    parser.add_argument(
        "--include-es-group1",
        action="store_true",
        help="Include ES_* covariates for Group 1 (human blood) with zero-fill elsewhere.",
    )
    parser.add_argument(
        "--include-es-all",
        action="store_true",
        help="Include all ES_* covariates present in the CSV (no group masking).",
    )
    parser.add_argument(
        "--bayes-a0",
        type=float,
        default=2.0,
        help="InvGamma prior shape a0 for Bayesian ridge noise variance (always used).",
    )
    parser.add_argument(
        "--bayes-b0",
        type=float,
        default=1e-6,
        help="InvGamma prior scale b0 for Bayesian ridge noise variance (always used).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_csv = args.data_csv
    if not data_csv.is_absolute():
        data_csv = (REPO_ROOT / data_csv).resolve()
    if not data_csv.exists():
        raise SystemExit(f"Training CSV not found: {data_csv}")

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = (REPO_ROOT / output_dir).resolve()

    artifacts = train_ridge_prod_model(
        data_csv=data_csv,
        include_es_group1=bool(args.include_es_group1),
        include_es_all=bool(args.include_es_all),
        lambda_ridge=float(args.lambda_ridge),
        bayes_a0=float(args.bayes_a0),
        bayes_b0=float(args.bayes_b0),
        chunk_size=int(args.chunk_size),
        max_train_rows=int(args.max_train_rows),
        seed=int(args.seed),
    )
    model_path = write_ridge_prod_artifacts(
        artifacts=artifacts,
        output_dir=output_dir,
        data_csv=data_csv,
    )

    feature_cols = list(artifacts.feature_cols)
    n_is = sum(c.startswith("IS") for c in feature_cols)
    n_rs = sum(c.startswith("RS") for c in feature_cols)
    n_es = sum(c.startswith("ES_") for c in feature_cols)

    print(f"[ridge] Training CSV: {data_csv}")
    print(f"[ridge] Output dir: {output_dir}")
    print(f"[ridge] Features: IS={n_is}, RS={n_rs}, ES={n_es} (total={len(feature_cols)})")
    print(f"[ridge] Wrote model to {model_path}")
    print(
        f"[ridge] n_obs_train={artifacts.n_obs_train:,}, n_models={artifacts.n_models:,}, "
        f"n_compounds={artifacts.n_compounds:,}"
    )


if __name__ == "__main__":
    main()
