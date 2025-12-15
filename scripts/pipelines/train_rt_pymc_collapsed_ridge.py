#!/usr/bin/env python3
"""
Train the production RT ridge models in PyMC (stage-1 coefficient summaries).

This is a small CLI wrapper around the two supported PyMC ridge model forms:
  - supercategory: fully pooled by species_cluster (collapsed slopes)
  - partial_pool: partial pooling by species nested in species_cluster (collapsed slopes)

Outputs under --output-dir:
  - models/stage1_coeff_summaries_posterior.npz
  - config.json
  - results/trace_*.nc (optional; method=advi)
  - results/comp_id_mapping_collisions.csv (optional)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.compassign.rt.pymc_partial_pool_ridge import (  # noqa: E402
    train_pymc_partial_pool_ridge_from_csv,
    write_pymc_partial_pool_ridge_artifacts,
)
from src.compassign.rt.pymc_supercategory_ridge import (  # noqa: E402
    train_pymc_supercategory_ridge_from_csv,
    write_pymc_supercategory_ridge_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PyMC RT ridge model and write stage-1 coefficient summaries.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-csv", type=Path, required=True, help="Production RT CSV to train on."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory (writes config.json, models/, results/).",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["supercategory", "partial_pool"],
        default="supercategory",
        help="Which PyMC ridge model form to train.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for determinism.")
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=0,
        help="Maximum number of training rows to process (0 = all).",
    )
    parser.add_argument(
        "--include-es-all",
        action="store_true",
        help="Include all ES_* covariates present in the CSV (no masking).",
    )
    parser.add_argument(
        "--lambda-slopes",
        type=float,
        default=3e-4,
        help="Ridge penalty (precision) for slopes; fixed.",
    )
    parser.add_argument(
        "--sigma-y-prior",
        type=float,
        default=0.05,
        help="HalfNormal prior scale for sigma_y.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["advi", "map"],
        default="advi",
        help="Inference method (ADVI recommended).",
    )
    parser.add_argument("--advi-steps", type=int, default=None, help="ADVI steps override.")
    parser.add_argument("--advi-draws", type=int, default=50, help="Posterior draws from ADVI.")
    parser.add_argument(
        "--advi-log-every",
        type=int,
        default=1000,
        help="Log ADVI progress every N iters (0 disables).",
    )
    parser.add_argument(
        "--map-maxeval",
        type=int,
        default=50_000,
        help="Max evaluations for MAP optimization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = str(args.model)
    method = str(args.method)
    if args.advi_steps is None:
        args.advi_steps = 5000 if model == "supercategory" else 10_000

    if model == "supercategory":
        artifacts = train_pymc_supercategory_ridge_from_csv(
            data_csv=args.data_csv,
            seed=int(args.seed),
            max_train_rows=int(args.max_train_rows),
            include_es_all=bool(args.include_es_all),
            feature_center="global",
            lambda_slopes=float(args.lambda_slopes),
            sigma_y_prior=float(args.sigma_y_prior),
            method=method,  # type: ignore[arg-type]
            advi_steps=int(args.advi_steps),
            advi_log_every=int(args.advi_log_every),
            advi_draws=int(args.advi_draws),
            map_maxeval=int(args.map_maxeval),
        )
        paths = write_pymc_supercategory_ridge_artifacts(
            artifacts=artifacts, output_dir=args.output_dir
        )
    else:
        artifacts = train_pymc_partial_pool_ridge_from_csv(
            data_csv=args.data_csv,
            seed=int(args.seed),
            max_train_rows=int(args.max_train_rows),
            include_es_all=bool(args.include_es_all),
            feature_center="global",
            lambda_slopes=float(args.lambda_slopes),
            sigma_y_prior=float(args.sigma_y_prior),
            method=method,  # type: ignore[arg-type]
            advi_steps=int(args.advi_steps),
            advi_log_every=int(args.advi_log_every),
            advi_draws=int(args.advi_draws),
            map_maxeval=int(args.map_maxeval),
        )
        paths = write_pymc_partial_pool_ridge_artifacts(
            artifacts=artifacts, output_dir=args.output_dir
        )

    print(f"[rt_pymc_ridge] Model: {model}")
    print(f"[rt_pymc_ridge] Training CSV: {args.data_csv}")
    print(f"[rt_pymc_ridge] Output dir: {args.output_dir}")
    print(f"[rt_pymc_ridge] Wrote coeff summaries: {paths['coeff_npz']}")
    print(f"[rt_pymc_ridge] Wrote config: {paths['config_json']}")
    if paths.get("trace_path"):
        print(f"[rt_pymc_ridge] Wrote trace: {paths['trace_path']}")


if __name__ == "__main__":
    main()
