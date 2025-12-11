#!/usr/bin/env python3
"""
Convert per-lib merged training Parquet (cap-sampled, with chem_id + compound_class)
into production-style RT CSVs expected by train_rt_prod.py.

Required input columns in the Parquet:
  - sample_set_id
  - worksheet_id
  - task_id
  - apex_rt
  - chemical_id  (CHEM_ID)
  - compound_class

Plus run covariates (IS_*, optionally ES_*, RS_*), and a species mapping CSV:
  - sample_set_id, species, species_cluster

Output CSV columns:
  - sampleset_id, worksheet_id, task_id
  - species, species_cluster
  - compound (chemical_id), compound_class
  - IS_* (and other numeric covariates)
  - rt (from apex_rt)

Usage example:
  python scripts/pipelines/make_rt_prod_csv_from_merged.py \\
      --input repo_export/merged_training_de194c2cc2114efaa1075ccf7539d0cb_lib209_cap5_chemclass.parquet \\
      --species-mapping repo_export/merged_training_de194c2cc2114efaa1075ccf7539d0cb_lib209_species_mapping.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Set

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build RT production CSV from merged per-lib Parquet with chem classes."
    )
    parser.add_argument("--input", type=Path, required=True, help="Per-lib *_chemclass.parquet file")
    parser.add_argument(
        "--species-mapping",
        type=Path,
        required=True,
        help="CSV with sample_set_id,species,species_cluster (per lib)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV (default: <stem>_rt_prod.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input Parquet not found: {args.input}")
    if not args.species_mapping.exists():
        raise SystemExit(f"Species mapping CSV not found: {args.species_mapping}")

    df = pd.read_parquet(args.input)
    sm = pd.read_csv(args.species_mapping)

    required_parq = [
        "sample_set_id",
        "worksheet_id",
        "task_id",
        "apex_rt",
        "chemical_id",
        "compound_class",
    ]
    missing_parq: Set[str] = {c for c in required_parq if c not in df.columns}
    if missing_parq:
        raise SystemExit(f"Input Parquet {args.input} missing columns: {sorted(missing_parq)}")

    required_sm = ["sample_set_id", "species", "species_cluster"]
    missing_sm: Set[str] = {c for c in required_sm if c not in sm.columns}
    if missing_sm:
        raise SystemExit(
            f"Species mapping {args.species_mapping} missing columns: {sorted(missing_sm)}"
        )

    # Normalize types for join
    df["sample_set_id"] = df["sample_set_id"].astype(int)
    sm["sample_set_id"] = sm["sample_set_id"].astype(int)

    df_merged = df.merge(
        sm[["sample_set_id", "species", "species_cluster"]],
        on="sample_set_id",
        how="left",
    )
    if df_merged["species"].isna().any():
        missing_ssids = df_merged.loc[df_merged["species"].isna(), "sample_set_id"].unique().tolist()
        raise SystemExit(
            f"Some sample_set_id values missing in species mapping: {missing_ssids[:10]} ..."
        )

    # Build production-style columns
    out = pd.DataFrame()
    out["sampleset_id"] = df_merged["sample_set_id"].astype(int)
    out["worksheet_id"] = df_merged["worksheet_id"].astype(int)
    out["task_id"] = df_merged["task_id"].astype(int)
    out["species"] = df_merged["species"].astype(int)
    out["species_cluster"] = df_merged["species_cluster"].astype(int)
    out["compound"] = df_merged["chemical_id"].astype(int)
    out["compound_class"] = df_merged["compound_class"].astype(int)
    out["rt"] = df_merged["apex_rt"].astype(float)

    # Append numeric covariates (IS_*/ES_*/RS_* etc.), excluding reserved identifiers
    reserved = set(out.columns)
    cov_cols: List[str] = []
    for col in df_merged.columns:
        if col in reserved or col in {"chemical_id", "sample_set_id", "species", "species_cluster"}:
            continue
        if pd.api.types.is_numeric_dtype(df_merged[col]):
            cov_cols.append(col)

    cov_cols_sorted = sorted(cov_cols)
    for col in cov_cols_sorted:
        out[col] = df_merged[col]

    # Sanity: train_rt_prod expects at least some IS_* numeric columns
    has_is = any(c.startswith("IS") for c in cov_cols_sorted)
    if not has_is:
        raise SystemExit("No IS* columns found among numeric covariates; check input schema.")

    output_path = args.output or args.input.with_name(args.input.stem + "_rt_prod.csv")
    out.to_csv(output_path, index=False)
    print(
        f"[make_rt_prod_csv] Wrote {len(out):,} rows, {out.shape[1]} columns to {output_path}"
    )


if __name__ == "__main__":
    main()

