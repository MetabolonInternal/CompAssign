#!/usr/bin/env python3
"""
Convert per-lib merged training Parquet (cap-sampled, with chem_id + compound_class)
into production-style RT CSVs expected by train_rt_prod.py.

This script streams Parquet batches to keep memory usage low.

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
  python -m compassign.rt.data_prep.make_rt_prod_csv_from_merged \\
      --input repo_export/merged_training_de194c2cc2114efaa1075ccf7539d0cb_lib209_cap5_chemclass.parquet \\
      --species-mapping repo_export/merged_training_de194c2cc2114efaa1075ccf7539d0cb_lib209_species_mapping.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Set

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build RT production CSV from merged per-lib Parquet with chem classes."
    )
    parser.add_argument(
        "--input", type=Path, required=True, help="Per-lib *_chemclass.parquet file"
    )
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


def _is_numeric_type(pa_type: pa.DataType) -> bool:
    return bool(
        pa.types.is_boolean(pa_type)
        or pa.types.is_integer(pa_type)
        or pa.types.is_floating(pa_type)
        or pa.types.is_decimal(pa_type)
    )


def _infer_covariate_columns(schema: pa.Schema, reserved: Set[str]) -> List[str]:
    cov_cols: List[str] = []
    for field in schema:
        if field.name in reserved:
            continue
        if _is_numeric_type(field.type):
            cov_cols.append(field.name)
    return sorted(cov_cols)


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input Parquet not found: {args.input}")
    if not args.species_mapping.exists():
        raise SystemExit(f"Species mapping CSV not found: {args.species_mapping}")

    pf = pq.ParquetFile(args.input)
    schema = pf.schema_arrow
    sm = pd.read_csv(args.species_mapping)

    required_parq = [
        "sample_set_id",
        "worksheet_id",
        "task_id",
        "apex_rt",
        "chemical_id",
        "compound_class",
    ]
    missing_parq: Set[str] = {c for c in required_parq if c not in schema.names}
    if missing_parq:
        raise SystemExit(f"Input Parquet {args.input} missing columns: {sorted(missing_parq)}")

    required_sm = ["sample_set_id", "species", "species_cluster"]
    missing_sm: Set[str] = {c for c in required_sm if c not in sm.columns}
    if missing_sm:
        raise SystemExit(
            f"Species mapping {args.species_mapping} missing columns: {sorted(missing_sm)}"
        )

    # Normalize types for join
    sm["sample_set_id"] = sm["sample_set_id"].astype(int)
    sm["species"] = sm["species"].astype(int)
    sm["species_cluster"] = sm["species_cluster"].astype(int)

    reserved = {
        "sampleset_id",
        "worksheet_id",
        "task_id",
        "species",
        "species_cluster",
        "compound",
        "compound_class",
        "rt",
        "chemical_id",
        "sample_set_id",
    }
    cov_cols_sorted = _infer_covariate_columns(schema, reserved=reserved)
    cols_to_read = list(dict.fromkeys([*required_parq, *cov_cols_sorted]))

    output_path = args.output or args.input.with_name(args.input.stem + "_rt_prod.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output_path.with_name(output_path.name + ".tmp")
    if tmp_output.exists():
        tmp_output.unlink()

    rows_written = 0
    try:
        for batch in pf.iter_batches(batch_size=200_000, columns=cols_to_read):
            df = batch.to_pandas()
            if df.empty:
                continue

            df["sample_set_id"] = df["sample_set_id"].astype(int)
            df_merged = df.merge(
                sm[["sample_set_id", "species", "species_cluster"]],
                on="sample_set_id",
                how="left",
            )
            if df_merged["species"].isna().any():
                missing_ssids = (
                    df_merged.loc[df_merged["species"].isna(), "sample_set_id"].unique().tolist()
                )
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

            for col in cov_cols_sorted:
                out[col] = df_merged[col]

            header = rows_written == 0
            out.to_csv(tmp_output, mode="w" if header else "a", header=header, index=False)
            rows_written += int(len(out))
    except Exception:
        if tmp_output.exists():
            tmp_output.unlink()
        raise

    # Sanity: train_rt_prod expects at least some IS_* numeric columns
    has_is = any(c.startswith("IS") for c in cov_cols_sorted)
    if not has_is:
        if tmp_output.exists():
            tmp_output.unlink()
        raise SystemExit("No IS* columns found among numeric covariates; check input schema.")

    if rows_written == 0:
        empty = pd.DataFrame(
            columns=[
                "sampleset_id",
                "worksheet_id",
                "task_id",
                "species",
                "species_cluster",
                "compound",
                "compound_class",
                "rt",
                *cov_cols_sorted,
            ]
        )
        empty.to_csv(tmp_output, index=False)

    tmp_output.replace(output_path)
    print(
        f"[make_rt_prod_csv] Wrote {rows_written:,} rows, "
        f"{8 + len(cov_cols_sorted)} columns to {output_path}"
    )


if __name__ == "__main__":
    main()
