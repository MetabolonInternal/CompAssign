#!/usr/bin/env python3
"""
Attach CHEM_ID and chemistry-based compound_class to per-lib cap5 Parquets,
and filter out compounds lacking embeddings/classes.

Inputs per lib:
  - cap5 Parquet (per-lib, per species×compound capped)
  - lib_comp_chem_mapping_lib<lib>.csv  (lib_id, comp_id, chemical_id)
  - resources/metabolites/chem_classes_kK.parquet (chem_id, compound_class)

Outputs:
  - A filtered Parquet with all original columns plus:
      - chemical_id
      - compound_class
    Rows where chemical_id has no class are dropped (including chem_id==0).

Usage example:
  python scripts/pipelines/attach_chem_classes_and_filter.py \\
      --input repo_export/merged_training_de194c2cc2114efaa1075ccf7539d0cb_lib209_cap5.parquet \\
      --lib-mapping repo_export/lib_comp_chem_mapping_lib209.csv \\
      --classes resources/metabolites/chem_classes_k32.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attach chem_id + compound_class to per-lib cap5 Parquet and filter unmapped chems."
    )
    parser.add_argument("--input", type=Path, required=True, help="Per-lib cap5 Parquet")
    parser.add_argument(
        "--lib-mapping",
        type=Path,
        required=True,
        help="CSV with lib_id, comp_id, chemical_id",
    )
    parser.add_argument(
        "--classes",
        type=Path,
        required=True,
        help="Parquet/CSV with chem_id, compound_class",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Parquet (default: <stem>_chemclass.parquet)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input Parquet not found: {args.input}")
    if not args.lib_mapping.exists():
        raise SystemExit(f"Lib mapping CSV not found: {args.lib_mapping}")
    if not args.classes.exists():
        raise SystemExit(f"Classes file not found: {args.classes}")

    # Load data
    df = pd.read_parquet(args.input)
    map_df = pd.read_csv(args.lib_mapping)
    if args.classes.suffix == ".parquet":
        classes_df = pd.read_parquet(args.classes)
    else:
        classes_df = pd.read_csv(args.classes)

    for col in ["comp_id"]:
        if col not in df.columns:
            raise SystemExit(f"Input Parquet {args.input} missing required column '{col}'")
    if "lib_id" not in df.columns and "lib_id" in map_df.columns:
        # If per-lib files don't carry lib_id, infer from mapping
        lib_ids = map_df["lib_id"].dropna().unique()
        if len(lib_ids) == 1:
            df["lib_id"] = lib_ids[0]
        else:
            raise SystemExit("Mapping has multiple lib_id values but input has no lib_id column")

    # Normalize types for join
    df["comp_id"] = df["comp_id"].astype("Int64")
    map_df["comp_id"] = map_df["comp_id"].astype("Int64")
    if "lib_id" in df.columns and "lib_id" in map_df.columns:
        on_cols = ["lib_id", "comp_id"]
    else:
        on_cols = ["comp_id"]

    merged = df.merge(map_df[on_cols + ["chemical_id"]], on=on_cols, how="left")

    # Drop rows with no chemical_id mapping (true unmapped compounds)
    before_map = len(merged)
    merged = merged.dropna(subset=["chemical_id"]).copy()
    after_map = len(merged)
    dropped_map = before_map - after_map
    if dropped_map > 0:
        print(
            f"[attach_chem_classes] Dropped {dropped_map:,} rows with comp_id not in mapping "
            f"({before_map:,} -> {after_map:,})"
        )
    merged["chemical_id"] = merged["chemical_id"].astype("Int64")

    # Attach compound_class from chem_classes
    if not {"chem_id", "compound_class"}.issubset(classes_df.columns):
        raise SystemExit("Classes file must contain 'chem_id' and 'compound_class' columns")
    classes_df = classes_df[["chem_id", "compound_class"]].copy()
    classes_df["chem_id"] = classes_df["chem_id"].astype("Int64")

    merged = merged.merge(
        classes_df.rename(columns={"chem_id": "chemical_id"}),
        on="chemical_id",
        how="left",
    )

    # Filter out rows with missing compound_class (i.e., chem_ids without embeddings/classes)
    before = len(merged)
    filtered = merged.dropna(subset=["compound_class"]).copy()
    after = len(filtered)
    dropped = before - after

    print(
        f"[attach_chem_classes] {args.input.name}: rows before={before:,}, "
        f"after={after:,} (dropped {dropped:,} rows without compound_class)"
    )

    out = args.output or args.input.with_name(args.input.stem + "_chemclass.parquet")
    filtered.to_parquet(out, index=False)
    print(f"[attach_chem_classes] Wrote filtered Parquet to {out}")


if __name__ == "__main__":
    main()
