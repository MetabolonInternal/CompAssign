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
  python -m compassign.rt.data_prep.attach_chem_classes_and_filter \\
      --input repo_export/merged_training_de194c2cc2114efaa1075ccf7539d0cb_lib209_cap5.parquet \\
      --lib-mapping repo_export/lib_comp_chem_mapping_lib209.csv \\
      --classes resources/metabolites/chem_classes_k32.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


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


def _read_classes(classes_path: Path) -> pd.DataFrame:
    if classes_path.suffix == ".parquet":
        classes_df = pd.read_parquet(classes_path)
    else:
        classes_df = pd.read_csv(classes_path)
    if not {"chem_id", "compound_class"}.issubset(classes_df.columns):
        raise SystemExit("Classes file must contain 'chem_id' and 'compound_class' columns")
    classes_df = classes_df[["chem_id", "compound_class"]].copy()
    classes_df["chem_id"] = classes_df["chem_id"].astype("Int64")
    classes_df["compound_class"] = classes_df["compound_class"].astype("Int64")
    return classes_df


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input Parquet not found: {args.input}")
    if not args.lib_mapping.exists():
        raise SystemExit(f"Lib mapping CSV not found: {args.lib_mapping}")
    if not args.classes.exists():
        raise SystemExit(f"Classes file not found: {args.classes}")

    pf = pq.ParquetFile(args.input)
    schema = pf.schema_arrow
    if "comp_id" not in schema.names:
        raise SystemExit(f"Input Parquet {args.input} missing required column 'comp_id'")

    map_df = pd.read_csv(args.lib_mapping)
    classes_df = _read_classes(args.classes)

    has_lib_id = "lib_id" in schema.names and "lib_id" in map_df.columns
    if not has_lib_id and "lib_id" in map_df.columns:
        lib_ids = map_df["lib_id"].dropna().unique()
        if len(lib_ids) != 1:
            raise SystemExit("Mapping has multiple lib_id values but input has no lib_id column")
        inferred_lib_id = lib_ids[0]
    else:
        inferred_lib_id = None

    on_cols = ["lib_id", "comp_id"] if has_lib_id else ["comp_id"]
    map_df = map_df[on_cols + ["chemical_id"]].copy()
    map_df["comp_id"] = map_df["comp_id"].astype("Int64")
    if "lib_id" in map_df.columns:
        map_df["lib_id"] = map_df["lib_id"].astype("Int64")
    map_df["chemical_id"] = map_df["chemical_id"].astype("Int64")
    classes_df = classes_df.rename(columns={"chem_id": "chemical_id"})

    cols_to_read: List[str] = list(schema.names)
    out = args.output or args.input.with_name(args.input.stem + "_chemclass.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()
    writer: pq.ParquetWriter | None = None

    total_before_map = 0
    total_after_map = 0
    total_before_class = 0
    total_after_class = 0

    try:
        for batch in pf.iter_batches(batch_size=200_000, columns=cols_to_read):
            df = batch.to_pandas()
            if df.empty:
                continue

            total_before_map += int(len(df))
            if inferred_lib_id is not None:
                df["lib_id"] = inferred_lib_id

            df["comp_id"] = df["comp_id"].astype("Int64")
            if "lib_id" in df.columns:
                df["lib_id"] = df["lib_id"].astype("Int64")

            merged = df.merge(map_df, on=on_cols, how="left")
            merged = merged.dropna(subset=["chemical_id"]).copy()
            total_after_map += int(len(merged))
            if merged.empty:
                continue

            merged["chemical_id"] = merged["chemical_id"].astype("Int64")
            merged = merged.merge(classes_df, on="chemical_id", how="left")
            total_before_class += int(len(merged))
            filtered = merged.dropna(subset=["compound_class"]).copy()
            total_after_class += int(len(filtered))
            if filtered.empty:
                continue

            table = pa.Table.from_pandas(filtered, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    dropped_map = total_before_map - total_after_map
    if dropped_map > 0:
        print(
            f"[attach_chem_classes] Dropped {dropped_map:,} rows with comp_id not in mapping "
            f"({total_before_map:,} -> {total_after_map:,})"
        )
    dropped = total_before_class - total_after_class
    print(
        f"[attach_chem_classes] {args.input.name}: rows before={total_before_class:,}, "
        f"after={total_after_class:,} (dropped {dropped:,} rows without compound_class)"
    )

    if total_after_class == 0:
        raise SystemExit(f"[attach_chem_classes] No rows written for {args.input}")
    print(f"[attach_chem_classes] Wrote filtered Parquet to {out}")


if __name__ == "__main__":
    main()
