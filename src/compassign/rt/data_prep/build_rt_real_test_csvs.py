#!/usr/bin/env python3
"""
Build RT production CSVs for real held-out test data using merged Pachyderm
training Parquets and the global train/test split.

Steps per lib:
  1) Filter merged_training_all_lib<lib>.parquet to rows whose sample_set_id
     is marked as 'test' in data/split_outputs/train_test_split_all.csv.
 2) Attach chem_id + compound_class. Rows with missing chemistry metadata are kept
     (chemical_id/compound_class filled with -1) to preserve RT model coverage.
 3) Convert the filtered Parquet into an RT production CSV compatible with
     the ridge RT trainers.

This script does NOT run RT model evaluation; use `./src/compassign/rt/eval.sh`
after training to evaluate coefficient-summary artifacts on the generated CSVs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa


_HERE = Path(__file__).resolve()
for _parent in _HERE.parents:
    if (_parent / "pyproject.toml").exists():
        REPO_ROOT = _parent
        break
else:
    REPO_ROOT = Path.cwd().resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build RT production CSVs for real held-out test data per library.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split-csv",
        type=Path,
        default=REPO_ROOT / "data" / "split_outputs" / "train_test_split_all.csv",
        help="Global train/test split CSV (lib,sample_set_id,group,fold)",
    )
    parser.add_argument(
        "--merged-208",
        type=Path,
        default=REPO_ROOT
        / "repo_export"
        / "merged_training"
        / "merged_training_all_lib208.parquet",
        help="Merged training Parquet for lib 208",
    )
    parser.add_argument(
        "--merged-209",
        type=Path,
        default=REPO_ROOT
        / "repo_export"
        / "merged_training"
        / "merged_training_all_lib209.parquet",
        help="Merged training Parquet for lib 209",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=REPO_ROOT / "repo_export",
        help="Output root (writes under repo_export/lib{lib}/realtest/).",
    )
    parser.add_argument(
        "--chem-classes",
        type=Path,
        default=REPO_ROOT / "resources" / "metabolites" / "chem_classes_k32.parquet",
        help="Chem classes Parquet/CSV (chem_id, compound_class)",
    )
    return parser.parse_args()


def filter_real_test(
    merged_path: Path, split_df: pd.DataFrame, lib_id: int, out_parquet: Path
) -> None:
    if not merged_path.exists():
        raise SystemExit(f"Merged Parquet for lib {lib_id} not found: {merged_path}")

    lib_split = split_df[(split_df["lib"] == lib_id) & (split_df["fold"] == "test")].copy()
    if lib_split.empty:
        raise SystemExit(f"No test rows found in split CSV for lib {lib_id}")

    test_ssids = lib_split["sample_set_id"].astype(int).unique().tolist()
    print(f"[realtest] lib {lib_id}: {len(test_ssids)} unique sample_set_id in test split")

    # Streaming filter by row group to keep memory usage bounded. We avoid
    # materialising the entire filtered table at once.
    pf = pq.ParquetFile(merged_path)
    values = {str(int(x)) for x in test_ssids}

    writer: pq.ParquetWriter | None = None
    n_rows = 0

    for rg_idx in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg_idx)
        df = tbl.to_pandas()
        if "sample_set_id" not in df.columns:
            raise SystemExit(f"[realtest] lib {lib_id}: 'sample_set_id' column missing")
        mask = df["sample_set_id"].astype(str).isin(values)
        if not mask.any():
            continue
        df_f = df[mask].copy()
        n_rows += len(df_f)
        table_f = pa.Table.from_pandas(df_f, preserve_index=False)
        if writer is None:
            out_parquet.parent.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(out_parquet, table_f.schema, compression="zstd")
        writer.write_table(table_f)

    if writer is not None:
        writer.close()

    if n_rows == 0:
        raise SystemExit(f"[realtest] lib {lib_id}: filter produced zero rows; check split/keys.")

    print(
        f"[realtest] lib {lib_id}: filtered to {n_rows:,} rows "
        f"out of merged_training at {merged_path}"
    )
    print(f"[realtest] lib {lib_id}: wrote filtered Parquet to {out_parquet}")


def run_attach_chem_classes(filtered_parq: Path, lib_id: int, chem_classes: Path) -> Path:
    lib_map = (
        REPO_ROOT
        / "repo_export"
        / f"lib{lib_id}"
        / "mappings"
        / f"lib_comp_chem_mapping_lib{lib_id}.csv"
    )
    if not lib_map.exists():
        raise SystemExit(f"Lib mapping CSV not found for lib {lib_id}: {lib_map}")

    import sys
    import subprocess

    cmd = [
        sys.executable,
        "-m",
        "compassign.rt.data_prep.attach_chem_classes_and_filter",
        "--input",
        str(filtered_parq),
        "--lib-mapping",
        str(lib_map),
        "--classes",
        str(chem_classes),
    ]
    print(f"[realtest] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    out_parq = filtered_parq.with_name(filtered_parq.stem + "_chemclass.parquet")
    if not out_parq.exists():
        raise SystemExit(f"[realtest] Expected chemclass Parquet not found: {out_parq}")
    return out_parq


def run_make_rt_csv(chemclass_parq: Path, lib_id: int, out_csv: Path) -> None:
    # Use the current strict species mapping for this lib.
    # This ensures consistency with cap dataset generation and multilevel modeling:
    #   - lib209: species_raw = species_matrix_type, species_group_raw = group
    #   - lib208: species_raw = JCJ_COMBO,          species_group_raw = group
    species_mapping = (
        REPO_ROOT
        / "repo_export"
        / f"lib{lib_id}"
        / "species_mapping"
        / f"merged_training_all_lib{lib_id}_species_mapping.csv"
    )

    if not species_mapping.exists():
        raise SystemExit(f"Species mapping CSV not found: {species_mapping}")

    import sys
    import subprocess

    cmd = [
        sys.executable,
        "-m",
        "compassign.rt.data_prep.make_rt_prod_csv_from_merged",
        "--input",
        str(chemclass_parq),
        "--species-mapping",
        str(species_mapping),
        "--output",
        str(out_csv),
    ]
    print(f"[realtest] Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()

    split_df = pd.read_csv(args.split_csv)
    required_cols = {"lib", "sample_set_id", "group", "fold"}
    if not required_cols.issubset(split_df.columns):
        raise SystemExit(f"Split CSV must contain columns {sorted(required_cols)}")

    # Lib 208
    out_208_dir = args.out_root / "lib208" / "realtest"
    real_208_parq = out_208_dir / "merged_training_realtest_lib208.parquet"
    filter_real_test(args.merged_208, split_df, lib_id=208, out_parquet=real_208_parq)
    chemclass_208 = run_attach_chem_classes(
        real_208_parq, lib_id=208, chem_classes=args.chem_classes
    )
    rt_csv_208 = out_208_dir / "merged_training_realtest_lib208_chemclass_rt_prod.csv"
    run_make_rt_csv(chemclass_208, lib_id=208, out_csv=rt_csv_208)

    # Lib 209
    out_209_dir = args.out_root / "lib209" / "realtest"
    real_209_parq = out_209_dir / "merged_training_realtest_lib209.parquet"
    filter_real_test(args.merged_209, split_df, lib_id=209, out_parquet=real_209_parq)
    chemclass_209 = run_attach_chem_classes(
        real_209_parq, lib_id=209, chem_classes=args.chem_classes
    )
    rt_csv_209 = out_209_dir / "merged_training_realtest_lib209_chemclass_rt_prod.csv"
    run_make_rt_csv(chemclass_209, lib_id=209, out_csv=rt_csv_209)

    print("[realtest] Done. RT production CSVs written to:")
    print(f"  {rt_csv_208}")
    print(f"  {rt_csv_209}")


if __name__ == "__main__":
    main()
