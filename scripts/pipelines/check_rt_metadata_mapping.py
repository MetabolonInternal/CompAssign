#!/usr/bin/env python3
"""
Validate that every sample_set_id in per-lib merged Parquet has a mapping row
in the corresponding lib input CSV, and materialize a simple species/species_group
mapping table for later use.

Lib 209:
  - mapping CSV: data/split_outputs/data/_lib_209_input.csv
  - species := species_matrix_type
  - species_group := group

Lib 208:
  - mapping CSV: data/split_outputs/data/_lib_208_input.csv
  - species := JCJ_SPECIES
  - species_group := group
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import pyarrow.parquet as pq


def collect_sample_set_ids(parquet_path: Path) -> pd.Series:
    pf = pq.ParquetFile(parquet_path)
    ssids = []
    for i in range(pf.num_row_groups):
        tbl = pf.read_row_group(i, columns=["sample_set_id"])
        df = tbl.to_pandas()
        ssids.append(df["sample_set_id"].astype(str))
    all_ssids = pd.concat(ssids, ignore_index=True).dropna().drop_duplicates().sort_values()
    return all_ssids


def load_mapping(lib_id: int, csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if lib_id == 209:
        species_col = "species_matrix_type"
        group_col = "group"
        ssid_col = "sample_set_id"
    elif lib_id == 208:
        # Normalize column names for JCJ_SPECIES and sample_set_id
        # Some headers contain spaces/upper-case; use case-insensitive match.
        cols_lower = {c.lower(): c for c in df.columns}
        ssid_col = cols_lower.get("sample_set_id", cols_lower.get("sample_set_id".lower(), "sample_set_id"))
        # JCJ SPECIES might appear with space; look for 'jcj species' normalized
        jcj_key = None
        for c in df.columns:
            if c.strip().lower().replace(" ", "_") == "jcj_species":
                jcj_key = c
                break
        if jcj_key is None:
            raise ValueError("Could not find JCJ_SPECIES column in 208 mapping CSV")
        species_col = jcj_key
        group_col = "group"
    else:
        raise ValueError(f"Unsupported lib_id {lib_id} for mapping loader")

    mapping = df[[ssid_col, species_col, group_col]].copy()
    mapping.columns = ["sample_set_id", "species_raw", "species_group_raw"]
    mapping["sample_set_id"] = mapping["sample_set_id"].astype(str)
    return mapping


def build_encoded_mapping(mapping: pd.DataFrame) -> pd.DataFrame:
    species_vals = sorted({str(s) for s in mapping["species_raw"].unique()})
    group_vals = sorted({str(g) for g in mapping["species_group_raw"].unique() if pd.notna(g)})
    species_codes = {s: i for i, s in enumerate(species_vals)}
    group_codes = {g: i for i, g in enumerate(group_vals)}

    out = mapping.copy()
    out["species_raw"] = out["species_raw"].astype(str)
    out["species_group_raw"] = out["species_group_raw"].where(out["species_group_raw"].notna())
    out["species_raw"] = out["species_raw"].astype(str)
    out["species_group_raw"] = out["species_group_raw"].astype(str)
    out["species"] = out["species_raw"].map(species_codes.__getitem__)
    # Map only non-null groups; leave species_cluster NaN where group is missing
    out["species_cluster"] = out["species_group_raw"].map(lambda g: group_codes.get(g))
    return out


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    repo_export = repo_root / "repo_export"
    data_root = repo_root / "data" / "split_outputs" / "data"

    configs: Dict[int, Tuple[Path, Path]] = {
        208: (
            repo_export / "merged_training_5684639a28c04bc5af7c4fd1a75e62b5_lib208.parquet",
            data_root / "_lib_208_input.csv",
        ),
        209: (
            repo_export / "merged_training_de194c2cc2114efaa1075ccf7539d0cb_lib209.parquet",
            data_root / "_lib_209_input.csv",
        ),
    }

    for lib_id, (parquet_path, csv_path) in configs.items():
        print(f"\n[check] lib_id={lib_id}")
        if not parquet_path.exists():
            print(f"  Parquet not found: {parquet_path}")
            continue
        if not csv_path.exists():
            print(f"  Mapping CSV not found: {csv_path}")
            continue

        ssids = collect_sample_set_ids(parquet_path)
        mapping = load_mapping(lib_id, csv_path)

        ssid_set = set(ssids.tolist())
        map_set = set(mapping["sample_set_id"].tolist())

        missing_in_map = sorted(ssid_set - map_set, key=lambda x: int(x))
        extra_in_map = sorted(map_set - ssid_set, key=lambda x: int(x))

        print(f"  sample_set_id in Parquet: {len(ssid_set)}")
        print(f"  sample_set_id in mapping: {len(map_set)}")
        if missing_in_map:
            print(f"  MISSING in mapping ({len(missing_in_map)}): first 20 -> {missing_in_map[:20]}")
        else:
            print("  All sample_set_id values have a mapping.")
        if extra_in_map:
            print(f"  Unused in data ({len(extra_in_map)}): first 20 -> {extra_in_map[:20]}")

        encoded = build_encoded_mapping(mapping)
        out_csv = parquet_path.with_name(parquet_path.stem + "_species_mapping.csv")
        encoded.to_csv(out_csv, index=False)
        print(f"  Wrote encoded species mapping to {out_csv}")


if __name__ == "__main__":
    main()
