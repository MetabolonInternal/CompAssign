#!/usr/bin/env python3
"""
Validate RT metadata mappings and materialize a species/species_cluster mapping CSV.

This is the single source of truth for how we encode `species` and `species_cluster`
in the RT production CSVs (and thus how we define subgroup/supercategory levels for
hierarchical modeling and for cap sampling).

Inputs:
- Per-lib merged training Parquet: repo_export/merged_training/merged_training_all_lib{lib}.parquet
- Split input CSVs:
  - data/split_outputs/data/_lib_208_input.csv
  - data/split_outputs/data/_lib_209_input.csv

Outputs:
- repo_export/lib{lib}/species_mapping/<name>.csv with columns:
    sample_set_id, species_raw, species_group_raw, species, species_cluster

Notes:
- `species_cluster` should correspond to a "supercategory" (e.g. human blood).
- `species` should correspond to a curate-like subgroup nested within `species_cluster`.
  For lib209 this is naturally `species_matrix_type` (e.g. human_plasma).
  For lib208 the closest nested proxy is `JCJ_COMBO` (species+matrix+type).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Tuple

import pandas as pd
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Validate RT mapping inputs and write an encoded species mapping CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--libs",
        type=str,
        default="208,209",
        help="Comma-separated list of libs to process (e.g. '208,209' or '209').",
    )
    ap.add_argument(
        "--output-name",
        type=str,
        default="merged_training_all_lib{lib}_species_mapping.csv",
        help="Output filename template written under repo_export/lib{lib}/species_mapping/.",
    )
    return ap.parse_args()


def collect_sample_set_ids(parquet_path: Path) -> pd.Series:
    pf = pq.ParquetFile(parquet_path)
    ssids = []
    for i in range(pf.num_row_groups):
        tbl = pf.read_row_group(i, columns=["sample_set_id"])
        df = tbl.to_pandas()
        ssids.append(df["sample_set_id"].astype(str))
    all_ssids = pd.concat(ssids, ignore_index=True).dropna().drop_duplicates().sort_values()
    return all_ssids


def _norm_key(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def _find_col(df: pd.DataFrame, key: str) -> str:
    want = _norm_key(key)
    for c in df.columns:
        if _norm_key(c) == want:
            return c
    raise KeyError(
        f"Could not find column '{key}' (normalized='{want}') in CSV columns: {list(df.columns)}"
    )


def load_mapping(
    lib_id: int,
    csv_path: Path,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    ssid_col = _find_col(df, "sample_set_id")
    group_col = _find_col(df, "group")

    if lib_id == 209:
        # Curate-like subgroup id (nested within supercategory).
        species_col = _find_col(df, "species_matrix_type")
        species_raw = df[species_col].astype(str)
    elif lib_id == 208:
        # Curate-like subgroup proxy (nested within supercategory).
        # JCJ_COMBO is effectively (matrix + matrix_type + organism + species) and is nested under `group`.
        species_col = _find_col(df, "jcj_combo")
        species_raw = df[species_col].astype(str)
    else:
        raise ValueError(f"Unsupported lib_id {lib_id} for mapping loader")

    mapping = df[[ssid_col, group_col]].copy()
    mapping["species_raw"] = species_raw
    mapping.rename(
        columns={ssid_col: "sample_set_id", group_col: "species_group_raw"}, inplace=True
    )
    mapping["sample_set_id"] = mapping["sample_set_id"].astype(str)
    return mapping[["sample_set_id", "species_raw", "species_group_raw"]]


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


def _require_nested_species_mapping(mapping: pd.DataFrame) -> None:
    """Ensure each species_raw belongs to exactly one non-null species_group_raw."""
    df = mapping.dropna(subset=["species_group_raw"]).copy()
    if df.empty:
        raise ValueError("Mapping has no rows with a non-null species_group_raw (group).")
    nunique = df.groupby("species_raw")["species_group_raw"].nunique(dropna=True)
    bad = nunique[nunique > 1]
    if not bad.empty:
        examples = bad.head(10).index.tolist()
        msg = ["species_raw must map to a single species_group_raw; found violations:"]
        for s in examples:
            vals = (
                df.loc[df["species_raw"] == s, "species_group_raw"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            msg.append(f"  - {s!r} -> {vals}")
        raise ValueError("\n".join(msg))


def main() -> None:
    args = parse_args()
    libs: list[int] = []
    for tok in str(args.libs).split(","):
        tok = tok.strip()
        if not tok:
            continue
        libs.append(int(tok))
    if not libs:
        raise SystemExit("--libs must include at least one lib id")

    repo_root = Path(__file__).resolve()
    for parent in repo_root.parents:
        if (parent / "pyproject.toml").exists():
            repo_root = parent
            break
    else:
        repo_root = Path.cwd().resolve()
    repo_export = repo_root / "repo_export"
    data_root = repo_root / "data" / "split_outputs" / "data"

    def configs_for(libs_in: Iterable[int]) -> Dict[int, Tuple[Path, Path]]:
        cfg: Dict[int, Tuple[Path, Path]] = {}
        for lib_id in libs_in:
            parquet_path = (
                repo_export / "merged_training" / f"merged_training_all_lib{lib_id}.parquet"
            )
            csv_path = data_root / f"_lib_{lib_id}_input.csv"
            cfg[int(lib_id)] = (parquet_path, csv_path)
        return cfg

    configs = configs_for(libs)

    for lib_id, (parquet_path, csv_path) in configs.items():
        print(f"\n[check] lib_id={lib_id}")
        if not parquet_path.exists():
            print(f"  Parquet not found: {parquet_path}")
            continue
        if not csv_path.exists():
            print(f"  Mapping CSV not found: {csv_path}")
            continue

        ssids = collect_sample_set_ids(parquet_path)
        mapping_all = load_mapping(lib_id, csv_path)

        ssid_set = set(ssids.tolist())
        map_all_set = set(mapping_all["sample_set_id"].tolist())
        missing_in_map = sorted(ssid_set - map_all_set, key=lambda x: int(x))
        if missing_in_map:
            raise SystemExit(
                f"Missing sample_set_id values in mapping input ({len(missing_in_map)}): "
                f"first 20 -> {missing_in_map[:20]}"
            )

        # Enforce: all encoded rows must have a non-null group.
        dropped_missing_group = mapping_all["species_group_raw"].isna()
        n_dropped = int(dropped_missing_group.sum())
        if n_dropped > 0:
            print(f"  Dropping {n_dropped} rows with missing group (species_cluster).")
        mapping = mapping_all.loc[~dropped_missing_group].copy()

        _require_nested_species_mapping(mapping)

        map_set = set(mapping["sample_set_id"].tolist())
        extra_in_map = sorted(map_set - ssid_set, key=lambda x: int(x))

        print(f"  sample_set_id in Parquet: {len(ssid_set)}")
        print(f"  sample_set_id in mapping (after drop-missing-group): {len(map_set)}")
        if extra_in_map:
            print(f"  Unused in data ({len(extra_in_map)}): first 20 -> {extra_in_map[:20]}")

        encoded = build_encoded_mapping(mapping)
        out_dir = repo_export / f"lib{lib_id}" / "species_mapping"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = str(args.output_name).format(lib=lib_id)
        out_csv = out_dir / out_name
        encoded.to_csv(out_csv, index=False)
        print(f"  Wrote encoded species mapping to {out_csv}")


if __name__ == "__main__":
    main()
