#!/usr/bin/env python3
"""
Summarise RT cap training datasets under repo_export.

For each lib (e.g. lib208, lib209) and cap directory (cap5, cap10, ...),
this script computes:
  - total number of rows (data points)
  - number of unique (species, compound) combinations
  - the same statistics grouped by species cluster / group

Results are written as two CSVs under the specified output directory:
  - rt_cap_summary.csv
      lib,cap,n_rows,n_species_compound_pairs
  - rt_cap_by_cluster_summary.csv
      lib,cap,cluster,n_rows,n_species_compound_pairs

Usage (from repo root):
  python -m compassign.rt.data_prep.summarize_rt_cap_datasets \
      --repo-export-dir repo_export \
      --output-dir output/rt_cap_stats
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


SPECIES_COL_CANDIDATES = ["species", "species_id"]
COMPOUND_COL_CANDIDATES = ["compound", "compound_id", "comp_id"]
CLUSTER_COL_CANDIDATES = ["species_cluster", "cluster", "species_group"]


@dataclass
class CapSummary:
    lib: str
    cap: int
    n_rows: int
    n_pairs: int


@dataclass
class ClusterSummary:
    lib: str
    cap: int
    cluster: int
    cluster_label: Optional[str]
    n_rows: int
    n_pairs: int


def _pick_column(candidates: List[str], columns: List[str]) -> Optional[str]:
    for name in candidates:
        if name in columns:
            return name
    return None


def _find_rt_csv(cap_dir: Path, lib: str) -> Optional[Path]:
    """
    Locate the chemclass_rt_prod CSV for a given lib/cap directory.

    We prefer the merged_training_all_libXXX_capYYY_chemclass_rt_prod.csv schema,
    but fall back to any *_chemclass_rt_prod.csv if needed.
    """
    all_pattern = f"merged_training_all_{lib}_*_chemclass_rt_prod.csv"
    candidates = sorted(cap_dir.glob(all_pattern))
    if not candidates:
        candidates = sorted(cap_dir.glob("*_chemclass_rt_prod.csv"))
    return candidates[0] if candidates else None


def summarise_lib_caps(
    lib_dir: Path,
    lib_name: str,
    cluster_labels: Optional[Dict[int, str]] = None,
) -> tuple[List[CapSummary], List[ClusterSummary]]:
    cap_summaries: List[CapSummary] = []
    cluster_summaries: List[ClusterSummary] = []

    for cap_dir in sorted(p for p in lib_dir.iterdir() if p.is_dir() and p.name.startswith("cap")):
        cap_name = cap_dir.name  # e.g. "cap200"
        # Extract numeric portion of cap (e.g. "cap200" -> 200)
        try:
            cap_num = int(cap_name.replace("cap", "", 1))
        except ValueError:
            logger.warning(
                "Unrecognised cap directory name %s under %s, skipping", cap_name, lib_dir
            )
            continue
        csv_path = _find_rt_csv(cap_dir, lib_name)
        if csv_path is None:
            logger.info("No chemclass_rt_prod CSV found for %s/%s, skipping", lib_name, cap_name)
            continue

        logger.info("Loading %s", csv_path)
        df = pd.read_csv(csv_path)
        n_rows = len(df)

        species_col = _pick_column(SPECIES_COL_CANDIDATES, list(df.columns))
        compound_col = _pick_column(COMPOUND_COL_CANDIDATES, list(df.columns))
        if species_col is None or compound_col is None:
            raise ValueError(
                f"Could not find species/compound columns in {csv_path}. "
                f"Columns={list(df.columns)}"
            )

        n_pairs = df[[species_col, compound_col]].drop_duplicates().shape[0]
        cap_summaries.append(CapSummary(lib=lib_name, cap=cap_num, n_rows=n_rows, n_pairs=n_pairs))

        cluster_col = _pick_column(CLUSTER_COL_CANDIDATES, list(df.columns))
        if cluster_col is None:
            logger.info(
                "No species cluster column found in %s, skipping cluster breakdown", csv_path
            )
            continue

        grouped = df.groupby(cluster_col, sort=True)
        for cluster, sub in grouped:
            try:
                cluster_id = int(cluster)
            except Exception:
                # Fall back to string representation if casting fails
                logger.warning("Non-integer cluster label %r in %s", cluster, csv_path)
                continue
            n_rows_c = len(sub)
            n_pairs_c = sub[[species_col, compound_col]].drop_duplicates().shape[0]
            label = cluster_labels.get(cluster_id) if cluster_labels else None
            cluster_summaries.append(
                ClusterSummary(
                    lib=lib_name,
                    cap=cap_num,
                    cluster=cluster_id,
                    cluster_label=label,
                    n_rows=n_rows_c,
                    n_pairs=n_pairs_c,
                )
            )

    return cap_summaries, cluster_summaries


def write_csv_summaries(
    cap_summaries: List[CapSummary],
    cluster_summaries: List[ClusterSummary],
    output_dir: Path,
    full_rows_by_lib: Dict[str, int],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    cap_records: List[Dict[str, object]] = [
        {
            "lib": s.lib,
            "cap": s.cap,
            "n_rows": s.n_rows,
            "n_species_compound_pairs": s.n_pairs,
            "rows_pct_of_full": (
                (s.n_rows / full_rows_by_lib[s.lib]) * 100.0
                if s.lib in full_rows_by_lib and full_rows_by_lib[s.lib] > 0
                else None
            ),
        }
        for s in cap_summaries
    ]
    cap_df = pd.DataFrame(cap_records)
    if "rows_pct_of_full" in cap_df.columns:
        cap_df["rows_pct_of_full"] = cap_df["rows_pct_of_full"].round(2)
    cap_df.sort_values(["lib", "cap"], inplace=True)
    cap_path = output_dir / "rt_cap_summary.csv"
    cap_df.to_csv(cap_path, index=False)
    logger.info("Wrote cap summary to %s", cap_path)

    cluster_records: List[Dict[str, object]] = [
        {
            "lib": s.lib,
            "cap": s.cap,
            "cluster": s.cluster,
            "cluster_label": s.cluster_label,
            "n_rows": s.n_rows,
            "n_species_compound_pairs": s.n_pairs,
        }
        for s in cluster_summaries
    ]
    cluster_df = pd.DataFrame(cluster_records)
    cluster_df.sort_values(["lib", "cap", "cluster"], inplace=True)
    cluster_path = output_dir / "rt_cap_by_cluster_summary.csv"
    cluster_df.to_csv(cluster_path, index=False)
    logger.info("Wrote cluster summary to %s", cluster_path)

    # Also print the high-level cap summary table to stdout for quick inspection.
    print()
    print("RT cap summary (rows, pairs, and fraction of full merged data):")
    print(cap_df.to_csv(index=False).strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarise RT cap training datasets under repo_export.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--repo-export-dir",
        type=Path,
        default=Path("repo_export"),
        help="Directory containing libXXX/capYYY RT training exports.",
    )
    parser.add_argument(
        "--libs",
        nargs="*",
        default=["lib208", "lib209"],
        help="Library IDs to summarise (subdirectories under repo_export).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/rt_cap_stats"),
        help="Directory to write CSV summaries into.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    args = parse_args()

    repo_export_dir: Path = args.repo_export_dir
    libs: List[str] = args.libs
    output_dir: Path = args.output_dir

    # Pre-compute full row counts for each lib from the uncapped merged Parquet,
    # and derive human-readable cluster labels from the species mapping.
    full_rows_by_lib: Dict[str, int] = {}
    cluster_labels_by_lib: Dict[str, Dict[int, str]] = {}

    for lib in libs:
        merged_path = repo_export_dir / "merged_training" / f"merged_training_all_{lib}.parquet"
        if merged_path.exists():
            try:
                pf = pq.ParquetFile(merged_path)
                full_rows_by_lib[lib] = pf.metadata.num_rows
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("Failed to read Parquet metadata for %s: %s", merged_path, exc)

        sm_dir = repo_export_dir / lib / "species_mapping"
        if sm_dir.is_dir():
            candidates = sorted(sm_dir.glob(f"merged_training_*_{lib}_species_mapping.csv"))
            if candidates:
                sm_path = candidates[0]
                try:
                    sm_df = pd.read_csv(sm_path)
                    if {"species_cluster", "species_group_raw"}.issubset(sm_df.columns):
                        mapping: Dict[int, str] = {}
                        for _, row in (
                            sm_df[["species_cluster", "species_group_raw"]].dropna().iterrows()
                        ):
                            cid = int(row["species_cluster"])
                            if cid not in mapping:
                                mapping[cid] = str(row["species_group_raw"])
                        if mapping:
                            cluster_labels_by_lib[lib] = mapping
                except Exception as exc:  # pragma: no cover - defensive
                    logger.warning("Failed to read species mapping for %s: %s", lib, exc)

    all_cap_summaries: List[CapSummary] = []
    all_cluster_summaries: List[ClusterSummary] = []

    for lib in libs:
        lib_dir = repo_export_dir / lib
        if not lib_dir.is_dir():
            logger.warning(
                "Library directory %s not found under %s, skipping", lib, repo_export_dir
            )
            continue
        cap_summaries, cluster_summaries = summarise_lib_caps(
            lib_dir,
            lib,
            cluster_labels=cluster_labels_by_lib.get(lib),
        )
        all_cap_summaries.extend(cap_summaries)
        all_cluster_summaries.extend(cluster_summaries)

    if not all_cap_summaries:
        logger.error("No cap datasets found. Check --repo-export-dir and --libs.")
        return

    write_csv_summaries(all_cap_summaries, all_cluster_summaries, output_dir, full_rows_by_lib)


if __name__ == "__main__":
    main()
