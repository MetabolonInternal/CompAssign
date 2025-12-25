#!/usr/bin/env python3
"""
End-to-end RT production data preparation pipeline.

Given a Pachyderm export root and a species-mapping CSV for that export,
this script:

1. Merges Pachyderm shards into a single merged Parquet.
2. Splits the merged Parquet into per-lib Parquets.
3. Down-samples each per-lib file to cap rows per (species_matrix_type, comp_id).
4. Attaches CHEM_ID (chemical_id) and chemistry-based compound_class, and
   filters out unmapped compounds.
5. Builds production-style RT CSVs compatible with train_rt_prod.py.

This wraps the individual helper scripts documented in docs/RT_PROD_PIPELINE.md.

Inputs are minimal:
- export root directory containing create_training_data/ and combine_predictors/
- a species-mapping CSV for the export (sample_set_id,species,species_cluster)

Example:

  python scripts/pipelines/run_rt_prod_prep.py \\
      --export-root repo_export/de194c2cc2114efaa1075ccf7539d0cb \\
      --species-mapping repo_export/merged_training_de194c2cc2114efaa1075ccf7539d0cb_lib209_species_mapping.csv \\
      --cap-per-pair 5
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List


def run(cmd: List[str]) -> None:
    """Run a subprocess, raising on failure."""
    print("[run]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run RT production prep pipeline for one or more export roots."
    )
    p.add_argument(
        "--export-root",
        type=Path,
        required=False,
        help=(
            "Pachyderm export root (contains create_training_data/ and combine_predictors/). "
            "When omitted, all such roots under repo_export/ are processed."
        ),
    )
    p.add_argument(
        "--species-mapping",
        type=Path,
        required=False,
        help=(
            "Species mapping CSV for an export "
            "(sample_set_id,species,species_cluster; any lib for the export is fine). "
            "When omitted, a matching file is inferred under repo_export/."
        ),
    )
    p.add_argument(
        "--cap-per-pair",
        type=int,
        default=5,
        help="Rows per (species_matrix_type, comp_id) pair to keep (default: 5).",
    )
    p.add_argument(
        "--with-cap10",
        action="store_true",
        help="Also generate cap-10 variants alongside cap-5.",
    )
    p.add_argument(
        "--n-classes",
        type=int,
        default=32,
        help="Number of chemistry clusters for chem_classes_kK (default: 32).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    repo_export = repo_root / "repo_export"
    resources_dir = repo_root / "resources" / "metabolites"
    chem_classes_parquet = resources_dir / f"chem_classes_k{args.n_classes}.parquet"

    # Discover export roots if not explicitly provided
    if args.export_root is not None:
        if not args.export_root.exists():
            raise SystemExit(f"Export root not found: {args.export_root}")
        export_roots = [args.export_root]
    else:
        candidates = [
            d
            for d in sorted(repo_export.iterdir())
            if d.is_dir()
            and (d / "create_training_data").is_dir()
            and (d / "combine_predictors").is_dir()
        ]
        if not candidates:
            raise SystemExit(
                "No export roots found under repo_export/. "
                "Expected directories with create_training_data/ and combine_predictors/."
            )
        export_roots = candidates

    # 1. Ensure chem_classes are built (global, once)
    if not chem_classes_parquet.exists():
        run(
            [
                sys.executable,
                "scripts/data_prep/build_chem_classes.py",
                "--n-clusters",
                str(args.n_classes),
            ]
        )

    # 2–5. Per-export pipeline
    for export_root in export_roots:
        print(f"[pipeline] Export root: {export_root}")

        # 2. Merge Pachyderm export into a single merged Parquet
        merged_dir = export_root / "merged_training"
        merged_dir.mkdir(parents=True, exist_ok=True)
        run(
            [
                sys.executable,
                "scripts/pipelines/merge_pachyderm_training.py",
                "--input-dir",
                str(export_root),
            ]
        )

        merged = merged_dir / "merged_training_all.parquet"
        if not merged.exists():
            raise SystemExit(f"Merged Parquet not found after merge step: {merged}")

        # 3. Split merged by lib_id into per-lib Parquets
        run(
            [
                sys.executable,
                "scripts/pipelines/split_merged_by_lib.py",
                str(merged),
            ]
        )

        # Discover per-lib Parquets from this merged file
        lib_parqs = sorted(merged_dir.glob(f"{merged.stem}_lib*.parquet"))
        if not lib_parqs:
            raise SystemExit(f"No per-lib Parquets found for stem {merged.stem} under {merged_dir}")

        # Determine species mapping for this export
        if args.species_mapping is not None:
            species_map = args.species_mapping
        else:
            pattern = f"merged_training_{export_root.name}_lib*_species_mapping.csv"
            candidates = sorted(repo_export.glob(pattern))
            if not candidates:
                raise SystemExit(
                    f"No species mapping found matching {pattern} under {repo_export}. "
                    "Provide --species-mapping explicitly."
                )
            species_map = candidates[0]
        if not species_map.exists():
            raise SystemExit(f"Species mapping CSV not found: {species_map}")

        # 4. For each lib Parquet, run cap sampling, attach classes, and build RT CSV
        for lib_parq in lib_parqs:
            stem = lib_parq.stem  # e.g. merged_training_all_lib209
            print(f"[pipeline] Processing {stem}")

            # (a) cap-5 (always)
            cap5 = merged_dir / f"{stem}_cap{args.cap_per_pair}.parquet"
            run(
                [
                    sys.executable,
                    "scripts/pipelines/sample_per_species_compound.py",
                    str(lib_parq),
                    "--cap-per-pair",
                    str(args.cap_per_pair),
                    "--seed",
                    "42",
                ]
            )

            cap5_chem = merged_dir / f"{stem}_cap{args.cap_per_pair}_chemclass.parquet"
            lib_id_str = stem.split("_lib")[-1]
            lib_mapping = (
                repo_export
                / f"lib{lib_id_str}"
                / "mappings"
                / f"lib_comp_chem_mapping_lib{lib_id_str}.csv"
            )
            if not lib_mapping.exists():
                lib_mapping = repo_export / f"lib_comp_chem_mapping_lib{lib_id_str}.csv"
            if not lib_mapping.exists():
                raise SystemExit(f"Lib mapping not found for lib {lib_id_str}: {lib_mapping}")

            run(
                [
                    sys.executable,
                    "scripts/pipelines/attach_chem_classes_and_filter.py",
                    "--input",
                    str(cap5),
                    "--lib-mapping",
                    str(lib_mapping),
                    "--classes",
                    str(chem_classes_parquet),
                ]
            )

            # Build RT CSV (cap-5)
            run(
                [
                    sys.executable,
                    "scripts/pipelines/make_rt_prod_csv_from_merged.py",
                    "--input",
                    str(cap5_chem),
                    "--species-mapping",
                    str(species_map),
                ]
            )

            # (b) cap-10 (optional)
            if args.with_cap10:
                cap10 = merged_dir / f"{stem}_cap10.parquet"
                run(
                    [
                        sys.executable,
                        "scripts/pipelines/sample_per_species_compound.py",
                        str(lib_parq),
                        "--cap-per-pair",
                        "10",
                        "--seed",
                        "42",
                    ]
                )
                cap10_chem = merged_dir / f"{stem}_cap10_chemclass.parquet"
                run(
                    [
                        sys.executable,
                        "scripts/pipelines/attach_chem_classes_and_filter.py",
                        "--input",
                        str(cap10),
                        "--lib-mapping",
                        str(lib_mapping),
                        "--classes",
                        str(chem_classes_parquet),
                    ]
                )
                run(
                    [
                        sys.executable,
                        "scripts/pipelines/make_rt_prod_csv_from_merged.py",
                        "--input",
                        str(cap10_chem),
                        "--species-mapping",
                        str(species_map),
                    ]
                )

    print("[pipeline] RT production prep complete.")


if __name__ == "__main__":
    main()
