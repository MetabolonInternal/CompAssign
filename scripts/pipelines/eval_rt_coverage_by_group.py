#!/usr/bin/env python3
"""
Compute RT prediction coverage by species group for a trained production model.

Here, "coverage" means:
  - row_coverage: fraction of RT rows in the test CSV for which the model can
    make predictions (species and compound were seen in training);
  - compound_coverage: fraction of unique compounds in the test CSV (per group)
    that the model can predict for.

This does *not* run MCMC or posterior predictions; it only uses the training
metadata (species/compound maps and species-group mapping) together with the
test CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.pipelines import train_rt_prod as train_mod  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute RT prediction coverage by species group.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train-output-dir",
        type=Path,
        required=True,
        help="Output directory from train_rt_prod.py (contains config.json and models/)",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        required=True,
        help="RT production CSV to evaluate coverage on (cap10 or realtest).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Where to write coverage metrics CSV "
            "(default: <train-output-dir>/results/rt_coverage_by_species_group_<label>.csv)."
        ),
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional label for this evaluation (e.g., cap10, realtest) used in output filenames.",
    )
    return parser.parse_args()


def _infer_label(test_csv: Path) -> str | None:
    name = test_csv.name
    if "_cap10_" in name:
        return "cap10"
    if "realtest" in name:
        return "realtest"
    return None


def _load_species_mapping(train_data_csv: Path) -> Dict[int, str]:
    """
    Load sample_set_id -> species_group_raw mapping using the same heuristic as
    eval_rt_prod_streaming.
    """
    import re

    stem = train_data_csv.stem
    root = stem.split("_cap")[0]
    species_mapping_path = train_data_csv.parent / f"{root}_species_mapping.csv"
    if not species_mapping_path.exists():
        parent = train_data_csv.parent.parent
        alt = parent / "species_mapping" / f"{root}_species_mapping.csv"
        if alt.exists():
            species_mapping_path = alt
        else:
            m = re.search(r"lib(\d+)", train_data_csv.name)
            if m:
                lib_id = m.group(1)
                candidates = sorted(
                    (REPO_ROOT / f"repo_export/lib{lib_id}/species_mapping").glob("*_species_mapping.csv")
                )
                if candidates:
                    species_mapping_path = candidates[0]
    ssid_to_group: Dict[int, str] = {}
    if species_mapping_path.exists():
        sm_df = pd.read_csv(species_mapping_path)
        if {"sample_set_id", "species_group_raw"}.issubset(sm_df.columns):
            for _, row in sm_df.iterrows():
                ssid_to_group[int(row["sample_set_id"])] = str(row["species_group_raw"])
    return ssid_to_group


def main() -> None:
    args = parse_args()

    out_dir = args.train_output_dir
    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    config_path = out_dir / "config.json"
    if not config_path.exists():
        raise SystemExit(f"config.json not found under {out_dir}")

    config = train_mod.json.loads(config_path.read_text())
    train_data_csv = Path(config["data_csv"])
    if not train_data_csv.is_absolute():
        train_data_csv = (REPO_ROOT / train_data_csv).resolve()
    if not train_data_csv.exists():
        raise SystemExit(f"Training data CSV from config not found: {train_data_csv}")

    # Use the production loader only to recover species/compound maps.
    include_es_group1 = bool(config.get("feature_group_map"))
    loaded = train_mod.load_production_csv(train_data_csv, include_es_group1=include_es_group1)
    species_map = loaded.species_map
    compound_map = loaded.compound_map

    ssid_to_group = _load_species_mapping(train_data_csv)

    test_csv = args.test_csv
    if not test_csv.is_absolute():
        test_csv = (REPO_ROOT / test_csv).resolve()
    if not test_csv.exists():
        raise SystemExit(f"Test CSV not found: {test_csv}")

    label = args.label or _infer_label(test_csv) or "test"
    print(f"[coverage] Using label={label}")

    print(f"[coverage] Reading test CSV {test_csv}")
    df = pd.read_csv(test_csv)

    required_cols = {"species", "compound"}
    if not required_cols.issubset(df.columns):
        raise SystemExit(f"Test CSV must contain columns {sorted(required_cols)}")
    if "sampleset_id" not in df.columns:
        raise SystemExit("Test CSV must contain 'sampleset_id' column for group mapping")

    # Determine which rows/compounds can be predicted by the model.
    known_species = {raw for (raw, _cluster) in species_map.keys()}
    species_in = df["species"].astype(int).isin(known_species)
    compound_in = df["compound"].isin(compound_map.keys())
    covered_row_mask = species_in & compound_in

    df["is_modeled"] = covered_row_mask

    # Map to species_group_raw (fall back to 'UNKNOWN' when mapping is missing).
    groups = df["sampleset_id"].astype(int).map(lambda s: ssid_to_group.get(int(s), "UNKNOWN"))
    df["species_group_raw"] = groups

    groups_unique = np.sort(df["species_group_raw"].astype(str).unique())
    rows = []
    for g in groups_unique:
        g_mask = df["species_group_raw"].astype(str) == g
        g_df = df[g_mask]
        if g_df.empty:
            continue
        n_rows_total = int(len(g_df))
        n_rows_modeled = int(g_df["is_modeled"].sum())

        # Compound-level coverage
        compounds_total = set(g_df["compound"].astype(int).unique())
        compounds_modeled = set(
            g_df.loc[g_df["is_modeled"], "compound"].astype(int).unique()
        )
        n_comp_total = len(compounds_total)
        n_comp_modeled = len(compounds_modeled)

        rows.append(
            {
                "species_group_raw": str(g),
                "n_rows_total": n_rows_total,
                "n_rows_modeled": n_rows_modeled,
                "row_coverage": float(n_rows_modeled / n_rows_total) if n_rows_total else float("nan"),
                "n_compounds_total": int(n_comp_total),
                "n_compounds_modeled": int(n_comp_modeled),
                "compound_coverage": (
                    float(n_comp_modeled / n_comp_total) if n_comp_total else float("nan")
                ),
            }
        )

    cov_df = pd.DataFrame(rows).sort_values("species_group_raw")

    out_csv = args.output_csv or (results_dir / f"rt_coverage_by_species_group_{label}.csv")
    cov_df.to_csv(out_csv, index=False)
    print(f"[coverage] Wrote coverage metrics to {out_csv}")

    # Simple bar plots for row and compound coverage.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        plt = None  # type: ignore

    if plt is not None and not cov_df.empty:
        x = np.arange(len(cov_df))
        labels = cov_df["species_group_raw"].astype(str)
        fig_width = max(6.0, 0.3 * len(cov_df))

        # Row coverage plot
        plt.figure(figsize=(fig_width, 4.0))
        plt.bar(x, cov_df["row_coverage"], color="tab:blue", alpha=0.8)
        plt.xticks(x, labels, rotation=90, fontsize=6)
        plt.ylabel("Row coverage")
        plt.xlabel("Species group")
        plt.ylim(0.0, 1.05)
        plt.title(f"RT prediction row coverage by species group ({label})")
        plt.tight_layout()
        row_plot = results_dir / f"rt_coverage_row_by_species_group_{label}.png"
        plt.savefig(row_plot, dpi=200)
        plt.close()
        print(f"[coverage] Wrote row coverage plot to {row_plot}")

        # Compound coverage plot
        plt.figure(figsize=(fig_width, 4.0))
        plt.bar(x, cov_df["compound_coverage"], color="tab:green", alpha=0.8)
        plt.xticks(x, labels, rotation=90, fontsize=6)
        plt.ylabel("Compound coverage")
        plt.xlabel("Species group")
        plt.ylim(0.0, 1.05)
        plt.title(f"RT prediction compound coverage by species group ({label})")
        plt.tight_layout()
        comp_plot = results_dir / f"rt_coverage_compound_by_species_group_{label}.png"
        plt.savefig(comp_plot, dpi=200)
        plt.close()
        print(f"[coverage] Wrote compound coverage plot to {comp_plot}")


if __name__ == "__main__":
    main()
