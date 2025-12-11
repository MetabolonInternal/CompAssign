#!/usr/bin/env python3
"""
Compare multiple RT model evaluations by species_cluster.

Reads per-species-cluster CSVs (columns: species_cluster, rmse) and produces a grouped bar plot.
Intended for quick side-by-side comparisons such as:
  lasso baseline vs Stage-1 ridge vs single PyMC model (collapsed).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare RT models by species_cluster (grouped RMSE bar plot).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--lasso-csv", type=Path, required=True, help="Lasso per-cluster CSV.")
    parser.add_argument("--ridge-csv", type=Path, required=True, help="Ridge per-cluster CSV.")
    parser.add_argument("--pymc-csv", type=Path, required=True, help="PyMC per-cluster CSV.")
    parser.add_argument("--label", type=str, default="realtest", help="Label for plot title.")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG path.")
    parser.add_argument("--lasso-label", type=str, default="Lasso", help="Legend label for lasso.")
    parser.add_argument(
        "--ridge-label", type=str, default="Ridge (Stage 1)", help="Legend label for ridge."
    )
    parser.add_argument(
        "--pymc-label", type=str, default="PyMC (Single)", help="Legend label for PyMC."
    )
    parser.add_argument(
        "--species-mapping-csv",
        type=Path,
        default=None,
        help=(
            "Optional species mapping CSV (from repo_export/*/species_mapping) used to replace "
            "numeric species_cluster tick labels with species_group_raw labels."
        ),
    )
    parser.add_argument(
        "--lib-id",
        type=int,
        default=None,
        help=(
            "Optional lib id used to auto-discover the species mapping CSV under repo_export/ "
            "(used only when --species-mapping-csv is not provided)."
        ),
    )
    return parser.parse_args()


def _load_csv(path: Path, *, name: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"{name} CSV not found: {path}")
    df = pd.read_csv(path)
    required = {"species_cluster", "rmse"}
    if not required.issubset(df.columns):
        raise SystemExit(f"{name} CSV must contain columns {sorted(required)}")
    df = df.copy()
    df["species_cluster"] = df["species_cluster"].astype(int)
    return df


def _default_species_mapping_csv(lib_id: int) -> Path:
    search_dirs = [
        REPO_ROOT / f"repo_export/lib{lib_id}/species_mapping",
        REPO_ROOT / "repo_export",
    ]
    for base in search_dirs:
        if base.is_dir():
            candidates = sorted(base.glob(f"**/*_lib{lib_id}_species_mapping.csv"))
            if candidates:
                return candidates[0]
    raise SystemExit(f"No species mapping found for lib {lib_id} under repo_export")


def _load_cluster_labels(mapping_csv: Path) -> dict[int, str]:
    if not mapping_csv.exists():
        raise SystemExit(f"Species mapping not found: {mapping_csv}")
    df = pd.read_csv(mapping_csv)
    required = {"species_cluster", "species_group_raw"}
    if not required.issubset(df.columns):
        raise SystemExit(f"Species mapping {mapping_csv} missing columns {sorted(required)}")
    df = df.dropna(subset=["species_cluster", "species_group_raw"]).copy()
    df["species_cluster"] = df["species_cluster"].astype(int)

    out: dict[int, str] = {}
    for cluster_id, sub in df.groupby("species_cluster"):
        labels = sorted({str(x).strip() for x in sub["species_group_raw"].dropna().tolist()})
        if not labels:
            continue
        if len(labels) == 1:
            out[int(cluster_id)] = labels[0]
        else:
            out[int(cluster_id)] = " / ".join(labels)
    return out


def main() -> None:
    args = parse_args()

    lasso_df = _load_csv(args.lasso_csv, name="Lasso")
    ridge_df = _load_csv(args.ridge_csv, name="Ridge")
    pymc_df = _load_csv(args.pymc_csv, name="PyMC")

    merged = (
        ridge_df[["species_cluster", "rmse", "n_obs"]]
        if "n_obs" in ridge_df.columns
        else ridge_df[["species_cluster", "rmse"]]
    ).rename(columns={"rmse": "rmse_ridge", "n_obs": "n_obs_ridge"})
    merged = merged.merge(
        (
            lasso_df[["species_cluster", "rmse", "n_obs"]]
            if "n_obs" in lasso_df.columns
            else lasso_df[["species_cluster", "rmse"]]
        ).rename(columns={"rmse": "rmse_lasso", "n_obs": "n_obs_lasso"}),
        on="species_cluster",
        how="outer",
    )
    merged = merged.merge(
        (
            pymc_df[["species_cluster", "rmse", "n_obs"]]
            if "n_obs" in pymc_df.columns
            else pymc_df[["species_cluster", "rmse"]]
        ).rename(columns={"rmse": "rmse_pymc", "n_obs": "n_obs_pymc"}),
        on="species_cluster",
        how="outer",
    )
    merged = merged.sort_values("species_cluster")

    out_dir = args.output.parent if args.output is not None else args.ridge_csv.parent
    out_path = args.output or (out_dir / f"rt_eval_compare_{args.label}_rmse_by_cluster.png")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise SystemExit("matplotlib is required for plotting but could not be imported") from exc

    x = np.arange(len(merged))
    width = 0.25
    fig_width = max(8.0, 1.0 + 0.8 * len(merged))

    cluster_label_map: dict[int, str] | None = None
    mapping_csv = args.species_mapping_csv
    if mapping_csv is None and args.lib_id is not None:
        mapping_csv = _default_species_mapping_csv(int(args.lib_id))
    if mapping_csv is not None:
        cluster_label_map = _load_cluster_labels(mapping_csv)

    if cluster_label_map is not None:
        x_tick_labels = [
            cluster_label_map.get(int(c), str(int(c)))
            for c in merged["species_cluster"].astype(int).tolist()
        ]
        x_axis_label = "Species group"
        title = f"RT RMSE by species group ({args.label})"
    else:
        x_tick_labels = merged["species_cluster"].astype(str).tolist()
        x_axis_label = "species_cluster"
        title = f"RT RMSE by species_cluster ({args.label})"

    plt.figure(figsize=(fig_width, 4.0))
    plt.bar(
        x - width,
        merged["rmse_lasso"],
        width,
        label=args.lasso_label,
        color="tab:orange",
        alpha=0.85,
    )
    plt.bar(
        x,
        merged["rmse_ridge"],
        width,
        label=args.ridge_label,
        color="tab:green",
        alpha=0.85,
    )
    plt.bar(
        x + width,
        merged["rmse_pymc"],
        width,
        label=args.pymc_label,
        color="tab:blue",
        alpha=0.85,
    )
    plt.xticks(
        x,
        x_tick_labels,
        rotation=35 if cluster_label_map is not None else 0,
        ha="right" if cluster_label_map is not None else "center",
        fontsize=9,
    )
    plt.xlabel(x_axis_label)
    plt.ylabel("RMSE (min)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[compare_cluster] Wrote plot to {out_path}")


if __name__ == "__main__":
    main()
