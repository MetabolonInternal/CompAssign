#!/usr/bin/env python3
"""
Compare hierarchical (streaming) vs lasso baseline RT performance by species group.

This script reads two per-species-group CSVs:
  - hierarchical metrics: e.g. rt_eval_streaming_by_species_group_cap10.csv
  - lasso metrics: e.g. rt_eval_lasso_by_species_group_cap10.csv

and produces a side-by-side bar plot of RMSE by species_group_raw.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare hierarchical vs lasso RT models by species group.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--hier-csv",
        type=Path,
        default=None,
        help="CSV with hierarchical per-species-group metrics (e.g. rt_eval_streaming_by_species_group_cap10.csv)",
    )
    parser.add_argument(
        "--lasso-csv",
        type=Path,
        default=None,
        help="CSV with lasso per-species-group metrics (e.g. rt_eval_lasso_by_species_group_cap10.csv)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="dataset",
        help="Label for the dataset (e.g., cap10, realtest) used in the plot title and filename.",
    )
    parser.add_argument(
        "--hier-label",
        type=str,
        default="Hierarchical",
        help="Legend/title label for the hier-csv series.",
    )
    parser.add_argument(
        "--lasso-label",
        type=str,
        default="Lasso",
        help="Legend/title label for the lasso-csv series.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: rt_eval_compare_<label>_rmse_by_species_group.png in hier CSV directory).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.hier_csv is None and args.lasso_csv is None:
        raise SystemExit("Provide at least one of --hier-csv or --lasso-csv")

    hier_df = None
    lasso_df = None

    if args.hier_csv is not None:
        if not args.hier_csv.exists():
            raise SystemExit(f"Hierarchical CSV not found: {args.hier_csv}")
        hier_df = pd.read_csv(args.hier_csv)

    if args.lasso_csv is not None:
        if not args.lasso_csv.exists():
            raise SystemExit(f"Lasso CSV not found: {args.lasso_csv}")
        lasso_df = pd.read_csv(args.lasso_csv)

    required_cols = {"species_group_raw", "rmse"}
    if hier_df is not None and not required_cols.issubset(hier_df.columns):
        raise SystemExit(f"Hierarchical CSV must contain columns {sorted(required_cols)}")
    if lasso_df is not None and not required_cols.issubset(lasso_df.columns):
        raise SystemExit(f"Lasso CSV must contain columns {sorted(required_cols)}")

    # Determine output directory
    if args.hier_csv is not None:
        out_dir = args.hier_csv.parent
    else:
        out_dir = args.lasso_csv.parent  # type: ignore[union-attr]

    out_path = args.output or (out_dir / f"rt_eval_compare_{args.label}_rmse_by_species_group.png")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        raise SystemExit("matplotlib is required for plotting but could not be imported")

    if hier_df is not None and lasso_df is not None:
        # Side-by-side RMSE comparison
        merged = hier_df[["species_group_raw", "rmse"]].merge(
            lasso_df[["species_group_raw", "rmse"]],
            on="species_group_raw",
            how="inner",
            suffixes=("_hier", "_lasso"),
        )
        if merged.empty:
            raise SystemExit(
                "No overlapping species groups between hierarchical and lasso metrics."
            )

        merged = merged.sort_values("rmse_hier")

        x = np.arange(len(merged))
        width = 0.4
        fig_width = max(6.0, 0.3 * len(merged))
        plt.figure(figsize=(fig_width, 4.0))
        plt.bar(
            x - width / 2,
            merged["rmse_hier"],
            width,
            label=args.hier_label,
            color="tab:blue",
            alpha=0.8,
        )
        plt.bar(
            x + width / 2,
            merged["rmse_lasso"],
            width,
            label=args.lasso_label,
            color="tab:orange",
            alpha=0.8,
        )
        plt.xticks(
            x,
            merged["species_group_raw"].astype(str),
            rotation=90,
            fontsize=6,
        )
        plt.ylabel("RMSE (min)")
        plt.xlabel("Species group")
        plt.title(
            f"RT RMSE by species group ({args.label}): {args.hier_label} vs {args.lasso_label}"
        )
        plt.tight_layout()
        plt.legend()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[compare] Wrote comparison plot to {out_path}")

        # Optional compound coverage comparison if available
        if "compound_coverage" in hier_df.columns and "compound_coverage" in lasso_df.columns:
            cov = hier_df[["species_group_raw", "compound_coverage"]].merge(
                lasso_df[["species_group_raw", "compound_coverage"]],
                on="species_group_raw",
                how="inner",
                suffixes=("_hier", "_lasso"),
            )
        elif {
            "n_compounds_total",
            "n_compounds_modeled",
        }.issubset(hier_df.columns) and {
            "n_compounds_total",
            "n_compounds_modeled",
        }.issubset(lasso_df.columns):
            cov = hier_df[["species_group_raw", "n_compounds_total", "n_compounds_modeled"]].merge(
                lasso_df[["species_group_raw", "n_compounds_total", "n_compounds_modeled"]],
                on="species_group_raw",
                how="inner",
                suffixes=("_hier", "_lasso"),
            )
            cov["compound_coverage_hier"] = (
                cov["n_compounds_modeled_hier"] / cov["n_compounds_total_hier"]
            )
            cov["compound_coverage_lasso"] = (
                cov["n_compounds_modeled_lasso"] / cov["n_compounds_total_lasso"]
            )
        else:
            cov = None

        if cov is not None and not cov.empty:
            # Align order with RMSE plot
            order = merged["species_group_raw"].tolist()
            cov = cov.set_index("species_group_raw").loc[order].reset_index()

            x_cov = np.arange(len(cov))
            width_cov = 0.4
            fig_width_cov = max(6.0, 0.3 * len(cov))

            # Standalone coverage comparison plot
            plt.figure(figsize=(fig_width_cov, 4.0))
            plt.bar(
                x_cov - width_cov / 2,
                cov["compound_coverage_hier"],
                width_cov,
                label=args.hier_label,
                color="tab:blue",
                alpha=0.8,
            )
            plt.bar(
                x_cov + width_cov / 2,
                cov["compound_coverage_lasso"],
                width_cov,
                label=args.lasso_label,
                color="tab:orange",
                alpha=0.8,
            )
            plt.xticks(
                x_cov,
                cov["species_group_raw"].astype(str),
                rotation=90,
                fontsize=6,
            )
            plt.ylabel("Compound coverage (fraction)")
            plt.xlabel("Species group")
            plt.ylim(0.0, 1.05)
            plt.title(
                f"RT compound coverage by species group ({args.label}): "
                f"{args.hier_label} vs {args.lasso_label}"
            )
            cov_out = out_path.with_name(f"{out_path.stem}_coverage_by_species_group.png")
            # Legend above the plot, centered, with extra top margin.
            plt.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, 1.20),
                ncol=2,
                borderaxespad=0.0,
            )
            plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))
            plt.savefig(cov_out, dpi=200)
            plt.close()
            print(f"[compare] Wrote coverage comparison plot to {cov_out}")
            # Joint RMSE + coverage figure (two subplots)
            fig_width_joint = max(fig_width, fig_width_cov)
            fig, (ax_rmse, ax_cov) = plt.subplots(2, 1, figsize=(fig_width_joint, 6.0), sharex=True)

            # Top: RMSE comparison
            ax_rmse.bar(
                x - width / 2,
                merged["rmse_hier"],
                width,
                label=args.hier_label,
                color="tab:blue",
                alpha=0.8,
            )
            ax_rmse.bar(
                x + width / 2,
                merged["rmse_lasso"],
                width,
                label=args.lasso_label,
                color="tab:orange",
                alpha=0.8,
            )
            ax_rmse.set_ylabel("RMSE (min)")

            # Bottom: compound coverage comparison
            ax_cov.bar(
                x_cov - width_cov / 2,
                cov["compound_coverage_hier"],
                width_cov,
                label=args.hier_label,
                color="tab:blue",
                alpha=0.8,
            )
            ax_cov.bar(
                x_cov + width_cov / 2,
                cov["compound_coverage_lasso"],
                width_cov,
                label=args.lasso_label,
                color="tab:orange",
                alpha=0.8,
            )
            ax_cov.set_ylabel("Compound coverage (fraction)")
            ax_cov.set_xlabel("Species group")
            ax_cov.set_ylim(0.0, 1.05)
            ax_cov.set_xticks(x_cov)
            ax_cov.set_xticklabels(
                cov["species_group_raw"].astype(str),
                rotation=90,
                fontsize=6,
            )

            # Legend inside the bottom (coverage) subplot, bottom-right.
            ax_cov.legend(loc="lower right")

            # Layout and overall title (draw title after layout so it sits close to the top axes)
            fig.tight_layout()
            fig.suptitle(f"{args.hier_label} vs {args.lasso_label} (Test Fold)", y=0.98)
            joint_out = out_path.with_name(f"{out_path.stem}_rmse_coverage_by_species_group.png")
            fig.savefig(joint_out, dpi=200)
            plt.close(fig)
            print(f"[compare] Wrote joint RMSE+coverage plot to {joint_out}")

    elif hier_df is not None:
        # Single-model: hierarchical only
        df = hier_df.sort_values("rmse")
        x = np.arange(len(df))
        fig_width = max(6.0, 0.3 * len(df))
        plt.figure(figsize=(fig_width, 4.0))
        plt.bar(
            x,
            df["rmse"],
            color="tab:blue",
            alpha=0.8,
            label=args.hier_label,
        )
        plt.xticks(
            x,
            df["species_group_raw"].astype(str),
            rotation=90,
            fontsize=6,
        )
        plt.ylabel("RMSE (min)")
        plt.xlabel("Species group")
        plt.title(f"RT RMSE by species group ({args.label}): {args.hier_label}")
        plt.tight_layout()
        plt.legend()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[compare] Wrote hierarchical-only plot to {out_path}")

    elif lasso_df is not None:
        # Single-model: lasso only
        df = lasso_df.sort_values("rmse")
        x = np.arange(len(df))
        fig_width = max(6.0, 0.3 * len(df))
        plt.figure(figsize=(fig_width, 4.0))
        plt.bar(
            x,
            df["rmse"],
            color="tab:orange",
            alpha=0.8,
            label=args.lasso_label,
        )
        plt.xticks(
            x,
            df["species_group_raw"].astype(str),
            rotation=90,
            fontsize=6,
        )
        plt.ylabel("RMSE (min)")
        plt.xlabel("Species group")
        plt.title(f"RT RMSE by species group ({args.label}): {args.lasso_label}")
        plt.tight_layout()
        plt.legend()
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[compare] Wrote lasso-only plot to {out_path}")


if __name__ == "__main__":
    main()
