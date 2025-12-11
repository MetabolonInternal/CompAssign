#!/usr/bin/env python3
"""
Memory-aware RT evaluation using chunked predictions.

This script mirrors eval_rt_prod_cap10.py but computes predictions in chunks
to avoid allocating a full (n_samples × n_obs) prediction matrix. It is
especially useful for very large test sets such as the lib209 real-test CSV.

Key differences vs eval_rt_prod_cap10.py:
  - Reads the test CSV in chunks.
  - For each chunk, calls HierarchicalRTModel.predict_new on that chunk only.
  - Aggregates global, per-species, and per-species-group metrics across chunks.
  - Does not store per-row predictions for the entire dataset in memory.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.pipelines import train_rt_prod as train_mod  # type: ignore
from scripts.pipelines import eval_rt_prod_cap10 as eval_mod  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RT model on large CSVs using chunked predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train-output-dir",
        type=Path,
        required=True,
        help="Output directory from train_rt_prod.py (contains config.json and models/rt_trace.nc)",
    )
    parser.add_argument(
        "--test-csv",
        type=Path,
        required=True,
        help="RT production CSV to evaluate.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Where to write evaluation metrics JSON (default: <train-output-dir>/results/rt_eval_streaming.json)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50_000,
        help="Number of rows per prediction chunk.",
    )
    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=0,
        help="Maximum number of test rows to evaluate (0 = all).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=200,
        help="Number of posterior samples to use for predictions (subsampled from full trace).",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional label for this evaluation (e.g., cap10, realtest) used in output filenames.",
    )
    return parser.parse_args()


@dataclass
class AggStats:
    n: int = 0
    sum_sq_err: float = 0.0
    sum_abs_err: float = 0.0
    sum_covered: float = 0.0

    def update(self, sq_err: np.ndarray, abs_err: np.ndarray, covered: np.ndarray) -> None:
        self.n += int(len(sq_err))
        self.sum_sq_err += float(sq_err.sum())
        self.sum_abs_err += float(abs_err.sum())
        self.sum_covered += float(covered.sum())

    def to_metrics(self) -> Dict[str, float]:
        if self.n == 0:
            return {"n_obs": 0, "rmse": float("nan"), "mae": float("nan"), "coverage_95": float("nan")}
        rmse = float(np.sqrt(self.sum_sq_err / self.n))
        mae = float(self.sum_abs_err / self.n)
        cov = float(self.sum_covered / self.n)
        return {"n_obs": int(self.n), "rmse": rmse, "mae": mae, "coverage_95": cov}


def main() -> None:
    args = parse_args()

    out_dir = args.train_output_dir
    results_dir = out_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[stream] Loading trained model from {out_dir}")
    rt_model, loaded, _train_df = eval_mod._load_trained_model(out_dir)  # type: ignore[attr-defined]

    test_csv = args.test_csv
    if not test_csv.is_absolute():
        test_csv = (REPO_ROOT / test_csv).resolve()
    if not test_csv.exists():
        raise SystemExit(f"Test CSV not found: {test_csv}")
    test_name = test_csv.name
    if "_cap10_" in test_name:
        inferred_label: str | None = "cap10"
    elif "realtest" in test_name:
        inferred_label = "realtest"
    else:
        inferred_label = None
    label = args.label or inferred_label
    label_suffix = f"_{label}" if label else ""

    print(f"[stream] Loading split and species-mapping metadata")
    # Recover species metadata from the species mapping CSV
    config_path = out_dir / "config.json"
    if not config_path.exists():
        raise SystemExit(f"config.json not found under {out_dir}")
    with config_path.open("r") as f:
        config = json.load(f)
    feature_group_map = config.get("feature_group_map", {})
    train_data_csv = Path(config["data_csv"])
    if not train_data_csv.is_absolute():
        train_data_csv = (REPO_ROOT / train_data_csv).resolve()
    stem = train_data_csv.stem
    root = stem.split("_cap")[0]
    species_mapping_path = train_data_csv.parent / f"{root}_species_mapping.csv"
    if not species_mapping_path.exists():
        parent = train_data_csv.parent.parent
        alt = parent / "species_mapping" / f"{root}_species_mapping.csv"
        if alt.exists():
            species_mapping_path = alt
        else:
            import re
            m = re.search(r"lib(\d+)", train_data_csv.name)
            if m:
                lib_id = m.group(1)
                candidates = sorted(
                    (REPO_ROOT / f"repo_export/lib{lib_id}/species_mapping").glob("*_species_mapping.csv")
                )
                if candidates:
                    species_mapping_path = candidates[0]
    species_mapping_df = None
    species_meta_df = None
    if species_mapping_path.exists():
        sm_df = pd.read_csv(species_mapping_path)
        # sample_set_id -> species_group_raw
        if {"sample_set_id", "species_group_raw"}.issubset(sm_df.columns):
            species_mapping_df = sm_df[["sample_set_id", "species_group_raw"]].copy()
        # species -> species_raw (organism)
        if {"species", "species_raw"}.issubset(sm_df.columns):
            species_meta_df = sm_df[["species", "species_raw"]].drop_duplicates(subset="species").copy()

    # Build mapping from raw species/compound ids to model indices
    species_map = loaded.species_map  # (raw_species, raw_species_cluster) -> species_idx
    compound_map = loaded.compound_map
    feature_names = loaded.feature_names

    # Build species_idx -> species_raw (for labelling). Multiple (species, group)
    # entries may share the same species_raw label.
    species_idx_to_raw: Dict[int, str] = {}
    if species_meta_df is not None:
        meta_indexed = species_meta_df.set_index("species")
        for (raw_species, _raw_cluster), idx in species_map.items():
            if raw_species in meta_indexed.index:
                species_idx_to_raw[idx] = str(meta_indexed.loc[raw_species, "species_raw"])

    # Build sample_set_id -> species_group_raw mapping (for group metrics)
    ssid_to_group: Dict[int, str] = {}
    if species_mapping_df is not None:
        for _, row in species_mapping_df.iterrows():
            ssid_to_group[int(row["sample_set_id"])] = str(row["species_group_raw"])

    # Aggregators
    global_stats = AggStats()
    species_stats: Dict[int, AggStats] = {}
    group_stats: Dict[str, AggStats] = {}
    # Per-group chemical coverage (unique compounds)
    group_compounds_total: Dict[str, set[int]] = {}
    group_compounds_modeled: Dict[str, set[int]] = {}

    remaining = args.max_test_rows if args.max_test_rows and args.max_test_rows > 0 else None
    total_rows = 0
    kept_rows = 0
    dropped_rows = 0

    print(f"[stream] Streaming test data from {test_csv} with chunk_size={args.chunk_size}")

    if "species_cluster" not in pd.read_csv(test_csv, nrows=1).columns:
        raise SystemExit("Test CSV must contain 'species_cluster' column")

    known_pairs = set(species_map.keys())

    for chunk in pd.read_csv(test_csv, chunksize=args.chunk_size):
        if remaining is not None and remaining <= 0:
            break

        total_rows += len(chunk)

        # Track total unique compounds per species group before any filtering
        if ssid_to_group:
            ssids_all = chunk["sampleset_id"].astype(int).to_numpy()
            groups_all = np.array(
                [ssid_to_group.get(int(s), "UNKNOWN") for s in ssids_all],
                dtype=object,
            )
            compounds_all = chunk["compound"].astype(int).to_numpy()
            for g in np.unique(groups_all):
                mask_g = groups_all == g
                if not np.any(mask_g):
                    continue
                comp_ids = np.unique(compounds_all[mask_g])
                total_set = group_compounds_total.setdefault(str(g), set())
                for cid in comp_ids:
                    total_set.add(int(cid))

        # Limit to remaining rows if max_test_rows is set
        if remaining is not None and len(chunk) > remaining:
            chunk = chunk.iloc[:remaining].copy()

        # Filter to (species, species_cluster) / compound combos seen in training
        species_series = chunk["species"].astype(int)
        cluster_series = chunk["species_cluster"].astype(int)
        pair_mask = [
            (int(s), int(c)) in known_pairs for s, c in zip(species_series, cluster_series)
        ]
        pair_mask = np.array(pair_mask, dtype=bool)
        compound_mask = chunk["compound"].astype(int).isin(compound_map.keys())
        mask = pair_mask & compound_mask

        dropped_rows += int((~mask).sum())
        chunk = chunk[mask].copy()
        if chunk.empty:
            if remaining is not None:
                remaining -= 0
            continue

        # Map to model indices
        raw_species = chunk["species"].astype(int).to_numpy()
        raw_cluster = chunk["species_cluster"].astype(int).to_numpy()
        raw_compound = chunk["compound"].astype(int).to_numpy()
        try:
            species_idx = np.array(
                [species_map[(int(s), int(c))] for s, c in zip(raw_species, raw_cluster)],
                dtype=int,
            )
        except KeyError as exc:
            raise SystemExit(
                f"Unexpected (species, species_cluster) pair in test data: {exc}"
            ) from exc
        compound_idx = np.array([compound_map[int(c)] for c in raw_compound], dtype=int)

        # Optionally zero group-specific covariates (e.g., ES_* for Group 1) for other groups.
        if feature_group_map:
            if not ssid_to_group:
                print(
                    "[stream] Warning: feature_group_map provided but species_group mapping is empty; "
                    "skipping group-based covariate masking."
                )
            else:
                groups = chunk["sampleset_id"].astype(int).map(ssid_to_group)
                for feat, allowed_groups in feature_group_map.items():
                    if not allowed_groups:
                        continue
                    if feat not in chunk.columns:
                        raise SystemExit(
                            f"Expected covariate column '{feat}' not found in test CSV {test_csv}"
                        )
                    mask_allowed = groups.isin(allowed_groups)
                    chunk.loc[~mask_allowed, feat] = 0.0
                # Replace any remaining NA in group-masked features with 0
                chunk[list(feature_group_map.keys())] = chunk[list(feature_group_map.keys())].fillna(
                    0.0
                )

        # Covariates and RT
        X = chunk[feature_names].to_numpy(dtype=float)
        y_true = chunk["rt"].to_numpy(dtype=float)

        # Predictions for this chunk
        pred_mean, pred_std = rt_model.predict_new(  # type: ignore[attr-defined]
            species_idx=species_idx,
            compound_idx=compound_idx,
            run_features=X,
            n_samples=args.n_samples,
        )

        sq_err = (pred_mean - y_true) ** 2
        abs_err = np.abs(pred_mean - y_true)
        lower = pred_mean - 1.96 * pred_std
        upper = pred_mean + 1.96 * pred_std
        covered = (y_true >= lower) & (y_true <= upper)

        # Global stats
        global_stats.update(sq_err, abs_err, covered)
        kept_rows += len(chunk)

        # Per-species stats
        for s_idx in np.unique(species_idx):
            mask_s = species_idx == s_idx
            if not np.any(mask_s):
                continue
            stats = species_stats.setdefault(int(s_idx), AggStats())
            stats.update(sq_err[mask_s], abs_err[mask_s], covered[mask_s])

        # Per-group stats (via sample_set_id -> species_group_raw) and chemical coverage
        if ssid_to_group:
            ssids = chunk["sampleset_id"].astype(int).to_numpy()
            groups = np.array(
                [ssid_to_group.get(int(s), "UNKNOWN") for s in ssids],
                dtype=object,
            )
            compounds = chunk["compound"].astype(int).to_numpy()
            for g in np.unique(groups):
                mask_g = groups == g
                if not np.any(mask_g):
                    continue
                key = str(g)
                stats = group_stats.setdefault(key, AggStats())
                stats.update(sq_err[mask_g], abs_err[mask_g], covered[mask_g])
                modeled_set = group_compounds_modeled.setdefault(key, set())
                for cid in np.unique(compounds[mask_g]):
                    modeled_set.add(int(cid))

        if remaining is not None:
            remaining -= len(chunk)

    print(
        f"[stream] Total rows seen={total_rows:,}, kept={kept_rows:,}, "
        f"dropped (unseen species/compound)={dropped_rows:,}"
    )

    # Global metrics
    global_metrics = global_stats.to_metrics()

    # Per-species DataFrame
    species_rows = []
    for s_idx, stats in species_stats.items():
        m = stats.to_metrics()
        row = {"species_idx": int(s_idx), **m}
        row["species_raw"] = species_idx_to_raw.get(int(s_idx))
        species_rows.append(row)
    species_df = pd.DataFrame(species_rows)

    # Per-group DataFrame
    group_rows = []
    for g, stats in group_stats.items():
        m = stats.to_metrics()
        total_set = group_compounds_total.get(g, set())
        modeled_set = group_compounds_modeled.get(g, set())
        n_total = len(total_set)
        n_modeled = len(modeled_set)
        coverage = float(n_modeled / n_total) if n_total > 0 else float("nan")
        row = {
            "species_group_raw": g,
            **m,
            "n_compounds_total": int(n_total),
            "n_compounds_modeled": int(n_modeled),
            "compound_coverage": coverage,
        }
        group_rows.append(row)
    group_df = pd.DataFrame(group_rows)

    out_json = args.output_json or (results_dir / f"rt_eval_streaming{label_suffix}.json")
    payload = {
        "metrics": global_metrics,
        "n_test": int(global_stats.n),
        "chunk_size": args.chunk_size,
        "n_samples": args.n_samples,
    }
    out_json.write_text(json.dumps(payload, indent=2))
    print(
        f"[stream] Global: n={global_stats.n:,}, RMSE={global_metrics['rmse']:.3f}, "
        f"MAE={global_metrics['mae']:.3f}, coverage95={global_metrics['coverage_95']:.3f}"
    )
    print(f"[stream] Wrote global metrics to {out_json}")

    species_csv = results_dir / f"rt_eval_streaming_by_species{label_suffix}.csv"
    species_df.to_csv(species_csv, index=False)
    print(f"[stream] Wrote per-species metrics to {species_csv}")

    group_csv = results_dir / f"rt_eval_streaming_by_species_group{label_suffix}.csv"
    group_df.to_csv(group_csv, index=False)
    print(f"[stream] Wrote per-species-group metrics to {group_csv}")

    # Plots: reuse the same style as eval_rt_prod_cap10, but using aggregated metrics.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        plt = None  # type: ignore

    if plt is not None and not species_df.empty:
        sg_sorted = species_df.sort_values("rmse")
        fig_width = max(6.0, 0.25 * len(sg_sorted))
        plt.figure(figsize=(fig_width, 4.0))
        plt.bar(
            np.arange(len(sg_sorted)),
            sg_sorted["rmse"],
            color="tab:blue",
            alpha=0.8,
        )
        labels = sg_sorted.get("species_raw", sg_sorted["species_idx"].astype(str))
        labels = labels.fillna(sg_sorted["species_idx"].astype(str))
        plt.xticks(
            np.arange(len(sg_sorted)),
            labels.astype(str),
            rotation=90,
            fontsize=6,
        )
        plt.ylabel("RMSE (min)")
        plt.xlabel("Species")
        plt.title("RT prediction RMSE by species (streaming)")
        plt.tight_layout()
        plot_path_species = results_dir / f"rt_eval_streaming_rmse_by_species{label_suffix}.png"
        plt.savefig(plot_path_species, dpi=200)
        plt.close()
        print(f"[stream] Wrote RMSE-by-species plot to {plot_path_species}")

    if plt is not None and not group_df.empty:
        g_sorted = group_df.sort_values("rmse")
        fig_width = max(6.0, 0.3 * len(g_sorted))
        plt.figure(figsize=(fig_width, 4.0))
        plt.bar(
            np.arange(len(g_sorted)),
            g_sorted["rmse"],
            color="tab:green",
            alpha=0.8,
        )
        plt.xticks(
            np.arange(len(g_sorted)),
            g_sorted["species_group_raw"].astype(str),
            rotation=90,
            fontsize=6,
        )
        plt.ylabel("RMSE (min)")
        plt.xlabel("Species group")
        plt.title("RT prediction RMSE by species group (streaming)")
        plt.tight_layout()
        plot_path_group = results_dir / f"rt_eval_streaming_rmse_by_species_group{label_suffix}.png"
        plt.savefig(plot_path_group, dpi=200)
        plt.close()
        print(f"[stream] Wrote RMSE-by-species-group plot to {plot_path_group}")


if __name__ == "__main__":
    main()
