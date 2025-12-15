#!/usr/bin/env python3
"""
Evaluate per-group coefficient summaries (Stage1CoeffSummaries) on RT production CSVs.

This evaluator is intentionally simple and focuses on *seen groups*:
  g = (species_cluster, comp_id)

It:
  - loads a Stage1CoeffSummaries .npz artifact (from ridge Stage-1 or a PyMC single model),
  - streams a production RT CSV in chunks,
  - looks up group coefficients by (species_cluster, comp_id),
  - computes point predictions and Normal-approx credible intervals using:
        y_hat = x1^T beta_hat
        Var(y) = sigma2_mean + x1^T beta_cov x1
    (falls back to diagonal variance if beta_cov is missing),
  - aggregates global + per-species-cluster metrics, and writes results/* outputs.

Outputs (under --output-dir, default: <coeff-npz-root>/results):
  - rt_eval_coeff_summaries_<label>.json
  - rt_eval_coeff_summaries_by_species_cluster_<label>.csv
  - rt_eval_coeff_summaries_by_species_group_<label>.csv (optional)
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
import sys
from typing import Dict

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.compassign.rt.pymc_collapsed_ridge import Stage1CoeffSummaries  # noqa: E402
from src.compassign.rt.pymc_collapsed_ridge import ChemHierBackoffSummaries  # noqa: E402


REQUIRED_COLS = ["rt", "compound", "comp_id", "compound_class", "species_cluster"]
SPECIES_GROUP_COLS = ["sampleset_id"]

_POLY2_SQ_RE = re.compile(r"^poly2_sq\((?P<name>[^)]+)\)$")
_POLY2_INT_RE = re.compile(r"^poly2_int\((?P<a>[^,]+),(?P<b>[^)]+)\)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Stage1CoeffSummaries by species_cluster on production RT CSVs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--coeff-npz",
        type=Path,
        required=True,
        help="Stage1CoeffSummaries .npz file (e.g., stage1_coeff_summaries.npz).",
    )
    parser.add_argument(
        "--test-csv", type=Path, required=True, help="RT production CSV to evaluate."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (writes results/*). Defaults next to the coeff artifact.",
    )
    parser.add_argument("--chunk-size", type=int, default=50_000, help="Rows per prediction chunk.")
    parser.add_argument(
        "--interval",
        type=float,
        default=0.95,
        help="Central interval mass for coverage calculation (Normal approx).",
    )
    parser.add_argument(
        "--pred-std-scale",
        type=float,
        default=1.0,
        help="Multiply predictive std dev by this factor before computing coverage.",
    )
    parser.add_argument(
        "--suggest-pred-std-scale",
        action="store_true",
        help=(
            "If set, suggests a pred-std scaling factor by sampling standardized residuals "
            "and matching the requested --interval coverage."
        ),
    )
    parser.add_argument(
        "--suggest-sample-per-chunk",
        type=int,
        default=2000,
        help="When suggesting pred-std scale, number of |z| samples per chunk.",
    )
    parser.add_argument(
        "--suggest-seed",
        type=int,
        default=42,
        help="RNG seed used when suggesting pred-std scale.",
    )
    parser.add_argument(
        "--require-seen-group",
        action="store_true",
        help="If set, skip rows whose (species_cluster, comp_id) is not present in coeff-npz.",
    )
    parser.add_argument(
        "--backoff-npz",
        type=Path,
        default=None,
        help=(
            "Optional ChemHierBackoffSummaries .npz (written by the chem_hier trainer). "
            "If provided, unseen groups are scored using the backoff model instead of being skipped."
        ),
    )
    parser.add_argument(
        "--backoff-mean-mode",
        type=str,
        default="chem",
        choices=["chem", "cluster_only"],
        help=(
            "How to compute the backoff mean for unseen groups. "
            "'chem' uses the chemistry regression (t0 + z·theta) for unseen chem_ids; "
            "'cluster_only' ignores chemistry and uses only the cluster offset (mu_cluster + t0)."
        ),
    )
    parser.add_argument(
        "--backoff-slope-mean-mode",
        type=str,
        default="auto",
        choices=["auto", "cluster_head", "zero", "global", "cluster"],
        help=(
            "How to set the slope mean for unseen groups during backoff prediction. "
            "'auto' uses the learned cluster slope head if available (otherwise 0); "
            "'cluster_head' uses the learned cluster slope head if available; "
            "'zero' uses the ridge prior mean 0; "
            "'global' uses the average slope across all seen groups; "
            "'cluster' uses the average slope within each species_cluster."
        ),
    )
    parser.add_argument(
        "--chem-embeddings-path",
        type=Path,
        default=None,
        help=(
            "Optional ChemBERTa PCA-20 embedding parquet used to compute chemistry features for unseen chem_ids. "
            "Defaults to the path stored in --backoff-npz (if present)."
        ),
    )
    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=0,
        help="Maximum number of test rows to evaluate (0 = all).",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional label for this evaluation used in output filenames.",
    )
    parser.add_argument(
        "--write-by-species-group",
        action="store_true",
        help=(
            "If set, also write per-species-group metrics using a species mapping "
            "(maps sampleset_id -> species_group_raw)."
        ),
    )
    parser.add_argument(
        "--species-mapping-csv",
        type=Path,
        default=None,
        help=(
            "Optional species mapping CSV with columns (sample_set_id, species_group_raw). "
            "If omitted, we try to infer lib id from the test CSV path and search under repo_export."
        ),
    )
    return parser.parse_args()


@dataclass
class AggStats:
    n: int = 0
    sum_sq_err: float = 0.0
    sum_abs_err: float = 0.0
    sum_covered: float = 0.0

    def update(self, sq_err: np.ndarray, abs_err: np.ndarray, covered: np.ndarray) -> None:
        if sq_err.size == 0:
            return
        self.n += int(sq_err.size)
        self.sum_sq_err += float(np.sum(sq_err))
        self.sum_abs_err += float(np.sum(abs_err))
        self.sum_covered += float(np.sum(covered))

    def to_metrics(self) -> Dict[str, float]:
        if self.n == 0:
            return {
                "n_obs": 0,
                "rmse": float("nan"),
                "mae": float("nan"),
                "coverage_95": float("nan"),
            }
        rmse = float(np.sqrt(self.sum_sq_err / self.n))
        mae = float(self.sum_abs_err / self.n)
        cov = float(self.sum_covered / self.n)
        return {"n_obs": int(self.n), "rmse": rmse, "mae": mae, "coverage_95": cov}


def _design_feature_raw_dependencies(feature_name: str) -> set[str]:
    m_sq = _POLY2_SQ_RE.match(feature_name)
    if m_sq:
        return {str(m_sq.group("name"))}
    m_int = _POLY2_INT_RE.match(feature_name)
    if m_int:
        return {str(m_int.group("a")), str(m_int.group("b"))}
    return {str(feature_name)}


def _split_design_features(
    feature_names: tuple[str, ...],
) -> tuple[list[str], set[str], bool]:
    """Split Stage1 design features into raw CSV columns and derived poly2 features.

    Returns:
      raw_features_in_order: list[str]  (all non-derived features, in design order)
      raw_dependencies: set[str]        (all raw columns needed to compute the design matrix)
      has_derived: bool
    """
    raw_features: list[str] = []
    raw_deps: set[str] = set()
    has_derived = False
    for name in feature_names:
        raw_deps |= _design_feature_raw_dependencies(name)
        if _POLY2_SQ_RE.match(name) or _POLY2_INT_RE.match(name):
            has_derived = True
            continue
        raw_features.append(str(name))
    return raw_features, raw_deps, has_derived


def _design_matrix_from_chunk(
    *,
    chunk: pd.DataFrame,
    design_features: tuple[str, ...],
    raw_features: list[str],
    fill_value: float,
) -> np.ndarray:
    """Build the design matrix X in the exact order of `design_features`.

    Supports poly2_sq(...) and poly2_int(...,...) derived features.
    """
    if not raw_features:
        raise SystemExit("No raw features found in coefficient artifact")
    chunk[raw_features] = chunk[raw_features].fillna(float(fill_value))
    raw_x = chunk[raw_features].to_numpy(dtype=np.float64, copy=False)
    raw_pos = {name: i for i, name in enumerate(raw_features)}

    n = int(raw_x.shape[0])
    p = int(len(design_features))
    x = np.empty((n, p), dtype=np.float64)

    for j, name in enumerate(design_features):
        if name in raw_pos:
            x[:, j] = raw_x[:, raw_pos[name]]
            continue
        m_sq = _POLY2_SQ_RE.match(name)
        if m_sq:
            src = str(m_sq.group("name"))
            if src not in raw_pos:
                raise SystemExit(f"Derived feature {name} references missing raw column {src}")
            v = raw_x[:, raw_pos[src]]
            x[:, j] = v * v
            continue
        m_int = _POLY2_INT_RE.match(name)
        if m_int:
            a = str(m_int.group("a"))
            b = str(m_int.group("b"))
            if a not in raw_pos or b not in raw_pos:
                raise SystemExit(f"Derived feature {name} references missing raw columns: {a}, {b}")
            x[:, j] = raw_x[:, raw_pos[a]] * raw_x[:, raw_pos[b]]
            continue
        raise SystemExit(f"Unknown feature name in artifact (not a CSV column or poly2_*): {name}")
    return x


def _default_output_dir(coeff_npz: Path) -> Path:
    # Common layouts:
    # - <out>/stage1_coeff_summaries.npz  -> <out>/results
    # - <out>/models/stage1_coeff_summaries_*.npz -> <out>/results
    if coeff_npz.parent.name == "models":
        return coeff_npz.parent.parent / "results"
    return coeff_npz.parent / "results"


def _infer_lib_id_from_path(path: Path) -> int | None:
    m = re.search(r"lib(\d+)", str(path))
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _find_species_mapping_csv(*, lib_id: int) -> Path:
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


def _load_sampleset_to_species_group(mapping_csv: Path) -> Dict[int, str]:
    if not mapping_csv.exists():
        raise SystemExit(f"Species mapping not found: {mapping_csv}")
    df = pd.read_csv(mapping_csv)
    if not {"sample_set_id", "species_group_raw"}.issubset(df.columns):
        raise SystemExit(
            f"Species mapping {mapping_csv} missing required columns 'sample_set_id'/'species_group_raw'"
        )
    out: Dict[int, str] = {}
    for _, row in df.iterrows():
        ssid = int(row["sample_set_id"])
        g_raw = row["species_group_raw"]
        if pd.isna(g_raw):
            continue
        out[ssid] = str(g_raw)
    if not out:
        raise SystemExit(
            f"Species mapping {mapping_csv} did not yield any valid sample_set_id rows"
        )
    return out


def _update_group_compound_sets(
    *,
    group_labels: np.ndarray,
    chem_ids: np.ndarray,
    out_sets: Dict[str, set[int]],
) -> None:
    if group_labels.size == 0:
        return
    for g in np.unique(group_labels).tolist():
        mask = group_labels == g
        if not np.any(mask):
            continue
        s = out_sets.setdefault(str(g), set())
        for cid in np.unique(chem_ids[mask]).tolist():
            s.add(int(cid))


def main() -> None:
    args = parse_args()
    if not (0.0 < float(args.interval) < 1.0):
        raise SystemExit("--interval must be in (0, 1)")
    if float(args.pred_std_scale) <= 0.0:
        raise SystemExit("--pred-std-scale must be > 0")
    require_seen = bool(args.require_seen_group)
    backoff_mean_mode = str(args.backoff_mean_mode)
    backoff_slope_mean_mode = str(args.backoff_slope_mean_mode)

    alpha = 0.5 + 0.5 * float(args.interval)
    q_norm = float(NormalDist().inv_cdf(alpha))
    pred_std_scale = float(args.pred_std_scale)

    suggest_scale = bool(args.suggest_pred_std_scale)
    suggest_per_chunk = int(args.suggest_sample_per_chunk)
    if suggest_per_chunk < 0:
        raise SystemExit("--suggest-sample-per-chunk must be >= 0")
    suggest_rng = np.random.default_rng(int(args.suggest_seed)) if suggest_scale else None
    abs_z_samples: list[np.ndarray] = []

    coeff_npz = args.coeff_npz
    if not coeff_npz.is_absolute():
        coeff_npz = (REPO_ROOT / coeff_npz).resolve()
    if not coeff_npz.exists():
        raise SystemExit(f"Coefficient summaries not found: {coeff_npz}")
    stage1 = Stage1CoeffSummaries.load_npz(coeff_npz)

    backoff: ChemHierBackoffSummaries | None = None
    if args.backoff_npz is not None:
        backoff_npz = args.backoff_npz
        if not backoff_npz.is_absolute():
            backoff_npz = (REPO_ROOT / backoff_npz).resolve()
        if not backoff_npz.exists():
            raise SystemExit(f"Backoff artifact not found: {backoff_npz}")
        backoff = ChemHierBackoffSummaries.load_npz(backoff_npz)

    test_csv = args.test_csv
    if not test_csv.is_absolute():
        test_csv = (REPO_ROOT / test_csv).resolve()
    if not test_csv.exists():
        raise SystemExit(f"Test CSV not found: {test_csv}")

    write_by_species_group = bool(args.write_by_species_group)
    ssid_to_group: Dict[int, str] | None = None
    if write_by_species_group:
        mapping_csv = args.species_mapping_csv
        if mapping_csv is None:
            lib_id = _infer_lib_id_from_path(test_csv)
            if lib_id is None:
                raise SystemExit(
                    "--write-by-species-group requires --species-mapping-csv or a test CSV path containing 'lib<id>'"
                )
            mapping_csv = _find_species_mapping_csv(lib_id=lib_id)
        if not mapping_csv.is_absolute():
            mapping_csv = (REPO_ROOT / mapping_csv).resolve()
        ssid_to_group = _load_sampleset_to_species_group(mapping_csv)

    header = pd.read_csv(test_csv, nrows=0)
    missing_req = [c for c in REQUIRED_COLS if c not in header.columns]
    if missing_req:
        raise SystemExit(f"CSV missing required columns: {missing_req}")
    if write_by_species_group:
        missing_group_cols = [c for c in SPECIES_GROUP_COLS if c not in header.columns]
        if missing_group_cols:
            raise SystemExit(
                f"CSV missing required columns for --write-by-species-group: {missing_group_cols}"
            )

    design_features = tuple(str(c) for c in stage1.feature_names)
    raw_features_in_order, raw_deps, has_derived = _split_design_features(design_features)

    raw_feature_cols = list(raw_features_in_order)
    for dep in sorted(raw_deps):
        if dep not in raw_feature_cols:
            raw_feature_cols.append(dep)

    missing_feats = [c for c in raw_feature_cols if c not in header.columns]
    if missing_feats:
        raise SystemExit(f"CSV missing required run covariate columns: {missing_feats}")

    feature_center = stage1.feature_center
    feature_rotation = stage1.feature_rotation
    if has_derived:
        if feature_center is not None or feature_rotation is not None:
            raise SystemExit(
                "Derived poly2_* features are not compatible with feature_center/feature_rotation; "
                "retrain without those transforms."
            )
        if backoff is not None and not require_seen:
            raise SystemExit("Backoff scoring does not support derived poly2_* features")
    else:
        if feature_center is not None:
            feature_center = np.asarray(feature_center, dtype=np.float64)
            if feature_center.shape != (len(design_features),):
                raise SystemExit(
                    f"Stage-1 feature_center has unexpected shape: {feature_center.shape} "
                    f"(expected {(len(design_features),)})"
                )
        if feature_rotation is not None:
            feature_rotation = np.asarray(feature_rotation, dtype=np.float64)
            if feature_rotation.shape != (len(design_features), len(design_features)):
                raise SystemExit(
                    f"Stage-1 feature_rotation has unexpected shape: {feature_rotation.shape} "
                    f"(expected {(len(design_features), len(design_features))})"
                )
    feature_cols = list(design_features)

    out_dir = args.output_dir or _default_output_dir(coeff_npz)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_suffix = f"_{args.label}" if args.label else ""
    out_json = out_dir / f"rt_eval_coeff_summaries{label_suffix}.json"
    out_csv = out_dir / f"rt_eval_coeff_summaries_by_species_cluster{label_suffix}.csv"
    out_group_csv = out_dir / f"rt_eval_coeff_summaries_by_species_group{label_suffix}.csv"

    keys = np.asarray(stage1.group_keys, dtype=np.int64)
    beta_hat = np.asarray(stage1.beta_hat, dtype=np.float64)
    beta_var_diag = np.asarray(stage1.beta_var_diag, dtype=np.float64)
    beta_cov = stage1.beta_cov
    if beta_cov is not None:
        beta_cov = np.asarray(beta_cov, dtype=np.float64)
    sigma2 = np.asarray(stage1.sigma2_mean, dtype=np.float64)

    backoff_cluster_ids = None
    backoff_mu = None
    backoff_chem_ids = None
    backoff_t = None
    backoff_t0 = None
    backoff_theta = None
    backoff_z_center = None
    backoff_tau_b2 = None
    backoff_tau_t2 = None
    backoff_sigma2 = None
    backoff_lambda = None
    emb_ids = None
    emb_z = None
    backoff_w_mean_global = None
    backoff_w_mean_by_cluster = None
    effective_backoff_slope_mode = backoff_slope_mean_mode
    if backoff is not None and not require_seen:
        backoff_cluster_ids = np.asarray(backoff.cluster_ids, dtype=np.int64)
        backoff_mu = np.asarray(backoff.mu_cluster, dtype=np.float64)
        backoff_chem_ids = np.asarray(backoff.chem_ids, dtype=np.int64)
        backoff_t = np.asarray(backoff.t_chem, dtype=np.float64)
        backoff_t0 = float(backoff.t0)
        backoff_theta = np.asarray(backoff.theta, dtype=np.float64)
        backoff_z_center = (
            np.asarray(backoff.z_center, dtype=np.float64) if backoff.z_center is not None else None
        )
        backoff_tau_b2 = float(backoff.tau_b) ** 2
        backoff_tau_t2 = float(backoff.tau_t) ** 2
        backoff_sigma2 = float(backoff.sigma2)
        backoff_lambda = float(backoff.lambda_slopes)

        if backoff_slope_mean_mode == "auto":
            effective_backoff_slope_mode = (
                "cluster_head" if backoff.w_cluster is not None else "zero"
            )
        if effective_backoff_slope_mode == "cluster_head" and backoff.w_cluster is None:
            raise SystemExit(
                "backoff_slope_mean_mode='cluster_head' requires w_cluster in the backoff artifact."
            )

        if backoff_mean_mode == "chem":
            emb_path = args.chem_embeddings_path
            if emb_path is None:
                if backoff.embeddings_path is None:
                    raise SystemExit(
                        "--chem-embeddings-path is required when --backoff-npz has no stored embeddings_path"
                    )
                emb_path = Path(backoff.embeddings_path)
            if not emb_path.is_absolute():
                emb_path = (REPO_ROOT / emb_path).resolve()

            try:
                from src.compassign.utils import load_chemberta_pca20  # type: ignore
            except Exception as exc:  # noqa: BLE001
                raise SystemExit("Failed to import ChemBERTa embedding loader") from exc

            emb = load_chemberta_pca20(emb_path)
            emb_ids = np.asarray(emb.chem_id, dtype=np.int64)
            emb_z = np.asarray(emb.features, dtype=np.float64)
            order = np.argsort(emb_ids, kind="mergesort")
            emb_ids = emb_ids[order]
            emb_z = emb_z[order]

        if effective_backoff_slope_mode in {"global", "cluster"}:
            w_hat = beta_hat[:, 1:]
            backoff_w_mean_global = w_hat.mean(axis=0)
            if effective_backoff_slope_mode == "cluster":
                group_cluster = (keys >> np.int64(32)).astype(np.int64, copy=False)
                backoff_w_mean_by_cluster = np.zeros(
                    (int(backoff_cluster_ids.size), int(w_hat.shape[1])), dtype=np.float64
                )
                for i, cl_id in enumerate(backoff_cluster_ids.tolist()):
                    mask = group_cluster == int(cl_id)
                    if not np.any(mask):
                        continue
                    backoff_w_mean_by_cluster[i] = w_hat[mask].mean(axis=0)

    if keys.ndim != 1 or beta_hat.ndim != 2:
        raise SystemExit("Invalid coefficient summaries: expected keys (G,) and beta_hat (G,D)")
    if beta_hat.shape[0] != keys.size:
        raise SystemExit("Invalid coefficient summaries: beta_hat and group_keys size mismatch")
    n_groups, n_coefs = beta_hat.shape
    if n_coefs != len(feature_cols) + 1:
        raise SystemExit(
            f"Invalid coefficient summaries: expected n_coefs={len(feature_cols)+1}, got {n_coefs}"
        )
    if beta_var_diag.shape != beta_hat.shape:
        raise SystemExit("Invalid coefficient summaries: beta_var_diag has wrong shape")
    if sigma2.shape != (n_groups,):
        raise SystemExit("Invalid coefficient summaries: sigma2_mean has wrong shape")
    if beta_cov is not None and beta_cov.shape != (n_groups, n_coefs, n_coefs):
        raise SystemExit("Invalid coefficient summaries: beta_cov has wrong shape")

    remaining = int(args.max_test_rows) if args.max_test_rows and args.max_test_rows > 0 else None
    global_stats = AggStats()
    cluster_stats: Dict[str, AggStats] = {}
    group_stats: Dict[str, AggStats] = {}
    group_compounds_total: Dict[str, set[int]] = {}
    group_compounds_modeled: Dict[str, set[int]] = {}

    total_rows = 0
    used_rows = 0
    skipped_no_group = 0
    skipped_missing_group = 0
    backoff_used = 0
    backoff_skipped = 0

    usecols = list(
        dict.fromkeys(
            [
                *REQUIRED_COLS,
                *(SPECIES_GROUP_COLS if write_by_species_group else []),
                *raw_feature_cols,
            ]
        )
    )
    for chunk in pd.read_csv(test_csv, chunksize=int(args.chunk_size), usecols=usecols):
        if remaining is not None and remaining <= 0:
            break

        total_rows += int(len(chunk))
        if remaining is not None and len(chunk) > remaining:
            chunk = chunk.iloc[:remaining].copy()

        group_raw_arr: np.ndarray | None = None
        if write_by_species_group:
            assert ssid_to_group is not None
            ssid = chunk["sampleset_id"].astype(int)
            group_raw = ssid.map(ssid_to_group)
            ok_group = group_raw.notna().to_numpy(dtype=bool, copy=False)
            n_drop = int((~ok_group).sum())
            if n_drop > 0:
                skipped_no_group += n_drop
                chunk = chunk.loc[ok_group].copy()
                group_raw = group_raw.loc[ok_group]
            if len(chunk) == 0:
                continue
            group_raw_arr = group_raw.astype(str).to_numpy(copy=False)

            chem_total = chunk["compound"].astype(int).to_numpy(dtype=np.int64, copy=False)
            _update_group_compound_sets(
                group_labels=group_raw_arr,
                chem_ids=chem_total,
                out_sets=group_compounds_total,
            )

        cluster_arr = chunk["species_cluster"].astype(int).to_numpy(dtype=np.int64, copy=False)
        comp_arr = chunk["comp_id"].astype(int).to_numpy(dtype=np.int64, copy=False)
        key_arr = (cluster_arr << np.int64(32)) + comp_arr

        idx = np.searchsorted(keys, key_arr)
        ok = (idx >= 0) & (idx < keys.size)
        if ok.any():
            ok[ok] &= keys[idx[ok]] == key_arr[ok]
        missing = ~ok
        n_missing = int(missing.sum())
        if n_missing > 0:
            skipped_missing_group += n_missing
            if require_seen or backoff is None:
                chunk = chunk.loc[ok].copy()
                if len(chunk) == 0:
                    continue
                if group_raw_arr is not None:
                    group_raw_arr = group_raw_arr[ok]
                cluster_arr = cluster_arr[ok]
                idx = idx[ok]
                ok = np.ones((int(len(chunk)),), dtype=bool)
                missing = np.zeros((int(len(chunk)),), dtype=bool)

        y = chunk["rt"].to_numpy(dtype=np.float64, copy=False)
        chem_id_arr = chunk["compound"].astype(int).to_numpy(dtype=np.int64, copy=False)
        x = _design_matrix_from_chunk(
            chunk=chunk,
            design_features=design_features,
            raw_features=raw_features_in_order,
            fill_value=0.0,
        )
        if feature_center is not None:
            x = x - feature_center[None, :]
        if feature_rotation is not None:
            x = x @ feature_rotation

        # Assemble x1 = [1, x] for variance computations.
        x1 = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)

        sq_err_all = []
        abs_err_all = []
        covered_all = []
        cluster_all = []
        group_all = []

        if ok.any():
            idx_ok = idx[ok]
            beta = beta_hat[idx_ok]
            pred_mean = beta[:, 0] + np.sum(beta[:, 1:] * x[ok], axis=1)

            sigma2_y = np.maximum(sigma2[idx_ok], 1e-12)
            if beta_cov is not None:
                cov = beta_cov[idx_ok]
                var_coef = np.einsum("ni,nij,nj->n", x1[ok], cov, x1[ok], optimize=True)
            else:
                var_coef = np.sum(np.square(x1[ok]) * beta_var_diag[idx_ok], axis=1)

            pred_var = np.maximum(sigma2_y + np.maximum(var_coef, 0.0), 0.0)
            pred_std_base = np.sqrt(pred_var)
            pred_std = pred_std_base * pred_std_scale

            err = pred_mean - y[ok]
            sq_err_all.append(np.square(err))
            abs_err_all.append(np.abs(err))
            covered_all.append(np.abs(err) <= (q_norm * pred_std))
            cluster_all.append(cluster_arr[ok])
            if suggest_scale and suggest_rng is not None and suggest_per_chunk > 0:
                abs_z = np.abs(err) / np.maximum(pred_std_base, 1e-12)
                k = min(int(abs_z.size), int(suggest_per_chunk))
                if k > 0:
                    if k == abs_z.size:
                        abs_z_samples.append(abs_z)
                    else:
                        sample_idx = suggest_rng.choice(abs_z.size, size=k, replace=False)
                        abs_z_samples.append(abs_z[sample_idx])
            if group_raw_arr is not None:
                group_all.append(group_raw_arr[ok])
                _update_group_compound_sets(
                    group_labels=group_raw_arr[ok],
                    chem_ids=chem_id_arr[ok],
                    out_sets=group_compounds_modeled,
                )

        if missing.any() and backoff is not None and not require_seen:
            if (
                backoff_cluster_ids is None
                or backoff_mu is None
                or backoff_chem_ids is None
                or backoff_t is None
                or backoff_t0 is None
                or backoff_theta is None
                or backoff_tau_b2 is None
                or backoff_tau_t2 is None
                or backoff_sigma2 is None
                or backoff_lambda is None
            ):
                raise SystemExit("Backoff state not initialized")
            if backoff_mean_mode == "chem" and (emb_ids is None or emb_z is None):
                raise SystemExit("Backoff embeddings not initialized (backoff_mean_mode='chem')")

            cl = cluster_arr[missing]
            chem = chem_id_arr[missing]
            x_m = x[missing]
            group_missing = group_raw_arr[missing] if group_raw_arr is not None else None

            c_idx = np.searchsorted(backoff_cluster_ids, cl)
            c_ok = (c_idx >= 0) & (c_idx < backoff_cluster_ids.size)
            if c_ok.any():
                c_ok[c_ok] &= backoff_cluster_ids[c_idx[c_ok]] == cl[c_ok]

            if not c_ok.any():
                backoff_skipped += int(cl.size)
            else:
                cl = cl[c_ok]
                chem = chem[c_ok]
                x_m = x_m[c_ok]
                c_idx = c_idx[c_ok]
                if group_missing is not None:
                    group_missing = group_missing[c_ok]

                k_idx = np.searchsorted(backoff_chem_ids, chem)
                k_ok = (k_idx >= 0) & (k_idx < backoff_chem_ids.size)
                if k_ok.any():
                    k_ok[k_ok] &= backoff_chem_ids[k_idx[k_ok]] == chem[k_ok]

                if backoff_mean_mode == "cluster_only":
                    t_mean = np.full((int(chem.size),), float(backoff_t0), dtype=np.float64)
                    var_b = np.full(
                        (int(chem.size),),
                        float(backoff_tau_b2) + float(backoff_tau_t2),
                        dtype=np.float64,
                    )
                else:
                    t_mean = np.empty((int(chem.size),), dtype=np.float64)
                    var_b = np.full((int(chem.size),), float(backoff_tau_b2), dtype=np.float64)
                    if k_ok.any():
                        t_mean[k_ok] = backoff_t[k_idx[k_ok]]
                    if (~k_ok).any():
                        assert emb_ids is not None and emb_z is not None
                        chem_new = chem[~k_ok]
                        emb_idx = np.searchsorted(emb_ids, chem_new)
                        e_ok = (emb_idx >= 0) & (emb_idx < emb_ids.size)
                        if e_ok.any():
                            e_ok[e_ok] &= emb_ids[emb_idx[e_ok]] == chem_new[e_ok]
                        z = np.zeros((int(chem_new.size), int(emb_z.shape[1])), dtype=np.float64)
                        if e_ok.any():
                            z[e_ok] = emb_z[emb_idx[e_ok]]
                        if backoff_z_center is not None:
                            z = z - backoff_z_center[None, :]
                        t_mean_new = float(backoff_t0) + (z @ backoff_theta)
                        t_mean[~k_ok] = t_mean_new
                        var_b[~k_ok] = float(backoff_tau_b2) + float(backoff_tau_t2)

                b_mean = backoff_mu[c_idx] + t_mean
                if effective_backoff_slope_mode == "cluster_head":
                    assert backoff.w_cluster is not None
                    w_mean = np.asarray(backoff.w_cluster, dtype=np.float64)[c_idx]
                    pred_mean = b_mean + np.sum(w_mean * x_m, axis=1)
                elif effective_backoff_slope_mode == "cluster":
                    if backoff_w_mean_by_cluster is None:
                        raise SystemExit("Backoff cluster slope means not initialized")
                    w_mean = backoff_w_mean_by_cluster[c_idx]
                    pred_mean = b_mean + np.sum(w_mean * x_m, axis=1)
                elif effective_backoff_slope_mode == "global":
                    if backoff_w_mean_global is None:
                        raise SystemExit("Backoff global slope mean not initialized")
                    pred_mean = b_mean + np.sum(backoff_w_mean_global[None, :] * x_m, axis=1)
                else:
                    pred_mean = b_mean

                var_w = (float(backoff_sigma2) / float(backoff_lambda)) * np.sum(
                    np.square(x_m), axis=1
                )
                pred_var = np.maximum(float(backoff_sigma2) + var_b + var_w, 0.0)
                pred_std_base = np.sqrt(pred_var)
                pred_std = pred_std_base * pred_std_scale

                err = pred_mean - y[missing][c_ok]
                sq_err_all.append(np.square(err))
                abs_err_all.append(np.abs(err))
                covered_all.append(np.abs(err) <= (q_norm * pred_std))
                cluster_all.append(cl)
                if suggest_scale and suggest_rng is not None and suggest_per_chunk > 0:
                    abs_z = np.abs(err) / np.maximum(pred_std_base, 1e-12)
                    k = min(int(abs_z.size), int(suggest_per_chunk))
                    if k > 0:
                        if k == abs_z.size:
                            abs_z_samples.append(abs_z)
                        else:
                            sample_idx = suggest_rng.choice(abs_z.size, size=k, replace=False)
                            abs_z_samples.append(abs_z[sample_idx])
                if group_missing is not None:
                    group_all.append(group_missing)
                    _update_group_compound_sets(
                        group_labels=group_missing,
                        chem_ids=chem,
                        out_sets=group_compounds_modeled,
                    )
                backoff_used += int(err.size)

        if not sq_err_all:
            continue

        sq_err = np.concatenate(sq_err_all, axis=0)
        abs_err = np.concatenate(abs_err_all, axis=0)
        covered = np.concatenate(covered_all, axis=0)
        cluster_used = np.concatenate(cluster_all, axis=0)
        group_used = np.concatenate(group_all, axis=0) if group_all else None

        used_rows += int(sq_err.size)
        global_stats.update(sq_err, abs_err, covered)
        for cl in np.unique(cluster_used).tolist():
            mask = cluster_used == int(cl)
            stats = cluster_stats.setdefault(str(int(cl)), AggStats())
            stats.update(sq_err[mask], abs_err[mask], covered[mask])
        if group_used is not None:
            for g in np.unique(group_used).tolist():
                mask = group_used == g
                stats = group_stats.setdefault(str(g), AggStats())
                stats.update(sq_err[mask], abs_err[mask], covered[mask])

        if remaining is not None:
            remaining -= int(len(chunk))

    global_metrics = global_stats.to_metrics()
    suggested_pred_std_scale = None
    suggested_scale_sample_size = 0
    if suggest_scale and abs_z_samples:
        abs_z_all = np.concatenate(abs_z_samples, axis=0)
        suggested_scale_sample_size = int(abs_z_all.size)
        q_abs = float(np.quantile(abs_z_all, float(args.interval)))
        suggested_pred_std_scale = float(q_abs / q_norm) if q_norm > 0 else None
    out_json.write_text(
        json.dumps(
            {
                "metrics": global_metrics,
                "n_test": int(total_rows),
                "n_used": int(used_rows),
                "chunk_size": int(args.chunk_size),
                "interval": float(args.interval),
                "pred_std_scale": float(pred_std_scale),
                "suggested_pred_std_scale": suggested_pred_std_scale,
                "suggested_pred_std_scale_sample_size": int(suggested_scale_sample_size),
                "require_seen_group": require_seen,
                "skipped_missing_group": int(skipped_missing_group),
                "skipped_no_species_group": int(skipped_no_group),
                "backoff_npz": str(args.backoff_npz) if args.backoff_npz is not None else None,
                "backoff_mean_mode": str(backoff_mean_mode)
                if args.backoff_npz is not None
                else None,
                "backoff_slope_mean_mode": str(backoff_slope_mean_mode)
                if args.backoff_npz is not None
                else None,
                "backoff_slope_mean_mode_effective": str(effective_backoff_slope_mode)
                if args.backoff_npz is not None
                else None,
                "backoff_used": int(backoff_used),
                "backoff_skipped": int(backoff_skipped),
                "coeff_npz": str(coeff_npz),
                "test_csv": str(test_csv),
            },
            indent=2,
        )
    )

    rows = []
    for cl, stats in sorted(cluster_stats.items(), key=lambda kv: int(kv[0])):
        m = stats.to_metrics()
        rows.append({"species_cluster": int(cl), **m})
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    if write_by_species_group:
        group_rows = []
        for label, stats in sorted(group_stats.items()):
            m = stats.to_metrics()
            total_set = group_compounds_total.get(label, set())
            modeled_set = group_compounds_modeled.get(label, set())
            n_total = len(total_set)
            n_modeled = len(modeled_set)
            coverage = float(n_modeled / n_total) if n_total > 0 else float("nan")
            group_rows.append(
                {
                    "species_group_raw": label,
                    **m,
                    "n_compounds_total": int(n_total),
                    "n_compounds_modeled": int(n_modeled),
                    "compound_coverage": coverage,
                }
            )
        pd.DataFrame(group_rows).to_csv(out_group_csv, index=False)

    print(
        f"[coeff_eval] Global: n={global_metrics['n_obs']:,}, RMSE={global_metrics['rmse']:.4f}, "
        f"coverage95={global_metrics['coverage_95']:.3f}"
    )
    if suggested_pred_std_scale is not None:
        print(
            f"[coeff_eval] Suggested --pred-std-scale ~ {suggested_pred_std_scale:.3f} "
            f"(sample n={suggested_scale_sample_size:,})"
        )
    print(f"[coeff_eval] Wrote {out_json}")
    print(f"[coeff_eval] Wrote {out_csv}")
    if write_by_species_group:
        print(f"[coeff_eval] Wrote {out_group_csv}")


if __name__ == "__main__":
    main()
