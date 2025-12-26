#!/usr/bin/env python3
"""
Baseline (sklearn ridge): build per-group Bayesian-ridge coefficient summaries.

This script reads a production RT CSV (cap100) and groups rows by:
  g = (species_cluster, comp_id)

For each group it fits a ridge regression (via scikit-learn) and writes approximate
Bayesian-ridge posterior summaries (mean + diagonal variance + sigma2 mean):
  rt = intercept + dot(slopes, x_run) + eps

where x_run are run covariates from the CSV (IS*/RS*/optional ES_*).

Outputs under --output-dir:
  - stage1_coeff_summaries.npz
  - config.json
  - results/comp_id_mapping_collisions.csv (if any)
  - results/excluded_pairs.csv (if --exclude-pairs-frac > 0)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .pymc_collapsed_ridge import (
    Stage1CoeffSummaries,
    _encode_cluster_comp_key,
)

_HERE = Path(__file__).resolve()
for _parent in _HERE.parents:
    if (_parent / "pyproject.toml").exists():
        REPO_ROOT = _parent
        break
else:
    REPO_ROOT = Path.cwd().resolve()

REQUIRED_COLS = ["rt", "compound", "comp_id", "compound_class", "species_cluster"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build per-group Bayesian-ridge coefficient summaries from a production RT CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-csv", type=Path, required=True, help="Production RT CSV to train on."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/rt_ridge_coeff_summaries"),
        help="Output directory (will contain stage1_coeff_summaries.npz and config.json).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for determinism.")
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=0,
        help="Maximum number of training rows to process (0 = all).",
    )
    parser.add_argument(
        "--lambda-ridge",
        type=float,
        default=1e-3,
        help="L2 regularization strength for ridge regression (slopes only; intercept unpenalized).",
    )
    parser.add_argument(
        "--anchor-expansion",
        type=str,
        choices=["none", "poly2"],
        default="none",
        help=(
            "Optional nonlinear expansion applied to IS*/RS* anchor covariates only. "
            "'poly2' appends squares and pairwise interactions for anchor features."
        ),
    )
    parser.add_argument(
        "--lambda-ridge-poly",
        type=float,
        default=None,
        help=(
            "Optional L2 regularization strength for anchor expansion terms. "
            "Only used when --anchor-expansion poly2. Defaults to --lambda-ridge."
        ),
    )
    parser.add_argument(
        "--bayesian",
        action="store_true",
        help=(
            "Kept for backward compatibility (this trainer always writes Bayesian-style summaries)."
        ),
    )
    parser.add_argument(
        "--include-es-all",
        action="store_true",
        help="Include all ES_* covariates present in the CSV (no masking).",
    )
    parser.add_argument(
        "--include-es",
        action="store_true",
        help="Alias for --include-es-all.",
    )
    parser.add_argument(
        "--bayes-a0",
        type=float,
        default=2.0,
        help="InvGamma prior shape a0 for Bayesian ridge noise variance.",
    )
    parser.add_argument(
        "--bayes-b0",
        type=float,
        default=1e-6,
        help="InvGamma prior scale b0 for Bayesian ridge noise variance.",
    )
    parser.add_argument(
        "--exclude-pairs-frac",
        type=float,
        default=0.0,
        help="Fraction of (species_cluster, comp_id) pairs to exclude from Stage-1 training.",
    )
    parser.add_argument(
        "--exclude-pairs-seed",
        type=int,
        default=42,
        help="Random seed used to hash (cluster, comp_id) pairs for exclusion.",
    )
    parser.add_argument(
        "--exclude-chems-frac",
        type=float,
        default=0.0,
        help="Fraction of chem_id values to exclude from Stage-1 training (holdout stress test).",
    )
    parser.add_argument(
        "--exclude-chems-seed",
        type=int,
        default=42,
        help="Random seed used to hash chem_id values for exclusion.",
    )
    parser.add_argument(
        "--exclude-clusters",
        type=str,
        default="",
        help=(
            "Comma-separated list of species_cluster values to exclude from Stage-1 training "
            "(e.g. '0,6'; holdout stress test)."
        ),
    )
    parser.add_argument(
        "--feature-center",
        type=str,
        choices=["none", "global"],
        default="global",
        help=(
            "Optional global centering for run covariates: use x_centered = x - mean_x. "
            "This is a pure reparameterization (does not change ridge slopes) but makes the intercept "
            "and its variance numerically well-behaved for coefficient-space modeling."
        ),
    )
    parser.add_argument(
        "--feature-rotation",
        type=str,
        choices=["none", "pca"],
        default="none",
        help=(
            "Optional orthonormal rotation applied to run covariates before fitting "
            "(stored in the Stage-1 artifact and applied at evaluation time). "
            "Use 'pca' to reduce collinearity in coefficient-space modeling."
        ),
    )
    return parser.parse_args()


def _infer_feature_columns(columns: Sequence[str], *, include_es: bool) -> List[str]:
    is_cols = sorted([str(c) for c in columns if str(c).startswith("IS")])
    rs_cols = sorted([str(c) for c in columns if str(c).startswith("RS")])
    es_cols: List[str] = []
    if include_es:
        es_cols = sorted([str(c) for c in columns if str(c).startswith("ES_")])
    if not is_cols:
        raise SystemExit("No IS* covariate columns detected in CSV")
    return [*is_cols, *rs_cols, *es_cols]


def _is_anchor_feature(name: str) -> bool:
    return bool(str(name).startswith("IS") or str(name).startswith("RS"))


def _poly2_anchor_feature_names(anchor_names: Sequence[str]) -> List[str]:
    names = [f"poly2_sq({str(a)})" for a in anchor_names]
    for i, a in enumerate(anchor_names):
        for b in anchor_names[i + 1 :]:
            names.append(f"poly2_int({str(a)},{str(b)})")
    return names


def _poly2_anchor_expand(x_raw: np.ndarray, *, anchor_indices: Sequence[int]) -> np.ndarray:
    """Degree-2 expansion for a subset of columns (anchor-only).

    Returns: [squares, pairwise interactions] in a stable, deterministic order.
    """
    if not anchor_indices:
        return np.empty((int(x_raw.shape[0]), 0), dtype=np.float64)
    a = x_raw[:, np.asarray(anchor_indices, dtype=int)]
    squares = a * a
    inter_cols = []
    for i in range(a.shape[1]):
        ai = a[:, i]
        for j in range(i + 1, a.shape[1]):
            inter_cols.append((ai * a[:, j])[:, None])
    inter = np.concatenate(inter_cols, axis=1) if inter_cols else np.empty((a.shape[0], 0))
    out = np.concatenate([squares, inter], axis=1)
    return out.astype(np.float64, copy=False)


def _resolve_comp_id_mapping(
    comp_id_to_mapping_counts: Dict[int, Counter[Tuple[int, int]]],
) -> tuple[Dict[int, Tuple[int, int]], List[dict]]:
    comp_id_to_choice: Dict[int, Tuple[int, int]] = {}
    collision_rows: List[dict] = []
    for cid, counter in comp_id_to_mapping_counts.items():
        if not counter:
            continue
        # Deterministic tie-break: highest count, then smallest (chem_id, class_id).
        (chem_cls, mode_count) = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        comp_id_to_choice[int(cid)] = (int(chem_cls[0]), int(chem_cls[1]))
        if len(counter) > 1:
            for (chem_id, cls_id), cnt in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
                collision_rows.append(
                    {
                        "comp_id": int(cid),
                        "chem_id": int(chem_id),
                        "compound_class": int(cls_id),
                        "count": int(cnt),
                        "mode_chem_id": int(chem_cls[0]),
                        "mode_compound_class": int(chem_cls[1]),
                        "mode_count": int(mode_count),
                        "n_unique_mappings": int(len(counter)),
                    }
                )
    return comp_id_to_choice, collision_rows


def _write_excluded_pairs_csv(out_dir: Path, excluded_pair_keys: Set[int]) -> str | None:
    if not excluded_pair_keys:
        return None
    rows_out = []
    for key in sorted(excluded_pair_keys):
        u = np.uint64(key)
        cluster = int(u >> np.uint64(32))
        comp_id = int(u & np.uint64(0xFFFFFFFF))
        rows_out.append({"species_cluster": cluster, "comp_id": comp_id, "key": int(key)})
    excluded_path = out_dir / "results" / "excluded_pairs.csv"
    pd.DataFrame(rows_out).to_csv(excluded_path, index=False)
    return str(excluded_path)


def _write_excluded_chems_csv(out_dir: Path, excluded_chems: Set[int]) -> str | None:
    if not excluded_chems:
        return None
    rows_out = [{"chem_id": int(chem_id)} for chem_id in sorted(excluded_chems)]
    excluded_path = out_dir / "results" / "excluded_chems.csv"
    pd.DataFrame(rows_out).to_csv(excluded_path, index=False)
    return str(excluded_path)


def _write_excluded_clusters_csv(out_dir: Path, excluded_clusters: Set[int]) -> str | None:
    if not excluded_clusters:
        return None
    rows_out = [{"species_cluster": int(cluster)} for cluster in sorted(excluded_clusters)]
    excluded_path = out_dir / "results" / "excluded_clusters.csv"
    pd.DataFrame(rows_out).to_csv(excluded_path, index=False)
    return str(excluded_path)


def _write_collisions_csv(out_dir: Path, collision_rows: List[dict]) -> str | None:
    if not collision_rows:
        return None
    collisions_path = out_dir / "results" / "comp_id_mapping_collisions.csv"
    pd.DataFrame(collision_rows).to_csv(collisions_path, index=False)
    return str(collisions_path)


def _train_stage1_sklearn_ridge(  # noqa: C901
    *,
    args: argparse.Namespace,
    data_csv: Path,
    feature_cols: List[str],
    out_dir: Path,
    exclude_pair_mask_fn,
    exclude_chem_mask_fn,
    excluded_clusters: Set[int],
) -> tuple[Stage1CoeffSummaries, dict]:
    try:
        from sklearn.linear_model import Ridge  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise SystemExit("scikit-learn is required for Stage-1 training") from exc

    anchor_expansion = str(getattr(args, "anchor_expansion", "none"))
    if anchor_expansion not in {"none", "poly2"}:
        raise SystemExit(f"Unknown --anchor-expansion: {anchor_expansion}")
    if anchor_expansion != "none":
        if str(args.feature_center) != "none":
            raise SystemExit("--anchor-expansion requires --feature-center none")
        if str(args.feature_rotation) != "none":
            raise SystemExit("--anchor-expansion requires --feature-rotation none")
        if args.lambda_ridge_poly is not None and float(args.lambda_ridge_poly) < 0:
            raise SystemExit("--lambda-ridge-poly must be >= 0")

    dtype: dict[str, object] = {c: np.float32 for c in feature_cols}
    dtype.update(
        {
            "rt": np.float32,
            "species_cluster": np.int64,
            "comp_id": np.int64,
            "compound": np.int64,
            "compound_class": np.float32,
        }
    )
    df = pd.read_csv(
        data_csv,
        usecols=list(dict.fromkeys([*REQUIRED_COLS, *feature_cols])),
        nrows=int(args.max_train_rows) if args.max_train_rows and args.max_train_rows > 0 else None,
        dtype=dtype,
    )
    total_rows = int(len(df))
    df["compound_class"] = df["compound_class"].fillna(-1).astype(int)

    cluster_arr = df["species_cluster"].to_numpy(dtype=np.int64, copy=False)
    comp_arr = df["comp_id"].to_numpy(dtype=np.int64, copy=False)
    chem_arr = df["compound"].to_numpy(dtype=np.int64, copy=False)
    class_arr = df["compound_class"].to_numpy(dtype=np.int64, copy=False)
    y_arr = df["rt"].to_numpy(dtype=np.float64, copy=False)
    x_arr = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    np.nan_to_num(x_arr, copy=False)

    key_arr = _encode_cluster_comp_key(cluster_arr, comp_arr)
    excluded_cluster_rows = 0
    if excluded_clusters:
        excluded_arr = np.asarray(sorted(excluded_clusters), dtype=np.int64)
        keep_rows = ~np.isin(cluster_arr, excluded_arr)
        excluded_cluster_rows = int((~keep_rows).sum())
        if excluded_cluster_rows > 0:
            cluster_arr = cluster_arr[keep_rows]
            comp_arr = comp_arr[keep_rows]
            chem_arr = chem_arr[keep_rows]
            class_arr = class_arr[keep_rows]
            y_arr = y_arr[keep_rows]
            x_arr = x_arr[keep_rows]
            key_arr = key_arr[keep_rows]

    excluded_chems: Set[int] = set()
    excluded_chem_rows = 0
    if float(args.exclude_chems_frac) > 0.0:
        uniq_chem, inv_chem = np.unique(chem_arr, return_inverse=True)
        excluded_mask = exclude_chem_mask_fn(uniq_chem)
        if excluded_mask.any():
            excluded_chems.update(int(c) for c in uniq_chem[excluded_mask].tolist())
            keep_rows = ~excluded_mask[inv_chem]
            excluded_chem_rows = int((~keep_rows).sum())
            if excluded_chem_rows > 0:
                cluster_arr = cluster_arr[keep_rows]
                comp_arr = comp_arr[keep_rows]
                chem_arr = chem_arr[keep_rows]
                class_arr = class_arr[keep_rows]
                y_arr = y_arr[keep_rows]
                x_arr = x_arr[keep_rows]
                key_arr = key_arr[keep_rows]

    excluded_pair_keys: Set[int] = set()
    excluded_pair_rows = 0
    if float(args.exclude_pairs_frac) > 0.0:
        uniq_all, inv_all = np.unique(key_arr, return_inverse=True)
        excluded_keys = exclude_pair_mask_fn(uniq_all)
        if excluded_keys.any():
            excluded_pair_keys.update(int(k) for k in uniq_all[excluded_keys].tolist())
            keep_rows = ~excluded_keys[inv_all]
            excluded_pair_rows = int((~keep_rows).sum())
            key_arr = key_arr[keep_rows]
            cluster_arr = cluster_arr[keep_rows]
            comp_arr = comp_arr[keep_rows]
            chem_arr = chem_arr[keep_rows]
            class_arr = class_arr[keep_rows]
            y_arr = y_arr[keep_rows]
            x_arr = x_arr[keep_rows]

    used_rows = int(y_arr.shape[0])
    global_n = np.int64(used_rows)
    global_sum_x = x_arr.sum(axis=0, dtype=np.float64)

    # comp_id -> (chem_id, class_id) mapping counts.
    comp_id_to_mapping_counts: Dict[int, Counter[Tuple[int, int]]] = defaultdict(Counter)
    mapping_mat = np.stack([comp_arr, chem_arr, class_arr], axis=1)
    uniq_map, map_counts = np.unique(mapping_mat, axis=0, return_counts=True)
    for row, cnt in zip(uniq_map.tolist(), map_counts.tolist(), strict=True):
        cid, chem, cls = (int(row[0]), int(row[1]), int(row[2]))
        comp_id_to_mapping_counts[cid][(chem, cls)] += int(cnt)

    comp_id_to_choice, collision_rows = _resolve_comp_id_mapping(comp_id_to_mapping_counts)
    collisions_csv = _write_collisions_csv(out_dir, collision_rows)
    if collision_rows:
        print(
            f"[stage1] Detected comp_id mapping collisions: {len(set(r['comp_id'] for r in collision_rows)):,} comp_ids "
            f"(wrote {collisions_csv})"
        )

    excluded_pairs_csv = _write_excluded_pairs_csv(out_dir, excluded_pair_keys)
    excluded_chems_csv = _write_excluded_chems_csv(out_dir, excluded_chems)
    excluded_clusters_csv = _write_excluded_clusters_csv(out_dir, excluded_clusters)

    # Global feature transform for stability (optional).
    feature_center = None
    if str(args.feature_center) == "global":
        feature_center = global_sum_x / float(global_n)
        x_arr = x_arr.astype(np.float32, copy=False)
        x_arr = x_arr - feature_center.astype(np.float32)[None, :]
        print("[stage1] Applied global feature centering")

    feature_rotation = None
    if str(args.feature_rotation) == "pca":
        if feature_center is None:
            raise SystemExit("--feature-rotation pca requires --feature-center global")
        cov = (x_arr.T @ x_arr).astype(np.float64, copy=False)
        cov = cov / float(max(int(global_n) - 1, 1))
        cov = 0.5 * (cov + cov.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        feature_rotation = eigvecs[:, order].astype(np.float64, copy=False)
        print(f"[stage1] Computed PCA feature rotation: shape={feature_rotation.shape}")

    # Process groups in deterministic key order.
    sort_idx = np.argsort(key_arr, kind="mergesort")
    keys_sorted = key_arr[sort_idx]
    bounds = np.flatnonzero(np.diff(keys_sorted)) + 1
    starts = np.concatenate([np.array([0], dtype=np.int64), bounds.astype(np.int64)])
    ends = np.concatenate([bounds.astype(np.int64), np.array([keys_sorted.size], dtype=np.int64)])

    base_feature_cols = list(feature_cols)
    anchor_names = [c for c in base_feature_cols if _is_anchor_feature(c)]
    anchor_indices = [i for i, c in enumerate(base_feature_cols) if _is_anchor_feature(c)]
    poly_feature_names: List[str] = []
    if anchor_expansion == "poly2":
        poly_feature_names = _poly2_anchor_feature_names(anchor_names)
        if not poly_feature_names:
            raise SystemExit("--anchor-expansion poly2 selected but no IS*/RS* features found")

    design_feature_cols = [*base_feature_cols, *poly_feature_names]
    p = int(len(design_feature_cols))
    d = p + 1
    if anchor_expansion == "poly2":
        lambda_poly = (
            float(args.lambda_ridge)
            if args.lambda_ridge_poly is None
            else float(args.lambda_ridge_poly)
        )
        lambda_vec = np.concatenate(
            [
                np.full((len(base_feature_cols),), float(args.lambda_ridge), dtype=np.float64),
                np.full((len(poly_feature_names),), float(lambda_poly), dtype=np.float64),
            ],
            axis=0,
        )
    else:
        lambda_vec = np.full((len(base_feature_cols),), float(args.lambda_ridge), dtype=np.float64)
    if lambda_vec.shape != (p,):
        raise SystemExit("Internal error: lambda_vec has wrong shape")
    eye = np.eye(p, dtype=np.float64)
    penalty_mat = np.diag(lambda_vec) if anchor_expansion == "poly2" else None

    group_keys: List[int] = []
    clusters: List[int] = []
    comp_ids: List[int] = []
    n_obs: List[int] = []
    beta_hat: List[np.ndarray] = []
    beta_var_diag: List[np.ndarray] = []
    beta_cov: List[np.ndarray] = []
    sigma2_mean: List[float] = []

    for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
        idx = sort_idx[start:end]
        key = int(keys_sorted[start])
        cluster0 = int(cluster_arr[idx[0]])
        comp0 = int(comp_arr[idx[0]])

        xg = x_arr[idx].astype(np.float64, copy=False)
        if anchor_expansion == "poly2":
            x_extra = _poly2_anchor_expand(xg, anchor_indices=anchor_indices)
            xg = np.concatenate([xg, x_extra], axis=1)
        else:
            if feature_rotation is not None:
                xg = xg @ feature_rotation
        yg = y_arr[idx].astype(np.float64, copy=False)

        ni = int(yg.size)
        xm = xg.mean(axis=0)
        ym = float(yg.mean())
        xc = xg - xm[None, :]
        yc = yg - ym
        xtx = xc.T @ xc
        xty = xc.T @ yc
        yty = float(yc.T @ yc)

        if anchor_expansion == "poly2":
            assert penalty_mat is not None
            a = xtx + penalty_mat + 1e-12 * eye
            try:
                chol = np.linalg.cholesky(a)
                w = np.linalg.solve(chol.T, np.linalg.solve(chol, xty))
                inv_chol = np.linalg.solve(chol, eye)
                a_inv = inv_chol.T @ inv_chol
                v_diag = np.sum(inv_chol * inv_chol, axis=0)
            except np.linalg.LinAlgError:
                a_inv = np.linalg.inv(a)
                v_diag = np.diag(a_inv)
                w = a_inv @ xty
            b = float(ym - xm.dot(w))
        else:
            model = Ridge(alpha=float(args.lambda_ridge), fit_intercept=True)
            model.fit(xg, yg)
            w = np.asarray(model.coef_, dtype=np.float64)
            b = float(model.intercept_)

            a = xtx + float(args.lambda_ridge) * eye
            a_inv = None
            try:
                chol = np.linalg.cholesky(a)
                inv_chol = np.linalg.solve(chol, eye)
                v_diag = np.sum(inv_chol * inv_chol, axis=0)
                a_inv = inv_chol.T @ inv_chol
            except np.linalg.LinAlgError:
                inv_a = np.linalg.inv(a)
                v_diag = np.diag(inv_a)
                a_inv = inv_a

        w_xty = float(w.dot(xty))
        sse_pen = max(yty - w_xty, 0.0)

        a_n = float(args.bayes_a0) + 0.5 * float(ni)
        b_n = float(args.bayes_b0) + 0.5 * float(sse_pen)
        sigma2_mean_i = b_n / (a_n - 1.0)

        assert a_inv is not None

        slopes_var = sigma2_mean_i * v_diag
        inv_n = 1.0 / float(max(ni, 1))
        intercept_var = sigma2_mean_i * (inv_n + float(xm.T @ a_inv @ xm))
        beta_var_diag_i = np.concatenate([[intercept_var], slopes_var], axis=0)
        beta_hat_i = np.concatenate([[b], w], axis=0)

        slopes_cov = sigma2_mean_i * a_inv
        cov_bw = -sigma2_mean_i * (a_inv @ xm)
        beta_cov_i = np.empty((d, d), dtype=np.float64)
        beta_cov_i[0, 0] = float(intercept_var)
        beta_cov_i[0, 1:] = cov_bw
        beta_cov_i[1:, 0] = cov_bw
        beta_cov_i[1:, 1:] = slopes_cov

        group_keys.append(key)
        clusters.append(cluster0)
        comp_ids.append(comp0)
        n_obs.append(ni)
        beta_hat.append(beta_hat_i.astype(np.float64, copy=False))
        beta_var_diag.append(beta_var_diag_i.astype(np.float64, copy=False))
        beta_cov.append(beta_cov_i.astype(np.float64, copy=False))
        sigma2_mean.append(float(sigma2_mean_i))

    keys_arr = np.asarray(group_keys, dtype=np.int64)
    clusters_arr = np.asarray(clusters, dtype=np.int64)
    comp_ids_arr = np.asarray(comp_ids, dtype=np.int64)
    n_arr = np.asarray(n_obs, dtype=np.int64)
    beta_hat_arr = np.stack(beta_hat, axis=0).astype(np.float64, copy=False)
    beta_var_diag_arr = np.stack(beta_var_diag, axis=0).astype(np.float64, copy=False)
    beta_cov_arr = np.stack(beta_cov, axis=0).astype(np.float64, copy=False)
    sigma2_mean_arr = np.asarray(sigma2_mean, dtype=np.float64)

    # Map each group to canonical (chem_id, compound_class) for its comp_id.
    chem_ids_arr = np.full(comp_ids_arr.shape, -1, dtype=np.int64)
    class_ids_arr = np.full(comp_ids_arr.shape, -1, dtype=np.int64)
    for i, cid in enumerate(comp_ids_arr.tolist()):
        chem_cls = comp_id_to_choice.get(int(cid))
        if chem_cls is None:
            continue
        chem_ids_arr[i] = int(chem_cls[0])
        class_ids_arr[i] = int(chem_cls[1])

    artifact = Stage1CoeffSummaries(
        feature_names=tuple(design_feature_cols),
        feature_center=feature_center,
        feature_rotation=feature_rotation,
        group_keys=keys_arr,
        species_cluster=clusters_arr,
        comp_id=comp_ids_arr,
        chem_id=chem_ids_arr,
        compound_class=class_ids_arr,
        n_obs=n_arr,
        beta_hat=beta_hat_arr,
        beta_var_diag=beta_var_diag_arr,
        beta_cov=beta_cov_arr,
        sigma2_mean=sigma2_mean_arr,
    )

    config = {
        "timestamp": datetime.now().isoformat(),
        "artifact_type": "rt_ridge_coeff_summaries",
        "backend": "sklearn_ridge",
        "data_csv": str(data_csv),
        "feature_names": feature_cols,
        "n_features": int(len(feature_cols)),
        "n_coefs": int(d),
        "max_train_rows": int(args.max_train_rows),
        "n_obs_train": int(used_rows),
        "lambda_ridge": float(args.lambda_ridge),
        "anchor_expansion": anchor_expansion,
        "lambda_ridge_poly": float(args.lambda_ridge_poly)
        if args.lambda_ridge_poly is not None
        else None,
        "bayes_a0": float(args.bayes_a0),
        "bayes_b0": float(args.bayes_b0),
        "bayesian": True,
        "include_es_all": bool(bool(args.include_es_all) or bool(args.include_es)),
        "exclude_pairs_frac": float(args.exclude_pairs_frac),
        "exclude_pairs_seed": int(args.exclude_pairs_seed),
        "excluded_pairs_n": int(len(excluded_pair_keys)),
        "excluded_pairs_csv": excluded_pairs_csv,
        "exclude_chems_frac": float(args.exclude_chems_frac),
        "exclude_chems_seed": int(args.exclude_chems_seed),
        "excluded_chems_n": int(len(excluded_chems)),
        "excluded_chems_csv": excluded_chems_csv,
        "exclude_clusters": sorted(int(c) for c in excluded_clusters),
        "excluded_clusters_n": int(len(excluded_clusters)),
        "excluded_clusters_csv": excluded_clusters_csv,
        "n_groups": int(keys_arr.size),
        "beta_cov": "full",
        "feature_center": str(args.feature_center),
        "feature_rotation": str(args.feature_rotation),
        "mapping_collisions_n_comp_ids": int(
            len(set(r["comp_id"] for r in collision_rows)) if collision_rows else 0
        ),
        "mapping_collisions_csv": collisions_csv,
        "total_rows_seen": int(total_rows),
        "excluded_pair_rows": int(excluded_pair_rows),
        "excluded_chem_rows": int(excluded_chem_rows),
        "excluded_cluster_rows": int(excluded_cluster_rows),
    }
    return artifact, config


def main() -> None:
    args = parse_args()
    np.random.seed(int(args.seed))

    include_es = bool(args.include_es_all) or bool(args.include_es)
    data_csv = args.data_csv
    if not data_csv.is_absolute():
        data_csv = (REPO_ROOT / data_csv).resolve()
    if not data_csv.exists():
        raise SystemExit(f"Training CSV not found: {data_csv}")

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results").mkdir(exist_ok=True)

    header = pd.read_csv(data_csv, nrows=0)
    missing_req = [c for c in REQUIRED_COLS if c not in header.columns]
    if missing_req:
        raise SystemExit(f"CSV missing required columns: {missing_req}")

    feature_cols = _infer_feature_columns(header.columns, include_es=include_es)
    print(f"[stage1] Training CSV: {data_csv}")
    print(
        f"[stage1] Features: IS={len([c for c in feature_cols if c.startswith('IS')])}, "
        f"RS={len([c for c in feature_cols if c.startswith('RS')])}, "
        f"ES={len([c for c in feature_cols if c.startswith('ES_')])} (total={len(feature_cols)})"
    )

    exclude_pairs_frac = float(args.exclude_pairs_frac)
    exclude_pairs_seed = int(args.exclude_pairs_seed)
    if exclude_pairs_frac < 0.0 or exclude_pairs_frac >= 1.0:
        raise SystemExit("--exclude-pairs-frac must be in [0, 1)")
    exclude_pairs_threshold = int(exclude_pairs_frac * float(2**64 - 1))

    def _exclude_pair_mask(keys: np.ndarray) -> np.ndarray:
        if exclude_pairs_threshold <= 0:
            return np.zeros(keys.shape, dtype=bool)
        x = np.asarray(keys, dtype=np.uint64)
        x ^= np.uint64(exclude_pairs_seed)
        x *= np.uint64(0x9E3779B97F4A7C15)
        x ^= x >> np.uint64(33)
        x *= np.uint64(0xC2B2AE3D27D4EB4F)
        x ^= x >> np.uint64(29)
        return x < np.uint64(exclude_pairs_threshold)

    exclude_chems_frac = float(args.exclude_chems_frac)
    exclude_chems_seed = int(args.exclude_chems_seed)
    if exclude_chems_frac < 0.0 or exclude_chems_frac >= 1.0:
        raise SystemExit("--exclude-chems-frac must be in [0, 1)")
    exclude_chems_threshold = int(exclude_chems_frac * float(2**64 - 1))

    def _exclude_chem_mask(chem_ids: np.ndarray) -> np.ndarray:
        if exclude_chems_threshold <= 0:
            return np.zeros(chem_ids.shape, dtype=bool)
        x = np.asarray(chem_ids, dtype=np.uint64)
        x ^= np.uint64(exclude_chems_seed)
        x *= np.uint64(0x9E3779B97F4A7C15)
        x ^= x >> np.uint64(33)
        x *= np.uint64(0xC2B2AE3D27D4EB4F)
        x ^= x >> np.uint64(29)
        return x < np.uint64(exclude_chems_threshold)

    excluded_clusters: Set[int] = set()
    if str(args.exclude_clusters).strip():
        try:
            excluded_clusters = set(
                int(tok.strip())
                for tok in str(args.exclude_clusters).split(",")
                if tok.strip() != ""
            )
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(
                "--exclude-clusters must be a comma-separated list of integers"
            ) from exc

    backend = "sklearn_ridge (in-memory)"
    if str(args.anchor_expansion) != "none":
        backend = "closed_form_ridge (+poly2 anchor terms)"
    print(f"[stage1] Backend: {backend}")
    artifact, config = _train_stage1_sklearn_ridge(
        args=args,
        data_csv=data_csv,
        feature_cols=feature_cols,
        out_dir=out_dir,
        exclude_pair_mask_fn=_exclude_pair_mask,
        exclude_chem_mask_fn=_exclude_chem_mask,
        excluded_clusters=excluded_clusters,
    )
    artifact_path = out_dir / "stage1_coeff_summaries.npz"
    artifact.save_npz(artifact_path)
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    print(f"[stage1] Wrote Stage-1 summaries: {artifact_path}")
    print(
        f"[stage1] n_groups={int(artifact.group_keys.size):,}, n_obs_train={int(config['n_obs_train']):,}"
    )


if __name__ == "__main__":
    main()
