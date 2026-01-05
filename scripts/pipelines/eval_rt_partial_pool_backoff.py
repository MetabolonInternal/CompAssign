#!/usr/bin/env python3
"""
Evaluate a partial-pooling Stage1CoeffSummaries artifact on a production RT CSV, with backoff.

Unlike the standard evaluators (which skip unseen groups), this script can score rows whose
(species, comp_id) key is missing from the coefficient artifact by constructing a "prior" coefficient
distribution using `PartialPoolBackoffSummaries`:

  (Optional) If `(species_cluster, comp_id)` was observed during training (through other species),
  we first back off to a derived `(species_cluster, comp_id)` coefficient by aggregating the fitted
  `(species, comp_id)` coefficients across species. This provides compound-specific slopes for
  unseen-species rows (and should match the supercategory ridge behaviour closely for `holdout=species`).

  b ~ Normal(t0 + mu_species + alpha_comp, tau_b)
  w ~ Normal(w_species, sigma2 * Lambda^{-1})

and falling back further to supercategory means (species_cluster) and/or compound_class means when a
species or comp_id is entirely unseen in training.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
import sys
import time

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.compassign.rt.pymc_collapsed_ridge import (  # noqa: E402
    PartialPoolBackoffSummaries,
    Stage1CoeffSummaries,
)
from src.compassign.utils.data_features import load_chemberta_pca20  # noqa: E402

BASE_REQUIRED_COLS = ["rt", "comp_id", "compound", "compound_class", "species", "species_cluster"]

_POLY2_SQ_RE = re.compile(r"^poly2_sq\((?P<name>[^)]+)\)$")
_POLY2_INT_RE = re.compile(r"^poly2_int\((?P<a>[^,]+),(?P<b>[^)]+)\)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate partial-pooling Stage1CoeffSummaries with backoff for unseen groups.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--coeff-npz", type=Path, required=True, help="Stage1CoeffSummaries .npz.")
    parser.add_argument(
        "--backoff-npz",
        type=Path,
        required=True,
        help="PartialPoolBackoffSummaries .npz written alongside the coeff artifact.",
    )
    parser.add_argument("--test-csv", type=Path, required=True, help="RT production CSV to score.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (writes results JSON). Defaults next to coeff artifact.",
    )
    parser.add_argument("--chunk-size", type=int, default=50_000, help="Rows per prediction chunk.")
    parser.add_argument(
        "--max-test-rows", type=int, default=0, help="Maximum number of test rows (0 = all)."
    )
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
        "--label",
        type=str,
        default=None,
        help="Optional label used in output filenames.",
    )
    parser.add_argument(
        "--chem-embeddings-path",
        type=Path,
        default=Path("resources/metabolites/embeddings_chemberta_pca20.parquet"),
        help="ChemBERTa PCA-20 embedding parquet (required for chem-linear compound backoff).",
    )
    parser.add_argument(
        "--log-every-chunks",
        type=int,
        default=20,
        help="Print a progress line every N chunks (0 disables).",
    )
    return parser.parse_args()


@dataclass
class SumStats:
    n: int = 0
    sum_sq_err: float = 0.0
    sum_abs_err: float = 0.0
    sum_covered: float = 0.0

    def update(self, *, err: np.ndarray, covered: np.ndarray) -> None:
        if err.size == 0:
            return
        self.n += int(err.size)
        self.sum_sq_err += float(np.sum(np.square(err)))
        self.sum_abs_err += float(np.sum(np.abs(err)))
        self.sum_covered += float(np.sum(covered.astype(np.float64, copy=False)))

    def to_metrics(self) -> dict[str, float]:
        if self.n <= 0:
            return {
                "n_obs": 0,
                "rmse": float("nan"),
                "mae": float("nan"),
                "coverage_95": float("nan"),
            }
        rmse = float(np.sqrt(self.sum_sq_err / float(self.n)))
        mae = float(self.sum_abs_err / float(self.n))
        cov = float(self.sum_covered / float(self.n))
        return {"n_obs": int(self.n), "rmse": rmse, "mae": mae, "coverage_95": cov}


def _default_output_dir(coeff_npz: Path) -> Path:
    if coeff_npz.parent.name == "models":
        return coeff_npz.parent.parent / "results"
    return coeff_npz.parent / "results"


def _design_feature_raw_dependencies(feature_name: str) -> set[str]:
    m_sq = _POLY2_SQ_RE.match(feature_name)
    if m_sq:
        return {str(m_sq.group("name"))}
    m_int = _POLY2_INT_RE.match(feature_name)
    if m_int:
        return {str(m_int.group("a")), str(m_int.group("b"))}
    return {str(feature_name)}


def _split_design_features(feature_names: tuple[str, ...]) -> tuple[list[str], list[str], bool]:
    raw_features: list[str] = []
    raw_deps: set[str] = set()
    has_derived = False
    for name in feature_names:
        raw_deps |= _design_feature_raw_dependencies(name)
        if _POLY2_SQ_RE.match(name) or _POLY2_INT_RE.match(name):
            has_derived = True
            continue
        raw_features.append(str(name))
    raw_dependencies = raw_features[:]
    for dep in sorted(raw_deps):
        if dep not in raw_dependencies:
            raw_dependencies.append(dep)
    return raw_features, raw_dependencies, has_derived


def _design_matrix_from_chunk(
    *, chunk: pd.DataFrame, design_features: tuple[str, ...], raw_features: list[str]
) -> np.ndarray:
    chunk[raw_features] = chunk[raw_features].fillna(0.0)
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
            v = raw_x[:, raw_pos[src]]
            x[:, j] = v * v
            continue
        m_int = _POLY2_INT_RE.match(name)
        if m_int:
            a = str(m_int.group("a"))
            b = str(m_int.group("b"))
            x[:, j] = raw_x[:, raw_pos[a]] * raw_x[:, raw_pos[b]]
            continue
        raise SystemExit(f"Unsupported derived feature: {name}")

    return x


def _row_variance_from_beta_cov(
    *, x1: np.ndarray, group_idx: np.ndarray, beta_cov: np.ndarray
) -> np.ndarray:
    out = np.empty((int(x1.shape[0]),), dtype=np.float64)
    for i in range(out.size):
        j = int(group_idx[i])
        out[i] = float(x1[i].T @ beta_cov[j] @ x1[i])
    return out


def _compute_supercat_means(
    *, supercat_ids: np.ndarray, cluster_supercat_id: np.ndarray, values: np.ndarray
) -> np.ndarray:
    """Compute per-supercat mean of an array aligned with clusters."""
    supercat_ids = np.asarray(supercat_ids, dtype=np.int64)
    cluster_supercat_id = np.asarray(cluster_supercat_id, dtype=np.int64)
    if values.ndim == 1:
        sums = np.zeros((int(supercat_ids.size),), dtype=np.float64)
    else:
        sums = np.zeros((int(supercat_ids.size), int(values.shape[1])), dtype=np.float64)
    counts = np.zeros((int(supercat_ids.size),), dtype=np.float64)
    idx = np.searchsorted(supercat_ids, cluster_supercat_id)
    for i in range(int(cluster_supercat_id.size)):
        j = int(idx[i])
        if (
            j < 0
            or j >= int(supercat_ids.size)
            or int(supercat_ids[j]) != int(cluster_supercat_id[i])
        ):
            continue
        sums[j] += values[i]
        counts[j] += 1.0
    counts = np.maximum(counts, 1.0)
    return sums / counts[:, None] if sums.ndim == 2 else (sums / counts)


def _compute_class_means(
    *, class_ids: np.ndarray, classes: np.ndarray, values: np.ndarray
) -> np.ndarray:
    """Compute per-class mean of values aligned with comp ids."""
    class_ids = np.asarray(class_ids, dtype=np.int64)
    classes = np.asarray(classes, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    sums = np.zeros((int(class_ids.size),), dtype=np.float64)
    counts = np.zeros((int(class_ids.size),), dtype=np.float64)
    idx = np.searchsorted(class_ids, classes)
    for i in range(int(classes.size)):
        j = int(idx[i])
        if j < 0 or j >= int(class_ids.size) or int(class_ids[j]) != int(classes[i]):
            continue
        sums[j] += float(values[i])
        counts[j] += 1.0
    counts = np.maximum(counts, 1.0)
    return sums / counts


def _build_supercat_comp_backoff_beta(
    *,
    supercat_id: np.ndarray,
    comp_id: np.ndarray,
    n_obs: np.ndarray,
    beta_hat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (keys, beta_hat) for derived (species_cluster, comp_id) coefficients.

    This aggregates fitted (species, comp_id) posterior means across species, weighted by n_obs.
    The resulting keys are encoded as (supercat_id<<32)+comp_id and sorted.
    """
    supercat_id = np.asarray(supercat_id, dtype=np.int64)
    comp_id = np.asarray(comp_id, dtype=np.int64)
    n_obs = np.asarray(n_obs, dtype=np.float64)
    beta_hat = np.asarray(beta_hat, dtype=np.float64)

    if supercat_id.ndim != 1 or comp_id.ndim != 1 or n_obs.ndim != 1:
        raise ValueError("Expected 1D arrays for supercat_id/comp_id/n_obs")
    if beta_hat.ndim != 2 or beta_hat.shape[0] != int(supercat_id.size):
        raise ValueError("beta_hat must be 2D and aligned with supercat_id")

    ok = supercat_id >= 0
    if not bool(np.any(ok)):
        return np.zeros((0,), dtype=np.int64), np.zeros(
            (0, int(beta_hat.shape[1])), dtype=np.float64
        )

    sc = supercat_id[ok]
    cid = comp_id[ok]
    w = n_obs[ok]
    beta = beta_hat[ok]

    key = (sc.astype(np.int64) << np.int64(32)) + cid.astype(np.int64)
    order = np.argsort(key, kind="mergesort")
    key_s = key[order]
    beta_s = beta[order]
    w_s = w[order]

    bounds = np.flatnonzero(np.diff(key_s)) + 1
    starts = np.concatenate([np.array([0], dtype=np.int64), bounds.astype(np.int64)])
    uniq = key_s[starts]

    w_sum = np.add.reduceat(w_s, starts, axis=0)
    w_sum = np.maximum(w_sum, 1e-12)
    beta_sum = np.add.reduceat(beta_s * w_s[:, None], starts, axis=0)
    beta_mean = beta_sum / w_sum[:, None]
    return uniq.astype(np.int64, copy=False), beta_mean.astype(np.float64, copy=False)


def _build_comp_backoff_beta_and_mu(
    *,
    comp_id: np.ndarray,
    n_obs: np.ndarray,
    beta_hat: np.ndarray,
    mu_group: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (comp_ids_unique, beta_hat_means, mu_means) for comp_id-only backoff.

    We aggregate:
      - beta_hat (intercept + slopes) across (species, comp_id) groups, weighted by n_obs
      - mu_group (species offsets for the same groups), weighted by n_obs
    """
    comp_id = np.asarray(comp_id, dtype=np.int64)
    n_obs = np.asarray(n_obs, dtype=np.float64)
    beta_hat = np.asarray(beta_hat, dtype=np.float64)
    mu_group = np.asarray(mu_group, dtype=np.float64)

    if comp_id.ndim != 1 or n_obs.ndim != 1 or mu_group.ndim != 1:
        raise ValueError("Expected 1D arrays for comp_id/n_obs/mu_group")
    if beta_hat.ndim != 2 or beta_hat.shape[0] != int(comp_id.size):
        raise ValueError("beta_hat must be 2D and aligned with comp_id")
    if mu_group.shape[0] != int(comp_id.size):
        raise ValueError("mu_group must be aligned with comp_id")

    order = np.argsort(comp_id, kind="mergesort")
    comp_sorted = comp_id[order]
    beta_sorted = beta_hat[order]
    mu_sorted = mu_group[order]
    w_sorted = np.maximum(n_obs[order], 1.0)

    bounds = np.flatnonzero(np.diff(comp_sorted)) + 1
    starts = np.concatenate([np.asarray([0], dtype=np.int64), bounds.astype(np.int64)])
    comp_ids_unique = comp_sorted[starts].astype(np.int64, copy=False)

    w_sum = np.add.reduceat(w_sorted, starts, axis=0)
    w_sum = np.maximum(w_sum, 1e-12)
    beta_sum = np.add.reduceat(beta_sorted * w_sorted[:, None], starts, axis=0)
    mu_sum = np.add.reduceat(mu_sorted * w_sorted, starts, axis=0)
    beta_mean = beta_sum / w_sum[:, None]
    mu_mean = mu_sum / w_sum
    return (
        comp_ids_unique.astype(np.int64, copy=False),
        beta_mean.astype(np.float64, copy=False),
        mu_mean.astype(np.float64, copy=False),
    )


def main() -> None:
    args = parse_args()
    if not (0.0 < float(args.interval) < 1.0):
        raise SystemExit("--interval must be in (0, 1)")
    if float(args.pred_std_scale) <= 0.0:
        raise SystemExit("--pred-std-scale must be > 0")

    coeff_npz = args.coeff_npz
    if not coeff_npz.is_absolute():
        coeff_npz = (REPO_ROOT / coeff_npz).resolve()
    stage1 = Stage1CoeffSummaries.load_npz(coeff_npz)

    backoff_npz = args.backoff_npz
    if not backoff_npz.is_absolute():
        backoff_npz = (REPO_ROOT / backoff_npz).resolve()
    backoff = PartialPoolBackoffSummaries.load_npz(backoff_npz)

    test_csv = args.test_csv
    if not test_csv.is_absolute():
        test_csv = (REPO_ROOT / test_csv).resolve()

    out_dir = args.output_dir or _default_output_dir(coeff_npz)
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    label_suffix = f"_{args.label}" if args.label else ""
    out_json = out_dir / f"rt_eval_partial_pool_backoff{label_suffix}.json"

    group_col = str(getattr(stage1, "group_col", "species_cluster"))
    if group_col != "species":
        raise SystemExit(f"Stage1CoeffSummaries group_col must be 'species' (got {group_col!r})")
    if tuple(str(s) for s in stage1.feature_names) != tuple(str(s) for s in backoff.feature_names):
        raise SystemExit("Feature mismatch between coeff-npz and backoff-npz")

    alpha = 0.5 + 0.5 * float(args.interval)
    q_norm = float(NormalDist().inv_cdf(alpha))
    pred_std_scale = float(args.pred_std_scale)

    keys = np.asarray(stage1.group_keys, dtype=np.int64)
    beta_hat = np.asarray(stage1.beta_hat, dtype=np.float64)
    beta_var_diag = np.asarray(stage1.beta_var_diag, dtype=np.float64)
    beta_cov = stage1.beta_cov
    sigma2 = np.asarray(stage1.sigma2_mean, dtype=np.float64)
    supercat_id_stage1 = stage1.supercat_id

    design_features = tuple(str(c) for c in stage1.feature_names)
    raw_features_in_order, raw_deps, has_derived = _split_design_features(design_features)
    if has_derived and (stage1.feature_center is not None or stage1.feature_rotation is not None):
        raise SystemExit("Derived features are incompatible with feature_center/feature_rotation")

    feature_center = stage1.feature_center
    feature_rotation = stage1.feature_rotation
    if feature_center is not None:
        feature_center = np.asarray(feature_center, dtype=np.float64)
    if feature_rotation is not None:
        feature_rotation = np.asarray(feature_rotation, dtype=np.float64)

    header = pd.read_csv(test_csv, nrows=0)
    missing_req = [c for c in BASE_REQUIRED_COLS if c not in header.columns]
    if missing_req:
        raise SystemExit(f"CSV missing required columns: {missing_req}")
    missing_feats = [c for c in raw_deps if c not in header.columns]
    if missing_feats:
        raise SystemExit(f"CSV missing required run covariate columns: {missing_feats}")

    # Precompute backoff lookups.
    cluster_ids = np.asarray(backoff.cluster_ids, dtype=np.int64)
    cluster_supercat_id = np.asarray(backoff.cluster_supercat_id, dtype=np.int64)
    comp_ids = np.asarray(backoff.comp_ids, dtype=np.int64)
    comp_class = np.asarray(backoff.comp_class, dtype=np.int64)
    mu_cluster = np.asarray(backoff.mu_cluster, dtype=np.float64)
    alpha_comp = np.asarray(backoff.alpha_comp, dtype=np.float64)
    w_cluster = np.asarray(backoff.w_cluster, dtype=np.float64)
    t0 = float(backoff.t0)
    tau_b2 = float(backoff.tau_b) ** 2
    sigma2_backoff = float(backoff.sigma2)
    lambda_slopes = float(backoff.lambda_slopes)
    if lambda_slopes <= 0.0:
        raise SystemExit("Invalid lambda_slopes in backoff artifact")
    tau_comp2 = float(backoff.tau_comp) ** 2

    supercat_ids = np.unique(cluster_supercat_id).astype(np.int64, copy=False)
    mu_supercat = _compute_supercat_means(
        supercat_ids=supercat_ids, cluster_supercat_id=cluster_supercat_id, values=mu_cluster
    )
    w_supercat = _compute_supercat_means(
        supercat_ids=supercat_ids, cluster_supercat_id=cluster_supercat_id, values=w_cluster
    )
    p = int(len(backoff.feature_names))
    w_global = np.mean(w_cluster, axis=0) if w_cluster.size else np.zeros((p,), dtype=np.float64)

    sc_comp_keys = None
    sc_comp_beta_hat = None
    if supercat_id_stage1 is not None:
        try:
            sc_comp_keys, sc_comp_beta_hat = _build_supercat_comp_backoff_beta(
                supercat_id=np.asarray(supercat_id_stage1, dtype=np.int64),
                comp_id=np.asarray(stage1.comp_id, dtype=np.int64),
                n_obs=np.asarray(stage1.n_obs, dtype=np.int64),
                beta_hat=beta_hat,
            )
        except Exception as e:
            raise SystemExit(f"Failed to build supercat-comp backoff coefficients: {e}") from e
        if sc_comp_keys.size == 0:
            sc_comp_keys = None
            sc_comp_beta_hat = None

    comp_backoff_ids = None
    comp_backoff_beta_hat = None
    comp_backoff_mu_mean = None
    try:
        species_train = np.asarray(stage1.species_cluster, dtype=np.int64)
        idx_mu = np.searchsorted(cluster_ids, species_train)
        ok_mu = (idx_mu >= 0) & (idx_mu < cluster_ids.size)
        if ok_mu.any():
            ok_mu[ok_mu] &= cluster_ids[idx_mu[ok_mu]] == species_train[ok_mu]
        mu_train = np.zeros((int(species_train.size),), dtype=np.float64)
        if ok_mu.any():
            mu_train[ok_mu] = mu_cluster[idx_mu[ok_mu]]

        (
            comp_backoff_ids,
            comp_backoff_beta_hat,
            comp_backoff_mu_mean,
        ) = _build_comp_backoff_beta_and_mu(
            comp_id=np.asarray(stage1.comp_id, dtype=np.int64),
            n_obs=np.asarray(stage1.n_obs, dtype=np.int64),
            beta_hat=beta_hat,
            mu_group=mu_train,
        )
    except Exception as e:
        raise SystemExit(f"Failed to build comp-id backoff coefficients: {e}") from e
    if comp_backoff_ids.size == 0:
        comp_backoff_ids = None
        comp_backoff_beta_hat = None
        comp_backoff_mu_mean = None

    class_ids = np.unique(comp_class[comp_class >= 0]).astype(np.int64, copy=False)
    alpha_class = (
        _compute_class_means(class_ids=class_ids, classes=comp_class, values=alpha_comp)
        if class_ids.size > 0
        else np.zeros((0,), dtype=np.float64)
    )

    emb_path = args.chem_embeddings_path
    if not emb_path.is_absolute():
        emb_path = (REPO_ROOT / emb_path).resolve()

    emb = load_chemberta_pca20(emb_path)
    order = np.argsort(emb.chem_id.astype(np.int64), kind="mergesort")
    embed_pos = emb.chem_id[order].astype(np.int64, copy=False)
    embed_features = emb.features[order].astype(np.float64, copy=False)

    theta_alpha = np.asarray(backoff.alpha_theta, dtype=np.float64)
    z_center = np.asarray(backoff.alpha_z_center, dtype=np.float64)
    if theta_alpha.ndim != 1 or z_center.ndim != 1 or theta_alpha.shape != z_center.shape:
        raise SystemExit("Invalid alpha_theta/alpha_z_center in backoff artifact")
    if int(embed_features.shape[1]) != int(theta_alpha.size):
        raise SystemExit(
            "Embedding dim mismatch: "
            f"alpha_theta has d={int(theta_alpha.size)}, embeddings have d={int(embed_features.shape[1])}"
        )

    total_rows = 0
    max_rows = int(args.max_test_rows)
    remaining = max_rows if max_rows > 0 else None

    stats_all = SumStats()
    stats_seen = SumStats()
    stats_sc_comp = SumStats()
    stats_comp_backoff = SumStats()
    stats_backoff = SumStats()
    n_seen = 0
    n_sc_comp = 0
    n_comp_backoff = 0
    n_backoff = 0

    chunk_i = 0
    t_start = time.time()
    log_every = int(args.log_every_chunks)

    usecols = list(dict.fromkeys([*BASE_REQUIRED_COLS, *raw_deps]))
    for chunk in pd.read_csv(test_csv, chunksize=int(args.chunk_size), usecols=usecols):
        chunk_i += 1
        if remaining is not None and remaining <= 0:
            break

        total_rows += int(len(chunk))
        if remaining is not None and len(chunk) > remaining:
            chunk = chunk.iloc[:remaining].copy()

        species = chunk["species"].astype(int).to_numpy(dtype=np.int64, copy=False)
        comp = chunk["comp_id"].astype(int).to_numpy(dtype=np.int64, copy=False)
        chem = chunk["compound"].astype(int).to_numpy(dtype=np.int64, copy=False)
        supercat = chunk["species_cluster"].astype(int).to_numpy(dtype=np.int64, copy=False)
        cls = chunk["compound_class"].fillna(-1).astype(int).to_numpy(dtype=np.int64, copy=False)
        y = chunk["rt"].to_numpy(dtype=np.float64, copy=False)

        x = _design_matrix_from_chunk(
            chunk=chunk, design_features=design_features, raw_features=raw_features_in_order
        )
        if feature_center is not None:
            x = x - feature_center[None, :]
        if feature_rotation is not None:
            x = x @ feature_rotation
        x1 = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)

        key = (species << np.int64(32)) + comp
        idx = np.searchsorted(keys, key)
        ok = (idx >= 0) & (idx < keys.size)
        if ok.any():
            ok[ok] &= keys[idx[ok]] == key[ok]

        # Seen groups: use stored posterior summaries.
        if ok.any():
            idx_ok = idx[ok]
            beta = beta_hat[idx_ok]
            pred_mean = beta[:, 0] + np.sum(beta[:, 1:] * x[ok], axis=1)

            sigma2_y = np.maximum(sigma2[idx_ok], 1e-12)
            if beta_cov is not None:
                var_coef = _row_variance_from_beta_cov(
                    x1=x1[ok], group_idx=idx_ok, beta_cov=np.asarray(beta_cov, dtype=np.float64)
                )
            else:
                var_coef = np.sum(np.square(x1[ok]) * beta_var_diag[idx_ok], axis=1)
            pred_var = np.maximum(sigma2_y + np.maximum(var_coef, 0.0), 0.0)
            pred_std = np.sqrt(pred_var) * pred_std_scale
            err = pred_mean - y[ok]
            covered = np.abs(err) <= (q_norm * pred_std)

            n_seen += int(err.size)
            stats_all.update(err=err, covered=covered)
            stats_seen.update(err=err, covered=covered)

        # Unseen groups: optional supercat-comp backoff, then comp-id backoff, then hierarchy prior.
        miss = ~ok
        if miss.any() and sc_comp_keys is not None and sc_comp_beta_hat is not None:
            super_m_all = supercat[miss]
            comp_m_all = comp[miss]
            key_sc = (super_m_all << np.int64(32)) + comp_m_all
            idx_sc = np.searchsorted(sc_comp_keys, key_sc)
            ok_sc = (idx_sc >= 0) & (idx_sc < sc_comp_keys.size)
            if ok_sc.any():
                ok_sc[ok_sc] &= sc_comp_keys[idx_sc[ok_sc]] == key_sc[ok_sc]
            if ok_sc.any():
                take = np.flatnonzero(miss)[ok_sc]
                beta_sc = sc_comp_beta_hat[idx_sc[ok_sc]]
                pred_mean = beta_sc[:, 0] + np.sum(beta_sc[:, 1:] * x[take], axis=1)

                slope_var = sigma2_backoff / float(lambda_slopes)
                var_coef = tau_b2 + slope_var * np.sum(np.square(x[take]), axis=1)
                pred_var = np.maximum(sigma2_backoff + np.maximum(var_coef, 0.0), 0.0)
                pred_std = np.sqrt(pred_var) * pred_std_scale
                err = pred_mean - y[take]
                covered = np.abs(err) <= (q_norm * pred_std)

                n_sc_comp += int(err.size)
                stats_all.update(err=err, covered=covered)
                stats_sc_comp.update(err=err, covered=covered)
                miss[take] = False

        if (
            miss.any()
            and comp_backoff_ids is not None
            and comp_backoff_beta_hat is not None
            and comp_backoff_mu_mean is not None
        ):
            comp_m_all = comp[miss]
            idx_c = np.searchsorted(comp_backoff_ids, comp_m_all)
            ok_c = (idx_c >= 0) & (idx_c < comp_backoff_ids.size)
            if ok_c.any():
                ok_c[ok_c] &= comp_backoff_ids[idx_c[ok_c]] == comp_m_all[ok_c]
            if ok_c.any():
                take = np.flatnonzero(miss)[ok_c]
                beta_c = comp_backoff_beta_hat[idx_c[ok_c]]
                mu_base = comp_backoff_mu_mean[idx_c[ok_c]]

                sp_take = species[take]
                super_take = supercat[take]
                idx_sp = np.searchsorted(cluster_ids, sp_take)
                ok_sp = (idx_sp >= 0) & (idx_sp < cluster_ids.size)
                if ok_sp.any():
                    ok_sp[ok_sp] &= cluster_ids[idx_sp[ok_sp]] == sp_take[ok_sp]

                mu_target = np.zeros((int(sp_take.size),), dtype=np.float64)
                if ok_sp.any():
                    mu_target[ok_sp] = mu_cluster[idx_sp[ok_sp]]
                miss_sp = ~ok_sp
                if miss_sp.any() and supercat_ids.size > 0:
                    idx_sc = np.searchsorted(supercat_ids, super_take[miss_sp])
                    ok_sc = (idx_sc >= 0) & (idx_sc < supercat_ids.size)
                    if ok_sc.any():
                        ok_sc[ok_sc] &= supercat_ids[idx_sc[ok_sc]] == super_take[miss_sp][ok_sc]
                    if ok_sc.any():
                        mu_target[np.flatnonzero(miss_sp)[ok_sc]] = mu_supercat[idx_sc[ok_sc]]

                beta0 = beta_c[:, 0] + (mu_target - mu_base)
                pred_mean = beta0 + np.sum(beta_c[:, 1:] * x[take], axis=1)

                slope_var = sigma2_backoff / float(lambda_slopes)
                var_coef = tau_b2 + slope_var * np.sum(np.square(x[take]), axis=1)
                pred_var = np.maximum(sigma2_backoff + np.maximum(var_coef, 0.0), 0.0)
                pred_std = np.sqrt(pred_var) * pred_std_scale
                err = pred_mean - y[take]
                covered = np.abs(err) <= (q_norm * pred_std)

                n_comp_backoff += int(err.size)
                stats_all.update(err=err, covered=covered)
                stats_comp_backoff.update(err=err, covered=covered)
                miss[take] = False

        if miss.any():
            sp_m = species[miss]
            comp_m = comp[miss]
            chem_m = chem[miss]
            super_m = supercat[miss]
            cls_m = cls[miss]

            idx_sp = np.searchsorted(cluster_ids, sp_m)
            ok_sp = (idx_sp >= 0) & (idx_sp < cluster_ids.size)
            if ok_sp.any():
                ok_sp[ok_sp] &= cluster_ids[idx_sp[ok_sp]] == sp_m[ok_sp]

            mu_m = np.zeros((int(sp_m.size),), dtype=np.float64)
            w_m = np.repeat(w_global[None, :], repeats=int(sp_m.size), axis=0).astype(
                np.float64, copy=False
            )
            if ok_sp.any():
                mu_m[ok_sp] = mu_cluster[idx_sp[ok_sp]]
                w_m[ok_sp] = w_cluster[idx_sp[ok_sp]]

            # Supercategory fallback for unseen species.
            miss_sp = ~ok_sp
            if miss_sp.any() and supercat_ids.size > 0:
                idx_sc = np.searchsorted(supercat_ids, super_m[miss_sp])
                ok_sc = (idx_sc >= 0) & (idx_sc < supercat_ids.size)
                if ok_sc.any():
                    ok_sc[ok_sc] &= supercat_ids[idx_sc[ok_sc]] == super_m[miss_sp][ok_sc]
                if ok_sc.any():
                    take = np.flatnonzero(miss_sp)[ok_sc]
                    mu_m[take] = mu_supercat[idx_sc[ok_sc]]
                    w_m[take] = w_supercat[idx_sc[ok_sc]]

            idx_comp = np.searchsorted(comp_ids, comp_m)
            ok_comp = (idx_comp >= 0) & (idx_comp < comp_ids.size)
            if ok_comp.any():
                ok_comp[ok_comp] &= comp_ids[idx_comp[ok_comp]] == comp_m[ok_comp]

            alpha_m = np.zeros((int(comp_m.size),), dtype=np.float64)
            alpha_set = np.zeros((int(comp_m.size),), dtype=bool)
            if ok_comp.any():
                alpha_m[ok_comp] = alpha_comp[idx_comp[ok_comp]]
            alpha_set[ok_comp] = True

            miss_comp = ~ok_comp
            if miss_comp.any():
                chem_m = chem[miss][miss_comp]
                idx_e = np.searchsorted(embed_pos, chem_m)
                ok_e = (idx_e >= 0) & (idx_e < embed_pos.size)
                if ok_e.any():
                    ok_e[ok_e] &= embed_pos[idx_e[ok_e]] == chem_m[ok_e]
                if ok_e.any():
                    z = embed_features[idx_e[ok_e]]
                    alpha_hat = (z - z_center[None, :]) @ theta_alpha
                    take = np.flatnonzero(miss_comp)[ok_e]
                    alpha_m[take] = alpha_hat
                    alpha_set[take] = True

            # compound_class fallback for any remaining unseen comp_ids.
            miss_alpha = ~alpha_set
            if miss_alpha.any() and class_ids.size > 0:
                idx_cls = np.searchsorted(class_ids, cls_m[miss_alpha])
                ok_cls = (idx_cls >= 0) & (idx_cls < class_ids.size)
                if ok_cls.any():
                    ok_cls[ok_cls] &= class_ids[idx_cls[ok_cls]] == cls_m[miss_alpha][ok_cls]
                if ok_cls.any():
                    take = np.flatnonzero(miss_alpha)[ok_cls]
                    alpha_m[take] = alpha_class[idx_cls[ok_cls]]
                    alpha_set[take] = True

            b_mean = t0 + mu_m + alpha_m
            pred_mean = b_mean + np.sum(w_m * x[miss], axis=1)

            slope_var = sigma2_backoff / float(lambda_slopes)
            var_coef = tau_b2 + slope_var * np.sum(np.square(x[miss]), axis=1)
            if tau_comp2 > 0.0:
                var_coef = var_coef + tau_comp2 * miss_comp.astype(np.float64, copy=False)
            pred_var = np.maximum(sigma2_backoff + np.maximum(var_coef, 0.0), 0.0)
            pred_std = np.sqrt(pred_var) * pred_std_scale
            err = pred_mean - y[miss]
            covered = np.abs(err) <= (q_norm * pred_std)

            n_backoff += int(err.size)
            stats_all.update(err=err, covered=covered)
            stats_backoff.update(err=err, covered=covered)

        if remaining is not None:
            remaining -= int(len(chunk))
        if log_every > 0 and (chunk_i == 1 or chunk_i % log_every == 0):
            dt = time.time() - t_start
            rate = (float(total_rows) / dt) if dt > 0 else float("nan")
            print(
                f"[eval] chunks={chunk_i} rows={total_rows:,} rate={rate:,.0f}/s "
                f"seen={n_seen:,} sc_comp={n_sc_comp:,} comp={n_comp_backoff:,} backoff={n_backoff:,}"
            )

    payload = {
        "coeff_npz": str(coeff_npz),
        "backoff_npz": str(backoff_npz),
        "test_csv": str(test_csv),
        "chem_embeddings_path": str(emb_path),
        "uses_supercat_comp_backoff": bool(
            sc_comp_keys is not None and sc_comp_beta_hat is not None
        ),
        "uses_comp_backoff": bool(
            comp_backoff_ids is not None and comp_backoff_beta_hat is not None
        ),
        "rows_total": int(total_rows),
        "rows_seen": int(n_seen),
        "rows_supercat_comp_backoff": int(n_sc_comp),
        "rows_comp_backoff": int(n_comp_backoff),
        "rows_backoff": int(n_backoff),
        "metrics_all": stats_all.to_metrics(),
        "metrics_seen": stats_seen.to_metrics(),
        "metrics_supercat_comp_backoff": stats_sc_comp.to_metrics(),
        "metrics_comp_backoff": stats_comp_backoff.to_metrics(),
        "metrics_backoff": stats_backoff.to_metrics(),
    }
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"[eval] Wrote {out_json}")


if __name__ == "__main__":
    main()
