"""
Stage-1 RT ridge artifacts and shared utilities.

The RT "stage-1" interface used throughout this repo is a compact, production-friendly
representation of per-(group_id, comp_id) linear regression coefficient posteriors:

  rt = b + x_run^T w + eps

where `x_run` are run covariates from a production RT CSV (IS*/RS*/optional ES_*).

We store the per-group posterior mean and covariance of beta=[b; w] in `Stage1CoeffSummaries`.
This artifact is produced by:
  - sklearn ridge (analytic / approximate Bayes),
  - PyMC ridge variants (collapsed-slope likelihood with hierarchical priors),

and is consumed by streaming evaluation and plotting scripts.

Notes on conventions:
  - `feature_names` excludes the intercept; beta_* arrays include intercept at column 0.
  - `group_keys` encodes (group_id, comp_id) as: (group_id << 32) + comp_id.
  - For historical reasons, the array holding `group_id` is named `species_cluster`. Use
    `group_col` to interpret it ("species_cluster" vs "species"). When `group_col="species"`,
    `supercat_id` may be present to provide the parent species_cluster id per group.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


def encode_group_comp_key(group_id: np.ndarray, comp_id: np.ndarray) -> np.ndarray:
    """Encode (group_id, comp_id) into a single int64 key: (group_id << 32) + comp_id."""
    group_i64 = np.asarray(group_id, dtype=np.int64)
    comp_i64 = np.asarray(comp_id, dtype=np.int64)
    return (group_i64 << np.int64(32)) + comp_i64


# Backward-compatible alias (used by older scripts/tests).
_encode_cluster_comp_key = encode_group_comp_key


def infer_feature_columns(columns: Sequence[str], *, es_candidates: Sequence[str]) -> List[str]:
    """
    Infer run covariate feature columns from a production RT CSV header.

    Always includes IS* (required), optionally includes RS*, and includes a caller-provided ES_*
    subset when present.
    """
    is_cols = sorted([str(c) for c in columns if str(c).startswith("IS")])
    rs_cols = sorted([str(c) for c in columns if str(c).startswith("RS")])
    cols_set = set(map(str, columns))
    es_cols = [str(c) for c in es_candidates if str(c) in cols_set]
    if not is_cols:
        raise ValueError("No IS* covariate columns detected in CSV")
    return [*is_cols, *rs_cols, *es_cols]


def infer_parent_ids(
    *,
    child_id: np.ndarray,
    parent_id: np.ndarray,
    child_name: str,
    parent_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Infer a deterministic parent id for each child id.

    Returns:
      child_ids_sorted: (K,) int64
      parent_ids: (K,) int64 aligned with child_ids_sorted
    """
    child_id = np.asarray(child_id, dtype=np.int64)
    parent_id = np.asarray(parent_id, dtype=np.int64)
    if child_id.shape != parent_id.shape:
        raise ValueError(
            f"child_id and parent_id must have the same shape; got {child_id.shape} vs {parent_id.shape}"
        )
    if child_id.ndim != 1:
        raise ValueError(f"child_id must be 1D; got shape {child_id.shape}")
    if parent_id.ndim != 1:
        raise ValueError(f"parent_id must be 1D; got shape {parent_id.shape}")

    n = int(child_id.size)
    if n == 0:
        empty = np.zeros((0,), dtype=np.int64)
        return empty, empty

    order = np.argsort(child_id, kind="mergesort")
    child_sorted = child_id[order]
    parent_sorted = parent_id[order]

    bounds = np.flatnonzero(np.diff(child_sorted)) + 1
    starts = np.concatenate([np.asarray([0], dtype=np.int64), bounds.astype(np.int64)])
    ends = np.concatenate([bounds.astype(np.int64), np.asarray([n], dtype=np.int64)])

    child_unique = child_sorted[starts]
    parent_unique = parent_sorted[starts].copy()
    for i, (start, end) in enumerate(zip(starts.tolist(), ends.tolist(), strict=True)):
        p0 = int(parent_sorted[start])
        if np.any(parent_sorted[start:end] != p0):
            vals = np.unique(parent_sorted[start:end]).astype(np.int64).tolist()
            raise ValueError(
                f"Found non-unique parent mapping: {child_name}={int(child_unique[i])} "
                f"maps to multiple {parent_name} values: {vals}"
            )
        parent_unique[i] = p0

    return child_unique.astype(np.int64, copy=False), parent_unique.astype(np.int64, copy=False)


# Backward-compatible alias.
_infer_parent_ids = infer_parent_ids


@dataclass(frozen=True)
class Stage1CoeffSummaries:
    """Per-group linear RT coefficient summaries (production-friendly)."""

    feature_names: Tuple[str, ...]  # (P,)
    group_keys: np.ndarray  # (G,) int64, sorted; (group_id<<32)+comp_id
    species_cluster: np.ndarray  # (G,) int64 (holds group_id; see group_col)
    comp_id: np.ndarray  # (G,) int64
    chem_id: np.ndarray  # (G,) int64
    compound_class: np.ndarray  # (G,) int64
    n_obs: np.ndarray  # (G,) int64
    beta_hat: np.ndarray  # (G, D) float; includes intercept as column 0
    beta_var_diag: np.ndarray  # (G, D) float
    sigma2_mean: np.ndarray  # (G,) float
    beta_cov: np.ndarray | None = None  # (G, D, D) optional full covariance (float32 preferred)
    feature_center: np.ndarray | None = None  # (P,) optional global mean subtracted from x
    feature_rotation: np.ndarray | None = None  # (P, P) optional orthonormal rotation matrix
    group_col: str = "species_cluster"
    supercat_id: np.ndarray | None = None  # (G,) optional parent supercategory id

    def save_npz(self, path: Path) -> None:
        """Save to a compressed .npz file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "feature_names": np.asarray(self.feature_names, dtype=object),
            "group_keys": np.asarray(self.group_keys, dtype=np.int64),
            "species_cluster": np.asarray(self.species_cluster, dtype=np.int64),
            "comp_id": np.asarray(self.comp_id, dtype=np.int64),
            "chem_id": np.asarray(self.chem_id, dtype=np.int64),
            "compound_class": np.asarray(self.compound_class, dtype=np.int64),
            "n_obs": np.asarray(self.n_obs, dtype=np.int64),
            "beta_hat": np.asarray(self.beta_hat, dtype=np.float32),
            "beta_var_diag": np.asarray(self.beta_var_diag, dtype=np.float32),
            "sigma2_mean": np.asarray(self.sigma2_mean, dtype=np.float32),
        }
        payload["group_col"] = np.asarray(str(self.group_col), dtype=object)
        if self.supercat_id is not None:
            payload["supercat_id"] = np.asarray(self.supercat_id, dtype=np.int64)
        if self.beta_cov is not None:
            payload["beta_cov"] = np.asarray(self.beta_cov, dtype=np.float32)
        if self.feature_center is not None:
            payload["feature_center"] = np.asarray(self.feature_center, dtype=np.float32)
        if self.feature_rotation is not None:
            payload["feature_rotation"] = np.asarray(self.feature_rotation, dtype=np.float32)
        np.savez_compressed(path, **payload)

    @staticmethod
    def load_npz(path: Path) -> "Stage1CoeffSummaries":
        """Load from a .npz file written by `save_npz`."""
        npz = np.load(path, allow_pickle=True)
        feature_names = tuple(str(s) for s in npz["feature_names"].tolist())
        group_col = "species_cluster"
        if "group_col" in npz.files:
            group_col = str(npz["group_col"].tolist())
        supercat_id = None
        if "supercat_id" in npz.files:
            supercat_id = np.asarray(npz["supercat_id"], dtype=np.int64)
        feature_center = None
        if "feature_center" in npz.files:
            feature_center = np.asarray(npz["feature_center"], dtype=float)
        feature_rotation = None
        if "feature_rotation" in npz.files:
            feature_rotation = np.asarray(npz["feature_rotation"], dtype=float)
        beta_cov = None
        if "beta_cov" in npz.files:
            beta_cov = np.asarray(npz["beta_cov"], dtype=np.float32)
        return Stage1CoeffSummaries(
            feature_names=feature_names,
            feature_center=feature_center,
            feature_rotation=feature_rotation,
            group_keys=np.asarray(npz["group_keys"], dtype=np.int64),
            species_cluster=np.asarray(npz["species_cluster"], dtype=np.int64),
            comp_id=np.asarray(npz["comp_id"], dtype=np.int64),
            chem_id=np.asarray(npz["chem_id"], dtype=np.int64),
            compound_class=np.asarray(npz["compound_class"], dtype=np.int64),
            n_obs=np.asarray(npz["n_obs"], dtype=np.int64),
            beta_hat=np.asarray(npz["beta_hat"], dtype=float),
            beta_var_diag=np.asarray(npz["beta_var_diag"], dtype=float),
            beta_cov=beta_cov,
            sigma2_mean=np.asarray(npz["sigma2_mean"], dtype=float),
            group_col=group_col,
            supercat_id=supercat_id,
        )


@dataclass(frozen=True)
class ChemHierBackoffSummaries:
    """Additional parameters needed to back off for unseen groups/compounds."""

    cluster_ids: np.ndarray  # (C,) int64, sorted unique species_cluster ids
    mu_cluster: np.ndarray  # (C,) float, cluster offsets (mean-zero)
    chem_ids: np.ndarray  # (K,) int64, sorted unique chem ids
    t_chem: np.ndarray  # (K,) float, per-chemical baseline RT
    t0: float  # global baseline RT (shared across all clusters and chemicals)
    theta: np.ndarray  # (D,) float, chemistry regression weights
    tau_mu: float
    tau_t: float
    tau_b: float
    sigma2: float
    lambda_slopes: float
    z_center: np.ndarray | None = None  # (D,) optional mean subtracted from embeddings
    embeddings_path: str | None = None
    class_ids: np.ndarray | None = None  # (J,) sorted unique class ids
    alpha_class: np.ndarray | None = None  # (J,) class offsets
    tau_class: float | None = None
    slope_head_mode: str | None = None
    feature_names: Tuple[str, ...] | None = None
    w0: np.ndarray | None = None
    w_cluster: np.ndarray | None = None
    tau_w: float | None = None

    def save_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "cluster_ids": np.asarray(self.cluster_ids, dtype=np.int64),
            "mu_cluster": np.asarray(self.mu_cluster, dtype=np.float32),
            "chem_ids": np.asarray(self.chem_ids, dtype=np.int64),
            "t_chem": np.asarray(self.t_chem, dtype=np.float32),
            "t0": np.asarray(float(self.t0), dtype=np.float32),
            "theta": np.asarray(self.theta, dtype=np.float32),
            "tau_mu": np.asarray(float(self.tau_mu), dtype=np.float32),
            "tau_t": np.asarray(float(self.tau_t), dtype=np.float32),
            "tau_b": np.asarray(float(self.tau_b), dtype=np.float32),
            "sigma2": np.asarray(float(self.sigma2), dtype=np.float32),
            "lambda_slopes": np.asarray(float(self.lambda_slopes), dtype=np.float32),
        }
        if self.slope_head_mode is not None:
            payload["slope_head_mode"] = np.asarray(str(self.slope_head_mode), dtype=object)
        if self.feature_names is not None:
            payload["feature_names"] = np.asarray(self.feature_names, dtype=object)
        if self.w0 is not None:
            payload["w0"] = np.asarray(self.w0, dtype=np.float32)
        if self.w_cluster is not None:
            payload["w_cluster"] = np.asarray(self.w_cluster, dtype=np.float32)
        if self.tau_w is not None:
            payload["tau_w"] = np.asarray(float(self.tau_w), dtype=np.float32)
        if self.z_center is not None:
            payload["z_center"] = np.asarray(self.z_center, dtype=np.float32)
        if self.embeddings_path is not None:
            payload["embeddings_path"] = np.asarray(str(self.embeddings_path), dtype=object)
        if self.class_ids is not None:
            payload["class_ids"] = np.asarray(self.class_ids, dtype=np.int64)
        if self.alpha_class is not None:
            payload["alpha_class"] = np.asarray(self.alpha_class, dtype=np.float32)
        if self.tau_class is not None:
            payload["tau_class"] = np.asarray(float(self.tau_class), dtype=np.float32)
        np.savez_compressed(path, **payload)

    @staticmethod
    def load_npz(path: Path) -> "ChemHierBackoffSummaries":
        npz = np.load(path, allow_pickle=True)
        slope_head_mode = None
        if "slope_head_mode" in npz.files:
            slope_head_mode = str(npz["slope_head_mode"].tolist())
        feature_names = None
        if "feature_names" in npz.files:
            feature_names = tuple(str(s) for s in npz["feature_names"].tolist())
        w0 = None
        if "w0" in npz.files:
            w0 = np.asarray(npz["w0"], dtype=float)
        w_cluster = None
        if "w_cluster" in npz.files:
            w_cluster = np.asarray(npz["w_cluster"], dtype=float)
        tau_w = None
        if "tau_w" in npz.files:
            tau_w = float(npz["tau_w"])
        z_center = None
        if "z_center" in npz.files:
            z_center = np.asarray(npz["z_center"], dtype=float)
        embeddings_path = None
        if "embeddings_path" in npz.files:
            embeddings_path = str(npz["embeddings_path"].tolist())
        class_ids = None
        if "class_ids" in npz.files:
            class_ids = np.asarray(npz["class_ids"], dtype=np.int64)
        alpha_class = None
        if "alpha_class" in npz.files:
            alpha_class = np.asarray(npz["alpha_class"], dtype=float)
        tau_class = None
        if "tau_class" in npz.files:
            tau_class = float(npz["tau_class"])
        return ChemHierBackoffSummaries(
            slope_head_mode=slope_head_mode,
            feature_names=feature_names,
            w0=w0,
            w_cluster=w_cluster,
            tau_w=tau_w,
            cluster_ids=np.asarray(npz["cluster_ids"], dtype=np.int64),
            mu_cluster=np.asarray(npz["mu_cluster"], dtype=float),
            chem_ids=np.asarray(npz["chem_ids"], dtype=np.int64),
            t_chem=np.asarray(npz["t_chem"], dtype=float),
            t0=float(npz["t0"]) if "t0" in npz.files else 0.0,
            theta=np.asarray(npz["theta"], dtype=float),
            z_center=z_center,
            tau_mu=float(npz["tau_mu"]),
            tau_t=float(npz["tau_t"]),
            tau_b=float(npz["tau_b"]),
            sigma2=float(npz["sigma2"]),
            lambda_slopes=float(npz["lambda_slopes"]),
            embeddings_path=embeddings_path,
            class_ids=class_ids,
            alpha_class=alpha_class,
            tau_class=tau_class,
        )


@dataclass(frozen=True)
class PartialPoolBackoffSummaries:
    """Backoff parameters for the partial-pooling ridge model (unseen species/compounds).

    This artifact stores the *hierarchy-level* posterior means needed to construct reasonable
    coefficients for groups that are not present in `Stage1CoeffSummaries.group_keys`, e.g.:
      - unseen (species, comp_id) pairs,
      - a new species nested in a known species_cluster,
      - an unseen comp_id with a known compound_class.

    It is intentionally small and does not store per-group sufficient stats.
    """

    feature_names: Tuple[str, ...]
    cluster_ids: np.ndarray  # (C,) int64 unique species ids (sorted)
    cluster_supercat_id: np.ndarray  # (C,) int64 parent species_cluster id per species
    comp_ids: np.ndarray  # (M,) int64 unique comp_id values (sorted)
    comp_chem_id: np.ndarray  # (M,) int64 chem_id per comp_id (or -1)
    comp_class: np.ndarray  # (M,) int64 compound_class id per comp_id (or -1)
    t0: float  # global intercept baseline
    mu_cluster: np.ndarray  # (C,) float species offsets (mean-zero)
    alpha_comp: np.ndarray  # (M,) float compound offsets (mean-zero)
    w_cluster: np.ndarray  # (C, P) float slope heads per species
    tau_b: float  # prior sd for per-(species, comp_id) intercepts
    sigma2: float  # noise variance (shared)
    lambda_slopes: float  # ridge precision for slopes (shared scalar)
    # Single-model chemistry regression for alpha_comp backoff (unseen compounds).
    alpha_z_center: np.ndarray  # (D,) float; mean chem embedding used for centering
    alpha_theta: np.ndarray  # (D,) float; embedding->alpha linear weights
    tau_comp: float  # sd for per-compound residual offsets (delta_comp)

    def save_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "feature_names": np.asarray(self.feature_names, dtype=object),
            "cluster_ids": np.asarray(self.cluster_ids, dtype=np.int64),
            "cluster_supercat_id": np.asarray(self.cluster_supercat_id, dtype=np.int64),
            "comp_ids": np.asarray(self.comp_ids, dtype=np.int64),
            "comp_chem_id": np.asarray(self.comp_chem_id, dtype=np.int64),
            "comp_class": np.asarray(self.comp_class, dtype=np.int64),
            "tau_comp": np.asarray(float(self.tau_comp), dtype=np.float32),
            "t0": np.asarray(float(self.t0), dtype=np.float32),
            "mu_cluster": np.asarray(self.mu_cluster, dtype=np.float32),
            "alpha_comp": np.asarray(self.alpha_comp, dtype=np.float32),
            "w_cluster": np.asarray(self.w_cluster, dtype=np.float32),
            "tau_b": np.asarray(float(self.tau_b), dtype=np.float32),
            "sigma2": np.asarray(float(self.sigma2), dtype=np.float32),
            "lambda_slopes": np.asarray(float(self.lambda_slopes), dtype=np.float32),
            "alpha_z_center": np.asarray(self.alpha_z_center, dtype=np.float32),
            "alpha_theta": np.asarray(self.alpha_theta, dtype=np.float32),
        }
        np.savez_compressed(path, **payload)

    @staticmethod
    def load_npz(path: Path) -> "PartialPoolBackoffSummaries":
        npz = np.load(path, allow_pickle=True)
        feature_names = tuple(str(s) for s in npz["feature_names"].tolist())
        if "alpha_z_center" not in npz.files or "alpha_theta" not in npz.files:
            raise ValueError(
                "Backoff artifact missing chem-linear parameters (alpha_z_center/alpha_theta). "
                "Re-train the partial pooling model with chem-linear enabled."
            )
        comp_ids = np.asarray(npz["comp_ids"], dtype=np.int64)
        comp_chem_id = np.asarray(npz["comp_chem_id"], dtype=np.int64)
        if comp_chem_id.shape != comp_ids.shape:
            comp_chem_id = np.full(comp_ids.shape, -1, dtype=np.int64)
        tau_comp_val = float(npz["tau_comp"]) if "tau_comp" in npz.files else float("nan")
        if not np.isfinite(tau_comp_val):
            raise ValueError("Backoff artifact missing tau_comp for chem-linear alpha prior.")
        return PartialPoolBackoffSummaries(
            feature_names=feature_names,
            cluster_ids=np.asarray(npz["cluster_ids"], dtype=np.int64),
            cluster_supercat_id=np.asarray(npz["cluster_supercat_id"], dtype=np.int64),
            comp_ids=comp_ids,
            comp_chem_id=comp_chem_id,
            comp_class=np.asarray(npz["comp_class"], dtype=np.int64),
            alpha_z_center=np.asarray(npz["alpha_z_center"], dtype=float),
            alpha_theta=np.asarray(npz["alpha_theta"], dtype=float),
            tau_comp=float(tau_comp_val),
            t0=float(npz["t0"]),
            mu_cluster=np.asarray(npz["mu_cluster"], dtype=float),
            alpha_comp=np.asarray(npz["alpha_comp"], dtype=float),
            w_cluster=np.asarray(npz["w_cluster"], dtype=float),
            tau_b=float(npz["tau_b"]),
            sigma2=float(npz["sigma2"]),
            lambda_slopes=float(npz["lambda_slopes"]),
        )


def apply_global_feature_transform(
    x_arr: np.ndarray, *, center_mode: str = "global", rotation_mode: str = "none"
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray]:
    """Apply an optional global centering / orthonormal rotation to covariates."""
    x_arr = np.asarray(x_arr, dtype=np.float32)
    p = int(x_arr.shape[1])

    feature_center: np.ndarray | None = None
    if center_mode == "global":
        feature_center = np.mean(x_arr.astype(np.float64), axis=0)
        x_arr = x_arr - feature_center.astype(np.float32)[None, :]
    elif center_mode != "none":
        raise ValueError(f"Unknown center_mode: {center_mode}")

    feature_rotation: np.ndarray | None = None
    if rotation_mode == "pca":
        if feature_center is None:
            raise ValueError("rotation_mode='pca' requires center_mode='global'")
        cov = (x_arr.T @ x_arr).astype(np.float64, copy=False)
        cov = cov / float(max(int(x_arr.shape[0]) - 1, 1))
        cov = 0.5 * (cov + cov.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        feature_rotation = eigvecs[:, order].astype(np.float64, copy=False)
    elif rotation_mode != "none":
        raise ValueError(f"Unknown rotation_mode: {rotation_mode}")

    if feature_rotation is not None and feature_rotation.shape != (p, p):
        raise ValueError(f"feature_rotation has shape {feature_rotation.shape}, expected {(p, p)}")

    return feature_center, feature_rotation, x_arr


def build_index_mapping(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (unique_sorted_values, index_per_value) for an int64 array."""
    values_i64 = np.asarray(values, dtype=np.int64)
    uniq = np.unique(values_i64)
    idx = np.searchsorted(uniq, values_i64)
    return uniq, idx.astype(np.int64, copy=False)


def compute_group_suffstats(
    *,
    key_arr: np.ndarray,
    group_id_arr: np.ndarray,
    comp_id_arr: np.ndarray,
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    feature_rotation: np.ndarray | None,
    max_groups: int = 0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Compute centered sufficient statistics per (group_id, comp_id)."""
    sort_idx = np.argsort(key_arr, kind="mergesort")
    keys_sorted = np.asarray(key_arr, dtype=np.int64)[sort_idx]
    bounds = np.flatnonzero(np.diff(keys_sorted)) + 1
    starts = np.concatenate([np.array([0], dtype=np.int64), bounds.astype(np.int64)])
    ends = np.concatenate([bounds.astype(np.int64), np.array([keys_sorted.size], dtype=np.int64)])

    group_keys: List[int] = []
    group_ids: List[int] = []
    comp_ids: List[int] = []
    n_obs: List[int] = []
    x_mean_list: List[np.ndarray] = []
    y_mean_list: List[float] = []
    xtx_list: List[np.ndarray] = []
    xty_list: List[np.ndarray] = []
    yty_list: List[float] = []

    n_groups_seen = 0
    for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
        if max_groups > 0 and n_groups_seen >= max_groups:
            break
        idx = sort_idx[start:end]
        key = int(keys_sorted[start])
        group0 = int(group_id_arr[idx[0]])
        comp0 = int(comp_id_arr[idx[0]])

        xg = x_arr[idx].astype(np.float64, copy=False)
        if feature_rotation is not None:
            xg = xg @ feature_rotation
        yg = y_arr[idx].astype(np.float64, copy=False)

        xm = xg.mean(axis=0)
        ym = float(yg.mean())
        xc = xg - xm[None, :]
        yc = yg - ym

        xtx = xc.T @ xc
        xty = xc.T @ yc
        yty = float(yc.T @ yc)

        group_keys.append(key)
        group_ids.append(group0)
        comp_ids.append(comp0)
        n_obs.append(int(yg.size))
        x_mean_list.append(xm.astype(np.float64, copy=False))
        y_mean_list.append(float(ym))
        xtx_list.append(xtx.astype(np.float64, copy=False))
        xty_list.append(xty.astype(np.float64, copy=False))
        yty_list.append(float(yty))
        n_groups_seen += 1

    keys_arr = np.asarray(group_keys, dtype=np.int64)
    group_ids_arr = np.asarray(group_ids, dtype=np.int64)
    comp_ids_arr = np.asarray(comp_ids, dtype=np.int64)
    n_arr = np.asarray(n_obs, dtype=np.int64)
    x_mean_arr = np.stack(x_mean_list, axis=0).astype(np.float64, copy=False)
    y_mean_arr = np.asarray(y_mean_list, dtype=np.float64)
    xtx_arr = np.stack(xtx_list, axis=0).astype(np.float64, copy=False)
    xty_arr = np.stack(xty_list, axis=0).astype(np.float64, copy=False)
    yty_arr = np.asarray(yty_list, dtype=np.float64)

    return (
        keys_arr,
        group_ids_arr,
        comp_ids_arr,
        n_arr,
        x_mean_arr,
        y_mean_arr,
        xtx_arr,
        xty_arr,
        yty_arr,
    )


def resolve_comp_id_mapping(
    *, comp_id: np.ndarray, chem_id: np.ndarray, compound_class: np.ndarray
) -> tuple[Dict[int, Tuple[int, int]], List[dict]]:
    """Resolve a deterministic (chem_id, compound_class) choice per comp_id (mode; tie->smallest)."""
    comp_id = np.asarray(comp_id, dtype=np.int64)
    chem_id = np.asarray(chem_id, dtype=np.int64)
    compound_class = np.asarray(compound_class, dtype=np.int64)

    comp_id_to_mapping_counts: Dict[int, Counter[Tuple[int, int]]] = defaultdict(Counter)
    mapping_mat = np.stack([comp_id, chem_id, compound_class], axis=1)
    uniq_map, map_counts = np.unique(mapping_mat, axis=0, return_counts=True)
    for row, cnt in zip(uniq_map.tolist(), map_counts.tolist(), strict=True):
        cid, chem, cls = (int(row[0]), int(row[1]), int(row[2]))
        comp_id_to_mapping_counts[cid][(chem, cls)] += int(cnt)

    comp_id_to_choice: Dict[int, Tuple[int, int]] = {}
    collision_rows: List[dict] = []
    for cid, counter in comp_id_to_mapping_counts.items():
        if not counter:
            continue
        (chem_cls, mode_count) = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0]
        comp_id_to_choice[int(cid)] = (int(chem_cls[0]), int(chem_cls[1]))
        if len(counter) > 1:
            for (chem0, cls0), cnt in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
                collision_rows.append(
                    {
                        "comp_id": int(cid),
                        "chem_id": int(chem0),
                        "compound_class": int(cls0),
                        "count": int(cnt),
                        "mode_chem_id": int(chem_cls[0]),
                        "mode_compound_class": int(chem_cls[1]),
                        "mode_count": int(mode_count),
                        "n_unique_mappings": int(len(counter)),
                    }
                )
    return comp_id_to_choice, collision_rows


def assign_group_metadata(
    *, comp_ids: np.ndarray, comp_id_to_choice: Dict[int, Tuple[int, int]]
) -> tuple[np.ndarray, np.ndarray]:
    """Attach (chem_id, compound_class) to each comp_id group using the resolved mapping."""
    comp_ids = np.asarray(comp_ids, dtype=np.int64)
    chem_ids = np.full(comp_ids.shape, -1, dtype=np.int64)
    class_ids = np.full(comp_ids.shape, -1, dtype=np.int64)
    for i, cid in enumerate(comp_ids.tolist()):
        chem_cls = comp_id_to_choice.get(int(cid))
        if chem_cls is None:
            continue
        chem_ids[i] = int(chem_cls[0])
        class_ids[i] = int(chem_cls[1])
    return chem_ids, class_ids


def compute_group_posterior_summaries(
    *,
    xtx_arr: np.ndarray,
    xty_arr: np.ndarray,
    x_mean_arr: np.ndarray,
    y_mean_arr: np.ndarray,
    n_arr: np.ndarray,
    lambda_diag: np.ndarray,
    slope_mean: np.ndarray | None = None,
    sigma2_mean: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Posterior mean/cov for beta=[b; w] with a flat intercept prior and ridge slopes."""
    n_groups = int(xtx_arr.shape[0])
    p = int(xtx_arr.shape[1])
    eye = np.eye(p, dtype=np.float64)

    lambda_diag = np.asarray(lambda_diag, dtype=np.float64)
    if lambda_diag.shape != (p,):
        raise ValueError(f"lambda_diag must have shape ({p},); got {lambda_diag.shape}")
    if not np.all(np.isfinite(lambda_diag)) or not np.all(lambda_diag > 0):
        raise ValueError("lambda_diag entries must be finite and > 0")
    penalty = np.diag(lambda_diag)

    if slope_mean is not None:
        slope_mean = np.asarray(slope_mean, dtype=np.float64)
        if slope_mean.shape != (n_groups, p):
            raise ValueError(
                f"slope_mean must have shape ({n_groups}, {p}); got {slope_mean.shape}"
            )

    beta_hat = np.empty((n_groups, p + 1), dtype=np.float64)
    beta_var_diag = np.empty((n_groups, p + 1), dtype=np.float64)
    beta_cov = np.empty((n_groups, p + 1, p + 1), dtype=np.float64)

    for i in range(n_groups):
        a = xtx_arr[i] + penalty + 1e-12 * eye
        xty = xty_arr[i]
        slope_mean_i = np.zeros((p,), dtype=np.float64)
        if slope_mean is not None:
            slope_mean_i = np.asarray(slope_mean[i], dtype=np.float64)
            xty = xty - (xtx_arr[i] @ slope_mean_i)
        try:
            chol = np.linalg.cholesky(a)
            u = np.linalg.solve(chol.T, np.linalg.solve(chol, xty))
            inv_chol = np.linalg.solve(chol, eye)
            a_inv = inv_chol.T @ inv_chol
        except np.linalg.LinAlgError:
            a_inv = np.linalg.inv(a)
            u = a_inv @ xty

        w = u + slope_mean_i
        b = float(y_mean_arr[i] - float(x_mean_arr[i].dot(w)))
        beta_hat[i, 0] = b
        beta_hat[i, 1:] = w

        slopes_cov = float(sigma2_mean) * a_inv
        cov_bw = -float(sigma2_mean) * (a_inv @ x_mean_arr[i])
        intercept_var = float(sigma2_mean) * (
            1.0 / float(max(int(n_arr[i]), 1)) + float(x_mean_arr[i].T @ a_inv @ x_mean_arr[i])
        )

        cov_i = np.empty((p + 1, p + 1), dtype=np.float64)
        cov_i[0, 0] = intercept_var
        cov_i[0, 1:] = cov_bw
        cov_i[1:, 0] = cov_bw
        cov_i[1:, 1:] = slopes_cov

        beta_cov[i] = cov_i
        beta_var_diag[i] = np.diagonal(cov_i)

    return beta_hat, beta_var_diag, beta_cov


def compute_group_posterior_summaries_with_b_prior_mean(
    *,
    xtx_arr: np.ndarray,
    xty_arr: np.ndarray,
    x_mean_arr: np.ndarray,
    y_mean_arr: np.ndarray,
    n_arr: np.ndarray,
    b_prior_mean: np.ndarray,
    tau_b: float,
    slope_mean: np.ndarray | None = None,
    lambda_diag: np.ndarray,
    sigma2_mean: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Posterior mean/cov for beta=[b; w] with a Normal prior on b (and optional slope mean)."""
    if float(tau_b) <= 0:
        raise ValueError("tau_b must be > 0")
    if float(sigma2_mean) <= 0:
        raise ValueError("sigma2_mean must be > 0")

    n_groups = int(xtx_arr.shape[0])
    p = int(xtx_arr.shape[1])
    eye = np.eye(p, dtype=np.float64)

    lambda_diag = np.asarray(lambda_diag, dtype=np.float64)
    if lambda_diag.shape != (p,):
        raise ValueError(f"lambda_diag must have shape ({p},); got {lambda_diag.shape}")
    if not np.all(np.isfinite(lambda_diag)) or not np.all(lambda_diag > 0):
        raise ValueError("lambda_diag entries must be finite and > 0")
    penalty = np.diag(lambda_diag)

    b_prior_mean = np.asarray(b_prior_mean, dtype=np.float64)
    if b_prior_mean.shape != (n_groups,):
        raise ValueError(f"b_prior_mean must have shape ({n_groups},); got {b_prior_mean.shape}")

    if slope_mean is not None:
        slope_mean = np.asarray(slope_mean, dtype=np.float64)
        if slope_mean.shape != (n_groups, p):
            raise ValueError(
                f"slope_mean must have shape ({n_groups}, {p}); got {slope_mean.shape}"
            )

    beta_hat = np.empty((n_groups, p + 1), dtype=np.float64)
    beta_var_diag = np.empty((n_groups, p + 1), dtype=np.float64)
    beta_cov = np.empty((n_groups, p + 1, p + 1), dtype=np.float64)

    tau_b2 = float(tau_b) ** 2
    sigma2 = float(sigma2_mean)

    for i in range(n_groups):
        n = float(max(int(n_arr[i]), 0))
        xm = np.asarray(x_mean_arr[i], dtype=np.float64)
        ym = float(y_mean_arr[i])
        xtx_c = np.asarray(xtx_arr[i], dtype=np.float64)
        xty_c = np.asarray(xty_arr[i], dtype=np.float64)

        sx = n * xm
        sy = n * ym
        xtx_unc = xtx_c + n * np.outer(xm, xm)
        sxy_unc = xty_c + xm * sy

        slope_mean_i = np.zeros((p,), dtype=np.float64)
        if slope_mean is not None:
            slope_mean_i = np.asarray(slope_mean[i], dtype=np.float64)
            sy = sy - float(sx @ slope_mean_i)
            sxy_unc = sxy_unc - (xtx_unc @ slope_mean_i)

        a = xtx_unc + penalty + 1e-12 * eye

        prior_mean_b = float(b_prior_mean[i])

        p_scaled = np.empty((p + 1, p + 1), dtype=np.float64)
        p_scaled[0, 0] = n + sigma2 / tau_b2
        p_scaled[0, 1:] = sx
        p_scaled[1:, 0] = sx
        p_scaled[1:, 1:] = a

        eta_scaled = np.empty((p + 1,), dtype=np.float64)
        eta_scaled[0] = sy + sigma2 * prior_mean_b / tau_b2
        eta_scaled[1:] = sxy_unc

        try:
            chol = np.linalg.cholesky(p_scaled)
            inv_chol = np.linalg.solve(chol, np.eye(p + 1, dtype=np.float64))
            p_scaled_inv = inv_chol.T @ inv_chol
            beta_mean = p_scaled_inv @ eta_scaled
        except np.linalg.LinAlgError:
            p_scaled_inv = np.linalg.inv(p_scaled)
            beta_mean = p_scaled_inv @ eta_scaled

        cov = sigma2 * p_scaled_inv

        beta_hat[i] = beta_mean
        if slope_mean is not None:
            beta_hat[i, 1:] = beta_hat[i, 1:] + slope_mean_i
        beta_cov[i] = cov
        beta_var_diag[i] = np.diagonal(cov)

    return beta_hat, beta_var_diag, beta_cov


def compute_group_posterior_summaries_with_b_prior(
    *,
    xtx_arr: np.ndarray,
    xty_arr: np.ndarray,
    x_mean_arr: np.ndarray,
    y_mean_arr: np.ndarray,
    n_arr: np.ndarray,
    group_cluster_idx: np.ndarray,
    group_chem_idx: np.ndarray,
    mu_cluster: np.ndarray,
    t_chem: np.ndarray,
    slope_mean_cluster: np.ndarray | None = None,
    tau_b: float,
    lambda_diag: np.ndarray,
    sigma2_mean: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convenience wrapper where b prior mean is indexed by (cluster, chem)."""
    group_cluster_idx = np.asarray(group_cluster_idx, dtype=np.int64)
    group_chem_idx = np.asarray(group_chem_idx, dtype=np.int64)
    mu_cluster = np.asarray(mu_cluster, dtype=np.float64)
    t_chem = np.asarray(t_chem, dtype=np.float64)

    b_prior_mean = mu_cluster[group_cluster_idx] + t_chem[group_chem_idx]

    slope_mean = None
    if slope_mean_cluster is not None:
        slope_mean_cluster = np.asarray(slope_mean_cluster, dtype=np.float64)
        slope_mean = slope_mean_cluster[group_cluster_idx]

    return compute_group_posterior_summaries_with_b_prior_mean(
        xtx_arr=xtx_arr,
        xty_arr=xty_arr,
        x_mean_arr=x_mean_arr,
        y_mean_arr=y_mean_arr,
        n_arr=n_arr,
        b_prior_mean=b_prior_mean,
        tau_b=tau_b,
        slope_mean=slope_mean,
        lambda_diag=lambda_diag,
        sigma2_mean=sigma2_mean,
    )


# Backward-compatible aliases used by older tests.
_compute_group_posterior_summaries = compute_group_posterior_summaries
_compute_group_posterior_summaries_with_b_prior_mean = (
    compute_group_posterior_summaries_with_b_prior_mean
)
_compute_group_posterior_summaries_with_b_prior = compute_group_posterior_summaries_with_b_prior
