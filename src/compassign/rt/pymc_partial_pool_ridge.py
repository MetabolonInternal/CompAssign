"""
PyMC ridge RT model: partial pooling (species nested in supercategory).

This is the recommended RT ridge model for peak assignment because it:
  - reduces RMSE vs supercategory-only ridge in the long tail,
  - maintains calibration (interval coverage closer to nominal),
  - shrinks sparse (species, comp_id) groups toward sensible supercategory-aware priors.

Groups and data
---------------
We fit one linear model per group:

  g = (species, comp_id)

with run covariates x_run (IS*/RS*/optional ES_*):

  rt = b_g + x_run^T w_g + eps,      eps ~ Normal(0, sigma2)

Collapsed slopes (key speed trick)
---------------------------------
We integrate out the per-group slopes w_g analytically in the likelihood using per-group
sufficient statistics. This keeps inference tractable even when the number of groups is large.

Hierarchical priors (partial pooling)
------------------------------------
Intercept hierarchy (supercategory-aware):
  t0 ~ Normal(0, t0_prior_sigma)
  mu_supercat ~ Normal(0, tau_mu_supercat)
  mu_species ~ Normal(mu_supercat[supercat(species)], tau_mu[supercat(species)])
  theta_alpha ~ Normal(0, theta_alpha_prior_sigma)
  delta_comp ~ Normal(0, tau_comp)
  alpha_comp[comp] = zc_comp^T theta_alpha + delta_comp[comp]
  b_g ~ Normal(t0 + mu_species[species(g)] + alpha_comp[comp(g)], tau_b)

Slope head (supercategory-aware):
  w0 ~ Normal(0, w0_prior_sigma)
  w_supercat ~ Normal(w0, tau_w_supercat)
  w_species ~ Normal(w_supercat[supercat(species)], tau_w[supercat(species)])

and w_g has a ridge prior around the species head:
  w_g ~ Normal(w_species[species(g)], sigma2 * Lambda^{-1})

After fitting hyperparameters in PyMC (ADVI/MAP), we compute per-group posterior mean/covariance
for beta=[b; w] in NumPy and write a `Stage1CoeffSummaries` artifact.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import logging
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .ridge_stage1 import (
    PartialPoolBackoffSummaries,
    Stage1CoeffSummaries,
    apply_global_feature_transform,
    assign_group_metadata,
    build_index_mapping,
    compute_group_posterior_summaries_with_b_prior_mean,
    compute_group_suffstats,
    encode_group_comp_key,
    infer_feature_columns,
    infer_parent_ids,
    resolve_comp_id_mapping,
)
from ..utils.data_features import load_chemberta_pca20

logger = logging.getLogger(__name__)

PymcMethod = Literal["advi", "map"]

REQUIRED_COLS: tuple[str, ...] = (
    "rt",
    "compound",
    "comp_id",
    "compound_class",
    "species_cluster",
    "species",
)


@dataclass(frozen=True)
class PartialPoolRidgeFitResult:
    sigma2_mean: float
    trace_idata: Any | None
    # Intercept hierarchy means.
    t0_mean: float
    mu_species_mean: np.ndarray  # (C,) (uncentered; will be centered post-fit)
    alpha_comp_mean: np.ndarray  # (M,) (uncentered; will be centered post-fit)
    tau_comp_mean: float
    tau_b_mean: float
    # Slope head means.
    w_species_mean: np.ndarray  # (C, P)
    theta_alpha_mean: np.ndarray  # (D,) chemistry regression weights


@dataclass(frozen=True)
class PartialPoolRidgeTrainArtifacts:
    coeff_summaries: Stage1CoeffSummaries
    backoff_summaries: PartialPoolBackoffSummaries
    trained_at: str
    seed: int
    data_csv: str
    max_train_rows: int
    include_es_all: bool
    feature_center: str
    lambda_slopes: float
    sigma2_mean: float
    sigma_y_prior: float
    tau_mu_prior: float
    tau_comp_prior: float
    chem_embeddings_path: str
    theta_alpha_prior_sigma: float
    tau_b_prior: float
    tau_w_prior: float
    w0_prior_sigma: float
    t0_prior_sigma: float
    method: PymcMethod
    advi_steps: int | None
    advi_draws: int | None
    map_maxeval: int | None
    trace_idata: Any | None
    mapping_collision_rows: list[dict]

    @property
    def n_groups(self) -> int:
        return int(self.coeff_summaries.group_keys.size)

    @property
    def n_obs_train(self) -> int:
        return int(np.asarray(self.coeff_summaries.n_obs, dtype=np.int64).sum())


def _infer_feature_columns_from_header(
    header: pd.DataFrame, *, include_es_all: bool
) -> tuple[str, ...]:
    cols = tuple(map(str, header.columns))
    es_candidates = sorted([c for c in cols if c.startswith("ES_")]) if include_es_all else []
    return tuple(infer_feature_columns(cols, es_candidates=es_candidates))


def _fit_partial_pool_hyperparams(
    *,
    xtx_arr: np.ndarray,
    xty_arr: np.ndarray,
    yty_arr: np.ndarray,
    x_mean_arr: np.ndarray,
    y_mean_arr: np.ndarray,
    n_arr: np.ndarray,
    group_cluster_idx: np.ndarray,
    group_comp_idx: np.ndarray,
    cluster_ids: np.ndarray,
    comp_ids_unique: np.ndarray,
    supercat_ids: np.ndarray,
    cluster_supercat_idx: np.ndarray,
    lambda_diag: np.ndarray,
    sigma_y_prior: float,
    tau_mu_prior: float,
    tau_comp_prior: float,
    tau_b_prior: float,
    tau_w_prior: float,
    w0_prior_sigma: float,
    t0_prior_sigma: float,
    comp_embed_zc: np.ndarray,
    theta_alpha_prior_sigma: float,
    method: PymcMethod,
    seed: int,
    advi_steps: int,
    advi_log_every: int,
    advi_draws: int,
    map_maxeval: int,
) -> PartialPoolRidgeFitResult:
    try:
        import pymc as pm  # type: ignore
        import pytensor.tensor as pt  # type: ignore
        import pytensor.tensor.slinalg as slinalg  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("pymc (and pytensor) are required for this trainer") from exc

    if float(lambda_diag.min()) <= 0:
        raise ValueError("lambda_diag must be > 0")
    if float(sigma_y_prior) <= 0:
        raise ValueError("sigma_y_prior must be > 0")
    comp_embed_zc = np.asarray(comp_embed_zc, dtype=np.float64)
    if comp_embed_zc.ndim != 2:
        raise ValueError(f"comp_embed_zc must be 2D (got shape {comp_embed_zc.shape})")
    if comp_embed_zc.shape[0] != int(comp_ids_unique.size):
        raise ValueError(
            f"comp_embed_zc has n_compounds={comp_embed_zc.shape[0]}, "
            f"expected {int(comp_ids_unique.size)}"
        )
    if float(theta_alpha_prior_sigma) <= 0:
        raise ValueError("theta_alpha_prior_sigma must be > 0")

    n_groups = int(n_arr.size)
    p = int(xtx_arr.shape[1])

    # Optional fixed-lambda caching (critical for ADVI speed, but can be memory heavy).
    cache_fixed = str(os.environ.get("COMPASSIGN_RT_CACHE_FIXED_CHOLESKY", "1")).strip() != "0"
    cached: dict[str, np.ndarray] = {}
    if cache_fixed:
        jitter = 1e-12
        n_obs_f_np = np.asarray(n_arr, dtype=np.float64)
        x_mean_f64 = np.asarray(x_mean_arr, dtype=np.float64)
        y_mean_f64 = np.asarray(y_mean_arr, dtype=np.float64)
        xty_f64 = np.asarray(xty_arr, dtype=np.float64)
        yty_f64 = np.asarray(yty_arr, dtype=np.float64)

        sy_np = y_mean_f64 * n_obs_f_np
        sx_np = x_mean_f64 * n_obs_f_np[:, None]
        sxy_np = xty_f64 + x_mean_f64 * sy_np[:, None]
        yty_unc_np = yty_f64 + n_obs_f_np * (y_mean_f64**2)
        xtx_unc_np = np.asarray(xtx_arr, dtype=np.float64) + (
            (x_mean_f64[:, :, None] * x_mean_f64[:, None, :]) * n_obs_f_np[:, None, None]
        )

        a_np = xtx_unc_np + np.diag(lambda_diag)[None, :, :] + jitter * np.eye(p)[None, :, :]
        chol_np = np.linalg.cholesky(a_np)
        logdet_a_np = 2.0 * np.sum(np.log(np.diagonal(chol_np, axis1=1, axis2=2)), axis=1)
        del xtx_unc_np, a_np

        cached.update(
            n_obs_f=n_obs_f_np,
            sx=sx_np,
            sy=sy_np,
            sxy=sxy_np,
            yty_unc=yty_unc_np,
            chol=chol_np,
            logdet_a=logdet_a_np,
        )

    logdet_lambda = float(np.sum(np.log(lambda_diag)))
    y_mean0 = float(np.mean(np.asarray(y_mean_arr, dtype=np.float64))) if n_groups > 0 else 0.0

    idata: Any | None = None
    map_est: dict | None = None

    with pm.Model():
        sigma_y = pm.HalfNormal("sigma_y", sigma=float(sigma_y_prior))
        sigma2 = pm.Deterministic("sigma2", sigma_y**2)

        group_cluster_idx_data = pm.Data(
            "group_cluster_idx", np.asarray(group_cluster_idx, dtype=np.int64)
        )
        group_comp_idx_data = pm.Data("group_comp_idx", np.asarray(group_comp_idx, dtype=np.int64))
        cluster_supercat_idx_data = pm.Data(
            "cluster_supercat_idx", np.asarray(cluster_supercat_idx, dtype=np.int64)
        )

        n_clusters = int(np.asarray(cluster_ids, dtype=np.int64).size)
        n_compounds = int(np.asarray(comp_ids_unique, dtype=np.int64).size)
        n_supercats = int(np.asarray(supercat_ids, dtype=np.int64).size)

        tau_mu_supercat = pm.HalfNormal("tau_mu_supercat", sigma=float(tau_mu_prior))
        mu_supercat = pm.Normal("mu_supercat", mu=0.0, sigma=tau_mu_supercat, shape=n_supercats)
        mu_supercat_centered = mu_supercat - pt.mean(mu_supercat)

        # Supercategory-specific shrinkage for species-level offsets.
        tau_mu = pm.HalfNormal("tau_mu", sigma=float(tau_mu_prior), shape=n_supercats)
        mu_species = pm.Normal(
            "mu_species",
            mu=mu_supercat_centered[cluster_supercat_idx_data],
            sigma=tau_mu[cluster_supercat_idx_data],
            shape=n_clusters,
        )
        mu_species_centered = mu_species - pt.mean(mu_species)

        tau_comp = pm.HalfNormal("tau_comp", sigma=float(tau_comp_prior))
        z_np = np.asarray(comp_embed_zc, dtype=np.float64)
        d = int(z_np.shape[1])
        z = pm.Data("comp_embed_zc", z_np)

        theta_alpha = pm.Normal(
            "theta_alpha", mu=0.0, sigma=float(theta_alpha_prior_sigma), shape=d
        )
        delta_comp = pm.Normal("delta_comp", mu=0.0, sigma=tau_comp, shape=n_compounds)
        delta_comp_centered = delta_comp - pt.mean(delta_comp)
        alpha_comp = pm.Deterministic("alpha_comp", pt.dot(z, theta_alpha) + delta_comp_centered)
        alpha_comp_centered = alpha_comp - pt.mean(alpha_comp)

        t0 = pm.Normal("t0", mu=0.0, sigma=float(t0_prior_sigma), initval=float(y_mean0))
        tau_b = pm.HalfNormal("tau_b", sigma=float(tau_b_prior))

        b_mean = (
            t0
            + mu_species_centered[group_cluster_idx_data]
            + alpha_comp_centered[group_comp_idx_data]
        )
        b = pm.Normal("b", mu=b_mean, sigma=tau_b, shape=(n_groups,))

        # Slope head: supercategory -> species.
        tau_w_supercat = pm.HalfNormal("tau_w_supercat", sigma=float(tau_w_prior))
        w0 = pm.Normal("w0", mu=0.0, sigma=float(w0_prior_sigma), shape=p)
        w_supercat = pm.Normal(
            "w_supercat", mu=w0[None, :], sigma=tau_w_supercat, shape=(n_supercats, p)
        )
        tau_w = pm.HalfNormal("tau_w", sigma=float(tau_w_prior), shape=n_supercats)
        w_species = pm.Normal(
            "w_species",
            mu=w_supercat[cluster_supercat_idx_data],
            sigma=tau_w[cluster_supercat_idx_data][:, None],
            shape=(n_clusters, p),
        )
        slope_mean = w_species[group_cluster_idx_data]

        if cache_fixed and "chol" in cached:
            n_obs_f = pm.Data("n_obs_f", np.asarray(cached["n_obs_f"], dtype=np.float64))
            sx = pm.Data("sx", np.asarray(cached["sx"], dtype=np.float64))
            sy = pm.Data("sy", np.asarray(cached["sy"], dtype=np.float64))
            sxy = pm.Data("sxy", np.asarray(cached["sxy"], dtype=np.float64))
            yty_unc = pm.Data("yty_unc", np.asarray(cached["yty_unc"], dtype=np.float64))
            chol = pm.Data("chol_a", np.asarray(cached["chol"], dtype=np.float64))
            logdet_a = pm.Data("logdet_a", np.asarray(cached["logdet_a"], dtype=np.float64))

            xtx_data = pm.Data("xtx", xtx_arr.astype(np.float64, copy=False))
            x_mean_data = pm.Data("x_mean", x_mean_arr.astype(np.float64, copy=False))
            xtx_unc = xtx_data + (
                (x_mean_data[:, :, None] * x_mean_data[:, None, :]) * n_obs_f[:, None, None]
            )
        else:
            xtx_data = pm.Data("xtx", xtx_arr.astype(np.float64, copy=False))
            xty_data = pm.Data("xty", xty_arr.astype(np.float64, copy=False))
            yty_data = pm.Data("yty", yty_arr.astype(np.float64, copy=False))
            x_mean_data = pm.Data("x_mean", x_mean_arr.astype(np.float64, copy=False))
            y_mean_data = pm.Data("y_mean", y_mean_arr.astype(np.float64, copy=False))
            n_obs_data = pm.Data("n_obs", n_arr.astype(np.int64, copy=False))

            n_obs_f = n_obs_data.astype("float64")
            sy = y_mean_data * n_obs_f
            sx = x_mean_data * n_obs_f[:, None]
            sxy = xty_data + x_mean_data * sy[:, None]
            yty_unc = yty_data + n_obs_f * y_mean_data**2

            xtx_unc = xtx_data + (
                (x_mean_data[:, :, None] * x_mean_data[:, None, :]) * n_obs_f[:, None, None]
            )

            a = xtx_unc + pt.diag(lambda_diag)[None, :, :] + 1e-12 * pt.eye(p)[None, :, :]
            chol = pt.linalg.cholesky(a)
            chol_diag = pt.diagonal(chol, axis1=1, axis2=2)
            logdet_a = 2.0 * pt.sum(pt.log(chol_diag), axis=1)

        xtx_m = pt.sum(xtx_unc * slope_mean[:, None, :], axis=2)
        m_sx = pt.sum(slope_mean * sx, axis=1)
        m_sxy = pt.sum(slope_mean * sxy, axis=1)
        m_xtx_m = pt.sum(slope_mean * xtx_m, axis=1)

        r2 = yty_unc - 2.0 * b * sy + n_obs_f * b**2 - 2.0 * m_sxy + 2.0 * b * m_sx + m_xtx_m
        s = sxy - b[:, None] * sx - xtx_m

        rhs = s[:, :, None]
        tmp = slinalg.solve_triangular(chol, rhs, lower=True, check_finite=False, b_ndim=2)
        w = slinalg.solve_triangular(
            chol, tmp, trans="T", lower=True, check_finite=False, b_ndim=2
        )[:, :, 0]
        quad2 = pt.sum(s * w, axis=1)
        sse = pt.maximum(r2 - quad2, 0.0)

        ll = -0.5 * (n_obs_f * pt.log(sigma2) + logdet_a - logdet_lambda + sse / sigma2)
        pm.Potential("explicit_intercept_collapsed_slopes_ll", pt.sum(ll))

        if method == "advi":
            import time

            try:
                from pymc.variational.callbacks import Callback  # type: ignore
            except Exception:  # noqa: BLE001
                Callback = object  # type: ignore[misc,assignment]

            class _AdviProgressLogger(Callback):  # type: ignore[misc,valid-type]
                def __init__(self, *, total_steps: int, log_every: int) -> None:
                    self._total_steps = int(total_steps)
                    self._log_every = int(log_every)
                    self._t0 = time.time()

                def __call__(self, approx: Any, loss: Any, i: int) -> None:
                    if self._log_every <= 0:
                        return
                    it = int(i)
                    if it != 1 and it != self._total_steps and it % self._log_every != 0:
                        return
                    try:
                        loss_f = float(np.asarray(loss, dtype=np.float64).reshape(-1)[-1])
                    except Exception:  # noqa: BLE001
                        loss_f = float("nan")
                    dt = time.time() - self._t0
                    rate = (float(it) / dt) if dt > 0 else float("nan")
                    eta = (float(self._total_steps - it) / rate) if rate > 0 else float("nan")
                    logger.info(
                        "[advi] iter=%s/%s loss=%.6g elapsed_min=%.1f eta_min=%.1f",
                        it,
                        self._total_steps,
                        loss_f,
                        dt / 60.0,
                        eta / 60.0,
                    )

            callbacks = []
            if int(advi_log_every) > 0:
                callbacks.append(
                    _AdviProgressLogger(total_steps=int(advi_steps), log_every=int(advi_log_every))
                )
            approx = pm.fit(
                n=int(advi_steps),
                method="advi",
                random_seed=int(seed),
                progressbar=True,
                callbacks=callbacks,
            )
            idata = approx.sample(draws=int(advi_draws), random_seed=int(seed))
        elif method == "map":
            map_est = pm.find_MAP(maxeval=int(map_maxeval))
        else:
            raise ValueError(f"Unknown method: {method}")

    if method == "advi":
        if idata is None:
            raise RuntimeError("Internal error: missing idata for advi fit")
        post = idata.posterior
        sigma2_mean = float(post["sigma2"].mean(dim=("chain", "draw")).to_numpy())
        t0_mean = float(post["t0"].mean(dim=("chain", "draw")).to_numpy())
        mu_species_mean = np.asarray(post["mu_species"].mean(dim=("chain", "draw")).to_numpy())
        alpha_comp_mean = np.asarray(post["alpha_comp"].mean(dim=("chain", "draw")).to_numpy())
        tau_comp_mean = float(post["tau_comp"].mean(dim=("chain", "draw")).to_numpy())
        theta_alpha_mean = np.asarray(post["theta_alpha"].mean(dim=("chain", "draw")).to_numpy())
        tau_b_mean = float(post["tau_b"].mean(dim=("chain", "draw")).to_numpy())
        w_species_mean = np.asarray(post["w_species"].mean(dim=("chain", "draw")).to_numpy())
    else:
        if map_est is None:
            raise RuntimeError("Internal error: missing MAP estimate")
        sigma2_mean = float(map_est["sigma_y"]) ** 2
        t0_mean = float(map_est["t0"])
        mu_species_mean = np.asarray(map_est["mu_species"], dtype=np.float64)
        alpha_comp_mean = np.asarray(map_est["alpha_comp"], dtype=np.float64)
        tau_comp_mean = float(map_est["tau_comp"])
        theta_alpha_mean = np.asarray(map_est["theta_alpha"], dtype=np.float64)
        tau_b_mean = float(map_est["tau_b"])
        w_species_mean = np.asarray(map_est["w_species"], dtype=np.float64)

    if not (np.isfinite(sigma2_mean) and sigma2_mean > 0):
        raise ValueError(f"Invalid sigma2 estimate: {sigma2_mean}")

    return PartialPoolRidgeFitResult(
        sigma2_mean=float(sigma2_mean),
        trace_idata=idata,
        t0_mean=float(t0_mean),
        mu_species_mean=mu_species_mean.astype(np.float64, copy=False),
        alpha_comp_mean=alpha_comp_mean.astype(np.float64, copy=False),
        tau_comp_mean=float(tau_comp_mean),
        theta_alpha_mean=theta_alpha_mean.astype(np.float64, copy=False),
        tau_b_mean=float(tau_b_mean),
        w_species_mean=w_species_mean.astype(np.float64, copy=False),
    )


def train_pymc_partial_pool_ridge_from_csv(
    *,
    data_csv: Path,
    seed: int = 42,
    max_train_rows: int = 0,
    include_es_all: bool = True,
    feature_center: Literal["none", "global"] = "global",
    lambda_slopes: float = 3e-4,
    sigma_y_prior: float = 0.05,
    tau_mu_prior: float = 0.5,
    tau_comp_prior: float = 0.5,
    tau_b_prior: float = 0.5,
    tau_w_prior: float = 0.5,
    w0_prior_sigma: float = 1.0,
    t0_prior_sigma: float = 10.0,
    chem_embeddings_path: Path = Path("resources/metabolites/embeddings_chemberta_pca20.parquet"),
    theta_alpha_prior_sigma: float = 1.0,
    method: PymcMethod = "advi",
    advi_steps: int = 10_000,
    advi_log_every: int = 1000,
    advi_draws: int = 50,
    map_maxeval: int = 50_000,
) -> PartialPoolRidgeTrainArtifacts:
    """Train the partial pooling ridge model from a production RT CSV."""
    seed = int(seed)
    np.random.seed(seed)

    data_csv = Path(data_csv)
    header = pd.read_csv(data_csv, nrows=0)
    missing_req = [c for c in REQUIRED_COLS if c not in header.columns]
    if missing_req:
        raise ValueError(f"CSV missing required columns: {missing_req}")

    feature_cols = _infer_feature_columns_from_header(header, include_es_all=bool(include_es_all))

    dtype: dict[str, object] = {c: np.float32 for c in feature_cols}
    dtype.update(
        {
            "rt": np.float32,
            "species_cluster": np.int64,
            "species": np.int64,
            "comp_id": np.int64,
            "compound": np.int64,
            "compound_class": np.float32,
        }
    )
    df = pd.read_csv(
        data_csv,
        usecols=list(dict.fromkeys([*REQUIRED_COLS, *feature_cols])),
        nrows=int(max_train_rows) if int(max_train_rows) > 0 else None,
        dtype=dtype,
    )
    df["compound_class"] = df["compound_class"].fillna(-1).astype(int)

    supercat_arr = df["species_cluster"].to_numpy(dtype=np.int64, copy=False)
    species_arr = df["species"].to_numpy(dtype=np.int64, copy=False)
    comp_arr = df["comp_id"].to_numpy(dtype=np.int64, copy=False)
    chem_arr = df["compound"].to_numpy(dtype=np.int64, copy=False)
    class_arr = df["compound_class"].to_numpy(dtype=np.int64, copy=False)
    y_arr = df["rt"].to_numpy(dtype=np.float64, copy=False)
    x_arr = df[list(feature_cols)].to_numpy(dtype=np.float32, copy=True)
    np.nan_to_num(x_arr, copy=False)

    key_arr = encode_group_comp_key(species_arr, comp_arr)

    comp_id_to_choice, mapping_collision_rows = resolve_comp_id_mapping(
        comp_id=comp_arr, chem_id=chem_arr, compound_class=class_arr
    )

    feature_center_arr, feature_rotation, x_arr = apply_global_feature_transform(
        x_arr, center_mode=str(feature_center), rotation_mode="none"
    )

    (
        keys_arr,
        clusters_arr,
        comp_ids_arr,
        n_arr,
        x_mean_arr,
        y_mean_arr,
        xtx_arr,
        xty_arr,
        yty_arr,
    ) = compute_group_suffstats(
        key_arr=key_arr,
        group_id_arr=species_arr,
        comp_id_arr=comp_arr,
        x_arr=x_arr,
        y_arr=y_arr,
        feature_rotation=feature_rotation,
        max_groups=0,
    )

    chem_ids_arr, class_ids_arr = assign_group_metadata(
        comp_ids=comp_ids_arr, comp_id_to_choice=comp_id_to_choice
    )

    # Deterministic mapping species -> species_cluster for supercategory-aware pooling.
    child_sorted, parent_sorted = infer_parent_ids(
        child_id=species_arr,
        parent_id=supercat_arr,
        child_name="species",
        parent_name="species_cluster",
    )
    if child_sorted.size == 0:
        raise ValueError("No rows found; cannot infer species->species_cluster mapping")
    pos = np.searchsorted(child_sorted, clusters_arr)
    if (
        np.any(pos < 0)
        or np.any(pos >= child_sorted.size)
        or not np.all(child_sorted[pos] == clusters_arr)
    ):
        raise RuntimeError("Internal error: missing species->species_cluster mapping for groups")
    supercat_id = parent_sorted[pos].astype(np.int64, copy=False)

    # Index mappings for hierarchical priors.
    cluster_ids, group_cluster_idx = build_index_mapping(clusters_arr)
    comp_ids_unique, group_comp_idx = build_index_mapping(comp_ids_arr)

    pos2 = np.searchsorted(child_sorted, cluster_ids)
    if (
        np.any(pos2 < 0)
        or np.any(pos2 >= child_sorted.size)
        or not np.all(child_sorted[pos2] == cluster_ids)
    ):
        raise RuntimeError(
            "Internal error: missing species->species_cluster mapping for cluster ids"
        )
    cluster_supercat_id = parent_sorted[pos2].astype(np.int64, copy=False)
    supercat_ids = np.unique(cluster_supercat_id).astype(np.int64, copy=False)
    cluster_supercat_idx = np.searchsorted(supercat_ids, cluster_supercat_id).astype(
        np.int64, copy=False
    )

    p = int(xtx_arr.shape[1])
    if float(lambda_slopes) <= 0:
        raise ValueError("lambda_slopes must be > 0")
    lambda_diag = np.full((p,), float(lambda_slopes), dtype=np.float64)

    # Metadata aligned with unique comp_ids (for chemistry-informed alpha priors and backoff summaries).
    comp_chem_unique, comp_class_unique = assign_group_metadata(
        comp_ids=comp_ids_unique, comp_id_to_choice=comp_id_to_choice
    )

    chem_embeddings_path_resolved = Path(chem_embeddings_path)
    if not chem_embeddings_path_resolved.is_absolute():
        chem_embeddings_path_resolved = (
            Path(__file__).resolve().parents[3] / chem_embeddings_path_resolved
        ).resolve()
    emb = load_chemberta_pca20(chem_embeddings_path_resolved)
    order = np.argsort(emb.chem_id.astype(np.int64), kind="mergesort")
    chem_ids_sorted = emb.chem_id[order].astype(np.int64, copy=False)
    features_sorted = emb.features[order].astype(np.float64, copy=False)

    chem = np.asarray(comp_chem_unique, dtype=np.int64)
    if np.any(chem < 0):
        missing = sorted(set(int(x) for x in chem[chem < 0].tolist()))
        raise ValueError(
            "comp_id->chem_id mapping missing for some compounds "
            f"(need chem_id for chem-linear alpha prior): {missing[:10]}"
        )
    idx = np.searchsorted(chem_ids_sorted, chem)
    ok = (idx >= 0) & (idx < chem_ids_sorted.size)
    if ok.any():
        ok[ok] &= chem_ids_sorted[idx[ok]] == chem[ok]
    if not bool(np.all(ok)):
        missing = sorted(set(int(x) for x in chem[~ok].tolist()))
        raise ValueError(
            "ChemBERTa embedding missing for some chem_ids in training compounds: "
            f"{missing[:10]} (n_missing={len(missing)})"
        )
    comp_embed = features_sorted[idx]
    alpha_z_center = np.mean(comp_embed, axis=0).astype(np.float64, copy=False)
    comp_embed_zc = (comp_embed - alpha_z_center[None, :]).astype(np.float64, copy=False)

    fit = _fit_partial_pool_hyperparams(
        xtx_arr=xtx_arr,
        xty_arr=xty_arr,
        yty_arr=yty_arr,
        x_mean_arr=x_mean_arr,
        y_mean_arr=y_mean_arr,
        n_arr=n_arr,
        group_cluster_idx=group_cluster_idx,
        group_comp_idx=group_comp_idx,
        cluster_ids=cluster_ids,
        comp_ids_unique=comp_ids_unique,
        supercat_ids=supercat_ids,
        cluster_supercat_idx=cluster_supercat_idx,
        lambda_diag=lambda_diag,
        sigma_y_prior=float(sigma_y_prior),
        tau_mu_prior=float(tau_mu_prior),
        tau_comp_prior=float(tau_comp_prior),
        tau_b_prior=float(tau_b_prior),
        tau_w_prior=float(tau_w_prior),
        w0_prior_sigma=float(w0_prior_sigma),
        t0_prior_sigma=float(t0_prior_sigma),
        comp_embed_zc=comp_embed_zc,
        theta_alpha_prior_sigma=float(theta_alpha_prior_sigma),
        method=method,
        seed=seed,
        advi_steps=int(advi_steps),
        advi_log_every=int(advi_log_every),
        advi_draws=int(advi_draws),
        map_maxeval=int(map_maxeval),
    )

    # Convert fitted hierarchy into per-group priors for (b, w).
    mu_species = np.asarray(fit.mu_species_mean, dtype=np.float64)
    mu_species = mu_species - float(mu_species.mean()) if mu_species.size else mu_species
    alpha_comp = np.asarray(fit.alpha_comp_mean, dtype=np.float64)
    alpha_comp = alpha_comp - float(alpha_comp.mean()) if alpha_comp.size else alpha_comp
    t0 = float(fit.t0_mean)

    # Backoff summaries for unseen (species, comp_id) groups.
    backoff_summaries = PartialPoolBackoffSummaries(
        feature_names=tuple(feature_cols),
        cluster_ids=np.asarray(cluster_ids, dtype=np.int64),
        cluster_supercat_id=np.asarray(cluster_supercat_id, dtype=np.int64),
        comp_ids=np.asarray(comp_ids_unique, dtype=np.int64),
        comp_chem_id=np.asarray(comp_chem_unique, dtype=np.int64),
        comp_class=np.asarray(comp_class_unique, dtype=np.int64),
        alpha_z_center=alpha_z_center,
        alpha_theta=np.asarray(fit.theta_alpha_mean, dtype=np.float64),
        tau_comp=float(fit.tau_comp_mean),
        t0=float(t0),
        mu_cluster=np.asarray(mu_species, dtype=np.float64),
        alpha_comp=np.asarray(alpha_comp, dtype=np.float64),
        w_cluster=np.asarray(fit.w_species_mean, dtype=np.float64),
        tau_b=float(fit.tau_b_mean),
        sigma2=float(fit.sigma2_mean),
        lambda_slopes=float(lambda_slopes),
    )

    b_prior_mean = (
        float(t0)
        + mu_species[np.asarray(group_cluster_idx, dtype=np.int64)]
        + alpha_comp[np.asarray(group_comp_idx, dtype=np.int64)]
    )
    slope_mean = np.asarray(fit.w_species_mean, dtype=np.float64)[
        np.asarray(group_cluster_idx, dtype=np.int64)
    ]

    beta_hat, beta_var_diag, beta_cov = compute_group_posterior_summaries_with_b_prior_mean(
        xtx_arr=xtx_arr,
        xty_arr=xty_arr,
        x_mean_arr=x_mean_arr,
        y_mean_arr=y_mean_arr,
        n_arr=n_arr,
        b_prior_mean=b_prior_mean,
        tau_b=float(fit.tau_b_mean),
        slope_mean=slope_mean,
        lambda_diag=lambda_diag,
        sigma2_mean=float(fit.sigma2_mean),
    )

    sigma2_vec = np.full((int(keys_arr.size),), float(fit.sigma2_mean), dtype=np.float64)

    coeff_summaries = Stage1CoeffSummaries(
        feature_names=tuple(feature_cols),
        feature_center=feature_center_arr,
        feature_rotation=feature_rotation,
        group_keys=keys_arr,
        species_cluster=clusters_arr,
        group_col="species",
        supercat_id=supercat_id,
        comp_id=comp_ids_arr,
        chem_id=chem_ids_arr,
        compound_class=class_ids_arr,
        n_obs=n_arr,
        beta_hat=beta_hat,
        beta_var_diag=beta_var_diag,
        beta_cov=beta_cov,
        sigma2_mean=sigma2_vec,
    )

    trained_at = datetime.now().isoformat()

    return PartialPoolRidgeTrainArtifacts(
        coeff_summaries=coeff_summaries,
        backoff_summaries=backoff_summaries,
        trained_at=trained_at,
        seed=seed,
        data_csv=str(data_csv),
        max_train_rows=int(max_train_rows),
        include_es_all=bool(include_es_all),
        feature_center=str(feature_center),
        lambda_slopes=float(lambda_slopes),
        sigma2_mean=float(fit.sigma2_mean),
        sigma_y_prior=float(sigma_y_prior),
        tau_mu_prior=float(tau_mu_prior),
        tau_comp_prior=float(tau_comp_prior),
        chem_embeddings_path=str(chem_embeddings_path_resolved),
        theta_alpha_prior_sigma=float(theta_alpha_prior_sigma),
        tau_b_prior=float(tau_b_prior),
        tau_w_prior=float(tau_w_prior),
        w0_prior_sigma=float(w0_prior_sigma),
        t0_prior_sigma=float(t0_prior_sigma),
        method=method,
        advi_steps=int(advi_steps) if method == "advi" else None,
        advi_draws=int(advi_draws) if method == "advi" else None,
        map_maxeval=int(map_maxeval) if method == "map" else None,
        trace_idata=fit.trace_idata,
        mapping_collision_rows=mapping_collision_rows,
    )


def write_pymc_partial_pool_ridge_artifacts(
    *, artifacts: PartialPoolRidgeTrainArtifacts, output_dir: Path
) -> dict[str, str | None]:
    """Write stage-1 coefficient summaries + config.json (+ optional trace/collision CSVs)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)
    (output_dir / "results").mkdir(exist_ok=True)

    coeff_npz = output_dir / "models" / "stage1_coeff_summaries_posterior.npz"
    artifacts.coeff_summaries.save_npz(coeff_npz)

    backoff_npz = output_dir / "models" / "partial_pool_backoff_summaries.npz"
    artifacts.backoff_summaries.save_npz(backoff_npz)

    trace_path: Path | None = None
    if artifacts.trace_idata is not None:
        trace_path = output_dir / "results" / f"trace_{artifacts.method}.nc"
        artifacts.trace_idata.to_netcdf(trace_path)

    collisions_csv: Path | None = None
    if artifacts.mapping_collision_rows:
        collisions_csv = output_dir / "results" / "comp_id_mapping_collisions.csv"
        pd.DataFrame(artifacts.mapping_collision_rows).to_csv(collisions_csv, index=False)

    config = {
        "timestamp": artifacts.trained_at,
        "artifact_type": "rt_pymc_partial_pool_ridge",
        "data_csv": artifacts.data_csv,
        "seed": int(artifacts.seed),
        "group_col": "species",
        "max_train_rows": int(artifacts.max_train_rows),
        "n_obs_train": int(artifacts.n_obs_train),
        "n_groups": int(artifacts.n_groups),
        "feature_names": list(artifacts.coeff_summaries.feature_names),
        "n_features": int(len(artifacts.coeff_summaries.feature_names)),
        "include_es_all": bool(artifacts.include_es_all),
        "feature_center": str(artifacts.feature_center),
        "lambda_slopes": float(artifacts.lambda_slopes),
        "sigma2_mean": float(artifacts.sigma2_mean),
        "sigma_y_prior": float(artifacts.sigma_y_prior),
        "tau_mu_prior": float(artifacts.tau_mu_prior),
        "tau_comp_prior": float(artifacts.tau_comp_prior),
        "chem_embeddings_path": artifacts.chem_embeddings_path,
        "theta_alpha_prior_sigma": float(artifacts.theta_alpha_prior_sigma),
        "tau_b_prior": float(artifacts.tau_b_prior),
        "tau_w_prior": float(artifacts.tau_w_prior),
        "w0_prior_sigma": float(artifacts.w0_prior_sigma),
        "t0_prior_sigma": float(artifacts.t0_prior_sigma),
        "method": str(artifacts.method),
        "advi_steps": int(artifacts.advi_steps) if artifacts.method == "advi" else None,
        "advi_draws": int(artifacts.advi_draws) if artifacts.method == "advi" else None,
        "map_maxeval": int(artifacts.map_maxeval) if artifacts.method == "map" else None,
        "trace_path": str(trace_path) if trace_path is not None else None,
        "coeff_npz": str(coeff_npz),
        "backoff_npz": str(backoff_npz),
        "mapping_collisions_n_comp_ids": int(
            len({int(r["comp_id"]) for r in artifacts.mapping_collision_rows})
            if artifacts.mapping_collision_rows
            else 0
        ),
        "mapping_collisions_csv": str(collisions_csv) if collisions_csv is not None else None,
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))

    return {
        "coeff_npz": str(coeff_npz),
        "backoff_npz": str(backoff_npz),
        "trace_path": str(trace_path) if trace_path is not None else None,
        "collisions_csv": str(collisions_csv) if collisions_csv is not None else None,
        "config_json": str(output_dir / "config.json"),
    }
