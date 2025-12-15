"""
PyMC ridge RT model: fully pooled by supercategory (species_cluster).

Model intent
------------
This is the simplest Bayesian ridge variant used in the RT pipeline. It trains an independent
ridge regression for each group:

  g = (species_cluster, comp_id)

using run covariates x_run (IS*/RS*/optional ES_*):

  rt = b_g + x_run^T w_g + eps,      eps ~ Normal(0, sigma2)
  w_g ~ Normal(0, sigma2 * Lambda^{-1})          (ridge prior; Lambda = diag(lambda_slopes))

Key implementation detail: collapsed slopes
-------------------------------------------
Instead of sampling (b_g, w_g) for every group (which is too expensive at scale), we:
  1) compute per-group centered sufficient statistics (Xc^T Xc, Xc^T yc, yc^T yc),
  2) integrate out slopes analytically inside the likelihood (collapsed ridge marginal),
  3) fit only the shared noise scale (sigma2) in PyMC, and
  4) compute per-group posterior mean/covariance of beta=[b; w] in NumPy.

This produces the `Stage1CoeffSummaries` artifact used downstream for evaluation and peak
assignment.

Why RMSE can match sklearn ridge
-------------------------------
When `lambda_slopes` is fixed, the posterior mean of w_g is the ridge solution
(X^T X + Lambda)^{-1} X^T y and does not depend on sigma2. Therefore mean predictions can match
sklearn ridge exactly when the same design matrix and lambda are used.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .ridge_stage1 import (
    Stage1CoeffSummaries,
    apply_global_feature_transform,
    assign_group_metadata,
    compute_group_posterior_summaries,
    compute_group_suffstats,
    encode_group_comp_key,
    infer_feature_columns,
    resolve_comp_id_mapping,
)

logger = logging.getLogger(__name__)

PymcMethod = Literal["advi", "map"]


REQUIRED_COLS: tuple[str, ...] = ("rt", "compound", "comp_id", "compound_class", "species_cluster")


@dataclass(frozen=True)
class SupercategoryRidgeTrainArtifacts:
    coeff_summaries: Stage1CoeffSummaries
    trained_at: str
    seed: int
    data_csv: str
    max_train_rows: int
    include_es_all: bool
    feature_center: str
    lambda_slopes: float
    sigma2_mean: float
    sigma_y_prior: float
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


def _fit_sigma2_collapsed_ridge(
    *,
    xtx_arr: np.ndarray,
    xty_arr: np.ndarray,
    yty_arr: np.ndarray,
    n_arr: np.ndarray,
    lambda_diag: np.ndarray,
    sigma_y_prior: float,
    method: PymcMethod,
    seed: int,
    advi_steps: int,
    advi_draws: int,
    advi_log_every: int,
    map_maxeval: int,
) -> tuple[float, Any | None]:
    try:
        import pymc as pm  # type: ignore
        import pytensor.tensor as pt  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("pymc (and pytensor) are required for this trainer") from exc

    if float(sigma_y_prior) <= 0:
        raise ValueError("sigma_y_prior must be > 0")

    xtx_arr = np.asarray(xtx_arr, dtype=np.float64)
    xty_arr = np.asarray(xty_arr, dtype=np.float64)
    yty_arr = np.asarray(yty_arr, dtype=np.float64)
    n_arr = np.asarray(n_arr, dtype=np.int64)
    lambda_diag = np.asarray(lambda_diag, dtype=np.float64)

    n_groups = int(n_arr.size)
    if n_groups == 0:
        raise ValueError("No groups found (n_groups=0); cannot fit sigma2.")

    p = int(xtx_arr.shape[1])
    jitter = 1e-12

    # Fixed-lambda fast path: precompute determinant and quadratic form per group.
    a_np = (
        xtx_arr
        + np.diag(lambda_diag)[None, :, :]
        + jitter * np.eye(p, dtype=np.float64)[None, :, :]
    )
    chol_np = np.linalg.cholesky(a_np)
    logdet_a_np = 2.0 * np.sum(np.log(np.diagonal(chol_np, axis1=1, axis2=2)), axis=1)
    rhs = xty_arr[:, :, None]
    tmp = np.linalg.solve(chol_np, rhs)
    w_np = np.linalg.solve(np.transpose(chol_np, (0, 2, 1)), tmp)[:, :, 0]
    quad2_np = np.sum(xty_arr * w_np, axis=1)
    sse_np = np.maximum(yty_arr - quad2_np, 0.0).astype(np.float64, copy=False)
    n_eff_np = np.maximum(n_arr - 1, 0).astype(np.float64, copy=False)

    logdet_lambda = float(np.sum(np.log(lambda_diag)))

    idata: Any | None = None
    map_est: dict | None = None

    with pm.Model():
        sigma_y = pm.HalfNormal("sigma_y", sigma=float(sigma_y_prior))
        sigma2 = pm.Deterministic("sigma2", sigma_y**2)

        n_eff_data = pm.Data("n_eff", n_eff_np)
        sse_data = pm.Data("sse", sse_np)
        logdet_a_data = pm.Data("logdet_a", logdet_a_np)
        logdet_lambda_t = pt.as_tensor_variable(logdet_lambda, dtype="float64")

        ll = -0.5 * (
            n_eff_data * pt.log(sigma2) + logdet_a_data - logdet_lambda_t + sse_data / sigma2
        )
        pm.Potential("collapsed_marginal_ll", pt.sum(ll))

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
    else:
        if map_est is None or "sigma_y" not in map_est:
            raise RuntimeError("Internal error: missing MAP estimate for sigma_y")
        sigma2_mean = float(map_est["sigma_y"]) ** 2

    if not (np.isfinite(sigma2_mean) and sigma2_mean > 0):
        raise ValueError(f"Invalid sigma2 estimate: {sigma2_mean}")

    return float(sigma2_mean), idata


def train_pymc_supercategory_ridge_from_csv(
    *,
    data_csv: Path,
    seed: int = 42,
    max_train_rows: int = 0,
    include_es_all: bool = True,
    feature_center: Literal["none", "global"] = "global",
    lambda_slopes: float = 3e-4,
    sigma_y_prior: float = 0.05,
    method: PymcMethod = "advi",
    advi_steps: int = 5000,
    advi_log_every: int = 1000,
    advi_draws: int = 50,
    map_maxeval: int = 50_000,
) -> SupercategoryRidgeTrainArtifacts:
    """Train the supercategory collapsed ridge model from a production RT CSV."""
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
    comp_arr = df["comp_id"].to_numpy(dtype=np.int64, copy=False)
    chem_arr = df["compound"].to_numpy(dtype=np.int64, copy=False)
    class_arr = df["compound_class"].to_numpy(dtype=np.int64, copy=False)
    y_arr = df["rt"].to_numpy(dtype=np.float64, copy=False)
    x_arr = df[list(feature_cols)].to_numpy(dtype=np.float32, copy=True)
    np.nan_to_num(x_arr, copy=False)

    key_arr = encode_group_comp_key(supercat_arr, comp_arr)

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
        group_id_arr=supercat_arr,
        comp_id_arr=comp_arr,
        x_arr=x_arr,
        y_arr=y_arr,
        feature_rotation=feature_rotation,
        max_groups=0,
    )

    chem_ids_arr, class_ids_arr = assign_group_metadata(
        comp_ids=comp_ids_arr, comp_id_to_choice=comp_id_to_choice
    )

    p = int(xtx_arr.shape[1])
    if float(lambda_slopes) <= 0:
        raise ValueError("lambda_slopes must be > 0")
    lambda_diag = np.full((p,), float(lambda_slopes), dtype=np.float64)

    sigma2_mean, trace_idata = _fit_sigma2_collapsed_ridge(
        xtx_arr=xtx_arr,
        xty_arr=xty_arr,
        yty_arr=yty_arr,
        n_arr=n_arr,
        lambda_diag=lambda_diag,
        sigma_y_prior=float(sigma_y_prior),
        method=method,
        seed=seed,
        advi_steps=int(advi_steps),
        advi_draws=int(advi_draws),
        advi_log_every=int(advi_log_every),
        map_maxeval=int(map_maxeval),
    )

    beta_hat, beta_var_diag, beta_cov = compute_group_posterior_summaries(
        xtx_arr=xtx_arr,
        xty_arr=xty_arr,
        x_mean_arr=x_mean_arr,
        y_mean_arr=y_mean_arr,
        n_arr=n_arr,
        lambda_diag=lambda_diag,
        slope_mean=None,
        sigma2_mean=float(sigma2_mean),
    )

    sigma2_vec = np.full((int(keys_arr.size),), float(sigma2_mean), dtype=np.float64)

    coeff_summaries = Stage1CoeffSummaries(
        feature_names=tuple(feature_cols),
        feature_center=feature_center_arr,
        feature_rotation=feature_rotation,
        group_keys=keys_arr,
        species_cluster=clusters_arr,
        group_col="species_cluster",
        supercat_id=None,
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

    return SupercategoryRidgeTrainArtifacts(
        coeff_summaries=coeff_summaries,
        trained_at=trained_at,
        seed=seed,
        data_csv=str(data_csv),
        max_train_rows=int(max_train_rows),
        include_es_all=bool(include_es_all),
        feature_center=str(feature_center),
        lambda_slopes=float(lambda_slopes),
        sigma2_mean=float(sigma2_mean),
        sigma_y_prior=float(sigma_y_prior),
        method=method,
        advi_steps=int(advi_steps) if method == "advi" else None,
        advi_draws=int(advi_draws) if method == "advi" else None,
        map_maxeval=int(map_maxeval) if method == "map" else None,
        trace_idata=trace_idata,
        mapping_collision_rows=mapping_collision_rows,
    )


def write_pymc_supercategory_ridge_artifacts(
    *, artifacts: SupercategoryRidgeTrainArtifacts, output_dir: Path
) -> dict[str, str | None]:
    """Write stage-1 coefficient summaries + config.json (+ optional trace/collision CSVs)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)
    (output_dir / "results").mkdir(exist_ok=True)

    coeff_npz = output_dir / "models" / "stage1_coeff_summaries_posterior.npz"
    artifacts.coeff_summaries.save_npz(coeff_npz)

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
        "artifact_type": "rt_pymc_supercategory_ridge",
        "data_csv": artifacts.data_csv,
        "seed": int(artifacts.seed),
        "group_col": "species_cluster",
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
        "method": str(artifacts.method),
        "advi_steps": int(artifacts.advi_steps) if artifacts.method == "advi" else None,
        "advi_draws": int(artifacts.advi_draws) if artifacts.method == "advi" else None,
        "map_maxeval": int(artifacts.map_maxeval) if artifacts.method == "map" else None,
        "trace_path": str(trace_path) if trace_path is not None else None,
        "coeff_npz": str(coeff_npz),
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
        "trace_path": str(trace_path) if trace_path is not None else None,
        "collisions_csv": str(collisions_csv) if collisions_csv is not None else None,
        "config_json": str(output_dir / "config.json"),
    }
