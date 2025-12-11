#!/usr/bin/env python3
"""
Posterior-predictive RT evaluation for PyMC collapsed ridge variants.

This script computes predictive intervals using posterior draws (Monte Carlo) rather than the
single-Normal approximation used by `eval_rt_coeff_summaries_by_species_cluster.py`.

It is intended primarily as a calibration/benchmarking tool:
  - Loads the trained PyMC output directory (coeff summaries + trace + config).
  - Streams a test CSV in chunks and, for each chunk:
      * computes the conditional predictive mean/variance per posterior draw,
      * samples one posterior-predictive value per draw, and
      * derives central intervals via empirical quantiles.

Notes:
  - Runtime scales ~O(n_rows * n_draws), so this is not intended for production-scale scoring.
  - When the trace contains per-group intercept draws `b` (chem_hier), we reconstruct the
    collapsed-slope conditional for each draw. Otherwise, we sample using the per-group
    coefficient covariance from the artifact.
  - Optionally supports "backoff" prediction for unseen groups using the chem_hier hierarchy
    (useful for held-out-chem experiments).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

import arviz as az

REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.compassign.rt.pymc_collapsed_ridge import Stage1CoeffSummaries  # noqa: E402
from src.compassign.rt.pymc_collapsed_ridge import ChemHierBackoffSummaries  # noqa: E402
from src.compassign.rt.pymc_collapsed_ridge import _encode_cluster_comp_key  # noqa: E402


REQUIRED_COLS = ["rt", "compound", "compound_class", "species_cluster", "comp_id"]
SPECIES_GROUP_COLS = ["sampleset_id"]


@dataclass
class AggStats:
    n: int = 0
    sum_sq_err: float = 0.0
    sum_abs_err: float = 0.0
    sum_covered: float = 0.0

    def update(
        self,
        *,
        pred_mean: np.ndarray,
        y_true: np.ndarray,
        covered: np.ndarray,
    ) -> None:
        if pred_mean.size == 0:
            return
        err = pred_mean - y_true
        self.n += int(err.size)
        self.sum_sq_err += float(np.sum(np.square(err)))
        self.sum_abs_err += float(np.sum(np.abs(err)))
        self.sum_covered += float(np.sum(covered))

    def to_metrics(self) -> dict[str, float]:
        if self.n == 0:
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


@dataclass(frozen=True)
class PosteriorDraws:
    sigma2: np.ndarray  # (D,)
    b: np.ndarray | None  # (D, G)
    mu_cluster: np.ndarray | None  # (D, C)
    w0: np.ndarray | None  # (D, P)
    w_cluster: np.ndarray | None  # (D, C, P)
    alpha_class: np.ndarray | None  # (D, J)
    theta: np.ndarray | None  # (D, Z)
    t0: np.ndarray | None  # (D,)
    tau_b: np.ndarray | None  # (D,)
    tau_t: np.ndarray | None  # (D,)
    tau_w: np.ndarray | None  # (D,)
    t_chem: np.ndarray | None  # (D, K)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Posterior-predictive evaluation for partially-collapsed PyMC RT model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--train-output-dir",
        type=Path,
        required=True,
        help="Training output directory (must contain config.json, models/*, results/trace_*.nc).",
    )
    parser.add_argument(
        "--test-csv", type=Path, required=True, help="RT production CSV to evaluate."
    )
    parser.add_argument(
        "--max-test-rows",
        type=int,
        default=100_000,
        help="Maximum number of test rows to evaluate (0 = all; large values will be slow).",
    )
    parser.add_argument("--chunk-size", type=int, default=50_000, help="Rows per chunk.")
    parser.add_argument(
        "--n-ppc-draws",
        type=int,
        default=200,
        help="Number of posterior draws used for posterior predictive intervals.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for posterior-predictive sampling."
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.95,
        help="Central interval mass for coverage calculation (e.g. 0.95).",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="ppc",
        help="Label included in output filenames under train-output-dir/results.",
    )
    parser.add_argument(
        "--require-seen-group",
        action="store_true",
        help=(
            "If set, skip rows whose (species_cluster, comp_id) is not present in the trained "
            "coefficient artifact (seen groups only)."
        ),
    )
    parser.add_argument(
        "--backoff-npz",
        type=Path,
        default=None,
        help=(
            "Optional ChemHierBackoffSummaries .npz (written by the chem_hier trainer). "
            "If provided and --require-seen-group is NOT set, unseen groups are scored using backoff."
        ),
    )
    parser.add_argument(
        "--backoff-mean-mode",
        type=str,
        default="chem",
        choices=["chem", "cluster_only"],
        help=(
            "How to compute the backoff mean for unseen groups. "
            "'chem' uses chemistry regression for unseen chem ids; "
            "'cluster_only' ignores chemistry and uses only cluster offset + t0."
        ),
    )
    parser.add_argument(
        "--backoff-slope-mean-mode",
        type=str,
        default="auto",
        choices=["auto", "cluster_head", "zero", "global", "cluster"],
        help=(
            "How to set the slope mean for unseen groups during backoff prediction. "
            "'auto' uses the learned cluster slope head if available (otherwise 0). "
            "'cluster_head' uses the learned cluster slope head (if present in the trace or backoff artifact). "
            "'zero' uses 0; 'global' uses average slope; 'cluster' uses per-cluster average slope."
        ),
    )
    parser.add_argument(
        "--chem-embeddings-path",
        type=Path,
        default=None,
        help=(
            "Optional ChemBERTa PCA-20 embedding parquet used to compute chemistry features for "
            "unseen chem ids during backoff prediction. Defaults to the path stored in --backoff-npz."
        ),
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


def _resolve_paths(train_output_dir: Path) -> tuple[Path, Path, Path]:
    cfg = train_output_dir / "config.json"
    coeff = train_output_dir / "models" / "stage1_coeff_summaries_posterior.npz"
    trace = train_output_dir / "results" / "trace_advi.nc"
    if not trace.exists():
        trace = train_output_dir / "results" / "trace_nuts.nc"
    if not cfg.exists():
        raise SystemExit(f"Missing config.json under {train_output_dir}")
    if not coeff.exists():
        raise SystemExit(f"Missing coeff summaries under {coeff}")
    if not trace.exists():
        raise SystemExit(
            f"Missing trace under {train_output_dir / 'results'} (expected trace_advi.nc or trace_nuts.nc)"
        )
    return cfg, coeff, trace


def _iter_test_chunks(
    *, test_csv: Path, usecols: list[str], chunk_size: int, max_rows: int
) -> Iterator[pd.DataFrame]:
    remaining = int(max_rows) if int(max_rows) > 0 else None
    for chunk in pd.read_csv(test_csv, usecols=usecols, chunksize=int(chunk_size)):
        if remaining is not None:
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining].copy()
        yield chunk
        if remaining is not None:
            remaining -= int(len(chunk))


def _extract_posterior_draws(*, trace_nc: Path, n_draws: int) -> PosteriorDraws:
    idata = az.from_netcdf(trace_nc)
    if not hasattr(idata, "posterior"):
        raise SystemExit(f"Trace missing posterior group: {trace_nc}")
    post = idata.posterior.stack(sample=("chain", "draw"))
    if "sigma2" not in post:
        raise SystemExit(f"Trace missing sigma2 variable: {trace_nc}")

    n_samples = int(post.sizes["sample"])
    if int(n_draws) > n_samples:
        raise SystemExit(f"--n-ppc-draws={n_draws} exceeds available samples={n_samples}")
    post = post.isel(sample=slice(0, int(n_draws)))

    def _get_1d(name: str) -> np.ndarray | None:
        if name not in post:
            return None
        arr = np.asarray(post[name].to_numpy(), dtype=np.float64).reshape(-1)
        if arr.ndim != 1:
            raise SystemExit(f"Expected {name} to be 1D after stacking, got shape {arr.shape}")
        return arr

    def _get_2d(name: str) -> np.ndarray | None:
        if name not in post:
            return None
        arr = post[name].transpose("sample", ...).to_numpy()
        out = np.asarray(arr, dtype=np.float64)
        if out.ndim != 2:
            raise SystemExit(f"Expected {name} to have shape (samples, dim), got {out.shape}")
        return out

    def _get_3d(name: str) -> np.ndarray | None:
        if name not in post:
            return None
        arr = post[name].transpose("sample", ...).to_numpy()
        out = np.asarray(arr, dtype=np.float64)
        if out.ndim != 3:
            raise SystemExit(
                f"Expected {name} to have shape (samples, dim0, dim1), got {out.shape}"
            )
        return out

    sigma2 = _get_1d("sigma2")
    assert sigma2 is not None
    return PosteriorDraws(
        sigma2=sigma2,
        b=_get_2d("b"),
        mu_cluster=_get_2d("mu_cluster"),
        w0=_get_2d("w0"),
        w_cluster=_get_3d("w_cluster"),
        alpha_class=_get_2d("alpha_class"),
        theta=_get_2d("theta"),
        t0=_get_1d("t0"),
        tau_b=_get_1d("tau_b"),
        tau_t=_get_1d("tau_t"),
        tau_w=_get_1d("tau_w"),
        t_chem=_get_2d("t_chem"),
    )


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


def _load_sampleset_to_species_group(mapping_csv: Path) -> dict[int, str]:
    if not mapping_csv.exists():
        raise SystemExit(f"Species mapping not found: {mapping_csv}")
    df = pd.read_csv(mapping_csv)
    if not {"sample_set_id", "species_group_raw"}.issubset(df.columns):
        raise SystemExit(
            f"Species mapping {mapping_csv} missing required columns 'sample_set_id'/'species_group_raw'"
        )
    out: dict[int, str] = {}
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
    *, group_labels: np.ndarray, chem_ids: np.ndarray, out_sets: dict[str, set[int]]
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


def _precompute_collapsed_slope_terms(
    *, stage1: Stage1CoeffSummaries
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if stage1.beta_cov is None:
        raise SystemExit(
            "Stage1CoeffSummaries missing beta_cov (required for posterior predictive eval)."
        )

    beta_hat = np.asarray(stage1.beta_hat, dtype=np.float64)
    beta_cov = np.asarray(stage1.beta_cov, dtype=np.float64)
    sigma2_ref = np.asarray(stage1.sigma2_mean, dtype=np.float64)

    if beta_cov.ndim != 3 or beta_hat.ndim != 2:
        raise SystemExit(
            "Invalid coefficient summaries: expected beta_hat (G,D) and beta_cov (G,D,D)"
        )
    if beta_cov.shape[0] != beta_hat.shape[0]:
        raise SystemExit("Invalid coefficient summaries: beta_cov and beta_hat size mismatch")
    if sigma2_ref.shape != (beta_hat.shape[0],):
        raise SystemExit("Invalid coefficient summaries: sigma2_mean has wrong shape")

    var_b = beta_cov[:, 0, 0]
    if not np.all(np.isfinite(var_b)) or np.any(var_b <= 0):
        raise SystemExit("Invalid beta_cov: intercept variance must be finite and > 0")

    sigma2_ref = np.maximum(sigma2_ref, 1e-12)
    s = sigma2_ref / var_b  # Schur complement scalar per group (S)

    cov_bw = beta_cov[:, 0, 1:]  # (G, P)
    wsx = -cov_bw * (s[:, None] / sigma2_ref[:, None])  # wsx = A^{-1} sx

    slopes_cov = beta_cov[:, 1:, 1:]  # (G, P, P)
    wsx_outer = np.einsum("ni,nj->nij", wsx, wsx, optimize=True)
    a_inv = slopes_cov / sigma2_ref[:, None, None] - wsx_outer / s[:, None, None]

    w0 = beta_hat[:, 1:] + wsx * beta_hat[:, 0][:, None]
    return a_inv.astype(np.float32), w0.astype(np.float32), wsx.astype(np.float32)


def _load_chem_embeddings(emb_path: Path) -> tuple[np.ndarray, np.ndarray]:
    try:
        from src.compassign.utils import load_chemberta_pca20  # type: ignore
    except Exception as exc:  # noqa: BLE001
        raise SystemExit("Failed to import ChemBERTa embedding loader") from exc

    emb = load_chemberta_pca20(emb_path)
    emb_ids = np.asarray(emb.chem_id, dtype=np.int64)
    emb_z = np.asarray(emb.features, dtype=np.float64)
    order = np.argsort(emb_ids, kind="mergesort")
    return emb_ids[order], emb_z[order]


def main() -> None:
    args = parse_args()
    if not (0.0 < float(args.interval) < 1.0):
        raise SystemExit("--interval must be in (0, 1)")

    train_output_dir = args.train_output_dir
    if not train_output_dir.is_absolute():
        train_output_dir = (REPO_ROOT / train_output_dir).resolve()
    cfg_path, coeff_path, trace_path = _resolve_paths(train_output_dir)

    test_csv = args.test_csv
    if not test_csv.is_absolute():
        test_csv = (REPO_ROOT / test_csv).resolve()
    if not test_csv.exists():
        raise SystemExit(f"Test CSV not found: {test_csv}")

    coeff = Stage1CoeffSummaries.load_npz(coeff_path)

    write_by_species_group = bool(args.write_by_species_group)
    ssid_to_group: dict[int, str] | None = None
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

    feature_cols = list(coeff.feature_names)
    missing_feats = [c for c in feature_cols if c not in header.columns]
    if missing_feats:
        raise SystemExit(f"CSV missing required run covariate columns: {missing_feats}")

    require_seen_group = bool(args.require_seen_group)
    backoff_mean_mode = str(args.backoff_mean_mode)
    backoff_slope_mean_mode = str(args.backoff_slope_mean_mode)
    backoff: ChemHierBackoffSummaries | None = None
    backoff_npz: Path | None = None
    if args.backoff_npz is not None:
        backoff_npz = args.backoff_npz
        if not backoff_npz.is_absolute():
            backoff_npz = (REPO_ROOT / backoff_npz).resolve()
        if not backoff_npz.exists():
            raise SystemExit(f"Backoff artifact not found: {backoff_npz}")
        backoff = ChemHierBackoffSummaries.load_npz(backoff_npz)

    emb_ids: np.ndarray | None = None
    emb_z: np.ndarray | None = None
    backoff_z_center: np.ndarray | None = None
    if backoff is not None and not require_seen_group and backoff_mean_mode == "chem":
        emb_path = args.chem_embeddings_path
        if emb_path is None:
            if backoff.embeddings_path is None:
                raise SystemExit(
                    "--chem-embeddings-path is required when --backoff-npz has no stored embeddings_path"
                )
            emb_path = Path(backoff.embeddings_path)
        if not emb_path.is_absolute():
            emb_path = (REPO_ROOT / emb_path).resolve()
        emb_ids, emb_z = _load_chem_embeddings(emb_path)
        if backoff.z_center is not None:
            backoff_z_center = np.asarray(backoff.z_center, dtype=np.float64)

    t_trace_start = time.time()
    draws = _extract_posterior_draws(trace_nc=trace_path, n_draws=int(args.n_ppc_draws))
    t_trace = time.time() - t_trace_start

    sigma2_draws = draws.sigma2
    b_draws = draws.b
    has_b = b_draws is not None
    if has_b and int(b_draws.shape[1]) != int(coeff.group_keys.size):
        raise SystemExit(
            f"Trace b has shape {b_draws.shape}, expected (n_draws, n_groups={coeff.group_keys.size})"
        )

    backoff_enabled = backoff is not None and not require_seen_group
    if backoff_enabled:
        if draws.mu_cluster is None or draws.theta is None or draws.t0 is None:
            raise SystemExit(
                "Backoff prediction requires trace variables mu_cluster, theta, and t0 (missing from trace)."
            )

    effective_backoff_slope_mode = backoff_slope_mean_mode
    if backoff_enabled and backoff_slope_mean_mode == "auto":
        if draws.w_cluster is not None or (backoff is not None and backoff.w_cluster is not None):
            effective_backoff_slope_mode = "cluster_head"
        else:
            effective_backoff_slope_mode = "zero"
    if backoff_enabled and effective_backoff_slope_mode == "cluster_head":
        if draws.w_cluster is None and (backoff is None or backoff.w_cluster is None):
            raise SystemExit(
                "backoff_slope_mean_mode='cluster_head' requires w_cluster in the trace or backoff artifact."
            )

    a_inv = None
    w0 = None
    wsx = None
    t_pre = 0.0
    if has_b:
        t1 = time.time()
        a_inv, w0, wsx = _precompute_collapsed_slope_terms(stage1=coeff)
        t_pre = time.time() - t1

    alpha_lo = 0.5 - 0.5 * float(args.interval)
    alpha_hi = 0.5 + 0.5 * float(args.interval)

    rng = np.random.default_rng(int(args.seed))

    global_stats = AggStats()
    cluster_stats: dict[int, AggStats] = {}
    group_stats: dict[str, AggStats] = {}
    group_compounds_total: dict[str, set[int]] = {}
    group_compounds_modeled: dict[str, set[int]] = {}
    total_rows_seen = 0
    used_rows = 0
    skipped_missing_group = 0
    skipped_no_group = 0
    backoff_used = 0
    backoff_skipped = 0

    keys = np.asarray(coeff.group_keys, dtype=np.int64)
    beta_hat = np.asarray(coeff.beta_hat, dtype=np.float64)
    beta_cov = np.asarray(coeff.beta_cov, dtype=np.float64) if coeff.beta_cov is not None else None
    sigma2_ref = np.asarray(coeff.sigma2_mean, dtype=np.float64)

    backoff_cluster_ids = None
    backoff_mu = None
    backoff_chem_ids = None
    backoff_class_ids = None
    backoff_alpha_class = None
    backoff_t_mean = None
    backoff_tau_t2 = None
    backoff_lambda = None
    backoff_w_mean_global = None
    backoff_w_mean_by_cluster = None
    if backoff_enabled:
        backoff_cluster_ids = np.asarray(backoff.cluster_ids, dtype=np.int64)
        backoff_mu = np.asarray(backoff.mu_cluster, dtype=np.float64)
        backoff_chem_ids = np.asarray(backoff.chem_ids, dtype=np.int64)
        if backoff.class_ids is not None and backoff.alpha_class is not None:
            backoff_class_ids = np.asarray(backoff.class_ids, dtype=np.int64)
            backoff_alpha_class = np.asarray(backoff.alpha_class, dtype=np.float64)
        backoff_t_mean = np.asarray(backoff.t_chem, dtype=np.float64)
        backoff_tau_t2 = float(backoff.tau_t) ** 2
        backoff_lambda = float(backoff.lambda_slopes)
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

    usecols = list(
        dict.fromkeys(
            [*REQUIRED_COLS, *(SPECIES_GROUP_COLS if write_by_species_group else []), *feature_cols]
        )
    )
    dtype: dict[str, object] = {c: np.float32 for c in feature_cols}
    dtype.update(
        {"rt": np.float32, "compound": np.int64, "species_cluster": np.int64, "comp_id": np.int64}
    )

    t_eval_start = time.time()
    for chunk in _iter_test_chunks(
        test_csv=test_csv,
        usecols=usecols,
        chunk_size=int(args.chunk_size),
        max_rows=int(args.max_test_rows),
    ):
        total_rows_seen += int(len(chunk))
        chunk = chunk.astype(dtype, errors="ignore").copy()

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
        key_arr = _encode_cluster_comp_key(cluster_arr, comp_arr)

        idx = np.searchsorted(keys, key_arr)
        ok = (idx >= 0) & (idx < keys.size)
        if ok.any():
            ok[ok] &= keys[idx[ok]] == key_arr[ok]
        missing = ~ok

        y = chunk["rt"].to_numpy(dtype=np.float64, copy=False)
        chem_id_arr = chunk["compound"].astype(int).to_numpy(dtype=np.int64, copy=False)
        class_id_arr = (
            chunk["compound_class"].fillna(-1).astype(int).to_numpy(dtype=np.int64, copy=False)
        )
        x = chunk[feature_cols].to_numpy(dtype=np.float32, copy=True)
        np.nan_to_num(x, copy=False)
        if coeff.feature_center is not None:
            x = x - np.asarray(coeff.feature_center, dtype=np.float32)[None, :]
        if coeff.feature_rotation is not None:
            x = x @ np.asarray(coeff.feature_rotation, dtype=np.float32)

        pred_parts = []
        covered_parts = []
        y_parts = []
        cluster_parts = []
        chem_parts = []
        group_parts = []

        if ok.any():
            idx_ok = idx[ok].astype(np.int64, copy=False)
            y_ok = y[ok]
            x_ok = x[ok]
            cluster_ok = cluster_arr[ok]
            chem_ok = chem_id_arr[ok]
            group_ok = group_raw_arr[ok] if group_raw_arr is not None else None

            if has_b:
                assert (
                    a_inv is not None and w0 is not None and wsx is not None and b_draws is not None
                )
                w0_row = w0[idx_ok]  # (N, P)
                wsx_row = wsx[idx_ok]  # (N, P)
                m0 = np.sum(w0_row * x_ok, axis=1, dtype=np.float64)  # (N,)
                m1 = 1.0 - np.sum(wsx_row * x_ok, axis=1, dtype=np.float64)  # (N,)

                a_inv_row = a_inv[idx_ok].astype(np.float64, copy=False)
                x64 = x_ok.astype(np.float64, copy=False)
                q = np.einsum("ni,nij,nj->n", x64, a_inv_row, x64, optimize=True)  # (N,)

                b_rows = b_draws[:, idx_ok]  # (D, N)
                mean = m0[None, :] + m1[None, :] * b_rows  # (D, N)
                var = sigma2_draws[:, None] * (1.0 + q[None, :])
            else:
                if beta_cov is None:
                    raise SystemExit(
                        "Stage1CoeffSummaries missing beta_cov (required for posterior predictive eval)."
                    )
                x1 = np.concatenate(
                    [np.ones((x_ok.shape[0], 1), dtype=np.float64), x_ok.astype(np.float64)],
                    axis=1,
                )
                beta = beta_hat[idx_ok]
                mean0 = beta[:, 0] + np.sum(beta[:, 1:] * x_ok.astype(np.float64), axis=1)
                cov0 = beta_cov[idx_ok]
                var_coef_ref = np.einsum("ni,nij,nj->n", x1, cov0, x1, optimize=True)
                scale = var_coef_ref / np.maximum(sigma2_ref[idx_ok], 1e-12)
                mean = mean0[None, :]
                var = sigma2_draws[:, None] * (1.0 + scale[None, :])

            std = np.sqrt(np.maximum(var, 0.0))
            z = rng.standard_normal(size=mean.shape, dtype=np.float64)
            y_ppc = mean + std * z
            lo = np.quantile(y_ppc, alpha_lo, axis=0)
            hi = np.quantile(y_ppc, alpha_hi, axis=0)

            pred_mean = mean.mean(axis=0)
            covered = (y_ok >= lo) & (y_ok <= hi)

            pred_parts.append(pred_mean)
            covered_parts.append(covered)
            y_parts.append(y_ok)
            cluster_parts.append(cluster_ok)
            chem_parts.append(chem_ok)
            if group_ok is not None:
                group_parts.append(group_ok)

        if missing.any():
            if not backoff_enabled:
                if require_seen_group or backoff is None:
                    skipped_missing_group += int(missing.sum())
            else:
                assert (
                    backoff_cluster_ids is not None
                    and backoff_mu is not None
                    and backoff_chem_ids is not None
                    and backoff_t_mean is not None
                    and backoff_tau_t2 is not None
                    and backoff_lambda is not None
                )
                mu_cluster_draws = draws.mu_cluster
                theta_draws = draws.theta
                t0_draws = draws.t0
                if mu_cluster_draws is None or theta_draws is None or t0_draws is None:
                    raise SystemExit("Backoff enabled but required trace variables are missing.")

                tau_b_draws = draws.tau_b
                if tau_b_draws is None:
                    tau_b_draws = np.full_like(sigma2_draws, float(backoff.tau_b), dtype=np.float64)
                tau_b2 = np.square(tau_b_draws)

                tau_t_draws = draws.tau_t
                if tau_t_draws is None:
                    tau_t2 = float(backoff_tau_t2)
                else:
                    tau_t2 = np.square(tau_t_draws)  # (D,)

                x_m = x[missing].astype(np.float64, copy=False)
                y_m = y[missing]
                cl_m = cluster_arr[missing]
                chem_m = chem_id_arr[missing]
                class_m = class_id_arr[missing]
                group_m = group_raw_arr[missing] if group_raw_arr is not None else None

                c_idx = np.searchsorted(backoff_cluster_ids, cl_m)
                c_ok = (c_idx >= 0) & (c_idx < backoff_cluster_ids.size)
                if c_ok.any():
                    c_ok[c_ok] &= backoff_cluster_ids[c_idx[c_ok]] == cl_m[c_ok]

                if not c_ok.any():
                    backoff_skipped += int(cl_m.size)
                else:
                    x_m = x_m[c_ok]
                    y_m = y_m[c_ok]
                    cl_m = cl_m[c_ok]
                    chem_m = chem_m[c_ok]
                    class_m = class_m[c_ok]
                    c_idx = c_idx[c_ok]
                    if group_m is not None:
                        group_m = group_m[c_ok]

                    n_draws = int(sigma2_draws.size)
                    slope_mean_term = np.zeros((n_draws, int(x_m.shape[0])), dtype=np.float64)
                    if effective_backoff_slope_mode == "global":
                        if backoff_w_mean_global is None:
                            raise SystemExit("Backoff global slope mean not initialized")
                        slope_mean_term[:] = (x_m @ backoff_w_mean_global)[None, :]
                    elif effective_backoff_slope_mode == "cluster":
                        if backoff_w_mean_by_cluster is None:
                            raise SystemExit("Backoff cluster slope means not initialized")
                        w_mean = backoff_w_mean_by_cluster[c_idx]
                        slope_mean_term[:] = np.sum(w_mean * x_m, axis=1, dtype=np.float64)[None, :]
                    elif effective_backoff_slope_mode == "cluster_head":
                        if draws.w_cluster is not None:
                            w_cl = draws.w_cluster[:, c_idx, :]  # (D, N, P)
                            slope_mean_term = np.einsum("np,dnp->dn", x_m, w_cl, optimize=True)
                        else:
                            assert backoff is not None and backoff.w_cluster is not None
                            w_mean = np.asarray(backoff.w_cluster, dtype=np.float64)[c_idx]
                            slope_mean_term[:] = np.sum(w_mean * x_m, axis=1, dtype=np.float64)[
                                None, :
                            ]

                    mu_part = mu_cluster_draws[:, c_idx]  # (D, N)

                    if backoff_mean_mode == "cluster_only":
                        t_mean = t0_draws[:, None]
                        var_b = tau_b2[:, None] + (
                            tau_t2[:, None] if isinstance(tau_t2, np.ndarray) else float(tau_t2)
                        )
                    else:
                        if emb_ids is None or emb_z is None:
                            raise SystemExit(
                                "Backoff embeddings not initialized (backoff_mean_mode='chem')"
                            )
                        t_mean = np.empty(
                            (int(sigma2_draws.size), int(chem_m.size)), dtype=np.float64
                        )

                        k_idx = np.searchsorted(backoff_chem_ids, chem_m)
                        k_ok = (k_idx >= 0) & (k_idx < backoff_chem_ids.size)
                        if k_ok.any():
                            k_ok[k_ok] &= backoff_chem_ids[k_idx[k_ok]] == chem_m[k_ok]

                        if k_ok.any():
                            if draws.t_chem is not None:
                                t_mean[:, k_ok] = draws.t_chem[:, k_idx[k_ok]]
                            else:
                                t_mean[:, k_ok] = backoff_t_mean[k_idx[k_ok]][None, :]

                        if (~k_ok).any():
                            chem_new = chem_m[~k_ok]
                            class_new = class_m[~k_ok]
                            emb_idx = np.searchsorted(emb_ids, chem_new)
                            e_ok = (emb_idx >= 0) & (emb_idx < emb_ids.size)
                            if e_ok.any():
                                e_ok[e_ok] &= emb_ids[emb_idx[e_ok]] == chem_new[e_ok]
                            z_new = np.zeros(
                                (int(chem_new.size), int(emb_z.shape[1])), dtype=np.float64
                            )
                            if e_ok.any():
                                z_new[e_ok] = emb_z[emb_idx[e_ok]]
                            if backoff_z_center is not None:
                                z_new = z_new - backoff_z_center[None, :]
                            class_term = np.zeros(
                                (int(sigma2_draws.size), int(class_new.size)), dtype=np.float64
                            )
                            if draws.alpha_class is not None and backoff_class_ids is not None:
                                cls_idx = np.searchsorted(backoff_class_ids, class_new)
                                ok_cls = (cls_idx >= 0) & (cls_idx < backoff_class_ids.size)
                                if ok_cls.any():
                                    ok_cls[ok_cls] &= (
                                        backoff_class_ids[cls_idx[ok_cls]] == class_new[ok_cls]
                                    )
                                if ok_cls.any():
                                    class_term[:, ok_cls] = draws.alpha_class[:, cls_idx[ok_cls]]
                            elif backoff_class_ids is not None and backoff_alpha_class is not None:
                                cls_idx = np.searchsorted(backoff_class_ids, class_new)
                                ok_cls = (cls_idx >= 0) & (cls_idx < backoff_class_ids.size)
                                if ok_cls.any():
                                    ok_cls[ok_cls] &= (
                                        backoff_class_ids[cls_idx[ok_cls]] == class_new[ok_cls]
                                    )
                                if ok_cls.any():
                                    class_term[:, ok_cls] = backoff_alpha_class[cls_idx[ok_cls]][
                                        None, :
                                    ]
                            t_mean_new = t0_draws[:, None] + class_term + (theta_draws @ z_new.T)
                            t_mean[:, ~k_ok] = t_mean_new

                        tau_t2_broadcast = (
                            tau_t2[:, None] if isinstance(tau_t2, np.ndarray) else float(tau_t2)
                        )
                        var_b = tau_b2[:, None] + tau_t2_broadcast * (~k_ok)[None, :]

                    mean_m = mu_part + t_mean + slope_mean_term
                    sum_x2 = np.sum(np.square(x_m), axis=1, dtype=np.float64)  # (N,)
                    var_w = sigma2_draws[:, None] * (sum_x2[None, :] / float(backoff_lambda))
                    var_m = sigma2_draws[:, None] + var_b + var_w

                    std_m = np.sqrt(np.maximum(var_m, 0.0))
                    z_m = rng.standard_normal(size=mean_m.shape, dtype=np.float64)
                    y_ppc_m = mean_m + std_m * z_m
                    lo_m = np.quantile(y_ppc_m, alpha_lo, axis=0)
                    hi_m = np.quantile(y_ppc_m, alpha_hi, axis=0)
                    pred_mean_m = mean_m.mean(axis=0)
                    covered_m = (y_m >= lo_m) & (y_m <= hi_m)

                    pred_parts.append(pred_mean_m)
                    covered_parts.append(covered_m)
                    y_parts.append(y_m)
                    cluster_parts.append(cl_m)
                    chem_parts.append(chem_m)
                    if group_m is not None:
                        group_parts.append(group_m)
                    backoff_used += int(y_m.size)

        if not pred_parts:
            continue

        pred_mean_used = np.concatenate(pred_parts, axis=0)
        covered_used = np.concatenate(covered_parts, axis=0)
        y_used = np.concatenate(y_parts, axis=0)
        cluster_used = np.concatenate(cluster_parts, axis=0)
        chem_used = np.concatenate(chem_parts, axis=0)
        used_rows += int(y_used.size)

        global_stats.update(pred_mean=pred_mean_used, y_true=y_used, covered=covered_used)

        for cl in np.unique(cluster_used).tolist():
            mask = cluster_used == int(cl)
            stats = cluster_stats.setdefault(int(cl), AggStats())
            stats.update(
                pred_mean=pred_mean_used[mask],
                y_true=y_used[mask],
                covered=covered_used[mask],
            )

        if write_by_species_group and group_parts:
            group_used = np.concatenate(group_parts, axis=0)
            for g in np.unique(group_used).tolist():
                mask = group_used == g
                stats = group_stats.setdefault(str(g), AggStats())
                stats.update(
                    pred_mean=pred_mean_used[mask],
                    y_true=y_used[mask],
                    covered=covered_used[mask],
                )
            _update_group_compound_sets(
                group_labels=group_used,
                chem_ids=chem_used,
                out_sets=group_compounds_modeled,
            )

    t_eval = time.time() - t_eval_start

    metrics = global_stats.to_metrics()
    print(
        f"[ppc_eval] Global: n={metrics['n_obs']:,}, RMSE={metrics['rmse']:.4f}, "
        f"coverage95={metrics['coverage_95']:.3f}"
    )
    if backoff_enabled:
        print(
            f"[ppc_eval] Backoff: used={backoff_used:,}, skipped={backoff_skipped:,}, "
            f"mean_mode={backoff_mean_mode}, slope_mean_mode={effective_backoff_slope_mode}"
        )
    print(
        f"[ppc_eval] Timing: load_trace={t_trace:.2f}s, precompute={t_pre:.2f}s, "
        f"eval={t_eval:.2f}s (rows_seen={total_rows_seen:,}, rows_used={used_rows:,})"
    )

    results_dir = train_output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / f"rt_eval_ppc_{args.label}.json"
    out_json.write_text(
        json.dumps(
            {
                "metrics": metrics,
                "n_test": int(total_rows_seen),
                "n_used": int(used_rows),
                "chunk_size": int(args.chunk_size),
                "max_test_rows": int(args.max_test_rows),
                "interval": float(args.interval),
                "n_ppc_draws": int(args.n_ppc_draws),
                "train_output_dir": str(train_output_dir),
                "config_json": str(cfg_path),
                "test_csv": str(test_csv),
                "trace_path": str(trace_path),
                "coeff_npz": str(coeff_path),
                "has_b": bool(has_b),
                "require_seen_group": bool(require_seen_group),
                "backoff_npz": str(backoff_npz) if backoff_npz is not None else None,
                "backoff_mean_mode": backoff_mean_mode,
                "backoff_slope_mean_mode": backoff_slope_mean_mode,
                "backoff_slope_mean_mode_effective": effective_backoff_slope_mode,
                "backoff_used": int(backoff_used),
                "backoff_skipped": int(backoff_skipped),
                "skipped_missing_group": int(skipped_missing_group),
                "skipped_no_species_group": int(skipped_no_group),
                "timing_s": {"load_trace": t_trace, "precompute": t_pre, "eval": t_eval},
            },
            indent=2,
        )
    )
    print(f"[ppc_eval] Wrote {out_json}")

    out_csv = results_dir / f"rt_eval_ppc_by_species_cluster_{args.label}.csv"
    rows = []
    for cl, stats in sorted(cluster_stats.items(), key=lambda kv: int(kv[0])):
        m = stats.to_metrics()
        rows.append({"species_cluster": int(cl), **m})
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[ppc_eval] Wrote {out_csv}")

    if write_by_species_group:
        out_group_csv = results_dir / f"rt_eval_ppc_by_species_group_{args.label}.csv"
        group_rows = []
        for label, stats in sorted(group_stats.items()):
            m = stats.to_metrics()
            total_set = group_compounds_total.get(label, set())
            modeled_set = group_compounds_modeled.get(label, set())
            n_total = len(total_set)
            n_modeled = len(modeled_set)
            coverage_comp = float(n_modeled / n_total) if n_total > 0 else float("nan")
            group_rows.append(
                {
                    "species_group_raw": label,
                    **m,
                    "n_compounds_total": int(n_total),
                    "n_compounds_modeled": int(n_modeled),
                    "compound_coverage": coverage_comp,
                }
            )
        pd.DataFrame(group_rows).to_csv(out_group_csv, index=False)
        print(f"[ppc_eval] Wrote {out_group_csv}")


if __name__ == "__main__":
    main()
