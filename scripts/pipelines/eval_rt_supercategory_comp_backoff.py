#!/usr/bin/env python3
"""
Evaluate a supercategory Stage1CoeffSummaries artifact on a production RT CSV, with comp-id backoff.

The supercategory ridge artifact contains coefficients keyed by:
  (species_cluster, comp_id)

When a (species_cluster, comp_id) group is missing at evaluation time, this script backs off to a
comp_id-only coefficient obtained by averaging all seen (species_cluster, comp_id) groups for that
comp_id (weighted by n_obs_train in the artifact).
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

from src.compassign.rt.pymc_collapsed_ridge import Stage1CoeffSummaries  # noqa: E402

BASE_REQUIRED_COLS = ["rt", "comp_id", "species_cluster"]

_POLY2_SQ_RE = re.compile(r"^poly2_sq\((?P<name>[^)]+)\)$")
_POLY2_INT_RE = re.compile(r"^poly2_int\((?P<a>[^,]+),(?P<b>[^)]+)\)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate supercategory Stage1CoeffSummaries with comp_id backoff.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--coeff-npz", type=Path, required=True, help="Stage1CoeffSummaries .npz.")
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


def _build_comp_backoff(
    stage1: Stage1CoeffSummaries,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, float, np.ndarray | None
]:
    """Compute weighted comp_id-only averages over (species_cluster, comp_id) groups.

    Returns:
      comp_ids_unique: (K,) int64 sorted
      beta_hat_comp: (K, D) float
      sigma2_comp: (K,) float
      beta_cov_comp: (K, D, D) float | None
      beta_hat_global: (D,) float
      sigma2_global: float
      beta_cov_global: (D, D) float | None
    """
    comp_id = np.asarray(stage1.comp_id, dtype=np.int64)
    n_obs = np.asarray(stage1.n_obs, dtype=np.float64)
    beta_hat = np.asarray(stage1.beta_hat, dtype=np.float64)
    sigma2 = np.asarray(stage1.sigma2_mean, dtype=np.float64)
    beta_cov = stage1.beta_cov

    order = np.argsort(comp_id, kind="mergesort")
    comp_sorted = comp_id[order]
    bounds = np.flatnonzero(np.diff(comp_sorted)) + 1
    starts = np.concatenate([np.asarray([0], dtype=np.int64), bounds.astype(np.int64)])
    ends = np.concatenate([bounds.astype(np.int64), np.asarray([comp_sorted.size], dtype=np.int64)])

    comp_ids_unique = comp_sorted[starts].astype(np.int64, copy=False)
    d = int(beta_hat.shape[1])

    beta_hat_comp = np.empty((int(comp_ids_unique.size), d), dtype=np.float64)
    sigma2_comp = np.empty((int(comp_ids_unique.size),), dtype=np.float64)
    beta_cov_comp: np.ndarray | None = None
    if beta_cov is not None:
        beta_cov_comp = np.empty((int(comp_ids_unique.size), d, d), dtype=np.float64)

    for i, (start, end) in enumerate(zip(starts.tolist(), ends.tolist(), strict=True)):
        idx = order[start:end]
        w = np.maximum(n_obs[idx], 1.0)
        w_sum = float(np.sum(w))
        beta_hat_comp[i] = np.sum(beta_hat[idx] * w[:, None], axis=0) / w_sum
        sigma2_comp[i] = float(np.sum(sigma2[idx] * w) / w_sum)
        if beta_cov_comp is not None and beta_cov is not None:
            cov = np.asarray(beta_cov, dtype=np.float64)[idx]
            beta_cov_comp[i] = np.sum(cov * w[:, None, None], axis=0) / w_sum

    w_all = np.maximum(n_obs, 1.0)
    w_all_sum = float(np.sum(w_all))
    beta_hat_global = np.sum(beta_hat * w_all[:, None], axis=0) / w_all_sum
    sigma2_global = float(np.sum(sigma2 * w_all) / w_all_sum)
    beta_cov_global = None
    if beta_cov is not None:
        beta_cov_global = (
            np.sum(np.asarray(beta_cov, dtype=np.float64) * w_all[:, None, None], axis=0)
            / w_all_sum
        )

    return (
        comp_ids_unique,
        beta_hat_comp,
        sigma2_comp,
        beta_cov_comp,
        beta_hat_global,
        sigma2_global,
        beta_cov_global,
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

    test_csv = args.test_csv
    if not test_csv.is_absolute():
        test_csv = (REPO_ROOT / test_csv).resolve()

    out_dir = args.output_dir or _default_output_dir(coeff_npz)
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    label_suffix = f"_{args.label}" if args.label else ""
    out_json = out_dir / f"rt_eval_supercategory_comp_backoff{label_suffix}.json"

    group_col = str(getattr(stage1, "group_col", "species_cluster"))
    if group_col != "species_cluster":
        raise SystemExit(
            f"Stage1CoeffSummaries group_col must be 'species_cluster' (got {group_col!r})"
        )

    alpha = 0.5 + 0.5 * float(args.interval)
    q_norm = float(NormalDist().inv_cdf(alpha))
    pred_std_scale = float(args.pred_std_scale)

    keys = np.asarray(stage1.group_keys, dtype=np.int64)
    beta_hat = np.asarray(stage1.beta_hat, dtype=np.float64)
    beta_var_diag = np.asarray(stage1.beta_var_diag, dtype=np.float64)
    beta_cov = stage1.beta_cov
    sigma2 = np.asarray(stage1.sigma2_mean, dtype=np.float64)

    (
        comp_ids_unique,
        beta_hat_comp,
        sigma2_comp,
        beta_cov_comp,
        beta_hat_global,
        sigma2_global,
        beta_cov_global,
    ) = _build_comp_backoff(stage1)

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

    total_rows = 0
    max_rows = int(args.max_test_rows)
    remaining = max_rows if max_rows > 0 else None

    stats_all = SumStats()
    stats_seen = SumStats()
    stats_backoff = SumStats()
    n_seen = 0
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

        cluster = chunk["species_cluster"].astype(int).to_numpy(dtype=np.int64, copy=False)
        comp = chunk["comp_id"].astype(int).to_numpy(dtype=np.int64, copy=False)
        y = chunk["rt"].to_numpy(dtype=np.float64, copy=False)

        x = _design_matrix_from_chunk(
            chunk=chunk, design_features=design_features, raw_features=raw_features_in_order
        )
        if feature_center is not None:
            x = x - feature_center[None, :]
        if feature_rotation is not None:
            x = x @ feature_rotation
        x1 = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)

        key = (cluster << np.int64(32)) + comp
        idx = np.searchsorted(keys, key)
        ok = (idx >= 0) & (idx < keys.size)
        if ok.any():
            ok[ok] &= keys[idx[ok]] == key[ok]

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

        miss = ~ok
        if miss.any():
            comp_m = comp[miss]
            idx_comp = np.searchsorted(comp_ids_unique, comp_m)
            ok_comp = (idx_comp >= 0) & (idx_comp < comp_ids_unique.size)
            if ok_comp.any():
                ok_comp[ok_comp] &= comp_ids_unique[idx_comp[ok_comp]] == comp_m[ok_comp]
            n_miss = int(comp_m.size)
            pred_mean = np.empty((n_miss,), dtype=np.float64)
            pred_std = np.empty((n_miss,), dtype=np.float64)

            # Comp-id backoff for comp_ids seen in training.
            if ok_comp.any():
                take = np.flatnonzero(ok_comp)
                beta = beta_hat_comp[idx_comp[ok_comp]]
                pred_mean[take] = beta[:, 0] + np.sum(beta[:, 1:] * x[miss][ok_comp], axis=1)

                sigma2_y = np.maximum(sigma2_comp[idx_comp[ok_comp]], 1e-12)
                if beta_cov_comp is not None:
                    var_coef = _row_variance_from_beta_cov(
                        x1=x1[miss][ok_comp],
                        group_idx=idx_comp[ok_comp],
                        beta_cov=beta_cov_comp,
                    )
                else:
                    var_coef = np.zeros((int(take.size),), dtype=np.float64)
                pred_var = np.maximum(sigma2_y + np.maximum(var_coef, 0.0), 0.0)
                pred_std[take] = np.sqrt(pred_var) * pred_std_scale

            # Global fallback for completely unseen comp_ids.
            miss_comp = ~ok_comp
            if miss_comp.any():
                take = np.flatnonzero(miss_comp)
                beta0 = np.asarray(beta_hat_global, dtype=np.float64)
                pred_mean[take] = beta0[0] + np.sum(beta0[1:][None, :] * x[miss][miss_comp], axis=1)
                sigma2_y = max(float(sigma2_global), 1e-12)
                if beta_cov_global is not None:
                    var_coef = _row_variance_from_beta_cov(
                        x1=x1[miss][miss_comp],
                        group_idx=np.zeros((int(take.size),), dtype=np.int64),
                        beta_cov=beta_cov_global[None, :, :],
                    )
                else:
                    var_coef = np.zeros((int(take.size),), dtype=np.float64)
                pred_var = np.maximum(sigma2_y + np.maximum(var_coef, 0.0), 0.0)
                pred_std[take] = np.sqrt(pred_var) * pred_std_scale

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
                f"seen={n_seen:,} backoff={n_backoff:,}"
            )

    payload = {
        "coeff_npz": str(coeff_npz),
        "test_csv": str(test_csv),
        "rows_total": int(total_rows),
        "rows_seen": int(n_seen),
        "rows_backoff": int(n_backoff),
        "metrics_all": stats_all.to_metrics(),
        "metrics_seen": stats_seen.to_metrics(),
        "metrics_backoff": stats_backoff.to_metrics(),
    }
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"[eval] Wrote {out_json}")


if __name__ == "__main__":
    main()
