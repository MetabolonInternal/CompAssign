#!/usr/bin/env python3
"""
Evaluate per-group coefficient summaries (Stage1CoeffSummaries) on RT production CSVs, with tail slices.

This is a Stage1CoeffSummaries evaluator that adds:
  - per-group metrics, and
  - aggregation by *training support* (n_obs in the coefficient artifact).

Primary use: detect whether Bayesian pooling helps sparse groups even when overall RMSE barely moves.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .pymc_collapsed_ridge import Stage1CoeffSummaries

_HERE = Path(__file__).resolve()
for _parent in _HERE.parents:
    if (_parent / "pyproject.toml").exists():
        REPO_ROOT = _parent
        break
else:
    REPO_ROOT = Path.cwd().resolve()


BASE_REQUIRED_COLS = ["rt", "comp_id"]

_POLY2_SQ_RE = re.compile(r"^poly2_sq\((?P<name>[^)]+)\)$")
_POLY2_INT_RE = re.compile(r"^poly2_int\((?P<a>[^,]+),(?P<b>[^)]+)\)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Stage1CoeffSummaries with tail slices by training support.",
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
        "--max-test-rows",
        type=int,
        default=0,
        help="Maximum number of test rows to evaluate (0 = all).",
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
        "--require-seen-group",
        action="store_true",
        help=(
            "If set, error if any row has an unseen (group_id, comp_id) not present in coeff-npz. "
            "Otherwise, unseen groups are skipped."
        ),
    )
    parser.add_argument(
        "--support-bins",
        type=str,
        default="1,2,5,10,20,50,100",
        help=(
            "Comma-separated upper edges for training support bins over n_obs_train. "
            "Example: '1,2,5,10,20,50,100' -> bins [<=1, 2, 3-5, 6-10, 11-20, 21-50, 51-100, >100]."
        ),
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Optional label for this evaluation used in output filenames.",
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
    sum_pred_std: float = 0.0
    sum_interval_width: float = 0.0

    def update(
        self,
        *,
        err: np.ndarray,
        covered: np.ndarray,
        pred_std: np.ndarray,
        interval_width: np.ndarray,
    ) -> None:
        if err.size == 0:
            return
        self.n += int(err.size)
        self.sum_sq_err += float(np.sum(np.square(err)))
        self.sum_abs_err += float(np.sum(np.abs(err)))
        self.sum_covered += float(np.sum(covered.astype(np.float64, copy=False)))
        self.sum_pred_std += float(np.sum(pred_std.astype(np.float64, copy=False)))
        self.sum_interval_width += float(np.sum(interval_width.astype(np.float64, copy=False)))

    def to_metrics(self) -> Dict[str, float]:
        if self.n <= 0:
            return {
                "n_obs": 0,
                "rmse": float("nan"),
                "mae": float("nan"),
                "coverage_95": float("nan"),
                "pred_std_mean": float("nan"),
                "interval_width_mean": float("nan"),
            }
        rmse = float(np.sqrt(self.sum_sq_err / self.n))
        mae = float(self.sum_abs_err / self.n)
        cov = float(self.sum_covered / self.n)
        pred_std_mean = float(self.sum_pred_std / self.n)
        width_mean = float(self.sum_interval_width / self.n)
        return {
            "n_obs": int(self.n),
            "rmse": rmse,
            "mae": mae,
            "coverage_95": cov,
            "pred_std_mean": pred_std_mean,
            "interval_width_mean": width_mean,
        }


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


def _split_design_features(
    feature_names: Tuple[str, ...],
) -> tuple[list[str], list[str], bool]:
    """Split Stage1 design features into raw CSV columns and derived poly2 features.

    Returns:
      raw_features_in_order: list[str]  (all non-derived features, in design order)
      raw_dependencies: list[str]       (all raw columns needed to compute the design matrix)
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
    raw_dependencies = raw_features[:]
    for dep in sorted(raw_deps):
        if dep not in raw_dependencies:
            raw_dependencies.append(dep)
    return raw_features, raw_dependencies, has_derived


def _design_matrix_from_chunk(
    *,
    chunk: pd.DataFrame,
    design_features: Tuple[str, ...],
    raw_features: list[str],
    fill_value: float,
) -> np.ndarray:
    """Build design matrix X in the exact order of `design_features`.

    Supports poly2_sq(...) and poly2_int(...,...) derived features.
    """
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
        raise SystemExit(f"Unknown feature name in artifact: {name}")
    return x


def _parse_support_bins(spec: str) -> List[int]:
    vals: List[int] = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            v = int(part)
        except ValueError as exc:
            raise SystemExit(f"Invalid --support-bins value: {part!r}") from exc
        if v <= 0:
            raise SystemExit("--support-bins must contain positive integers")
        vals.append(v)
    if not vals:
        raise SystemExit("--support-bins must not be empty")
    vals = sorted(set(vals))
    return vals


def _bin_labels(edges: List[int]) -> List[str]:
    labels = [f"<= {edges[0]}"]
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        labels.append(f"{lo + 1}-{hi}")
    labels.append(f"> {edges[-1]}")
    return labels


def _group_rmse_stats(group_rmse: np.ndarray) -> Dict[str, float]:
    group_rmse = np.asarray(group_rmse, dtype=np.float64)
    group_rmse = group_rmse[np.isfinite(group_rmse)]
    if group_rmse.size == 0:
        return {
            "n_groups": 0,
            "rmse_mean": float("nan"),
            "rmse_std": float("nan"),
            "rmse_p50": float("nan"),
            "rmse_p90": float("nan"),
            "rmse_p99": float("nan"),
        }
    return {
        "n_groups": int(group_rmse.size),
        "rmse_mean": float(np.mean(group_rmse)),
        "rmse_std": float(np.std(group_rmse)),
        "rmse_p50": float(np.quantile(group_rmse, 0.50)),
        "rmse_p90": float(np.quantile(group_rmse, 0.90)),
        "rmse_p99": float(np.quantile(group_rmse, 0.99)),
    }


def _row_variance_from_beta_cov(
    *,
    x1: np.ndarray,
    group_idx: np.ndarray,
    beta_cov: np.ndarray,
) -> np.ndarray:
    """Compute per-row coefficient variance x1^T Cov[group] x1 without expanding Cov per row.

    Avoids allocating a (n_rows, D, D) tensor, which can be prohibitively large for poly2 models.
    """
    x1 = np.asarray(x1, dtype=np.float64)
    group_idx = np.asarray(group_idx, dtype=np.int64)
    if group_idx.ndim != 1:
        raise ValueError(f"group_idx must be 1D; got shape {group_idx.shape}")
    if x1.ndim != 2:
        raise ValueError(f"x1 must be 2D; got shape {x1.shape}")
    if x1.shape[0] != group_idx.size:
        raise ValueError(f"x1 rows ({x1.shape[0]}) must match group_idx size ({group_idx.size})")

    n = int(group_idx.size)
    if n == 0:
        return np.zeros((0,), dtype=np.float64)

    order = np.argsort(group_idx, kind="mergesort")
    idx_sorted = group_idx[order]
    x1_sorted = x1[order]

    # Identify contiguous runs of the same group index in the sorted order.
    bounds = np.flatnonzero(np.diff(idx_sorted)) + 1
    starts = np.concatenate([np.asarray([0], dtype=np.int64), bounds.astype(np.int64)])
    ends = np.concatenate([bounds.astype(np.int64), np.asarray([n], dtype=np.int64)])

    out_sorted = np.empty((n,), dtype=np.float64)
    for start, end in zip(starts.tolist(), ends.tolist(), strict=True):
        g = int(idx_sorted[start])
        cov_g = np.asarray(beta_cov[g], dtype=np.float64)
        x1_g = x1_sorted[start:end]
        # diag(X @ C @ X^T) = sum((X @ C) * X, axis=1)
        out_sorted[start:end] = np.sum((x1_g @ cov_g) * x1_g, axis=1, dtype=np.float64)

    out = np.empty((n,), dtype=np.float64)
    out[order] = out_sorted
    return out


def main() -> None:
    args = parse_args()
    if not (0.0 < float(args.interval) < 1.0):
        raise SystemExit("--interval must be in (0, 1)")
    if float(args.pred_std_scale) <= 0.0:
        raise SystemExit("--pred-std-scale must be > 0")

    coeff_npz = args.coeff_npz
    if not coeff_npz.is_absolute():
        coeff_npz = (REPO_ROOT / coeff_npz).resolve()
    if not coeff_npz.exists():
        raise SystemExit(f"Coefficient summaries not found: {coeff_npz}")
    stage1 = Stage1CoeffSummaries.load_npz(coeff_npz)

    test_csv = args.test_csv
    if not test_csv.is_absolute():
        test_csv = (REPO_ROOT / test_csv).resolve()
    if not test_csv.exists():
        raise SystemExit(f"Test CSV not found: {test_csv}")

    label_suffix = f"_{args.label}" if args.label else ""
    out_dir = args.output_dir or _default_output_dir(coeff_npz)
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / f"rt_eval_coeff_summaries_by_support{label_suffix}.json"
    out_group_csv = out_dir / f"rt_eval_coeff_summaries_by_group{label_suffix}.csv"
    out_support_csv = out_dir / f"rt_eval_coeff_summaries_by_support{label_suffix}.csv"

    alpha = 0.5 + 0.5 * float(args.interval)
    q_norm = float(NormalDist().inv_cdf(alpha))
    pred_std_scale = float(args.pred_std_scale)
    require_seen = bool(args.require_seen_group)

    group_col = str(getattr(stage1, "group_col", "species_cluster"))
    if group_col not in {"species_cluster", "species"}:
        raise SystemExit(f"Unsupported group_col in coeff artifact: {group_col}")

    group_cols: list[str]
    group_cols = [group_col]
    required_cols = [*BASE_REQUIRED_COLS, *group_cols]

    keys = np.asarray(stage1.group_keys, dtype=np.int64)
    group_ids = np.asarray(stage1.species_cluster, dtype=np.int64)
    comp_ids = np.asarray(stage1.comp_id, dtype=np.int64)
    n_obs_train = np.asarray(stage1.n_obs, dtype=np.int64)
    beta_hat = np.asarray(stage1.beta_hat, dtype=np.float64)
    beta_var_diag = np.asarray(stage1.beta_var_diag, dtype=np.float64)
    beta_cov = stage1.beta_cov
    sigma2 = np.asarray(stage1.sigma2_mean, dtype=np.float64)

    if beta_hat.shape[0] != keys.size:
        raise SystemExit("Invalid coefficient summaries: beta_hat and group_keys size mismatch")
    if beta_hat.shape[1] != beta_var_diag.shape[1]:
        raise SystemExit("Invalid coefficient summaries: beta_hat and beta_var_diag size mismatch")

    design_features = tuple(str(c) for c in stage1.feature_names)
    raw_features_in_order, raw_deps, has_derived = _split_design_features(design_features)

    feature_center = stage1.feature_center
    feature_rotation = stage1.feature_rotation
    if has_derived:
        if feature_center is not None or feature_rotation is not None:
            raise SystemExit(
                "Derived poly2_* features are not compatible with feature_center/feature_rotation; "
                "retrain without those transforms."
            )
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

    header = pd.read_csv(test_csv, nrows=0)
    missing_req = [c for c in required_cols if c not in header.columns]
    if missing_req:
        raise SystemExit(f"CSV missing required columns: {missing_req}")
    missing_feats = [c for c in raw_deps if c not in header.columns]
    if missing_feats:
        raise SystemExit(f"CSV missing required run covariate columns: {missing_feats}")

    usecols = list(dict.fromkeys([*required_cols, *raw_deps]))

    global_stats = SumStats()

    n_groups = int(keys.size)
    group_n = np.zeros((n_groups,), dtype=np.int64)
    group_sum_sq = np.zeros((n_groups,), dtype=np.float64)
    group_sum_abs = np.zeros((n_groups,), dtype=np.float64)
    group_sum_cov = np.zeros((n_groups,), dtype=np.float64)
    group_sum_pred_std = np.zeros((n_groups,), dtype=np.float64)
    group_sum_interval_width = np.zeros((n_groups,), dtype=np.float64)

    total_rows = 0
    used_rows = 0
    skipped_missing_group = 0
    max_rows = int(args.max_test_rows)
    remaining = max_rows if max_rows > 0 else None
    log_every_chunks = int(args.log_every_chunks)
    t0 = time.time()
    chunk_i = 0

    for chunk in pd.read_csv(test_csv, chunksize=int(args.chunk_size), usecols=usecols):
        chunk_i += 1
        if remaining is not None and remaining <= 0:
            break

        total_rows += int(len(chunk))
        if remaining is not None and len(chunk) > remaining:
            chunk = chunk.iloc[:remaining].copy()

        group_arr = chunk[group_col].astype(int).to_numpy(dtype=np.int64, copy=False)
        comp_arr = chunk["comp_id"].astype(int).to_numpy(dtype=np.int64, copy=False)
        key_arr = (group_arr << np.int64(32)) + comp_arr

        idx = np.searchsorted(keys, key_arr)
        ok = (idx >= 0) & (idx < keys.size)
        if ok.any():
            ok[ok] &= keys[idx[ok]] == key_arr[ok]
        if not ok.any():
            continue

        n_missing = int((~ok).sum())
        if n_missing > 0:
            skipped_missing_group += n_missing
            chunk = chunk.loc[ok].copy()
            if len(chunk) == 0:
                continue
            idx = idx[ok]

        y = chunk["rt"].to_numpy(dtype=np.float64, copy=False)
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
        x1 = np.concatenate([np.ones((x.shape[0], 1), dtype=np.float64), x], axis=1)

        idx_ok = idx
        beta = beta_hat[idx_ok]
        pred_mean = beta[:, 0] + np.sum(beta[:, 1:] * x, axis=1)

        sigma2_y = np.maximum(sigma2[idx_ok], 1e-12)
        if beta_cov is not None:
            var_coef = _row_variance_from_beta_cov(x1=x1, group_idx=idx_ok, beta_cov=beta_cov)
        else:
            var_coef = np.sum(np.square(x1) * beta_var_diag[idx_ok], axis=1)

        pred_var = np.maximum(sigma2_y + np.maximum(var_coef, 0.0), 0.0)
        pred_std = np.sqrt(pred_var) * pred_std_scale
        interval_width = (2.0 * q_norm) * pred_std

        err = pred_mean - y
        covered = np.abs(err) <= (q_norm * pred_std)

        used_rows += int(err.size)
        global_stats.update(
            err=err, covered=covered, pred_std=pred_std, interval_width=interval_width
        )

        np.add.at(group_n, idx_ok, 1)
        np.add.at(group_sum_sq, idx_ok, np.square(err))
        np.add.at(group_sum_abs, idx_ok, np.abs(err))
        np.add.at(group_sum_cov, idx_ok, covered.astype(np.float64, copy=False))
        np.add.at(group_sum_pred_std, idx_ok, pred_std.astype(np.float64, copy=False))
        np.add.at(group_sum_interval_width, idx_ok, interval_width.astype(np.float64, copy=False))

        if remaining is not None:
            remaining -= int(len(chunk))

        if log_every_chunks > 0 and (
            chunk_i == 1
            or chunk_i % log_every_chunks == 0
            or (remaining is not None and remaining <= 0)
        ):
            dt = time.time() - t0
            metrics = global_stats.to_metrics()
            rate = (float(used_rows) / dt) if dt > 0 else float("nan")
            if remaining is not None and max_rows > 0:
                processed = int(max_rows - remaining)
                pct = 100.0 * float(processed) / float(max_rows)
                eta = (float(max_rows - processed) / rate) if rate > 0 else float("nan")
                print(
                    f"[coeff_eval_support] chunk={chunk_i} used={used_rows:,} read={total_rows:,} "
                    f"rmse={float(metrics['rmse']):.4f} cov95={float(metrics['coverage_95']):.3f} "
                    f"elapsed_min={dt/60.0:.1f} eta_min={eta/60.0:.1f} ({pct:.1f}%)"
                )
            else:
                print(
                    f"[coeff_eval_support] chunk={chunk_i} used={used_rows:,} read={total_rows:,} "
                    f"rmse={float(metrics['rmse']):.4f} cov95={float(metrics['coverage_95']):.3f} "
                    f"krows_per_s={rate/1000.0:.1f} elapsed_min={dt/60.0:.1f}"
                )

    group_rmse = np.full((n_groups,), np.nan, dtype=np.float64)
    group_mae = np.full((n_groups,), np.nan, dtype=np.float64)
    group_cov = np.full((n_groups,), np.nan, dtype=np.float64)
    group_pred_std_mean = np.full((n_groups,), np.nan, dtype=np.float64)
    group_interval_width_mean = np.full((n_groups,), np.nan, dtype=np.float64)
    ok_g = group_n > 0
    group_rmse[ok_g] = np.sqrt(group_sum_sq[ok_g] / group_n[ok_g])
    group_mae[ok_g] = group_sum_abs[ok_g] / group_n[ok_g]
    group_cov[ok_g] = group_sum_cov[ok_g] / group_n[ok_g]
    group_pred_std_mean[ok_g] = group_sum_pred_std[ok_g] / group_n[ok_g]
    group_interval_width_mean[ok_g] = group_sum_interval_width[ok_g] / group_n[ok_g]

    support_edges = _parse_support_bins(args.support_bins)
    labels = _bin_labels(support_edges)
    support_idx = np.digitize(
        n_obs_train, bins=np.asarray(support_edges, dtype=np.int64), right=True
    )

    global_metrics = global_stats.to_metrics()

    support_rows = []
    for bin_i, label in enumerate(labels):
        g_mask = (support_idx == int(bin_i)) & ok_g
        if not np.any(g_mask):
            support_rows.append(
                {
                    "support_bin": label,
                    "n_groups": int(np.sum(support_idx == int(bin_i))),
                    "n_groups_with_test": 0,
                    "n_obs_test": 0,
                    "rmse": float("nan"),
                    "mae": float("nan"),
                    "coverage_95": float("nan"),
                    **_group_rmse_stats(np.array([], dtype=np.float64)),
                }
            )
            continue
        n_bin = int(np.sum(group_n[g_mask]))
        sum_sq = float(np.sum(group_sum_sq[g_mask]))
        sum_abs = float(np.sum(group_sum_abs[g_mask]))
        sum_cov = float(np.sum(group_sum_cov[g_mask]))
        sum_std = float(np.sum(group_sum_pred_std[g_mask]))
        sum_width = float(np.sum(group_sum_interval_width[g_mask]))
        row_rmse = float(np.sqrt(sum_sq / n_bin)) if n_bin > 0 else float("nan")
        row_mae = float(sum_abs / n_bin) if n_bin > 0 else float("nan")
        row_cov = float(sum_cov / n_bin) if n_bin > 0 else float("nan")
        row_std_mean = float(sum_std / n_bin) if n_bin > 0 else float("nan")
        row_width_mean = float(sum_width / n_bin) if n_bin > 0 else float("nan")
        g_stats = _group_rmse_stats(group_rmse[g_mask])
        support_rows.append(
            {
                "support_bin": label,
                "n_groups": int(np.sum(support_idx == int(bin_i))),
                "n_groups_with_test": int(np.sum(g_mask)),
                "n_obs_test": int(n_bin),
                "rmse": row_rmse,
                "mae": row_mae,
                "coverage_95": row_cov,
                "pred_std_mean": row_std_mean,
                "interval_width_mean": row_width_mean,
                **g_stats,
            }
        )

    # Per-group CSV
    group_df_dict: dict[str, object] = {
        "group_key": keys.astype(np.int64, copy=False),
        "comp_id": comp_ids.astype(np.int64, copy=False),
        "n_obs_train": n_obs_train.astype(np.int64, copy=False),
        "n_obs_test": group_n.astype(np.int64, copy=False),
        "rmse": group_rmse,
        "mae": group_mae,
        "coverage_95": group_cov,
        "pred_std_mean": group_pred_std_mean,
        "interval_width_mean": group_interval_width_mean,
        "support_bin": [labels[int(i)] for i in support_idx.tolist()],
    }
    group_df_dict[group_col] = group_ids.astype(np.int64, copy=False)
    group_df = pd.DataFrame(group_df_dict)
    if stage1.supercat_id is not None:
        group_df["supercat_id"] = np.asarray(stage1.supercat_id, dtype=np.int64)
    group_df.to_csv(out_group_csv, index=False)

    support_df = pd.DataFrame(support_rows)
    support_df.to_csv(out_support_csv, index=False)

    out_json.write_text(
        json.dumps(
            {
                "metrics": global_metrics,
                "n_test": int(total_rows),
                "n_used": int(used_rows),
                "skipped_missing_group": int(skipped_missing_group),
                "chunk_size": int(args.chunk_size),
                "interval": float(args.interval),
                "pred_std_scale": float(pred_std_scale),
                "require_seen_group": bool(require_seen),
                "group_col": group_col,
                "support_bins": support_edges,
                "support_metrics": support_rows,
                "coeff_npz": str(coeff_npz),
                "test_csv": str(test_csv),
            },
            indent=2,
        )
    )

    if require_seen and skipped_missing_group > 0:
        raise SystemExit(
            f"--require-seen-group set but encountered {skipped_missing_group:,} rows with unseen groups"
        )

    print(
        f"[coeff_eval_support] Global: n={global_metrics['n_obs']:,}, RMSE={global_metrics['rmse']:.4f}, "
        f"coverage95={global_metrics['coverage_95']:.3f}"
    )
    print(f"[coeff_eval_support] Wrote {out_json}")
    print(f"[coeff_eval_support] Wrote {out_group_csv}")
    print(f"[coeff_eval_support] Wrote {out_support_csv}")


if __name__ == "__main__":
    main()
