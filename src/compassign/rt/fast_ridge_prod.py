"""
Fast ridge RT model and production CSV training utilities (no MCMC).

This module consolidates:
  - production CSV helpers (lib detection, species mapping, feature selection),
  - the core per-(supercategory × comp_id) ridge model (`RidgeGroupCompoundRTModel`),
  - analytic ridge/Bayesian-ridge fitting from sufficient stats, and
  - streaming production training helpers that write `rt_ridge_model.npz`.

It is intended as the single entrypoint for the "fast ridge production model" pathway.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd


# Repo root for locating repo_export artifacts when called from scripts.
REPO_ROOT = Path(__file__).resolve().parents[3]

RUN_KEY_COLS: tuple[str, ...] = ("sampleset_id", "worksheet_id", "task_id")


@dataclass(frozen=True)
class SpeciesMapping:
    """
    Species mapping derived from repo export mapping CSVs.

    Attributes:
        sample_set_id_to_group_raw: maps sample_set_id -> species_group_raw label (string).
        sample_set_id_to_supercategory: maps sample_set_id -> supercategory number (int), parsed
            from species_group_raw (leading integer token).
        group1_sample_set_ids: set of sample_set_id values that are exactly "1 human blood".
        sample_set_id_to_species_raw: optional mapping sample_set_id -> species_raw, when present.
    """

    sample_set_id_to_group_raw: Dict[int, str]
    sample_set_id_to_supercategory: Dict[int, int]
    group1_sample_set_ids: Set[int]
    sample_set_id_to_species_raw: Dict[int, str]


def detect_lib_id(header_df: pd.DataFrame, data_csv: Path) -> Optional[int]:
    """Detect lib_id from a CSV header (preferred) or the filename."""
    if "lib_id" in header_df.columns:
        libs = sorted(header_df["lib_id"].dropna().unique().tolist())
        if len(libs) == 1:
            try:
                return int(libs[0])
            except Exception:
                pass
    m = re.search(r"lib(\d+)", data_csv.name)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
    return None


def detect_lib_id_from_paths(paths: Iterable[Path]) -> Optional[int]:
    """Detect lib_id by regex search over one or more paths."""
    for p in paths:
        m = re.search(r"lib(\d+)", str(p))
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return None


def find_species_mapping_for_lib(lib_id: int, *, repo_root: Path = REPO_ROOT) -> Optional[Path]:
    """
    Locate the repo-export species mapping CSV for a given library.

    Searches typical repo_export layouts and returns the first match (sorted) if found.
    """
    search_dirs = [
        repo_root / f"repo_export/lib{lib_id}/species_mapping",
        repo_root / "repo_export",
    ]
    for base in search_dirs:
        if base.is_dir():
            candidates = sorted(base.glob(f"**/*_lib{lib_id}_species_mapping.csv"))
            if candidates:
                return candidates[0]
    return None


def _parse_supercategory_number(species_group_raw: str) -> Optional[int]:
    m = re.match(r"\s*(\d+)", str(species_group_raw))
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def load_species_mapping(lib_id: int, *, repo_root: Path = REPO_ROOT) -> SpeciesMapping:
    """Load species mapping for `lib_id` from repo_export."""
    mapping_path = find_species_mapping_for_lib(lib_id, repo_root=repo_root)
    if mapping_path is None or not mapping_path.exists():
        raise FileNotFoundError(f"No species mapping found for lib {lib_id} under repo_export")

    sm_df = pd.read_csv(mapping_path)
    required_cols = {"sample_set_id", "species_group_raw"}
    if not required_cols.issubset(sm_df.columns):
        raise ValueError(
            f"Species mapping {mapping_path} missing required columns: {sorted(required_cols)}"
        )

    ssid_to_group_raw: Dict[int, str] = {}
    ssid_to_super: Dict[int, int] = {}
    group1_ssids: Set[int] = set()
    ssid_to_species_raw: Dict[int, str] = {}

    for _, row in sm_df.iterrows():
        try:
            ssid = int(row["sample_set_id"])
        except Exception:
            continue

        group_raw = str(row["species_group_raw"])
        ssid_to_group_raw[ssid] = group_raw

        super_num = _parse_supercategory_number(group_raw)
        if super_num is not None:
            ssid_to_super[ssid] = super_num

        if group_raw == "1 human blood":
            group1_ssids.add(ssid)

        if "species_raw" in sm_df.columns:
            val = row.get("species_raw")
            if val is not None and not pd.isna(val):
                ssid_to_species_raw[ssid] = str(val)

    return SpeciesMapping(
        sample_set_id_to_group_raw=ssid_to_group_raw,
        sample_set_id_to_supercategory=ssid_to_super,
        group1_sample_set_ids=group1_ssids,
        sample_set_id_to_species_raw=ssid_to_species_raw,
    )


def group1_es_by_lib(lib_id: int) -> List[str]:
    """Return the historical Group 1 ES_* feature subset used in older pipelines."""
    if lib_id == 208:
        return ["ES_15506", "ES_1649", "ES_27718", "ES_32198", "ES_33941", "ES_33955"]
    if lib_id == 209:
        return [
            "ES_1299",
            "ES_1564",
            "ES_1604",
            "ES_32425",
            "ES_32553",
            "ES_33955",
            "ES_34419",
            "ES_36103",
            "ES_42398",
            "ES_606",
        ]
    return []


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


def _encode_super_comp_key(super_num: np.ndarray, comp_id: np.ndarray) -> np.ndarray:
    """
    Encode (supercategory_number, comp_id) into a single int64 key.

    Uses: key = (super_num << 32) + comp_id
    """
    super_num_i64 = np.asarray(super_num, dtype=np.int64)
    comp_id_i64 = np.asarray(comp_id, dtype=np.int64)
    return (super_num_i64 << np.int64(32)) + comp_id_i64


@dataclass(frozen=True)
class RidgeGroupCompoundRTModel:
    """
    Fast RT model: per (species-group supercategory × comp_id) ridge regression.

    Each model is: rt = intercept + dot(coefs, x_run), where x_run are run covariates
    (IS*/RS*/optional ES_* for Group 1).

    The primary mapping uses (super_num, comp_id). When not found, an optional
    compound-only fallback model can be used.
    """

    feature_names: Tuple[str, ...]
    keys_super_comp: np.ndarray  # int64, sorted
    coefs_super_comp: np.ndarray  # (n_models, n_features)
    intercepts_super_comp: np.ndarray  # (n_models,)
    sigma_super_comp: np.ndarray  # (n_models,)
    comp_ids: np.ndarray  # int64, sorted
    coefs_compound: np.ndarray  # (n_compounds, n_features)
    intercepts_compound: np.ndarray  # (n_compounds,)
    sigma_compound: np.ndarray  # (n_compounds,)

    # Optional Bayesian ridge posterior info (separate from `sigma_*`, which remains
    # the residual scale sqrt(SSE/n) used by `predict()` for backward compatibility).
    bayes_a0: float | None = None
    bayes_b0: float | None = None
    n_super_comp: np.ndarray | None = None  # (n_models,)
    x_mean_super_comp: np.ndarray | None = None  # (n_models, n_features)
    v_diag_super_comp: np.ndarray | None = None  # (n_models, n_features) diag((XtX+λI)^-1)
    sigma2_mean_super_comp: np.ndarray | None = None  # (n_models,) posterior mean E[σ²|data]
    n_compound: np.ndarray | None = None  # (n_compounds,)
    x_mean_compound: np.ndarray | None = None  # (n_compounds, n_features)
    v_diag_compound: np.ndarray | None = None  # (n_compounds, n_features)
    sigma2_mean_compound: np.ndarray | None = None  # (n_compounds,) posterior mean E[σ²|data]

    def _match_model_indices(
        self, *, super_num: np.ndarray, comp_id: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Match each (super_num, comp_id) row to a group-compound model, or fall back
        to a compound-only model if present.

        Returns:
            idx_super_comp: int64 shape (n,), -1 for missing
            idx_compound: int64 shape (n,), -1 for missing
            used_fallback: bool shape (n,)
        """
        super_num = np.asarray(super_num, dtype=np.int64)
        comp_id = np.asarray(comp_id, dtype=np.int64)
        if super_num.shape != comp_id.shape:
            raise ValueError("super_num and comp_id must have the same shape")
        n = int(comp_id.size)

        idx_super_comp = np.full(n, -1, dtype=np.int64)
        idx_compound = np.full(n, -1, dtype=np.int64)
        used_fallback = np.zeros(n, dtype=bool)

        key = _encode_super_comp_key(super_num, comp_id)
        idx = np.searchsorted(self.keys_super_comp, key)
        ok = (idx >= 0) & (idx < self.keys_super_comp.size)
        if ok.any():
            idx_ok = idx[ok]
            ok_key = self.keys_super_comp[idx_ok] == key[ok]
            ok[ok] &= ok_key
        if ok.any():
            idx_super_comp[ok] = idx[ok].astype(np.int64, copy=False)

        miss = ~ok
        if miss.any() and self.comp_ids.size > 0:
            idx_c = np.searchsorted(self.comp_ids, comp_id[miss])
            ok_c = (idx_c >= 0) & (idx_c < self.comp_ids.size)
            if ok_c.any():
                idx_c_ok = idx_c[ok_c]
                ok_comp = self.comp_ids[idx_c_ok] == comp_id[miss][ok_c]
                ok_c[ok_c] &= ok_comp
            if ok_c.any():
                miss_idx = np.flatnonzero(miss)
                use_rows = miss_idx[ok_c]
                idx_compound[use_rows] = idx_c[ok_c].astype(np.int64, copy=False)
                used_fallback[use_rows] = True

        return idx_super_comp, idx_compound, used_fallback

    def predict(
        self,
        *,
        super_num: np.ndarray,
        comp_id: np.ndarray,
        x: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Vectorized prediction.

        Returns:
            pred_mean: shape (n,)
            pred_std: shape (n,)
            used_fallback: shape (n,) boolean mask (True if compound-only fallback used)
        """
        super_num = np.asarray(super_num, dtype=np.int64)
        comp_id = np.asarray(comp_id, dtype=np.int64)
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.feature_names):
            raise ValueError(f"x must have shape (n, {len(self.feature_names)}); got {x.shape}")
        n = int(comp_id.size)

        pred_mean = np.full(n, np.nan, dtype=float)
        pred_std = np.full(n, np.nan, dtype=float)

        idx_sc, idx_c, used_fallback = self._match_model_indices(
            super_num=super_num, comp_id=comp_id
        )

        ok_sc = idx_sc >= 0
        if ok_sc.any():
            idx_ok = idx_sc[ok_sc]
            pred_mean[ok_sc] = self.intercepts_super_comp[idx_ok] + np.sum(
                self.coefs_super_comp[idx_ok] * x[ok_sc], axis=1
            )
            pred_std[ok_sc] = self.sigma_super_comp[idx_ok]

        ok_c = idx_c >= 0
        if ok_c.any():
            idx_ok = idx_c[ok_c]
            pred_mean[ok_c] = self.intercepts_compound[idx_ok] + np.sum(
                self.coefs_compound[idx_ok] * x[ok_c], axis=1
            )
            pred_std[ok_c] = self.sigma_compound[idx_ok]

        return pred_mean, pred_std, used_fallback

    def predict_posterior(
        self,
        *,
        super_num: np.ndarray,
        comp_id: np.ndarray,
        x: np.ndarray,
        include_intercept_uncertainty: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Bayesian ridge posterior predictive (Student-t) using stored summary arrays.

        Requires `bayes_a0/bayes_b0` and posterior summary arrays from training
        (see `bayesian_ridge_fit_from_sufficient_stats`).

        Uses a diagonal approximation to diag((XtX_centered + λI)^-1) for compactness.

        Returns:
            pred_mean: shape (n,)
            pred_std: shape (n,) posterior predictive standard deviation
            pred_df: shape (n,) Student-t degrees of freedom (df = n + 2*a0)
            used_fallback: shape (n,) boolean mask (True if compound-only fallback used)
        """
        super_num = np.asarray(super_num, dtype=np.int64)
        comp_id = np.asarray(comp_id, dtype=np.int64)
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.feature_names):
            raise ValueError(f"x must have shape (n, {len(self.feature_names)}); got {x.shape}")

        if self.bayes_a0 is None or self.bayes_b0 is None:
            raise ValueError("Model is missing bayes_a0/bayes_b0; retrain with Bayesian stats.")
        if (
            self.n_super_comp is None
            or self.x_mean_super_comp is None
            or self.v_diag_super_comp is None
            or self.sigma2_mean_super_comp is None
            or self.n_compound is None
            or self.x_mean_compound is None
            or self.v_diag_compound is None
            or self.sigma2_mean_compound is None
        ):
            raise ValueError(
                "Model is missing Bayesian posterior arrays; retrain with Bayesian stats."
            )

        n = int(comp_id.size)
        pred_mean = np.full(n, np.nan, dtype=float)
        pred_std = np.full(n, np.nan, dtype=float)
        pred_df = np.full(n, np.nan, dtype=float)

        idx_sc, idx_c, used_fallback = self._match_model_indices(
            super_num=super_num, comp_id=comp_id
        )

        ok_sc = idx_sc >= 0
        if ok_sc.any():
            idx_ok = idx_sc[ok_sc]
            pred_mean[ok_sc] = self.intercepts_super_comp[idx_ok] + np.sum(
                self.coefs_super_comp[idx_ok] * x[ok_sc], axis=1
            )
            x_centered = x[ok_sc] - self.x_mean_super_comp[idx_ok]
            quad = np.sum(x_centered * x_centered * self.v_diag_super_comp[idx_ok], axis=1)
            n_train = self.n_super_comp[idx_ok].astype(np.float64, copy=False)
            infl = 1.0 + quad
            if include_intercept_uncertainty:
                infl = infl + (1.0 / np.maximum(n_train, 1.0))
            var = self.sigma2_mean_super_comp[idx_ok] * infl
            pred_std[ok_sc] = np.sqrt(np.maximum(var, 0.0))
            pred_df[ok_sc] = float(2.0 * self.bayes_a0) + n_train

        ok_c = idx_c >= 0
        if ok_c.any():
            idx_ok = idx_c[ok_c]
            pred_mean[ok_c] = self.intercepts_compound[idx_ok] + np.sum(
                self.coefs_compound[idx_ok] * x[ok_c], axis=1
            )
            x_centered = x[ok_c] - self.x_mean_compound[idx_ok]
            quad = np.sum(x_centered * x_centered * self.v_diag_compound[idx_ok], axis=1)
            n_train = self.n_compound[idx_ok].astype(np.float64, copy=False)
            infl = 1.0 + quad
            if include_intercept_uncertainty:
                infl = infl + (1.0 / np.maximum(n_train, 1.0))
            var = self.sigma2_mean_compound[idx_ok] * infl
            pred_std[ok_c] = np.sqrt(np.maximum(var, 0.0))
            pred_df[ok_c] = float(2.0 * self.bayes_a0) + n_train

        return pred_mean, pred_std, pred_df, used_fallback

    def save_npz(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray] = {
            "feature_names": np.asarray(self.feature_names, dtype=object),
            "keys_super_comp": np.asarray(self.keys_super_comp, dtype=np.int64),
            "coefs_super_comp": np.asarray(self.coefs_super_comp, dtype=np.float32),
            "intercepts_super_comp": np.asarray(self.intercepts_super_comp, dtype=np.float32),
            "sigma_super_comp": np.asarray(self.sigma_super_comp, dtype=np.float32),
            "comp_ids": np.asarray(self.comp_ids, dtype=np.int64),
            "coefs_compound": np.asarray(self.coefs_compound, dtype=np.float32),
            "intercepts_compound": np.asarray(self.intercepts_compound, dtype=np.float32),
            "sigma_compound": np.asarray(self.sigma_compound, dtype=np.float32),
        }

        if self.bayes_a0 is not None and self.bayes_b0 is not None:
            payload["bayes_a0"] = np.asarray(float(self.bayes_a0), dtype=np.float32)
            payload["bayes_b0"] = np.asarray(float(self.bayes_b0), dtype=np.float32)
        if self.n_super_comp is not None:
            payload["n_super_comp"] = np.asarray(self.n_super_comp, dtype=np.int64)
        if self.x_mean_super_comp is not None:
            payload["x_mean_super_comp"] = np.asarray(self.x_mean_super_comp, dtype=np.float32)
        if self.v_diag_super_comp is not None:
            payload["v_diag_super_comp"] = np.asarray(self.v_diag_super_comp, dtype=np.float32)
        if self.sigma2_mean_super_comp is not None:
            payload["sigma2_mean_super_comp"] = np.asarray(
                self.sigma2_mean_super_comp, dtype=np.float32
            )
        if self.n_compound is not None:
            payload["n_compound"] = np.asarray(self.n_compound, dtype=np.int64)
        if self.x_mean_compound is not None:
            payload["x_mean_compound"] = np.asarray(self.x_mean_compound, dtype=np.float32)
        if self.v_diag_compound is not None:
            payload["v_diag_compound"] = np.asarray(self.v_diag_compound, dtype=np.float32)
        if self.sigma2_mean_compound is not None:
            payload["sigma2_mean_compound"] = np.asarray(
                self.sigma2_mean_compound, dtype=np.float32
            )

        np.savez_compressed(path, **payload)

    @staticmethod
    def load_npz(path: Path) -> "RidgeGroupCompoundRTModel":
        npz = np.load(path, allow_pickle=True)
        feature_names = tuple(str(s) for s in npz["feature_names"].tolist())
        files = set(npz.files)
        return RidgeGroupCompoundRTModel(
            feature_names=feature_names,
            keys_super_comp=np.asarray(npz["keys_super_comp"], dtype=np.int64),
            coefs_super_comp=np.asarray(npz["coefs_super_comp"], dtype=float),
            intercepts_super_comp=np.asarray(npz["intercepts_super_comp"], dtype=float),
            sigma_super_comp=np.asarray(npz["sigma_super_comp"], dtype=float),
            comp_ids=np.asarray(npz["comp_ids"], dtype=np.int64),
            coefs_compound=np.asarray(npz["coefs_compound"], dtype=float),
            intercepts_compound=np.asarray(npz["intercepts_compound"], dtype=float),
            sigma_compound=np.asarray(npz["sigma_compound"], dtype=float),
            bayes_a0=float(npz["bayes_a0"]) if "bayes_a0" in files else None,
            bayes_b0=float(npz["bayes_b0"]) if "bayes_b0" in files else None,
            n_super_comp=np.asarray(npz["n_super_comp"], dtype=np.int64)
            if "n_super_comp" in files
            else None,
            x_mean_super_comp=np.asarray(npz["x_mean_super_comp"], dtype=float)
            if "x_mean_super_comp" in files
            else None,
            v_diag_super_comp=np.asarray(npz["v_diag_super_comp"], dtype=float)
            if "v_diag_super_comp" in files
            else None,
            sigma2_mean_super_comp=np.asarray(npz["sigma2_mean_super_comp"], dtype=float)
            if "sigma2_mean_super_comp" in files
            else None,
            n_compound=np.asarray(npz["n_compound"], dtype=np.int64)
            if "n_compound" in files
            else None,
            x_mean_compound=np.asarray(npz["x_mean_compound"], dtype=float)
            if "x_mean_compound" in files
            else None,
            v_diag_compound=np.asarray(npz["v_diag_compound"], dtype=float)
            if "v_diag_compound" in files
            else None,
            sigma2_mean_compound=np.asarray(npz["sigma2_mean_compound"], dtype=float)
            if "sigma2_mean_compound" in files
            else None,
        )


def ridge_fit_from_sufficient_stats(
    *,
    sum_x: np.ndarray,
    sum_y: np.ndarray,
    sum_y2: np.ndarray,
    sum_xx: np.ndarray,
    sum_xy: np.ndarray,
    n: np.ndarray,
    lambda_ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit per-row ridge models given sufficient statistics (with intercept).

    For each row i, solves:
        y = b + x·w + eps
        w = argmin Σ (y - b - x·w)^2 + lambda_ridge ||w||^2

    Returns:
        coefs: (m, d)
        intercepts: (m,)
        sigma: (m,) sqrt(SSE / n)
    """
    sum_x = np.asarray(sum_x, dtype=np.float64)
    sum_y = np.asarray(sum_y, dtype=np.float64)
    sum_y2 = np.asarray(sum_y2, dtype=np.float64)
    sum_xx = np.asarray(sum_xx, dtype=np.float64)
    sum_xy = np.asarray(sum_xy, dtype=np.float64)
    n = np.asarray(n, dtype=np.int64)

    if sum_x.ndim != 2:
        raise ValueError("sum_x must have shape (m, d)")
    m, d = sum_x.shape
    if sum_xx.shape != (m, d, d):
        raise ValueError(f"sum_xx must have shape (m, d, d); got {sum_xx.shape}")
    if sum_xy.shape != (m, d):
        raise ValueError(f"sum_xy must have shape (m, d); got {sum_xy.shape}")
    if sum_y.shape != (m,) or sum_y2.shape != (m,) or n.shape != (m,):
        raise ValueError("sum_y, sum_y2, and n must have shape (m,)")

    coefs = np.zeros((m, d), dtype=np.float64)
    intercepts = np.zeros(m, dtype=np.float64)
    sigma = np.full(m, np.nan, dtype=np.float64)
    eye = np.eye(d, dtype=np.float64)

    for i in range(m):
        ni = int(n[i])
        if ni <= 0:
            continue
        xm = sum_x[i] / float(ni)
        ym = float(sum_y[i] / float(ni))

        xtx_centered = sum_xx[i] - float(ni) * np.outer(xm, xm)
        xty_centered = sum_xy[i] - float(ni) * xm * ym
        a = xtx_centered + float(lambda_ridge) * eye
        try:
            w = np.linalg.solve(a, xty_centered)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(a, xty_centered, rcond=None)[0]
        b = ym - float(xm.dot(w))

        # SSE from sufficient statistics (no extra pass).
        # SSE = yTy - 2b*sum_y - 2 w·sum_xy + 2b*w·sum_x + b^2*n + w^T sum_xx w
        yty = float(sum_y2[i])
        w_sum_xy = float(w.dot(sum_xy[i]))
        w_sum_x = float(w.dot(sum_x[i]))
        w_xx_w = float(w.dot(sum_xx[i].dot(w)))
        sse = yty - 2.0 * b * float(sum_y[i]) - 2.0 * w_sum_xy + 2.0 * b * w_sum_x
        sse += (b * b) * float(ni) + w_xx_w
        sse = max(sse, 0.0)

        coefs[i] = w
        intercepts[i] = b
        sigma[i] = float(np.sqrt(sse / float(ni)))

    return coefs, intercepts, sigma


def bayesian_ridge_fit_from_sufficient_stats(
    *,
    sum_x: np.ndarray,
    sum_y: np.ndarray,
    sum_y2: np.ndarray,
    sum_xx: np.ndarray,
    sum_xy: np.ndarray,
    n: np.ndarray,
    lambda_ridge: float,
    a0: float = 2.0,
    b0: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit per-row ridge models and compute Bayesian ridge posterior summaries from sufficient stats.

    Model (with flat/unpenalized intercept):
        y = b + x·w + eps,  eps ~ Normal(0, σ²)
        w | σ² ~ Normal(0, σ²/λ I)
        σ² ~ InvGamma(a0, b0)

    The posterior predictive is Student-t with df = n + 2*a0. We return compact
    summaries needed for fast prediction with a diagonal covariance approximation:
        diag((XtX_centered + λI)^-1), E[σ² | data], and x means for centering.

    Returns:
        coefs: (m, d)
        intercepts: (m,)
        sigma_resid: (m,) sqrt(SSE/n) (frequentist residual scale)
        x_mean: (m, d)
        v_diag: (m, d) diagonal of (XtX_centered + λI)^-1
        sigma2_mean: (m,) posterior mean of σ², E[σ² | data]
        n_obs: (m,) int64
    """
    sum_x = np.asarray(sum_x, dtype=np.float64)
    sum_y = np.asarray(sum_y, dtype=np.float64)
    sum_y2 = np.asarray(sum_y2, dtype=np.float64)
    sum_xx = np.asarray(sum_xx, dtype=np.float64)
    sum_xy = np.asarray(sum_xy, dtype=np.float64)
    n = np.asarray(n, dtype=np.int64)

    if sum_x.ndim != 2:
        raise ValueError("sum_x must have shape (m, d)")
    m, d = sum_x.shape
    if sum_xx.shape != (m, d, d):
        raise ValueError(f"sum_xx must have shape (m, d, d); got {sum_xx.shape}")
    if sum_xy.shape != (m, d):
        raise ValueError(f"sum_xy must have shape (m, d); got {sum_xy.shape}")
    if sum_y.shape != (m,) or sum_y2.shape != (m,) or n.shape != (m,):
        raise ValueError("sum_y, sum_y2, and n must have shape (m,)")

    if a0 <= 1.0:
        raise ValueError("a0 must be > 1 so E[σ²|data] exists")
    if b0 < 0.0:
        raise ValueError("b0 must be >= 0")

    coefs = np.zeros((m, d), dtype=np.float64)
    intercepts = np.zeros(m, dtype=np.float64)
    sigma_resid = np.full(m, np.nan, dtype=np.float64)
    x_mean = np.full((m, d), np.nan, dtype=np.float64)
    v_diag = np.full((m, d), np.nan, dtype=np.float64)
    sigma2_mean = np.full(m, np.nan, dtype=np.float64)
    n_obs = n.copy()

    eye = np.eye(d, dtype=np.float64)

    for i in range(m):
        ni = int(n[i])
        if ni <= 0:
            continue

        xm = sum_x[i] / float(ni)
        ym = float(sum_y[i] / float(ni))

        xtx_centered = sum_xx[i] - float(ni) * np.outer(xm, xm)
        xty_centered = sum_xy[i] - float(ni) * xm * ym
        yty_centered = float(sum_y2[i] - float(ni) * ym * ym)

        a = xtx_centered + float(lambda_ridge) * eye
        try:
            w = np.linalg.solve(a, xty_centered)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(a, xty_centered, rcond=None)[0]
        b = ym - float(xm.dot(w))

        # Residual SSE and sigma (frequentist).
        w_xty = float(w.dot(xty_centered))
        w_xtx_w = float(w.dot(xtx_centered.dot(w)))
        sse = yty_centered - 2.0 * w_xty + w_xtx_w
        sse = max(sse, 0.0)

        # Bayesian noise posterior mean E[σ²|data] from Normal-InvGamma conjugacy.
        a_n = float(a0) + 0.5 * float(ni)
        # yTy - w^T (XtY) equals SSE + λ||w||^2 (penalized objective).
        sse_pen = float(yty_centered - w_xty)
        sse_pen = max(sse_pen, 0.0)
        b_n = float(b0) + 0.5 * sse_pen
        sigma2_mean_i = b_n / (a_n - 1.0)

        # Diagonal of (XtX_centered+λI)^-1 via Cholesky.
        try:
            chol = np.linalg.cholesky(a)
            inv_chol = np.linalg.solve(chol, eye)
            v_diag_i = np.sum(inv_chol * inv_chol, axis=0)
        except np.linalg.LinAlgError:
            inv_a = np.linalg.inv(a)
            v_diag_i = np.diag(inv_a)

        coefs[i] = w
        intercepts[i] = b
        sigma_resid[i] = float(np.sqrt(sse / float(ni)))
        x_mean[i] = xm
        v_diag[i] = v_diag_i
        sigma2_mean[i] = sigma2_mean_i

    return coefs, intercepts, sigma_resid, x_mean, v_diag, sigma2_mean, n_obs


# ---- Production CSV streaming trainer -----------------------------------------------------------


REQUIRED_COLS_RIDGE_PROD: tuple[str, ...] = (*RUN_KEY_COLS, "rt", "comp_id")


def _accum_init(
    d: int,
) -> tuple[np.ndarray, np.float64, np.float64, np.ndarray, np.ndarray, np.int64]:
    return (
        np.zeros(d, dtype=np.float64),  # sum_x
        np.float64(0.0),  # sum_y
        np.float64(0.0),  # sum_y2
        np.zeros((d, d), dtype=np.float64),  # sum_xx
        np.zeros(d, dtype=np.float64),  # sum_xy
        np.int64(0),  # n
    )


@dataclass
class _SufficientStatsAccumulator:
    n_features: int
    key_to_idx: Dict[int, int]
    keys: List[int]
    sum_x: List[np.ndarray]
    sum_y: List[np.float64]
    sum_y2: List[np.float64]
    sum_xx: List[np.ndarray]
    sum_xy: List[np.ndarray]
    n_obs: List[np.int64]

    @classmethod
    def create(cls, *, n_features: int) -> "_SufficientStatsAccumulator":
        return cls(
            n_features=n_features,
            key_to_idx={},
            keys=[],
            sum_x=[],
            sum_y=[],
            sum_y2=[],
            sum_xx=[],
            sum_xy=[],
            n_obs=[],
        )

    def _ensure_key(self, key: int) -> int:
        idx = self.key_to_idx.get(key)
        if idx is not None:
            return idx
        idx = len(self.key_to_idx)
        self.key_to_idx[key] = idx
        self.keys.append(int(key))
        sx, sy, sy2, sxx, sxy, nn = _accum_init(self.n_features)
        self.sum_x.append(sx)
        self.sum_y.append(sy)
        self.sum_y2.append(sy2)
        self.sum_xx.append(sxx)
        self.sum_xy.append(sxy)
        self.n_obs.append(nn)
        return idx

    def update(self, *, key: np.ndarray, x: np.ndarray, y: np.ndarray) -> None:
        if key.ndim != 1:
            raise ValueError("key must be 1D")
        if x.ndim != 2 or x.shape[1] != self.n_features:
            raise ValueError(f"x must have shape (n, {self.n_features}); got {x.shape}")
        if y.ndim != 1 or y.shape[0] != x.shape[0] or key.shape[0] != x.shape[0]:
            raise ValueError("key, x, y must have consistent first dimension")

        uniq, inv = np.unique(np.asarray(key, dtype=np.int64), return_inverse=True)
        for j, key_j in enumerate(uniq.tolist()):
            idx = self._ensure_key(int(key_j))
            rows = inv == j
            xj = x[rows]
            yj = y[rows]
            self.sum_x[idx] += xj.sum(axis=0)
            self.sum_y[idx] += np.float64(yj.sum())
            self.sum_y2[idx] += np.float64(np.square(yj).sum())
            self.sum_xx[idx] += xj.T @ xj
            self.sum_xy[idx] += xj.T @ yj
            self.n_obs[idx] += np.int64(rows.sum())

    def to_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not self.keys:
            d = self.n_features
            return (
                np.zeros(0, dtype=np.int64),
                np.zeros((0, d), dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                np.zeros((0, d, d), dtype=np.float64),
                np.zeros((0, d), dtype=np.float64),
                np.zeros(0, dtype=np.int64),
            )
        return (
            np.asarray(self.keys, dtype=np.int64),
            np.stack(self.sum_x),
            np.asarray(self.sum_y, dtype=np.float64),
            np.asarray(self.sum_y2, dtype=np.float64),
            np.stack(self.sum_xx),
            np.stack(self.sum_xy),
            np.asarray(self.n_obs, dtype=np.int64),
        )


@dataclass(frozen=True)
class RidgeProdTrainArtifacts:
    model: RidgeGroupCompoundRTModel
    lib_id: int
    feature_cols: Tuple[str, ...]
    include_es_group1: bool
    include_es_all: bool
    lambda_ridge: float
    bayes_a0: float
    bayes_b0: float
    n_obs_train: int
    n_models: int
    n_compounds: int
    species_group_example: Tuple[str, ...]
    trained_at: str


def _select_feature_columns(
    columns: Sequence[str],
    *,
    lib_id: int,
    include_es_group1: bool,
    include_es_all: bool,
) -> Tuple[str, ...]:
    es_candidates: List[str] = []
    if include_es_group1:
        es_candidates = group1_es_by_lib(lib_id)
    elif include_es_all:
        es_candidates = sorted([str(c) for c in columns if str(c).startswith("ES_")])
    feature_cols = infer_feature_columns(columns, es_candidates=es_candidates)
    return tuple(feature_cols)


def _attach_supercategory_numbers(
    chunk: pd.DataFrame, mapping: SpeciesMapping
) -> tuple[np.ndarray, pd.DataFrame]:
    super_num = chunk["sampleset_id"].astype(int).map(mapping.sample_set_id_to_supercategory)
    ok = super_num.notna()
    if not ok.all():
        chunk = chunk.loc[ok].copy()
        super_num = super_num.loc[ok]
    return super_num.astype(int).to_numpy(dtype=np.int64, copy=False), chunk


def train_ridge_prod_model(
    *,
    data_csv: Path,
    include_es_group1: bool,
    include_es_all: bool,
    lambda_ridge: float,
    bayes_a0: float,
    bayes_b0: float,
    chunk_size: int = 200_000,
    max_train_rows: int = 0,
    seed: int = 42,
) -> RidgeProdTrainArtifacts:
    """
    Train the fast ridge RT production model from a production RT CSV.

    Returns the trained model plus metadata for writing artifacts.
    """
    if include_es_group1 and include_es_all:
        raise ValueError("Use at most one of include_es_group1 or include_es_all")

    np.random.seed(int(seed))

    header = pd.read_csv(data_csv, nrows=0)
    missing_req = [c for c in REQUIRED_COLS_RIDGE_PROD if c not in header.columns]
    if missing_req:
        raise ValueError(f"CSV missing required columns: {missing_req}")

    lib_id = detect_lib_id(header, data_csv)
    if lib_id is None:
        raise ValueError("Unable to detect lib_id from CSV; required for species mapping.")

    species_mapping = load_species_mapping(int(lib_id))
    feature_cols = _select_feature_columns(
        header.columns,
        lib_id=int(lib_id),
        include_es_group1=bool(include_es_group1),
        include_es_all=bool(include_es_all),
    )
    es_cols = [c for c in feature_cols if c.startswith("ES_")]

    acc_gc = _SufficientStatsAccumulator.create(n_features=len(feature_cols))
    acc_c = _SufficientStatsAccumulator.create(n_features=len(feature_cols))

    remaining: Optional[int] = int(max_train_rows) if int(max_train_rows) > 0 else None
    used_rows = 0

    for chunk in pd.read_csv(data_csv, chunksize=int(chunk_size)):
        if remaining is not None and remaining <= 0:
            break
        if remaining is not None and len(chunk) > remaining:
            chunk = chunk.iloc[:remaining].copy()

        super_arr, chunk = _attach_supercategory_numbers(chunk, species_mapping)
        if len(chunk) == 0:
            continue

        if es_cols:
            chunk[es_cols] = chunk[es_cols].fillna(0.0)
            if include_es_group1:
                mask1 = (
                    chunk["sampleset_id"].astype(int).isin(species_mapping.group1_sample_set_ids)
                )
                chunk.loc[~mask1, es_cols] = 0.0

        x = chunk[list(feature_cols)].to_numpy(dtype=np.float64, copy=False)
        y = chunk["rt"].to_numpy(dtype=np.float64, copy=False)
        comp = chunk["comp_id"].to_numpy(dtype=np.int64, copy=False)

        used_rows += int(len(chunk))

        key_gc = _encode_super_comp_key(super_arr, comp)
        acc_gc.update(key=key_gc, x=x, y=y)
        acc_c.update(key=comp, x=x, y=y)

        if remaining is not None:
            remaining -= int(len(chunk))

    (
        keys_arr,
        sum_x_arr,
        sum_y_arr,
        sum_y2_arr,
        sum_xx_arr,
        sum_xy_arr,
        n_arr,
    ) = acc_gc.to_arrays()
    (
        comp_ids_arr,
        sum_x_c_arr,
        sum_y_c_arr,
        sum_y2_c_arr,
        sum_xx_c_arr,
        sum_xy_c_arr,
        n_c_arr,
    ) = acc_c.to_arrays()

    (
        coefs_gc,
        intercepts_gc,
        sigma_gc,
        x_mean_gc,
        v_diag_gc,
        sigma2_mean_gc,
        n_arr,
    ) = bayesian_ridge_fit_from_sufficient_stats(
        sum_x=sum_x_arr,
        sum_y=sum_y_arr,
        sum_y2=sum_y2_arr,
        sum_xx=sum_xx_arr,
        sum_xy=sum_xy_arr,
        n=n_arr,
        lambda_ridge=float(lambda_ridge),
        a0=float(bayes_a0),
        b0=float(bayes_b0),
    )

    (
        coefs_c,
        intercepts_c,
        sigma_c,
        x_mean_c,
        v_diag_c,
        sigma2_mean_c,
        n_c_arr,
    ) = bayesian_ridge_fit_from_sufficient_stats(
        sum_x=sum_x_c_arr,
        sum_y=sum_y_c_arr,
        sum_y2=sum_y2_c_arr,
        sum_xx=sum_xx_c_arr,
        sum_xy=sum_xy_c_arr,
        n=n_c_arr,
        lambda_ridge=float(lambda_ridge),
        a0=float(bayes_a0),
        b0=float(bayes_b0),
    )

    # Sort keys for vectorized lookups at prediction time.
    sort_gc = np.argsort(keys_arr)
    keys_arr = keys_arr[sort_gc]
    coefs_gc = coefs_gc[sort_gc]
    intercepts_gc = intercepts_gc[sort_gc]
    sigma_gc = sigma_gc[sort_gc]
    n_arr = n_arr[sort_gc]
    x_mean_gc = x_mean_gc[sort_gc]
    v_diag_gc = v_diag_gc[sort_gc]
    sigma2_mean_gc = sigma2_mean_gc[sort_gc]

    sort_c = np.argsort(comp_ids_arr)
    comp_ids_arr = comp_ids_arr[sort_c]
    coefs_c = coefs_c[sort_c]
    intercepts_c = intercepts_c[sort_c]
    sigma_c = sigma_c[sort_c]
    n_c_arr = n_c_arr[sort_c]
    x_mean_c = x_mean_c[sort_c]
    v_diag_c = v_diag_c[sort_c]
    sigma2_mean_c = sigma2_mean_c[sort_c]

    model = RidgeGroupCompoundRTModel(
        feature_names=tuple(feature_cols),
        keys_super_comp=keys_arr,
        coefs_super_comp=coefs_gc,
        intercepts_super_comp=intercepts_gc,
        sigma_super_comp=sigma_gc,
        comp_ids=comp_ids_arr,
        coefs_compound=coefs_c,
        intercepts_compound=intercepts_c,
        sigma_compound=sigma_c,
        bayes_a0=float(bayes_a0),
        bayes_b0=float(bayes_b0),
        n_super_comp=n_arr,
        x_mean_super_comp=x_mean_gc,
        v_diag_super_comp=v_diag_gc,
        sigma2_mean_super_comp=sigma2_mean_gc,
        n_compound=n_c_arr,
        x_mean_compound=x_mean_c,
        v_diag_compound=v_diag_c,
        sigma2_mean_compound=sigma2_mean_c,
    )

    species_group_example = tuple(
        sorted(set(species_mapping.sample_set_id_to_group_raw.values()))[:5]
    )
    trained_at = datetime.now().isoformat()

    return RidgeProdTrainArtifacts(
        model=model,
        lib_id=int(lib_id),
        feature_cols=tuple(feature_cols),
        include_es_group1=bool(include_es_group1),
        include_es_all=bool(include_es_all),
        lambda_ridge=float(lambda_ridge),
        bayes_a0=float(bayes_a0),
        bayes_b0=float(bayes_b0),
        n_obs_train=int(used_rows),
        n_models=int(keys_arr.size),
        n_compounds=int(comp_ids_arr.size),
        species_group_example=species_group_example,
        trained_at=trained_at,
    )


def write_ridge_prod_artifacts(
    *,
    artifacts: RidgeProdTrainArtifacts,
    output_dir: Path,
    data_csv: Path,
) -> Path:
    """
    Write `config.json`, `results/rt_ridge_train_summary.json`, and the model `.npz`.

    Returns:
        Path to the written model file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "models").mkdir(exist_ok=True)
    (output_dir / "results").mkdir(exist_ok=True)

    model_path = output_dir / "models" / "rt_ridge_model.npz"
    artifacts.model.save_npz(model_path)

    (output_dir / "config.json").write_text(
        json.dumps(
            {
                "timestamp": artifacts.trained_at,
                "model_type": "ridge_supercategory_compound",
                "lib_id": int(artifacts.lib_id),
                "data_csv": str(data_csv),
                "feature_cols": list(artifacts.feature_cols),
                "lambda_ridge": float(artifacts.lambda_ridge),
                "include_es_group1": bool(artifacts.include_es_group1),
                "include_es_all": bool(artifacts.include_es_all),
                "bayesian": True,
                "bayes_a0": float(artifacts.bayes_a0),
                "bayes_b0": float(artifacts.bayes_b0),
                "n_models": int(artifacts.n_models),
                "n_compounds": int(artifacts.n_compounds),
                "n_obs_train": int(artifacts.n_obs_train),
                "species_group_example": list(artifacts.species_group_example),
            },
            indent=2,
        )
    )

    (output_dir / "results" / "rt_ridge_train_summary.json").write_text(
        json.dumps(
            {
                "n_obs_train": int(artifacts.n_obs_train),
                "n_models": int(artifacts.n_models),
                "n_compounds": int(artifacts.n_compounds),
                "lib_id": int(artifacts.lib_id),
                "feature_cols": list(artifacts.feature_cols),
                "lambda_ridge": float(artifacts.lambda_ridge),
                "bayesian": True,
                "bayes_a0": float(artifacts.bayes_a0),
                "bayes_b0": float(artifacts.bayes_b0),
            },
            indent=2,
        )
    )

    return model_path
