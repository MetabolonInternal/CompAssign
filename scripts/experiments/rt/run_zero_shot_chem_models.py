#!/usr/bin/env python3
"""
Zero-shot compound experiment (single-model; no backoff).

We hold out a set of chem_ids entirely from training (true compound cold-start) and compare
single ridge models that can still score those held-out chemicals because chemistry enters as
features (compound_class and/or ChemBERTa embeddings), not as a per-compound lookup table.

Models (all single-model ridge regressions):
  1) cluster_only: run covariates + one-hot species_cluster
  2) cluster_class: + one-hot compound_class
  3) cluster_emb: + ChemBERTa PCA-20 embedding
  4) cluster_emb_inter: + cluster-specific embedding interactions (emb × one-hot cluster)

Notes
-----
- This script intentionally avoids per-(cluster, comp_id) coefficient tables and "backoff" logic.
  If a chem_id is unseen at training time, it is still scored via its chemistry features.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.compassign.rt.ridge_stage1 import infer_feature_columns  # noqa: E402
from src.compassign.rt.pymc_partial_pool_ridge import (  # noqa: E402
    train_pymc_partial_pool_ridge_from_csv,
    write_pymc_partial_pool_ridge_artifacts,
)
from src.compassign.utils.data_features import load_chemberta_pca20  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run single-model zero-shot compound experiments (hold out chem_ids).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-csv", type=Path, required=True, help="Input RT CSV (already filtered).")
    p.add_argument(
        "--chem-embeddings-path",
        type=Path,
        default=Path("resources/metabolites/embeddings_chemberta_pca20.parquet"),
        help="ChemBERTa PCA-20 embedding parquet.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory (writes summary.json and config.json).",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip work if output summary.json already exists.",
    )
    p.add_argument(
        "--holdout-n",
        type=int,
        default=10,
        help="Number of unique chem_ids to hold out (eligible chems only).",
    )
    p.add_argument(
        "--holdout-strategy",
        type=str,
        choices=["random", "stratified_rt"],
        default="random",
        help="How to select held-out chem_ids among eligible chems.",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for selecting holdout chems.")
    p.add_argument(
        "--alphas",
        type=str,
        default="1e-4,3e-4,1e-3,3e-3,1e-2,3e-2,1e-1,3e-1,1,3,10,30,100,300,1e3,3e3,1e4",
        help="Comma-separated ridge alpha candidates for RidgeCV.",
    )
    p.add_argument(
        "--include-mlp",
        action="store_true",
        help="Also train MLPRegressor variants that use chemistry features (single model, non-linear).",
    )
    p.add_argument(
        "--include-pymc",
        action="store_true",
        help=(
            "Also train the PyMC ridge partial pooling model with a chemistry-informed compound prior "
            "(single model; produces calibrated posterior uncertainty)."
        ),
    )
    p.add_argument(
        "--pymc-advi-steps",
        type=int,
        default=5000,
        help="ADVI steps for the PyMC partial pooling run.",
    )
    p.add_argument(
        "--theta-alpha-prior-sigma",
        type=float,
        default=1.0,
        help="Prior sigma for theta_alpha in the chem-linear compound prior.",
    )
    p.add_argument(
        "--report-minimal",
        action="store_true",
        help=(
            "Run only the two report comparisons: PyMC chem-linear partial pooling and "
            "MLP (ChemBERTa PCA-20 + cluster interactions)."
        ),
    )
    return p.parse_args()


def _parse_float_list(s: str) -> list[float]:
    out: list[float] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise SystemExit("Empty --alphas")
    return out


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    return float(np.sqrt(np.mean(err * err)))


def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    err = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    return float(np.mean(np.abs(err)))


def _one_hot(values: np.ndarray, *, categories: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    categories = np.asarray(categories, dtype=np.int64)
    n = int(values.size)
    k = int(categories.size)
    out = np.zeros((n, k), dtype=np.float32)
    if n == 0 or k == 0:
        return out
    idx = np.searchsorted(categories, values)
    ok = (idx >= 0) & (idx < k)
    if ok.any():
        ok[ok] &= categories[idx[ok]] == values[ok]
    if ok.any():
        rows = np.flatnonzero(ok)
        out[rows, idx[ok]] = 1.0
    return out


def _embed_rows(
    *, chem_id: np.ndarray, emb_chem_id: np.ndarray, emb_features: np.ndarray
) -> np.ndarray:
    chem_id = np.asarray(chem_id, dtype=np.int64)
    emb_chem_id = np.asarray(emb_chem_id, dtype=np.int64)
    emb_features = np.asarray(emb_features, dtype=np.float32)
    order = np.argsort(emb_chem_id, kind="mergesort")
    emb_id_sorted = emb_chem_id[order].astype(np.int64, copy=False)
    emb_feat_sorted = emb_features[order].astype(np.float32, copy=False)
    idx = np.searchsorted(emb_id_sorted, chem_id)
    ok = (idx >= 0) & (idx < emb_id_sorted.size)
    if ok.any():
        ok[ok] &= emb_id_sorted[idx[ok]] == chem_id[ok]
    if not bool(np.all(ok)):
        missing = np.unique(chem_id[~ok]).tolist()
        raise SystemExit(
            f"Missing ChemBERTa embeddings for chem_ids: {missing[:10]} (n_missing={len(missing)})"
        )
    return emb_feat_sorted[idx]


def _choose_holdout_chems(
    *,
    df: pd.DataFrame,
    holdout_n: int,
    seed: int,
    strategy: str,
    chem_col: str = "compound",
) -> list[int]:
    chem_df = (
        df[[chem_col, "compound_class"]]
        .copy()
        .fillna({"compound_class": -1})
        .astype({chem_col: int, "compound_class": int})
        .drop_duplicates()
    )
    cls_counts = (
        chem_df.groupby("compound_class")[chem_col].nunique().rename("n_chems").reset_index()
    )
    eligible_cls = set(cls_counts.loc[cls_counts["n_chems"] >= 2, "compound_class"].astype(int))
    eligible = chem_df.loc[chem_df["compound_class"].isin(eligible_cls), chem_col].astype(int)
    eligible_chems = np.asarray(sorted(set(eligible.tolist())), dtype=np.int64)
    if eligible_chems.size == 0:
        raise SystemExit("No eligible chem_ids (need compound_class with >=2 chems)")
    if not (1 <= int(holdout_n) < int(eligible_chems.size)):
        raise SystemExit(
            f"--holdout-n must be in [1, {int(eligible_chems.size) - 1}] (got {holdout_n})"
        )

    rng = np.random.default_rng(int(seed))
    chosen: np.ndarray

    if strategy == "random":
        chosen = rng.choice(eligible_chems, size=int(holdout_n), replace=False)
        return sorted(int(x) for x in chosen.tolist())

    if strategy != "stratified_rt":
        raise SystemExit(f"Unknown holdout strategy: {strategy}")

    chem_rt = (
        df.loc[df[chem_col].isin(eligible_chems), [chem_col, "rt"]]
        .copy()
        .astype({chem_col: int})
        .groupby(chem_col)["rt"]
        .mean()
        .reset_index()
        .sort_values("rt", kind="mergesort")
    )
    ordered_chems = chem_rt[chem_col].to_numpy(dtype=np.int64, copy=False)
    if ordered_chems.size != eligible_chems.size:
        ordered_chems = np.asarray(sorted(set(ordered_chems.tolist())), dtype=np.int64)
    if ordered_chems.size < int(holdout_n):
        raise SystemExit("Not enough eligible chem_ids for stratified holdout")

    # Choose one chem per RT bin (with random tie-breaking within each bin).
    n = int(ordered_chems.size)
    bins = np.array_split(np.arange(n, dtype=np.int64), int(holdout_n))
    picks: list[int] = []
    for b in bins:
        idx = int(rng.choice(b, size=1)[0])
        picks.append(int(ordered_chems[idx]))
    return sorted(set(picks))


def _infer_feature_cols(cols: Iterable[str]) -> list[str]:
    cols = tuple(map(str, cols))
    es_candidates = sorted([c for c in cols if c.startswith("ES_")])
    return list(infer_feature_columns(cols, es_candidates=es_candidates))


def _build_x(
    *,
    df: pd.DataFrame,
    feature_cols: list[str],
    clusters: np.ndarray,
    classes: np.ndarray,
    emb_chem_id: np.ndarray,
    emb_features: np.ndarray,
    mode: str,
) -> np.ndarray:
    run_x = df[feature_cols].fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    cluster_oh = _one_hot(df["species_cluster"].astype(int).to_numpy(), categories=clusters)
    class_oh = _one_hot(df["compound_class"].fillna(-1).astype(int).to_numpy(), categories=classes)
    emb = _embed_rows(
        chem_id=df["compound"].astype(int).to_numpy(),
        emb_chem_id=emb_chem_id,
        emb_features=emb_features,
    ).astype(np.float32, copy=False)

    parts: list[np.ndarray] = [run_x, cluster_oh]
    if mode in {"cluster_class"}:
        parts.append(class_oh)
    if mode in {"cluster_emb", "cluster_emb_inter"}:
        parts.append(emb)
    if mode in {"cluster_emb_inter"}:
        inter = (cluster_oh[:, :, None] * emb[:, None, :]).reshape(
            int(emb.shape[0]), int(cluster_oh.shape[1]) * int(emb.shape[1])
        )
        parts.append(inter.astype(np.float32, copy=False))

    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def _fit_and_eval(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    clusters: np.ndarray,
    classes: np.ndarray,
    emb_chem_id: np.ndarray,
    emb_features: np.ndarray,
    mode: str,
    alphas: list[float],
) -> dict[str, Any]:
    x_train = _build_x(
        df=train_df,
        feature_cols=feature_cols,
        clusters=clusters,
        classes=classes,
        emb_chem_id=emb_chem_id,
        emb_features=emb_features,
        mode=mode,
    )
    y_train = train_df["rt"].to_numpy(dtype=np.float64, copy=False)

    x_test = _build_x(
        df=test_df,
        feature_cols=feature_cols,
        clusters=clusters,
        classes=classes,
        emb_chem_id=emb_chem_id,
        emb_features=emb_features,
        mode=mode,
    )
    y_test = test_df["rt"].to_numpy(dtype=np.float64, copy=False)

    model = Pipeline(
        steps=[
            ("scale", StandardScaler(with_mean=True, with_std=True)),
            ("ridge", RidgeCV(alphas=np.asarray(alphas, dtype=np.float64))),
        ]
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    ridge: RidgeCV = model.named_steps["ridge"]  # type: ignore[assignment]
    return {
        "mode": mode,
        "n_train": int(y_train.size),
        "n_test": int(y_test.size),
        "alpha": float(ridge.alpha_),
        "rmse": _rmse(y_test, y_pred),
        "mae": _mae(y_test, y_pred),
    }


def _fit_and_eval_mlp(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    clusters: np.ndarray,
    classes: np.ndarray,
    emb_chem_id: np.ndarray,
    emb_features: np.ndarray,
    mode: str,
    seed: int,
) -> dict[str, Any]:
    x_train = _build_x(
        df=train_df,
        feature_cols=feature_cols,
        clusters=clusters,
        classes=classes,
        emb_chem_id=emb_chem_id,
        emb_features=emb_features,
        mode=mode,
    )
    y_train = train_df["rt"].to_numpy(dtype=np.float64, copy=False)

    x_test = _build_x(
        df=test_df,
        feature_cols=feature_cols,
        clusters=clusters,
        classes=classes,
        emb_chem_id=emb_chem_id,
        emb_features=emb_features,
        mode=mode,
    )
    y_test = test_df["rt"].to_numpy(dtype=np.float64, copy=False)

    mlp = MLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=200,
        early_stopping=True,
        n_iter_no_change=10,
        validation_fraction=0.1,
        random_state=int(seed),
    )
    model = Pipeline(
        steps=[
            ("scale", StandardScaler(with_mean=True, with_std=True)),
            ("mlp", mlp),
        ]
    )
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return {
        "mode": f"mlp_{mode}",
        "n_train": int(y_train.size),
        "n_test": int(y_test.size),
        "n_iter": int(mlp.n_iter_),
        "rmse": _rmse(y_test, y_pred),
        "mae": _mae(y_test, y_pred),
    }


def _run(cmd: list[str], *, cwd: Path) -> None:
    subprocess.run(cmd, check=True, cwd=str(cwd))


def _fit_and_eval_pymc_partial_pool(
    *,
    out_dir: Path,
    train_csv: Path,
    test_csv: Path,
    emb_path: Path,
    seed: int,
    theta_alpha_prior_sigma: float,
    advi_steps: int,
) -> dict[str, Any]:
    pymc_out = out_dir / "pymc_partial_pool"
    pymc_out.mkdir(parents=True, exist_ok=True)

    artifacts = train_pymc_partial_pool_ridge_from_csv(
        data_csv=train_csv,
        seed=int(seed),
        include_es_all=True,
        feature_center="global",
        lambda_slopes=3e-4,
        sigma_y_prior=0.05,
        chem_embeddings_path=emb_path,
        theta_alpha_prior_sigma=float(theta_alpha_prior_sigma),
        method="advi",
        advi_steps=int(advi_steps),
        advi_draws=30,
        advi_log_every=500,
        map_maxeval=50_000,
    )
    paths = write_pymc_partial_pool_ridge_artifacts(artifacts=artifacts, output_dir=pymc_out)

    eval_dir = pymc_out / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    label = "pymc_partial_pool"
    _run(
        [
            sys.executable,
            "-u",
            str(REPO_ROOT / "scripts/pipelines/eval_rt_partial_pool_backoff.py"),
            "--coeff-npz",
            str(paths["coeff_npz"]),
            "--backoff-npz",
            str(paths["backoff_npz"]),
            "--test-csv",
            str(test_csv),
            "--output-dir",
            str(eval_dir),
            "--label",
            label,
            "--chem-embeddings-path",
            str(emb_path),
            "--log-every-chunks",
            "0",
        ],
        cwd=REPO_ROOT,
    )

    metrics_json = eval_dir / f"rt_eval_partial_pool_backoff_{label}.json"
    if not metrics_json.exists():
        raise RuntimeError(f"Missing PyMC eval JSON: {metrics_json}")
    d = json.loads(metrics_json.read_text())
    m = d.get("metrics_all", {})
    return {
        "mode": "pymc_partial_pool_chem_linear",
        "rmse": float(m.get("rmse", float("nan"))),
        "mae": float(m.get("mae", float("nan"))),
        "coverage_95": float(m.get("coverage_95", float("nan"))),
        "paths": {
            "coeff_npz": str(paths["coeff_npz"]),
            "backoff_npz": str(paths["backoff_npz"]),
            "eval_json": str(metrics_json),
        },
    }


def main() -> None:
    args = parse_args()

    data_csv = args.data_csv
    if not data_csv.is_absolute():
        data_csv = (REPO_ROOT / data_csv).resolve()
    if not data_csv.exists():
        raise SystemExit(f"Input CSV not found: {data_csv}")

    out_dir = args.output_dir
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if bool(args.skip_existing) and (out_dir / "summary.json").exists():
        print(f"[skip-existing] Found {(out_dir / 'summary.json')}; skipping.")
        return

    emb_path = args.chem_embeddings_path
    if not emb_path.is_absolute():
        emb_path = (REPO_ROOT / emb_path).resolve()

    df = pd.read_csv(data_csv)
    required_cols = ["rt", "compound", "compound_class", "species_cluster"]
    if bool(args.include_pymc):
        required_cols += ["comp_id", "species"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise SystemExit(f"CSV missing required columns: {missing}")

    feature_cols = _infer_feature_cols(df.columns)
    needed_cols = list(dict.fromkeys([*required_cols, *feature_cols]))
    df = df[needed_cols].copy()
    df["compound_class"] = df["compound_class"].fillna(-1).astype(int)
    df["compound"] = df["compound"].astype(int)
    if "comp_id" in df.columns:
        df["comp_id"] = df["comp_id"].astype(int)
    df["species_cluster"] = df["species_cluster"].astype(int)
    if "species" in df.columns:
        df["species"] = df["species"].astype(int)

    holdout_chems = _choose_holdout_chems(
        df=df,
        holdout_n=int(args.holdout_n),
        seed=int(args.seed),
        strategy=str(args.holdout_strategy),
    )
    is_test = df["compound"].isin(set(holdout_chems))
    train_df = df.loc[~is_test].copy()
    test_df = df.loc[is_test].copy()
    if train_df.empty or test_df.empty:
        raise SystemExit("Split produced empty train or test")

    clusters = np.asarray(
        sorted(set(train_df["species_cluster"].astype(int).tolist())), dtype=np.int64
    )
    classes = np.asarray(
        sorted(set(train_df["compound_class"].astype(int).tolist())), dtype=np.int64
    )

    emb = load_chemberta_pca20(emb_path)
    alphas = _parse_float_list(args.alphas)

    results: list[dict[str, Any]] = []
    if bool(args.report_minimal):
        results.append(
            _fit_and_eval_mlp(
                train_df=train_df,
                test_df=test_df,
                feature_cols=feature_cols,
                clusters=clusters,
                classes=classes,
                emb_chem_id=emb.chem_id.astype(np.int64, copy=False),
                emb_features=emb.features.astype(np.float32, copy=False),
                mode="cluster_emb_inter",
                seed=int(args.seed),
            )
        )
    else:
        for mode in ["cluster_only", "cluster_class", "cluster_emb", "cluster_emb_inter"]:
            results.append(
                _fit_and_eval(
                    train_df=train_df,
                    test_df=test_df,
                    feature_cols=feature_cols,
                    clusters=clusters,
                    classes=classes,
                    emb_chem_id=emb.chem_id.astype(np.int64, copy=False),
                    emb_features=emb.features.astype(np.float32, copy=False),
                    mode=mode,
                    alphas=alphas,
                )
            )
        if bool(args.include_mlp):
            for mode in ["cluster_emb", "cluster_emb_inter"]:
                results.append(
                    _fit_and_eval_mlp(
                        train_df=train_df,
                        test_df=test_df,
                        feature_cols=feature_cols,
                        clusters=clusters,
                        classes=classes,
                        emb_chem_id=emb.chem_id.astype(np.int64, copy=False),
                        emb_features=emb.features.astype(np.float32, copy=False),
                        mode=mode,
                        seed=int(args.seed),
                    )
                )

    if bool(args.include_pymc):
        split_dir = out_dir / "splits"
        split_dir.mkdir(parents=True, exist_ok=True)
        train_csv = split_dir / "train.csv"
        test_csv = split_dir / "test.csv"
        train_df.to_csv(train_csv, index=False)
        test_df.to_csv(test_csv, index=False)
        results.append(
            _fit_and_eval_pymc_partial_pool(
                out_dir=out_dir,
                train_csv=train_csv,
                test_csv=test_csv,
                emb_path=emb_path,
                seed=int(args.seed),
                theta_alpha_prior_sigma=float(args.theta_alpha_prior_sigma),
                advi_steps=int(args.pymc_advi_steps),
            )
        )

    config = {
        "data_csv": str(data_csv),
        "chem_embeddings_path": str(emb_path),
        "seed": int(args.seed),
        "holdout_n": int(args.holdout_n),
        "holdout_strategy": str(args.holdout_strategy),
        "holdout_chems": holdout_chems,
        "rows_total": int(df.shape[0]),
        "rows_train": int(train_df.shape[0]),
        "rows_test": int(test_df.shape[0]),
        "feature_cols": feature_cols,
        "report_minimal": bool(args.report_minimal),
        "include_mlp": bool(args.include_mlp),
        "include_pymc": bool(args.include_pymc),
        "pymc_advi_steps": int(args.pymc_advi_steps),
        "theta_alpha_prior_sigma": float(args.theta_alpha_prior_sigma),
        "species_clusters_train": clusters.tolist(),
        "compound_classes_train": classes.tolist(),
    }

    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(results, indent=2))
    print(f"[done] Wrote {(out_dir / 'summary.json')}")
    for row in results:
        alpha = row.get("alpha")
        alpha_str = f" alpha={alpha:.3g}" if isinstance(alpha, (float, int)) else ""
        n_iter = row.get("n_iter")
        it_str = f" iters={n_iter}" if isinstance(n_iter, int) else ""
        cov = row.get("coverage_95")
        cov_str = f" cov95={float(cov):.3f}" if isinstance(cov, (float, int)) else ""
        print(
            f"{row['mode']}: rmse={row['rmse']:.4f} mae={row['mae']:.4f}{cov_str}{alpha_str}{it_str}"
        )


if __name__ == "__main__":
    main()
