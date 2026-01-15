#!/usr/bin/env python3
"""
Plot and summarize RT multilevel experiment outputs.

This script is intended for analyzing outputs produced by:
  - src/compassign/rt/train.sh

It reads:
  - per-model JSON summaries:
      */results/rt_eval_coeff_summaries_by_support_realtest.json
  - per-model per-group CSVs:
      */results/rt_eval_coeff_summaries_by_group_realtest.csv
  - species mapping:
      repo_export/lib<id>/species_mapping/merged_training_all_lib<id>_species_mapping.csv

and writes plots + compact CSV summaries under an output directory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)

DEFAULT_SUPPORT_BIN_ORDER = ["<= 1", "2-2", "3-5", "6-10", "11-20", "21-50", "51-100", "> 100"]

_ADVI_ELAPSED_MIN_RE = re.compile(r"\[advi\] iter=\d+/\d+ .*elapsed_min=(?P<min>\d+\.?\d*)")
_TAG_SAFE_RE = re.compile(r"[^A-Za-z0-9_-]+")

# More muted, paper-friendly colors (loosely based on seaborn's "deep" palette).
_MODEL_COLORS = {
    "pymc_collapsed_group_species": "#C44E52",  # muted red
    "pymc_pooled_species_comp_hier_supercat": "#8172B3",  # muted purple
    "pymc_pooled_species_comp_hier_supercat_cluster_supercat": "#55A868",  # muted green
    "pymc_pooled_species_chem_hier_cluster_supercat": "#8172B3",  # muted purple
    "pymc_collapsed_group_species_cluster": "#64B5CD",  # muted teal
    "pymc_collapsed_group_species_cluster_poly2": "#64B5CD",  # muted teal
    "sklearn_ridge_species_cluster": "#4C72B0",  # muted blue
    "lasso_eslasso_species_cluster": "#DD8452",  # muted orange
}

LASSO_CLUSTER_CSV = {
    "lasso_eslasso_species_cluster": "rt_eval_lasso_by_species_cluster_realtest.csv",
}

LASSO_SUPPORT_CSV = {
    "lasso_eslasso_species_cluster": "rt_eval_lasso_by_support_realtest.csv",
}

LASSO_GLOBAL_JSON = {
    "lasso_eslasso_species_cluster": "rt_eval_lasso_realtest.json",
}

COEFF_SUMMARY_MODELS_NO_ANCHOR = {
    "sklearn_ridge_species_cluster",
}


def _is_lasso_model(model: str) -> bool:
    return model in LASSO_GLOBAL_JSON or model in LASSO_SUPPORT_CSV or model in LASSO_CLUSTER_CSV


def _format_tag_for_title(tag_suffix: str) -> str:
    if not tag_suffix:
        return ""
    tag = tag_suffix.removeprefix("_")
    if tag == "full":
        return " (full evaluation)"
    return f" ({tag} evaluation)"


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot and summarize RT multilevel results under a run directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Run directory under output/ (e.g. output/rt_prod_YYYYMMDD_HHMMSS). "
            "If omitted, uses output/rt_prod_latest.txt."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for plots/CSVs (default: <run-dir>/plots).",
    )
    parser.add_argument(
        "--libs",
        type=str,
        default=None,
        help="Comma-separated lib ids to include (default: infer from run-dir/lib* folders).",
    )
    parser.add_argument(
        "--cap", type=str, default="cap100", help="Cap label under the run directory."
    )
    parser.add_argument(
        "--anchor",
        type=str,
        default="none",
        help="Anchor expansion to focus on for per-group plots (default: none).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=(
            "sklearn_ridge_species_cluster,"
            "pymc_pooled_species_comp_hier_supercat_cluster_supercat@none,"
            "lasso_eslasso_species_cluster"
        ),
        help=(
            "Comma-separated model directory names to compare in plots. "
            "Optionally specify a per-model anchor override as '<model>@<anchor>' (e.g. "
            "'pymc_collapsed_group_species_cluster@poly2'). "
            "Defaults to the report baselines (sklearn ridge supercategory + partial pooling + lasso supercategory)."
        ),
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help=(
            "Optional tag appended to output filenames (e.g. 'full' or 'candidates'). "
            "Useful for writing multiple plot sets into the same output directory."
        ),
    )
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_run_dir(run_dir: Path | None) -> Path:
    if run_dir is not None:
        p = run_dir
        if not p.is_absolute():
            p = (_repo_root() / p).resolve()
        return p
    latest = _repo_root() / "output" / "rt_prod_latest.txt"
    if not latest.exists():
        raise SystemExit(f"--run-dir not provided and latest pointer missing: {latest}")
    p = Path(latest.read_text().strip())
    if not p.is_absolute():
        p = (_repo_root() / p).resolve()
    if not p.exists():
        raise SystemExit(f"Run directory does not exist: {p}")
    return p


def _infer_libs(run_dir: Path, libs_csv: str | None) -> list[int]:
    if libs_csv:
        libs = [int(x.strip()) for x in libs_csv.split(",") if x.strip()]
        if not libs:
            raise SystemExit(f"--libs provided but empty: {libs_csv!r}")
        return sorted(set(libs))
    libs: list[int] = []
    for p in run_dir.glob("lib*"):
        if not p.is_dir():
            continue
        if not p.name.startswith("lib"):
            continue
        try:
            libs.append(int(p.name.removeprefix("lib")))
        except ValueError:
            continue
    if not libs:
        raise SystemExit(f"No lib*/ directories found under run dir: {run_dir}")
    return sorted(set(libs))


def _load_species_mapping(*, lib: int) -> pd.DataFrame:
    path = (
        _repo_root()
        / "repo_export"
        / f"lib{lib}"
        / "species_mapping"
        / (f"merged_training_all_lib{lib}_species_mapping.csv")
    )
    if not path.exists():
        raise SystemExit(f"Missing species mapping CSV: {path}")
    df = pd.read_csv(
        path, usecols=["species", "species_cluster", "species_group_raw"]
    ).drop_duplicates()
    df["species"] = df["species"].astype(int)
    df["species_cluster"] = df["species_cluster"].astype(int)
    df["species_group_raw"] = df["species_group_raw"].astype(str)
    return df


def _cluster_name_map(mapping_df: pd.DataFrame) -> dict[int, str]:
    return (
        mapping_df[["species_cluster", "species_group_raw"]]
        .drop_duplicates()
        .set_index("species_cluster")["species_group_raw"]
        .to_dict()
    )


def _cluster_category(species_group_raw: str) -> str:
    s = str(species_group_raw).lower()
    if "blood" in s:
        return "blood"
    if "urine" in s:
        return "urine"
    if "cells" in s or "plants" in s:
        return "cells"
    return "other"


def _safe_model_label(model_dir_name: str) -> str:
    mapping = {
        "pymc_collapsed_group_species": "No pooling (species)",
        "pymc_pooled_species_comp_hier_supercat": "Partial pooling (intercepts)",
        "pymc_collapsed_group_species_cluster": "Ridge (supercategory)",
        "pymc_collapsed_group_species_cluster_poly2": "Ridge (supercategory) + poly2",
        "pymc_pooled_species_comp_hier_supercat_cluster_supercat": "Ridge (partial pooling)",
        "pymc_pooled_species_chem_hier_cluster_supercat": "Chem hier",
        "sklearn_ridge_species_cluster": "Ridge (supercategory)",
        "lasso_eslasso_species_cluster": "Lasso (supercategory)",
    }
    return mapping.get(model_dir_name, model_dir_name)


def _build_model_labels(
    *,
    run_dir: Path,
    libs: Sequence[int],
    cap: str,
    default_anchor: str,
    model_specs: Sequence[ModelSpec],
) -> dict[str, str]:
    return {spec.display: _safe_model_label(spec.display) for spec in model_specs}


def _tag_suffix(tag: str | None) -> str:
    if tag is None:
        return ""
    tag = str(tag).strip()
    if not tag:
        return ""
    safe = _TAG_SAFE_RE.sub("-", tag).strip("-")
    if not safe:
        return ""
    return f"_{safe}"


@dataclass(frozen=True)
class ModelSpec:
    model: str
    anchor: str
    display: str


def _parse_model_specs(models_csv: str, *, default_anchor: str) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for raw in str(models_csv).split(","):
        raw = raw.strip()
        if not raw:
            continue
        if "@" in raw:
            model, anchor = raw.split("@", 1)
            model = model.strip()
            anchor = anchor.strip()
        else:
            model = raw
            anchor = str(default_anchor)
        if not model:
            continue
        if not anchor:
            raise SystemExit(f"Invalid --models entry (empty anchor): {raw!r}")

        display = str(model)
        if anchor != str(default_anchor):
            # Preserve stable naming for the baseline poly2 variant.
            if model == "pymc_collapsed_group_species_cluster" and anchor == "poly2":
                display = "pymc_collapsed_group_species_cluster_poly2"
            # Models that do not live under features_<anchor>/; ignore anchor override.
            elif (
                model in LASSO_CLUSTER_CSV
                or model in LASSO_SUPPORT_CSV
                or model in COEFF_SUMMARY_MODELS_NO_ANCHOR
            ):
                display = str(model)
            else:
                display = f"{model}_{anchor}"

        specs.append(ModelSpec(model=str(model), anchor=str(anchor), display=str(display)))

    if not specs:
        raise SystemExit(f"--models provided but empty: {models_csv!r}")
    return specs


def _scan_support_jsons(run_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for p in sorted(run_dir.rglob("rt_eval_coeff_summaries_by_support_realtest.json")):
        rel = p.relative_to(run_dir)
        if len(rel.parts) < 5:
            continue
        if not (rel.parts[0].startswith("lib") and rel.parts[1].startswith("cap")):
            continue
        try:
            lib = int(rel.parts[0].removeprefix("lib"))
        except ValueError:
            continue
        cap = str(rel.parts[1])

        # Expected:
        #   - libX/capY/features_Z/<model>/results/file.json
        #   - libX/capY/<model>/results/file.json (e.g. sklearn baselines)
        if (
            len(rel.parts) >= 6
            and rel.parts[2].startswith("features_")
            and rel.parts[4] == "results"
        ):
            anchor = str(rel.parts[2]).removeprefix("features_")
            model = str(rel.parts[3])
        elif len(rel.parts) >= 5 and rel.parts[3] == "results":
            anchor = "none"
            model = str(rel.parts[2])
        else:
            continue

        with p.open("r") as f:
            d = json.load(f)
        m = d.get("metrics", {})
        rows.append(
            {
                "lib": lib,
                "cap": cap,
                "anchor": anchor,
                "model": model,
                "group_col": d.get("group_col", ""),
                "rmse": float(m.get("rmse", float("nan"))),
                "mae": float(m.get("mae", float("nan"))),
                "cov95": float(m.get("coverage_95", float("nan"))),
                "pred_std_mean": float(m.get("pred_std_mean", float("nan"))),
                "interval_width_mean": float(m.get("interval_width_mean", float("nan"))),
                "n_test": int(d.get("n_test", 0)),
                "n_used": int(d.get("n_used", 0)),
                "skipped_missing_group": int(d.get("skipped_missing_group", 0)),
                "json_path": str(p),
            }
        )
    if not rows:
        raise SystemExit(f"No support JSONs found under run dir: {run_dir}")
    return pd.DataFrame(rows)


def _scan_lasso_global_jsons(run_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for p in sorted(run_dir.rglob("rt_eval_lasso*_realtest.json")):
        rel = p.relative_to(run_dir)
        if len(rel.parts) < 4:
            continue
        # Expected: libX/capY/<model>/results/file.json
        if not (rel.parts[0].startswith("lib") and rel.parts[1].startswith("cap")):
            continue
        model = str(rel.parts[2])
        if model not in LASSO_GLOBAL_JSON:
            continue
        if rel.parts[-1] != LASSO_GLOBAL_JSON[model]:
            continue
        try:
            lib = int(str(rel.parts[0]).removeprefix("lib"))
        except ValueError:
            continue
        cap = str(rel.parts[1])
        with p.open("r") as f:
            d = json.load(f)
        m = d.get("metrics", {})
        n_test = int(d.get("n_test_rows_seen", 0))
        n_used = int(d.get("n_rows_evaluated", 0))
        cov = float(m.get("coverage_window", m.get("coverage_95", float("nan"))))
        width = float(m.get("window_width_mean", m.get("interval_width_mean", float("nan"))))
        rows.append(
            {
                "lib": lib,
                "cap": cap,
                "anchor": "none",
                "model": model,
                "group_col": "species_cluster",
                "rmse": float(m.get("rmse", float("nan"))),
                "mae": float(m.get("mae", float("nan"))),
                "cov95": cov,
                "pred_std_mean": float("nan"),
                "interval_width_mean": width,
                "n_test": n_test,
                "n_used": n_used,
                "skipped_missing_group": max(0, int(n_test - n_used)),
                "json_path": str(p),
            }
        )
    return pd.DataFrame(rows)


def _train_elapsed_min_from_log(log_path: Path) -> float | None:
    if not log_path.exists():
        return None
    last: float | None = None
    for line in log_path.read_text(errors="ignore").splitlines():
        m = _ADVI_ELAPSED_MIN_RE.search(line)
        if not m:
            continue
        try:
            last = float(m.group("min"))
        except ValueError:
            continue
    return last


def _attach_train_times(*, global_df: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    df = global_df.copy()
    log_dir = run_dir / "logs"
    minutes: list[float] = []
    for row in df.itertuples(index=False):
        log_path = log_dir / f"lib{int(row.lib)}_{row.cap}_{row.anchor}_{row.model}.train.log"
        if not log_path.exists() and str(row.model) in COEFF_SUMMARY_MODELS_NO_ANCHOR:
            log_path = log_dir / f"lib{int(row.lib)}_{row.cap}_{row.model}.train.log"
        elapsed = _train_elapsed_min_from_log(log_path)
        minutes.append(float("nan") if elapsed is None else float(elapsed))
    df["train_elapsed_min"] = np.asarray(minutes, dtype=float)
    return df


def _support_metrics_long(run_dir: Path, *, libs: Sequence[int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    libs_set = set(map(int, libs))
    for p in sorted(run_dir.rglob("rt_eval_coeff_summaries_by_support_realtest.json")):
        rel = p.relative_to(run_dir)
        if len(rel.parts) < 5:
            continue
        if not (rel.parts[0].startswith("lib") and rel.parts[1].startswith("cap")):
            continue
        try:
            lib = int(rel.parts[0].removeprefix("lib"))
        except ValueError:
            continue
        if lib not in libs_set:
            continue
        cap = str(rel.parts[1])
        if (
            len(rel.parts) >= 6
            and rel.parts[2].startswith("features_")
            and rel.parts[4] == "results"
        ):
            anchor = str(rel.parts[2]).removeprefix("features_")
            model = str(rel.parts[3])
        elif len(rel.parts) >= 5 and rel.parts[3] == "results":
            anchor = "none"
            model = str(rel.parts[2])
        else:
            continue

        with p.open("r") as f:
            d = json.load(f)
        for row in d.get("support_metrics", []):
            rows.append(
                {
                    "lib": lib,
                    "cap": cap,
                    "anchor": anchor,
                    "model": model,
                    "support_bin": str(row.get("support_bin", "")),
                    "n_groups": int(row.get("n_groups", 0)),
                    "n_obs_test": int(row.get("n_obs_test", 0)),
                    "rmse": float(row.get("rmse", float("nan"))),
                    "mae": float(row.get("mae", float("nan"))),
                    "cov95": float(row.get("coverage_95", float("nan"))),
                    "pred_std_mean": float(row.get("pred_std_mean", float("nan"))),
                    "interval_width_mean": float(row.get("interval_width_mean", float("nan"))),
                }
            )
    if not rows:
        raise SystemExit("No support metrics rows found (support_metrics missing from JSONs?)")
    df = pd.DataFrame(rows)
    df["support_bin"] = pd.Categorical(
        df["support_bin"], categories=DEFAULT_SUPPORT_BIN_ORDER, ordered=True
    )
    return df


def _append_lasso_support_metrics(
    *,
    support_df: pd.DataFrame,
    run_dir: Path,
    libs: Sequence[int],
    cap: str,
    anchor: str,
    models: Sequence[str],
) -> pd.DataFrame:
    extra: list[pd.DataFrame] = []
    for lib in libs:
        for model in models:
            if model not in LASSO_SUPPORT_CSV:
                continue
            path = run_dir / f"lib{lib}" / cap / model / "results" / LASSO_SUPPORT_CSV[model]
            if not path.exists():
                LOGGER.info("Skip missing lasso support CSV: %s", path)
                continue
            df = pd.read_csv(path)
            required_base = {"support_bin", "n_obs_test", "rmse"}
            missing_base = required_base - set(df.columns)
            if missing_base:
                raise SystemExit(
                    f"Lasso by-support CSV missing columns {sorted(missing_base)}: {path}"
                )
            cov_col = "coverage_window" if "coverage_window" in df.columns else "coverage_95"
            width_col = (
                "window_width_mean" if "window_width_mean" in df.columns else "interval_width_mean"
            )
            if cov_col not in df.columns:
                raise SystemExit(
                    (
                        "Lasso by-support CSV missing coverage column "
                        f"(expected coverage_window or coverage_95): {path}"
                    )
                )
            if width_col not in df.columns:
                raise SystemExit(
                    (
                        "Lasso by-support CSV missing width column "
                        f"(expected window_width_mean or interval_width_mean): {path}"
                    )
                )
            df = df[df["n_obs_test"].fillna(0).astype(int) > 0].copy()
            if df.empty:
                continue
            extra.append(
                pd.DataFrame(
                    {
                        "lib": int(lib),
                        "cap": str(cap),
                        "anchor": str(anchor),
                        "model": str(model),
                        "support_bin": df["support_bin"].astype(str),
                        "n_groups": df.get("n_groups", 0),
                        "n_obs_test": df["n_obs_test"].astype(int),
                        "rmse": df["rmse"].astype(float),
                        "mae": df.get("mae", float("nan")),
                        "cov95": df[cov_col].astype(float),
                        "pred_std_mean": df.get("pred_std_mean", float("nan")),
                        "interval_width_mean": df[width_col].astype(float),
                    }
                )
            )

    if not extra:
        return support_df
    merged = pd.concat([support_df, *extra], ignore_index=True)
    merged["support_bin"] = pd.Categorical(
        merged["support_bin"].astype(str), categories=DEFAULT_SUPPORT_BIN_ORDER, ordered=True
    )
    return merged


def _load_group_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "n_obs_test" not in df.columns:
        raise SystemExit(f"Unexpected by-group CSV missing n_obs_test: {path}")
    df = df[df["n_obs_test"].fillna(0).astype(int) > 0].copy()
    df = df[df["rmse"].notna()].copy()
    return df


def _aggregate_by_cluster(
    *, group_csv_path: Path, mapping_df: pd.DataFrame, model_dir: str
) -> pd.DataFrame:
    df = _load_group_csv(group_csv_path)
    if df.empty:
        return df

    if "species_cluster" in df.columns:
        df["species_cluster"] = df["species_cluster"].astype(int)
    elif "species" in df.columns:
        df["species"] = df["species"].astype(int)
        df = df.merge(mapping_df, on="species", how="left", validate="m:1")
        if df["species_cluster"].isna().any():
            missing = int(df["species_cluster"].isna().sum())
            LOGGER.warning(
                "Missing species_cluster mapping for %s rows (%s)", missing, group_csv_path
            )
        df["species_cluster"] = df["species_cluster"].fillna(-1).astype(int)
    else:
        raise SystemExit(
            f"By-group CSV missing both species and species_cluster columns: {group_csv_path}"
        )

    n = df["n_obs_test"].astype(float)
    sse = np.square(df["rmse"].astype(float)) * n
    covered = df["coverage_95"].astype(float) * n
    width = df["interval_width_mean"].astype(float) * n

    tmp = df.assign(sse=sse, covered=covered, width=width)
    agg = (
        tmp.groupby("species_cluster", as_index=False)
        .agg(
            n_obs_test=("n_obs_test", "sum"),
            sse=("sse", "sum"),
            covered=("covered", "sum"),
            width=("width", "sum"),
        )
        .copy()
    )
    agg["rmse"] = np.sqrt(agg["sse"] / agg["n_obs_test"].astype(float))
    agg["cov95"] = agg["covered"] / agg["n_obs_test"].astype(float)
    agg["interval_width_mean"] = agg["width"] / agg["n_obs_test"].astype(float)
    agg["model"] = model_dir
    return agg


def _load_lasso_cluster_csv(*, path: Path, model_dir: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    df = pd.read_csv(path)
    required_base = {"species_cluster", "n_obs", "rmse"}
    missing_base = required_base - set(df.columns)
    if missing_base:
        raise SystemExit(f"Lasso by-cluster CSV missing columns {sorted(missing_base)}: {path}")
    cov_col = "coverage_window" if "coverage_window" in df.columns else "coverage_95"
    width_col = "window_width_mean" if "window_width_mean" in df.columns else "interval_width_mean"
    if cov_col not in df.columns:
        raise SystemExit(
            (
                "Lasso by-cluster CSV missing coverage column "
                f"(expected coverage_window or coverage_95): {path}"
            )
        )
    if width_col not in df.columns:
        raise SystemExit(
            (
                "Lasso by-cluster CSV missing width column "
                f"(expected window_width_mean or interval_width_mean): {path}"
            )
        )
    df = df[df["n_obs"].fillna(0).astype(int) > 0].copy()
    if df.empty:
        return df

    df["species_cluster"] = df["species_cluster"].astype(int)
    n = df["n_obs"].astype(float)
    sse = np.square(df["rmse"].astype(float)) * n
    covered = df[cov_col].astype(float) * n
    width = df[width_col].astype(float) * n
    return pd.DataFrame(
        {
            "species_cluster": df["species_cluster"].astype(int),
            "n_obs_test": df["n_obs"].astype(int),
            "sse": sse.astype(float),
            "covered": covered.astype(float),
            "width": width.astype(float),
            "rmse": df["rmse"].astype(float),
            "cov95": df[cov_col].astype(float),
            "interval_width_mean": df[width_col].astype(float),
            "model": str(model_dir),
        }
    )


def _write_cluster_and_category_summaries(
    *,
    run_dir: Path,
    out_dir: Path,
    libs: Sequence[int],
    cap: str,
    anchor: str,
    models: Sequence[ModelSpec],
    tag_suffix: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)

    cluster_rows: list[pd.DataFrame] = []
    for lib in libs:
        mapping_df = _load_species_mapping(lib=int(lib))
        cluster_name = _cluster_name_map(mapping_df)

        for spec in models:
            model = str(spec.model)
            display = str(spec.display)
            model_anchor = str(spec.anchor)
            if model in LASSO_CLUSTER_CSV:
                lasso_csv = (
                    run_dir / f"lib{lib}" / cap / model / "results" / LASSO_CLUSTER_CSV[model]
                )
                if not lasso_csv.exists():
                    LOGGER.info("Skip missing lasso cluster CSV: %s", lasso_csv)
                    continue
                agg = _load_lasso_cluster_csv(path=lasso_csv, model_dir=display)
                if agg.empty:
                    continue
                agg["lib"] = int(lib)
                agg["cap"] = str(cap)
                agg["anchor"] = str(anchor)
                agg["species_group_raw"] = (
                    agg["species_cluster"].map(cluster_name).fillna("UNKNOWN")
                )
                agg["category"] = agg["species_group_raw"].map(_cluster_category)
                cluster_rows.append(agg)
                continue

            if model in COEFF_SUMMARY_MODELS_NO_ANCHOR:
                group_csv = (
                    run_dir
                    / f"lib{lib}"
                    / cap
                    / model
                    / "results"
                    / "rt_eval_coeff_summaries_by_group_realtest.csv"
                )
                if not group_csv.exists():
                    LOGGER.info("Skip missing group CSV: %s", group_csv)
                    continue
                agg = _aggregate_by_cluster(
                    group_csv_path=group_csv, mapping_df=mapping_df, model_dir=display
                )
                if agg.empty:
                    continue
                agg["lib"] = int(lib)
                agg["cap"] = str(cap)
                agg["anchor"] = str(anchor)
                agg["species_group_raw"] = (
                    agg["species_cluster"].map(cluster_name).fillna("UNKNOWN")
                )
                agg["category"] = agg["species_group_raw"].map(_cluster_category)
                cluster_rows.append(agg)
                continue

            group_csv = (
                run_dir
                / f"lib{lib}"
                / cap
                / f"features_{model_anchor}"
                / model
                / "results"
                / "rt_eval_coeff_summaries_by_group_realtest.csv"
            )
            if not group_csv.exists():
                LOGGER.info("Skip missing group CSV: %s", group_csv)
                continue
            agg = _aggregate_by_cluster(
                group_csv_path=group_csv, mapping_df=mapping_df, model_dir=display
            )
            if agg.empty:
                continue
            agg["lib"] = int(lib)
            agg["cap"] = str(cap)
            agg["anchor"] = str(anchor)
            agg["species_group_raw"] = agg["species_cluster"].map(cluster_name).fillna("UNKNOWN")
            agg["category"] = agg["species_group_raw"].map(_cluster_category)
            cluster_rows.append(agg)

    if not cluster_rows:
        raise SystemExit(
            "No per-group CSVs found for requested libs/models; cannot build cluster summaries."
        )
    cluster_df = pd.concat(cluster_rows, ignore_index=True)
    cluster_csv = out_dir / f"by_species_cluster{tag_suffix}.csv"
    cluster_df.to_csv(cluster_csv, index=False)
    LOGGER.info("Wrote %s", cluster_csv)

    cat_df = (
        cluster_df.groupby(["lib", "cap", "anchor", "model", "category"], as_index=False)
        .agg(
            n_obs_test=("n_obs_test", "sum"),
            sse=("sse", "sum"),
            covered=("covered", "sum"),
            width=("width", "sum"),
        )
        .copy()
    )
    cat_df["rmse"] = np.sqrt(cat_df["sse"] / cat_df["n_obs_test"].astype(float))
    cat_df["cov95"] = cat_df["covered"] / cat_df["n_obs_test"].astype(float)
    cat_df["interval_width_mean"] = cat_df["width"] / cat_df["n_obs_test"].astype(float)
    cat_csv = out_dir / f"by_category{tag_suffix}.csv"
    cat_df.to_csv(cat_csv, index=False)
    LOGGER.info("Wrote %s", cat_csv)

    return cluster_df, cat_df


def _plot_per_lib_cluster_panels(
    *,
    cluster_df: pd.DataFrame,
    out_dir: Path,
    libs: Sequence[int],
    models: Sequence[str],
    model_labels: dict[str, str],
    cap: str,
    anchor: str,
    tag_suffix: str = "",
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        raise SystemExit("matplotlib is required for plotting but could not be imported") from exc

    present_models = set(cluster_df["model"].astype(str))
    order_models = [m for m in models if m in present_models]
    if not order_models:
        LOGGER.warning("No requested models present in by-species-cluster summary; skipping plots.")
        return

    tag_label = _format_tag_for_title(tag_suffix)

    for lib in libs:
        df = cluster_df[cluster_df["lib"] == int(lib)].copy()
        if df.empty:
            continue

        # Stable x-axis order.
        order = (
            df[["species_cluster", "species_group_raw"]]
            .drop_duplicates()
            .sort_values("species_cluster")
            .reset_index(drop=True)
        )
        cluster_order = order["species_cluster"].astype(int).tolist()
        x_labels = order["species_group_raw"].astype(str).tolist()
        x = np.arange(len(cluster_order))

        fig, axes = plt.subplots(
            nrows=3,
            ncols=1,
            figsize=(max(9.0, 1.3 * len(x_labels)), 9.0),
            sharex=True,
        )
        metrics = [
            ("rmse", "RMSE (min)"),
            ("cov95", "Cov95"),
            ("interval_width_mean", "Mean RT window width (min)"),
        ]

        for ax, (metric, ylabel) in zip(axes, metrics, strict=True):
            plot_models = order_models
            if metric == "cov95":
                plot_models = [m for m in order_models if not _is_lasso_model(m)]
                if not plot_models:
                    ax.text(
                        0.5,
                        0.5,
                        "Cov95 not defined for lasso window",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_axis_off()
                    continue

            bar_width = 0.8 / max(1, len(plot_models))
            for i, model in enumerate(plot_models):
                sub = df[df["model"] == model].set_index("species_cluster")
                y = [
                    float(sub.loc[sc, metric]) if sc in sub.index else float("nan")
                    for sc in cluster_order
                ]
                ax.bar(
                    x + (i - (len(plot_models) - 1) / 2) * bar_width,
                    y,
                    width=bar_width,
                    label=model_labels.get(model, _safe_model_label(model)),
                    color=_MODEL_COLORS.get(model, None),
                    alpha=0.85,
                )
            ax.set_ylabel(ylabel)
            ax.grid(True, axis="y", alpha=0.3)
            if metric == "cov95":
                ax.axhline(0.95, color="black", linewidth=1, linestyle="--", alpha=0.6)

        for ax in axes[:-1]:
            ax.label_outer()

        axes[-1].set_xticks(x)
        axes[-1].set_xticklabels(x_labels, rotation=90, ha="center")
        fig.suptitle(
            f"Library {lib}: Metrics by species_cluster for models trained on {cap} and evaluated on realtest{tag_label}.",
            y=1.01,
        )

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                borderaxespad=0.0,
                fontsize=8,
                frameon=False,
            )

        fig.tight_layout()
        out_path = out_dir / f"lib{lib}_by_species_cluster_anchor_{anchor}{tag_suffix}.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        LOGGER.info("Wrote %s", out_path)


def _plot_support_curves(
    *,
    support_df: pd.DataFrame,
    out_dir: Path,
    libs: Sequence[int],
    models: Sequence[str],
    model_labels: dict[str, str],
    anchor: str,
    cap: str,
    tag_suffix: str = "",
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        raise SystemExit("matplotlib is required for plotting but could not be imported") from exc

    present_models = set(support_df["model"].astype(str))
    order_models = [m for m in models if m in present_models]
    if not order_models:
        LOGGER.warning("No requested models present in support metrics; skipping support plots.")
        return

    tag_label = _format_tag_for_title(tag_suffix)

    for lib in libs:
        df = support_df[
            (support_df["lib"] == int(lib))
            & (support_df["cap"] == str(cap))
            & (support_df["anchor"] == str(anchor))
            & (support_df["model"].isin(order_models))
        ].copy()
        if df.empty:
            continue

        bins = [b for b in DEFAULT_SUPPORT_BIN_ORDER if b in set(df["support_bin"].astype(str))]
        x = np.arange(len(bins))

        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10.0, 8.0), sharex=True)
        metrics = [
            ("rmse", "RMSE (min)"),
            ("cov95", "Cov95"),
            ("interval_width_mean", "Mean RT window width (min)"),
        ]
        for ax, (metric, ylabel) in zip(axes, metrics, strict=True):
            plot_models = order_models
            if metric == "cov95":
                plot_models = [m for m in order_models if not _is_lasso_model(m)]
                if not plot_models:
                    ax.text(
                        0.5,
                        0.5,
                        "Cov95 not defined for lasso window",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_axis_off()
                    continue

            bar_width = 0.8 / max(1, len(plot_models))
            for i, model in enumerate(plot_models):
                sub = df[df["model"] == model].set_index("support_bin")
                y = [float(sub.loc[b, metric]) if b in sub.index else float("nan") for b in bins]
                ax.bar(
                    x + (i - (len(plot_models) - 1) / 2) * bar_width,
                    y,
                    width=bar_width,
                    label=model_labels.get(model, _safe_model_label(model)),
                    color=_MODEL_COLORS.get(model, None),
                    alpha=0.85,
                )
            ax.set_ylabel(ylabel)
            ax.grid(True, axis="y", alpha=0.3)
            if metric == "cov95":
                ax.axhline(0.95, color="black", linewidth=1, linestyle="--", alpha=0.6)

        axes[2].set_xticks(x)
        axes[2].set_xticklabels(bins, rotation=90, ha="center")
        fig.suptitle(
            f"Library {lib}: Metrics by training support bin for models trained on {cap} and evaluated on realtest{tag_label}.",
            y=1.01,
        )

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(
                handles,
                labels,
                loc="upper left",
                bbox_to_anchor=(1.01, 1.0),
                borderaxespad=0.0,
                fontsize=8,
                frameon=False,
            )

        fig.tight_layout()
        out_path = out_dir / f"lib{lib}_by_support_bin_anchor_{anchor}{tag_suffix}.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        LOGGER.info("Wrote %s", out_path)


def _plot_global_comparison(
    *,
    global_df: pd.DataFrame,
    out_dir: Path,
    libs: Sequence[int],
    models: Sequence[str],
    model_labels: dict[str, str],
    cap: str,
    anchor: str,
    tag_suffix: str = "",
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        raise SystemExit("matplotlib is required for plotting but could not be imported") from exc

    present_models = set(global_df["model"].astype(str))
    order_models = [m for m in models if m in present_models]
    if not order_models:
        LOGGER.warning("No requested models present in global metrics; skipping global plots.")
        return

    metrics = [
        ("rmse", "RMSE (min)", "linear"),
        ("cov95", "Cov95", "coverage"),
        ("interval_width_mean", "Mean RT window width (min)", "linear"),
    ]

    tag_label = _format_tag_for_title(tag_suffix)

    for lib in libs:
        df = global_df[
            (global_df["lib"] == int(lib))
            & (global_df["cap"] == str(cap))
            & (global_df["anchor"] == str(anchor))
            & (global_df["model"].isin(order_models))
        ].copy()
        if df.empty:
            continue
        df = df.set_index("model").reindex(order_models)

        n_metrics = int(len(metrics))
        # Make the global comparison taller so the x tick labels (rotated 90 degrees) are readable.
        fig, axes = plt.subplots(
            nrows=1, ncols=n_metrics, figsize=(3.6 * n_metrics, 4.8), squeeze=False
        )
        axes_flat = axes.reshape(-1).tolist()

        for ax, (col, ylabel, kind) in zip(axes_flat[:n_metrics], metrics, strict=True):
            plot_models = order_models
            if kind == "coverage":
                plot_models = [m for m in order_models if not _is_lasso_model(m)]
                if not plot_models:
                    ax.text(
                        0.5,
                        0.5,
                        "Cov95 not defined for lasso window",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    ax.set_axis_off()
                    continue

            df_metric = df.reindex(plot_models)
            x = np.arange(len(plot_models))
            labels = [model_labels.get(m, _safe_model_label(m)) for m in plot_models]
            colors = [_MODEL_COLORS.get(m, None) for m in plot_models]
            y = df_metric[col].astype(float).to_numpy()
            ax.bar(x, y, color=colors, alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=90, ha="center")
            ax.set_ylabel(ylabel)
            ax.grid(True, axis="y", alpha=0.3)
            if kind == "coverage":
                ax.axhline(0.95, color="black", linewidth=1, linestyle="--", alpha=0.6)
            if kind == "log":
                # Some runs intentionally omit training-time logging, in which case the extracted
                # values can be all-NaN. Guard log scaling to avoid hard failures.
                has_positive = bool(np.any(np.isfinite(y) & (y > 0)))
                if has_positive:
                    ax.set_yscale("log")

        for ax in axes_flat[n_metrics:]:
            ax.set_axis_off()

        fig.suptitle(
            f"Library {lib}: Global metrics for models trained on {cap} and evaluated on realtest{tag_label}.",
            y=1.01,
        )
        plt.tight_layout()
        out_path = out_dir / f"lib{lib}_global_comparison_anchor_{anchor}{tag_suffix}.png"
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        LOGGER.info("Wrote %s", out_path)


def main() -> None:
    _setup_logging()
    args = parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"Run directory does not exist: {run_dir}")

    libs = _infer_libs(run_dir, args.libs)
    cap = str(args.cap)
    default_anchor = str(args.anchor)
    model_specs = _parse_model_specs(str(args.models), default_anchor=default_anchor)
    anchors = sorted(set(s.anchor for s in model_specs))
    anchor_label = default_anchor if len(anchors) == 1 else "mixed"
    models = [s.display for s in model_specs]
    out_dir = args.out_dir or (run_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag_suffix = _tag_suffix(args.tag)
    model_labels = _build_model_labels(
        run_dir=run_dir,
        libs=libs,
        cap=cap,
        default_anchor=default_anchor,
        model_specs=model_specs,
    )

    LOGGER.info("RUN_DIR=%s", run_dir)
    LOGGER.info("OUT_DIR=%s", out_dir)
    LOGGER.info("LIBS=%s", libs)
    LOGGER.info("CAP=%s ANCHOR=%s", cap, anchor_label)
    LOGGER.info("MODELS=%s", [f"{s.model}@{s.anchor}->{s.display}" for s in model_specs])
    if tag_suffix:
        LOGGER.info("TAG=%s", tag_suffix.removeprefix("_"))

    global_df = _scan_support_jsons(run_dir)
    lasso_global = _scan_lasso_global_jsons(run_dir)
    if not lasso_global.empty:
        global_df = pd.concat([global_df, lasso_global], ignore_index=True)
    global_df = _attach_train_times(global_df=global_df, run_dir=run_dir)
    global_df["model_label"] = (
        global_df["model"]
        .astype(str)
        .map(lambda m: model_labels.get(str(m), _safe_model_label(str(m))))
    )
    global_metrics_csv = out_dir / f"global_metrics{tag_suffix}.csv"
    global_df.to_csv(global_metrics_csv, index=False)
    LOGGER.info("Wrote %s", global_metrics_csv)

    libs_set = set(map(int, libs))
    selected_parts: list[pd.DataFrame] = []
    for spec in model_specs:
        mask = (
            (global_df["lib"].isin(libs_set))
            & (global_df["cap"] == cap)
            & (global_df["model"] == spec.model)
        )
        if spec.model not in LASSO_GLOBAL_JSON and spec.model not in COEFF_SUMMARY_MODELS_NO_ANCHOR:
            mask &= global_df["anchor"] == spec.anchor
        sub = global_df[mask].copy()
        if sub.empty:
            LOGGER.warning(
                "No global metrics for model=%s anchor=%s (cap=%s)", spec.model, spec.anchor, cap
            )
            continue
        sub["anchor"] = str(anchor_label)
        sub["model"] = str(spec.display)
        selected_parts.append(sub)
    global_selected = (
        pd.concat(selected_parts, ignore_index=True)
        if selected_parts
        else global_df.iloc[:0].copy()
    )
    if not global_selected.empty:
        global_selected["model_label"] = (
            global_selected["model"]
            .astype(str)
            .map(lambda m: model_labels.get(str(m), _safe_model_label(str(m))))
        )
    global_selected_csv = out_dir / f"global_metrics_selected{tag_suffix}.csv"
    global_selected.to_csv(global_selected_csv, index=False)
    LOGGER.info("Wrote %s", global_selected_csv)

    cluster_df, cat_df = _write_cluster_and_category_summaries(
        run_dir=run_dir,
        out_dir=out_dir,
        libs=libs,
        cap=cap,
        anchor=anchor_label,
        models=model_specs,
        tag_suffix=tag_suffix,
    )
    _plot_per_lib_cluster_panels(
        cluster_df=cluster_df,
        out_dir=out_dir,
        libs=libs,
        models=models,
        model_labels=model_labels,
        cap=cap,
        anchor=anchor_label,
        tag_suffix=tag_suffix,
    )

    support_long = _support_metrics_long(run_dir, libs=libs)
    support_parts: list[pd.DataFrame] = []
    for spec in model_specs:
        if spec.model in LASSO_SUPPORT_CSV:
            continue
        mask = (support_long["lib"].isin(libs_set)) & (support_long["cap"] == cap)
        mask &= support_long["model"] == spec.model
        if spec.model not in COEFF_SUMMARY_MODELS_NO_ANCHOR:
            mask &= support_long["anchor"] == spec.anchor
        sub = support_long[mask].copy()
        if sub.empty:
            LOGGER.warning(
                "No support metrics for model=%s anchor=%s (cap=%s)", spec.model, spec.anchor, cap
            )
            continue
        sub["anchor"] = str(anchor_label)
        sub["model"] = str(spec.display)
        support_parts.append(sub)
    support_df = (
        pd.concat(support_parts, ignore_index=True)
        if support_parts
        else support_long.iloc[:0].copy()
    )
    support_df = _append_lasso_support_metrics(
        support_df=support_df,
        run_dir=run_dir,
        libs=libs,
        cap=cap,
        anchor=anchor_label,
        models=[s.model for s in model_specs],
    )
    _plot_support_curves(
        support_df=support_df,
        out_dir=out_dir,
        libs=libs,
        models=models,
        model_labels=model_labels,
        anchor=anchor_label,
        cap=cap,
        tag_suffix=tag_suffix,
    )

    _plot_global_comparison(
        global_df=global_selected,
        out_dir=out_dir,
        libs=libs,
        models=models,
        model_labels=model_labels,
        cap=cap,
        anchor=anchor_label,
        tag_suffix=tag_suffix,
    )


if __name__ == "__main__":
    main()
