#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd


LOGGER = logging.getLogger(__name__)


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize PyMC collapsed ridge vs sklearn ridge (Stage1CoeffSummaries) on realtest."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Run directory under output/ (e.g. output/rt_pymc_multilevel_cap100_YYYYMMDD_HHMMSS). "
            "If omitted, uses output/rt_pymc_multilevel_cap100_latest.txt."
        ),
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
        "--out-csv",
        type=Path,
        default=None,
        help="Output CSV path (default: <run-dir>/analysis/pymc_vs_sklearn_collapsed_ridge.csv).",
    )
    parser.add_argument(
        "--out-support-csv",
        type=Path,
        default=None,
        help=(
            "Output per-support-bin CSV path (default: "
            "<run-dir>/analysis/pymc_vs_sklearn_collapsed_ridge_support.csv)."
        ),
    )
    return parser.parse_args()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_run_dir(run_dir: Path | None) -> Path:
    if run_dir is not None:
        p = run_dir
        if not p.is_absolute():
            p = (_repo_root() / p).resolve()
        return p
    latest = _repo_root() / "output" / "rt_pymc_multilevel_cap100_latest.txt"
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
    out: list[int] = []
    for p in sorted(run_dir.glob("lib*")):
        if not p.is_dir() or not p.name.startswith("lib"):
            continue
        try:
            out.append(int(p.name.removeprefix("lib")))
        except ValueError:
            continue
    if not out:
        raise SystemExit(f"Could not infer libs under {run_dir}; pass --libs explicitly.")
    return out


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Missing evaluation JSON: {path}")
    return json.loads(path.read_text())


def _global_row(*, lib: int, kind: str, payload: dict[str, Any]) -> dict[str, object]:
    m = payload.get("metrics", {})
    n_test = int(payload.get("n_test", 0) or 0)
    n_used = int(payload.get("n_used", 0) or 0)
    frac_used = (float(n_used) / float(n_test)) if n_test > 0 else float("nan")
    return {
        "lib": int(lib),
        "model_kind": str(kind),
        "rmse": float(m.get("rmse", float("nan"))),
        "mae": float(m.get("mae", float("nan"))),
        "cov95": float(m.get("coverage_95", float("nan"))),
        "pred_std_mean": float(m.get("pred_std_mean", float("nan"))),
        "interval_width_mean": float(m.get("interval_width_mean", float("nan"))),
        "n_test": int(n_test),
        "n_used": int(n_used),
        "frac_used": float(frac_used),
        "skipped_missing_group": int(payload.get("skipped_missing_group", 0) or 0),
        "pred_std_scale": float(payload.get("pred_std_scale", float("nan"))),
        "interval_mass": float(payload.get("interval", float("nan"))),
        "coeff_npz": str(payload.get("coeff_npz", "")),
        "test_csv": str(payload.get("test_csv", "")),
    }


def _support_rows(*, lib: int, kind: str, payload: dict[str, Any]) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for row in payload.get("support_metrics", []) or []:
        out.append(
            {
                "lib": int(lib),
                "model_kind": str(kind),
                "support_bin": str(row.get("support_bin", "")),
                "n_groups": int(row.get("n_groups", 0) or 0),
                "n_groups_with_test": int(row.get("n_groups_with_test", 0) or 0),
                "n_obs_test": int(row.get("n_obs_test", 0) or 0),
                "rmse": float(row.get("rmse", float("nan"))),
                "mae": float(row.get("mae", float("nan"))),
                "cov95": float(row.get("coverage_95", float("nan"))),
                "pred_std_mean": float(row.get("pred_std_mean", float("nan"))),
                "interval_width_mean": float(row.get("interval_width_mean", float("nan"))),
                "group_rmse_mean": float(row.get("rmse_mean", float("nan"))),
                "group_rmse_p90": float(row.get("rmse_p90", float("nan"))),
                "group_rmse_p99": float(row.get("rmse_p99", float("nan"))),
            }
        )
    return out


def main() -> None:
    _setup_logging()
    args = parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    libs = _infer_libs(run_dir, args.libs)
    cap = str(args.cap)

    out_csv = args.out_csv or (run_dir / "analysis" / "pymc_vs_sklearn_collapsed_ridge.csv")
    out_support_csv = args.out_support_csv or (
        run_dir / "analysis" / "pymc_vs_sklearn_collapsed_ridge_support.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_support_csv.parent.mkdir(parents=True, exist_ok=True)

    global_rows: list[dict[str, object]] = []
    support_rows: list[dict[str, object]] = []

    for lib in libs:
        pymc_json = (
            run_dir
            / f"lib{lib}"
            / cap
            / "features_none"
            / "pymc_collapsed_group_species_cluster"
            / "results"
            / "rt_eval_coeff_summaries_by_support_realtest.json"
        )
        skl_json = (
            run_dir
            / f"lib{lib}"
            / cap
            / "sklearn_ridge_species_cluster"
            / "results"
            / "rt_eval_coeff_summaries_by_support_realtest.json"
        )

        pymc = _read_json(pymc_json)
        skl = _read_json(skl_json)

        global_rows.append(_global_row(lib=lib, kind="pymc_collapsed", payload=pymc))
        global_rows.append(_global_row(lib=lib, kind="sklearn_ridge", payload=skl))

        support_rows.extend(_support_rows(lib=lib, kind="pymc_collapsed", payload=pymc))
        support_rows.extend(_support_rows(lib=lib, kind="sklearn_ridge", payload=skl))

    global_df = pd.DataFrame(global_rows).sort_values(["lib", "model_kind"]).reset_index(drop=True)
    global_df.to_csv(out_csv, index=False)
    LOGGER.info("Wrote %s", out_csv)

    support_df = (
        pd.DataFrame(support_rows)
        .sort_values(["lib", "model_kind", "support_bin"])
        .reset_index(drop=True)
    )
    support_df.to_csv(out_support_csv, index=False)
    LOGGER.info("Wrote %s", out_support_csv)

    # Console summary (compact).
    keep = [
        "lib",
        "model_kind",
        "rmse",
        "mae",
        "cov95",
        "interval_width_mean",
        "n_used",
        "frac_used",
    ]
    print("\n=== Global metrics (cap100 -> realtest) ===")
    print(global_df[keep].to_string(index=False))


if __name__ == "__main__":
    main()
