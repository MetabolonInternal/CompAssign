#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Iterable


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize rt_eval_coeff_summaries_by_support_*.json files under a run directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory (e.g., output/rt_pymc_overnight_YYYYMMDD_HHMMSS).",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use output/rt_pymc_overnight_latest.txt to locate the run dir.",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="lib*/cap*/**/results/rt_eval_coeff_summaries_by_support_*.json",
        help="Glob pattern (relative to run-dir) for JSON result files.",
    )
    parser.add_argument(
        "--out-summary",
        type=Path,
        default=None,
        help="Output summary CSV path. Defaults to <run-dir>/summary.csv.",
    )
    parser.add_argument(
        "--out-support",
        type=Path,
        default=None,
        help="Output per-support-bin CSV path. Defaults to <run-dir>/summary_support.csv.",
    )
    return parser.parse_args()


def _infer_run_dir(args: argparse.Namespace) -> Path:
    if args.latest:
        latest_path = Path("output/rt_pymc_overnight_latest.txt")
        if not latest_path.exists():
            raise SystemExit(f"Missing {latest_path}; pass --run-dir explicitly.")
        run_dir = Path(latest_path.read_text().strip())
        if not run_dir.exists():
            raise SystemExit(f"Run dir from {latest_path} does not exist: {run_dir}")
        return run_dir
    if args.run_dir is None:
        raise SystemExit("Pass --run-dir or --latest.")
    return args.run_dir


def _iter_result_jsons(run_dir: Path, pattern: str) -> Iterable[Path]:
    yield from sorted(run_dir.glob(pattern))


def _parse_identity(result_path: Path) -> tuple[str, str, str, str]:
    """
    Expected layout:
      <run_dir>/lib208/cap100/<model>/results/rt_eval_coeff_summaries_by_support_<label>.json
    """
    label = result_path.stem.removeprefix("rt_eval_coeff_summaries_by_support_")
    model_dir = result_path.parent.parent
    cap_dir = model_dir.parent
    lib_dir = cap_dir.parent
    lib = lib_dir.name.removeprefix("lib")
    cap = cap_dir.name
    model = model_dir.name
    return lib, cap, model, label


def _get(d: dict[str, Any], key: str) -> float:
    v = d.get(key, float("nan"))
    try:
        return float(v)
    except Exception:  # noqa: BLE001
        return float("nan")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    run_dir = _infer_run_dir(args).resolve()

    out_summary = args.out_summary or (run_dir / "summary.csv")
    out_support = args.out_support or (run_dir / "summary_support.csv")

    paths = list(_iter_result_jsons(run_dir, args.pattern))
    if not paths:
        raise SystemExit(f"No result JSONs found under {run_dir} with pattern: {args.pattern}")

    LOGGER.info("[summarize] Found %d result JSONs under %s", len(paths), run_dir)

    summary_fields = [
        "lib",
        "cap",
        "model",
        "label",
        "n_obs_test",
        "rmse",
        "mae",
        "cov95",
        "rmse_bin_3_5",
        "group_rmse_p90_bin_3_5",
        "json_path",
    ]
    support_fields = [
        "lib",
        "cap",
        "model",
        "label",
        "support_bin",
        "n_groups_with_test",
        "n_obs_test",
        "rmse",
        "mae",
        "cov95",
        "group_rmse_mean",
        "group_rmse_std",
        "group_rmse_p50",
        "group_rmse_p90",
        "group_rmse_p99",
        "json_path",
    ]

    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_support.parent.mkdir(parents=True, exist_ok=True)

    with out_summary.open("w", newline="") as f_sum, out_support.open("w", newline="") as f_sup:
        w_sum = csv.DictWriter(f_sum, fieldnames=summary_fields)
        w_sup = csv.DictWriter(f_sup, fieldnames=support_fields)
        w_sum.writeheader()
        w_sup.writeheader()

        for p in paths:
            lib, cap, model, label = _parse_identity(p)
            d = json.loads(p.read_text())
            metrics = d.get("metrics", {})

            rmse = _get(metrics, "rmse")
            mae = _get(metrics, "mae")
            cov95 = _get(metrics, "coverage_95")
            n_obs_test = int(metrics.get("n_obs", 0) or 0)

            rmse_3_5 = float("nan")
            p90_3_5 = float("nan")
            for row in d.get("support_metrics", []):
                if row.get("support_bin") == "3-5":
                    rmse_3_5 = _get(row, "rmse")
                    p90_3_5 = _get(row, "rmse_p90")
                w_sup.writerow(
                    {
                        "lib": lib,
                        "cap": cap,
                        "model": model,
                        "label": label,
                        "support_bin": row.get("support_bin", ""),
                        "n_groups_with_test": int(row.get("n_groups_with_test", 0) or 0),
                        "n_obs_test": int(row.get("n_obs_test", 0) or 0),
                        "rmse": _get(row, "rmse"),
                        "mae": _get(row, "mae"),
                        "cov95": _get(row, "coverage_95"),
                        "group_rmse_mean": _get(row, "rmse_mean"),
                        "group_rmse_std": _get(row, "rmse_std"),
                        "group_rmse_p50": _get(row, "rmse_p50"),
                        "group_rmse_p90": _get(row, "rmse_p90"),
                        "group_rmse_p99": _get(row, "rmse_p99"),
                        "json_path": str(p),
                    }
                )

            w_sum.writerow(
                {
                    "lib": lib,
                    "cap": cap,
                    "model": model,
                    "label": label,
                    "n_obs_test": n_obs_test,
                    "rmse": rmse,
                    "mae": mae,
                    "cov95": cov95,
                    "rmse_bin_3_5": rmse_3_5,
                    "group_rmse_p90_bin_3_5": p90_3_5,
                    "json_path": str(p),
                }
            )

    LOGGER.info("[summarize] Wrote %s", out_summary)
    LOGGER.info("[summarize] Wrote %s", out_support)


if __name__ == "__main__":
    main()
