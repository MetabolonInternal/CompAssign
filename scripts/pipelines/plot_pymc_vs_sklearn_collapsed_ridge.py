#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np


LOGGER = logging.getLogger(__name__)


DEFAULT_SUPPORT_BIN_ORDER = ["<= 1", "2-2", "3-5", "6-10", "11-20", "21-50", "51-100", "> 100"]


def _setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot PyMC collapsed ridge vs sklearn ridge baseline (cap100 -> realtest).",
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
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for plots (default: <run-dir>/analysis/pymc_vs_sklearn_plots).",
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


def _metric(payload: dict[str, Any], key: str) -> float:
    try:
        return float(payload.get("metrics", {}).get(key, float("nan")))
    except Exception:  # noqa: BLE001
        return float("nan")


def _support_metric(payload: dict[str, Any], support_bin: str, key: str) -> float:
    for row in payload.get("support_metrics", []) or []:
        if str(row.get("support_bin", "")) == str(support_bin):
            try:
                return float(row.get(key, float("nan")))
            except Exception:  # noqa: BLE001
                return float("nan")
    return float("nan")


def _plot_global(
    *, out_path: Path, lib: int, cap: str, pymc: dict[str, Any], skl: dict[str, Any]
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        raise SystemExit("matplotlib is required for plotting but could not be imported") from exc

    models = ["PyMC (collapsed)", "sklearn ridge"]
    x = np.arange(len(models))
    colors = ["tab:orange", "tab:blue"]

    metrics = [
        ("rmse", "RMSE (min)", "linear"),
        ("coverage_95", "Coverage @ 95%", "coverage"),
        ("interval_width_mean", "Mean 95% interval width (min)", "linear"),
    ]

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(11.0, 3.2))
    for ax, (m_key, ylabel, kind) in zip(axes, metrics, strict=True):
        y = [_metric(pymc, m_key), _metric(skl, m_key)]
        ax.bar(x, y, color=colors, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
        if kind == "coverage":
            ax.axhline(0.95, color="black", linewidth=1, linestyle="--", alpha=0.6)

    fig.suptitle(f"lib{lib} ({cap} -> realtest): PyMC collapsed vs sklearn ridge", y=1.03)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_support(
    *,
    out_path: Path,
    lib: int,
    cap: str,
    bins: Sequence[str],
    pymc: dict[str, Any],
    skl: dict[str, Any],
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        raise SystemExit("matplotlib is required for plotting but could not be imported") from exc

    models = ["PyMC (collapsed)", "sklearn ridge"]
    colors = ["tab:orange", "tab:blue"]
    x = np.arange(len(bins))
    width = 0.35

    metrics = [
        ("rmse", "RMSE (min)"),
        ("coverage_95", "Coverage @ 95%"),
        ("interval_width_mean", "Mean 95% interval width (min)"),
    ]

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10.5, 8.0), sharex=True)
    for ax, (m_key, ylabel) in zip(axes, metrics, strict=True):
        y_pymc = [_support_metric(pymc, b, m_key) for b in bins]
        y_skl = [_support_metric(skl, b, m_key) for b in bins]
        ax.bar(x - width / 2, y_pymc, width=width, color=colors[0], alpha=0.85, label=models[0])
        ax.bar(x + width / 2, y_skl, width=width, color=colors[1], alpha=0.85, label=models[1])
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
        if m_key == "coverage_95":
            ax.axhline(0.95, color="black", linewidth=1, linestyle="--", alpha=0.6)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(bins)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper left",
            bbox_to_anchor=(1.01, 1.0),
            borderaxespad=0.0,
            frameon=False,
        )

    fig.suptitle(f"lib{lib} ({cap} -> realtest): by training support bin", y=1.01)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _setup_logging()
    args = parse_args()

    run_dir = _resolve_run_dir(args.run_dir)
    libs = _infer_libs(run_dir, args.libs)
    cap = str(args.cap)
    out_dir = args.out_dir or (run_dir / "analysis" / "pymc_vs_sklearn_plots")
    out_dir.mkdir(parents=True, exist_ok=True)

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

        bins = [
            b for b in DEFAULT_SUPPORT_BIN_ORDER if not np.isnan(_support_metric(pymc, b, "rmse"))
        ]
        if not bins:
            bins = DEFAULT_SUPPORT_BIN_ORDER

        global_out = out_dir / f"lib{lib}_pymc_vs_sklearn_global.png"
        support_out = out_dir / f"lib{lib}_pymc_vs_sklearn_support_bin.png"

        _plot_global(out_path=global_out, lib=lib, cap=cap, pymc=pymc, skl=skl)
        _plot_support(out_path=support_out, lib=lib, cap=cap, bins=bins, pymc=pymc, skl=skl)

        LOGGER.info("Wrote %s", global_out)
        LOGGER.info("Wrote %s", support_out)


if __name__ == "__main__":
    main()
