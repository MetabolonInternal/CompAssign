from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class MixtureModelDiagnostics:
    ssid: int
    lib_id: int
    run_root: Path
    parquet_path: Path
    task_parquet_path: Path | None


_REJECTED_COL = "rejected"
_LEGACY_REJECTED_COL = "v" + "etoed"


def _normal_pdf(x: np.ndarray, mean: float, var: float) -> np.ndarray:
    var = float(var)
    if not np.isfinite(var) or var <= 0:
        raise ValueError(f"Invalid variance for normal pdf: var={var}")
    denom = np.sqrt(2.0 * np.pi * var)
    return np.exp(-0.5 * (x - float(mean)) ** 2 / var) / denom


def _load_diagnostics(parquet_path: Path) -> pd.DataFrame:
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Missing diagnostics parquet: {parquet_path}")
    return pd.read_parquet(parquet_path)


def _parse_diagnostics(df: pd.DataFrame) -> tuple[int, int]:
    if "sample_set_id" not in df.columns or "lib_id" not in df.columns:
        raise ValueError("Diagnostics parquet is missing required columns sample_set_id and lib_id")
    ssids = pd.to_numeric(df["sample_set_id"], errors="coerce").dropna().unique()
    libs = pd.to_numeric(df["lib_id"], errors="coerce").dropna().unique()
    if ssids.size != 1 or libs.size != 1:
        raise ValueError(
            "Diagnostics parquet expects a single sample_set_id and lib_id; "
            f"got sample_set_id={sorted(ssids.tolist())} lib_id={sorted(libs.tolist())}"
        )
    return int(ssids[0]), int(libs[0])


def find_mixture_model_diagnostics(out_root: Path, *, run_glob: str) -> list[MixtureModelDiagnostics]:
    pattern = f"{run_glob}/sally_output/mixture_model_diagnostics_ssid*_lib*.parquet"
    parquet_paths = sorted(out_root.glob(pattern))
    diagnostics: list[MixtureModelDiagnostics] = []
    for parquet_path in parquet_paths:
        run_root = parquet_path.parents[1]
        df = _load_diagnostics(parquet_path)
        ssid, lib_id = _parse_diagnostics(df)
        task_parquet_path = parquet_path.with_name(
            f"mixture_model_task_diagnostics_ssid{ssid}_lib{lib_id}.parquet"
        )
        if not task_parquet_path.is_file():
            task_parquet_path = None
        diagnostics.append(
            MixtureModelDiagnostics(
                ssid=ssid,
                lib_id=lib_id,
                run_root=run_root,
                parquet_path=parquet_path,
                task_parquet_path=task_parquet_path,
            )
        )
    return diagnostics


def _plot_one(
    df: pd.DataFrame,
    df_tasks: pd.DataFrame | None,
    *,
    ssid: int,
    lib_id: int,
    out_path: Path,
) -> None:
    required = {
        "calls",
        "mean_loglik",
        "n_tasks",
        "posterior_coherent",
        "coherent_component",
        "w0",
        "w1",
        "mu0",
        "mu1",
        "var0",
        "var1",
    }
    missing = required.difference(df.columns)
    if _REJECTED_COL not in df.columns and _LEGACY_REJECTED_COL not in df.columns:
        missing.add(_REJECTED_COL)
    if missing:
        raise ValueError(f"Diagnostics parquet is missing required columns: {sorted(missing)}")

    x = pd.to_numeric(df["mean_loglik"], errors="coerce").astype(float)
    p = pd.to_numeric(df["posterior_coherent"], errors="coerce").astype(float)
    n_tasks = pd.to_numeric(df["n_tasks"], errors="coerce").astype(float)
    rejected_col = _REJECTED_COL if _REJECTED_COL in df.columns else _LEGACY_REJECTED_COL
    rejected = df[rejected_col].astype(bool)

    mask = np.isfinite(x) & np.isfinite(p) & np.isfinite(n_tasks)
    df_fit = df.loc[mask].copy()
    if df_fit.empty:
        raise ValueError(f"No finite mixture_model diagnostics rows for ssid={ssid} lib={lib_id}")

    x_fit = pd.to_numeric(df_fit["mean_loglik"], errors="coerce").astype(float).to_numpy()
    p_fit = pd.to_numeric(df_fit["posterior_coherent"], errors="coerce").astype(float).to_numpy()
    n_tasks_fit = pd.to_numeric(df_fit["n_tasks"], errors="coerce").astype(float).to_numpy()
    rejected_fit = df_fit[rejected_col].astype(bool).to_numpy()

    coherent_component = int(pd.to_numeric(df_fit["coherent_component"], errors="raise").iloc[0])
    weights = np.array(
        [
            float(pd.to_numeric(df_fit["w0"], errors="raise").iloc[0]),
            float(pd.to_numeric(df_fit["w1"], errors="raise").iloc[0]),
        ],
        dtype=float,
    )
    means = np.array(
        [
            float(pd.to_numeric(df_fit["mu0"], errors="raise").iloc[0]),
            float(pd.to_numeric(df_fit["mu1"], errors="raise").iloc[0]),
        ],
        dtype=float,
    )
    vars_ = np.array(
        [
            float(pd.to_numeric(df_fit["var0"], errors="raise").iloc[0]),
            float(pd.to_numeric(df_fit["var1"], errors="raise").iloc[0]),
        ],
        dtype=float,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)

    # Left panel: histogram + mixture components overlay
    ax = axes[0]
    bins = 30
    counts, bin_edges, _ = ax.hist(
        x_fit, bins=bins, color="#e0e0e0", edgecolor="#444444", linewidth=0.5
    )
    ax.set_xlabel(r"Per-compound score $x_c$ (mean RT log-likelihood)")
    ax.set_ylabel("Compounds")
    ax.set_title("Mixture fit on $x_c$")

    x_grid = np.linspace(float(np.min(x_fit)), float(np.max(x_fit)), 400)
    bin_width = float(bin_edges[1] - bin_edges[0]) if len(bin_edges) > 1 else 1.0
    n = float(x_fit.size)
    pdf0 = _normal_pdf(x_grid, means[0], vars_[0])
    pdf1 = _normal_pdf(x_grid, means[1], vars_[1])
    y0 = n * bin_width * weights[0] * pdf0
    y1 = n * bin_width * weights[1] * pdf1
    y_mix = y0 + y1

    coh_color = "#1b9e77"
    bad_color = "#d95f02"
    if coherent_component == 0:
        ax.plot(x_grid, y0, color=coh_color, linewidth=2.0, label="Coherent component")
        ax.plot(x_grid, y1, color=bad_color, linewidth=2.0, label="Incoherent component")
    else:
        ax.plot(x_grid, y1, color=coh_color, linewidth=2.0, label="Coherent component")
        ax.plot(x_grid, y0, color=bad_color, linewidth=2.0, label="Incoherent component")
    ax.plot(x_grid, y_mix, color="#4c4c4c", linewidth=1.5, linestyle="--", label="Mixture")

    boundary_idx = int(np.argmin(np.abs(p_fit - 0.5)))
    boundary_x = float(x_fit[boundary_idx])
    ax.axvline(boundary_x, color="#377eb8", linestyle=":", linewidth=2.0, label="p=0.5 boundary")

    ax.legend(loc="best", fontsize=9)
    ax.text(
        0.02,
        0.98,
        f"n={x_fit.size}  rejected={int(rejected_fit.sum())}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )

    # Right panel: per-task normalized residuals for example compounds
    ax = axes[1]
    if df_tasks is None:
        # Backwards-compatible fallback for older runs that only dumped per-compound diagnostics.
        sc = ax.scatter(
            n_tasks_fit,
            x_fit,
            c=p_fit,
            cmap="viridis",
            s=12,
            alpha=0.9,
            linewidths=0.0,
        )
        if bool(rejected_fit.any()):
            ax.scatter(
                n_tasks_fit[rejected_fit],
                x_fit[rejected_fit],
                facecolors="none",
                edgecolors="#d62728",
                s=36,
                linewidths=0.7,
                label="Rejected",
            )
            ax.legend(loc="best", fontsize=9)
        ax.set_xlabel("Tasks per compound ($n_c$)")
        ax.set_ylabel(r"$x_c$ (mean RT log-likelihood)")
        ax.set_title("Per-compound score vs tasks")
        cb = fig.colorbar(sc, ax=ax)
        cb.set_label(r"$p(\mathrm{coherent}\mid x_c)$")
        fig.suptitle(f"mixture_model diagnostics (SSID {ssid}, lib{lib_id})", fontsize=12)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return

    required_tasks = {"calls", "task_id", "rt_z", "example_role"}
    missing_tasks = required_tasks.difference(df_tasks.columns)
    if missing_tasks:
        raise ValueError(
            "Task diagnostics parquet is missing required columns: "
            f"{sorted(missing_tasks)}"
        )

    df_tasks = df_tasks.copy()
    df_tasks["calls"] = pd.to_numeric(df_tasks["calls"], errors="coerce")
    df_tasks["task_id"] = pd.to_numeric(df_tasks["task_id"], errors="coerce")
    df_tasks["rt_z"] = pd.to_numeric(df_tasks["rt_z"], errors="coerce")
    df_tasks = df_tasks.dropna(subset=["calls", "task_id", "rt_z", "example_role"]).copy()
    df_tasks["calls"] = df_tasks["calls"].astype(int)
    df_tasks["task_id"] = df_tasks["task_id"].astype(int)
    df_tasks["rt_z"] = df_tasks["rt_z"].astype(float)
    df_tasks["example_role"] = df_tasks["example_role"].astype(str)
    df_tasks["example_role"] = df_tasks["example_role"].replace({_LEGACY_REJECTED_COL: _REJECTED_COL})

    ax.axvline(0.0, color="#444444", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.axvline(2.0, color="#888888", linewidth=0.8, linestyle=":", alpha=0.8)
    ax.axvline(-2.0, color="#888888", linewidth=0.8, linestyle=":", alpha=0.8)

    role_colors = {
        "kept": "#1b9e77",
        _REJECTED_COL: "#d62728",
        "lowest_p": "#d95f02",
    }
    role_labels = {
        "kept": "Kept",
        _REJECTED_COL: "Rejected",
        "lowest_p": "Least coherent",
    }
    seen_roles = sorted(df_tasks["example_role"].unique().tolist())
    role_order = [r for r in ["kept", _REJECTED_COL, "lowest_p"] if r in seen_roles] + [
        r for r in seen_roles if r not in {"kept", _REJECTED_COL, "lowest_p"}
    ]

    for role in role_order:
        sub = df_tasks[df_tasks["example_role"] == role].copy()
        if sub.empty:
            continue
        call_id = int(sub["calls"].iloc[0])
        posterior = None
        if "posterior_coherent" in sub.columns:
            posterior_val = pd.to_numeric(sub["posterior_coherent"], errors="coerce").iloc[0]
            if np.isfinite(posterior_val):
                posterior = float(posterior_val)

        n_tasks_call = int(sub.shape[0])
        if "n_tasks" in sub.columns:
            n_tasks_val = pd.to_numeric(sub["n_tasks"], errors="coerce").iloc[0]
            if np.isfinite(n_tasks_val):
                n_tasks_call = int(n_tasks_val)
        label = role_labels.get(role, role)
        if posterior is not None:
            label = f"{label} call={call_id} (p={posterior:.2f}, n={n_tasks_call})"
        else:
            label = f"{label} call={call_id} (n={n_tasks_call})"

        sub = sub.sort_values("task_id")
        y = sub["rt_z"].to_numpy(dtype=float)
        y_task = np.arange(y.size, dtype=int)
        ax.scatter(
            y,
            y_task,
            color=role_colors.get(role, "#377eb8"),
            s=14,
            alpha=0.75,
            edgecolors="none",
            label=label,
        )

    ax.set_xlabel(r"Normalized residual $z_{t,c}$")
    ax.set_ylabel("LC tasks (sorted by task_id)")
    ax.set_title("Per-task RT residual scatter (examples)")
    ax.legend(loc="best", fontsize=8)

    fig.suptitle(f"mixture_model diagnostics (SSID {ssid}, lib{lib_id})", fontsize=12)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def render_all(
    diagnostics: Iterable[MixtureModelDiagnostics],
    *,
    output_subdir: str,
) -> dict[tuple[int, int], list[Path]]:
    rendered: dict[tuple[int, int], list[Path]] = {}
    for diag in diagnostics:
        df = _load_diagnostics(diag.parquet_path)
        df_tasks = (
            _load_diagnostics(diag.task_parquet_path)
            if diag.task_parquet_path is not None
            else None
        )
        out_path = (
            diag.run_root
            / output_subdir
            / f"mixture_model_diagnostics_ssid{diag.ssid}_lib{diag.lib_id}.png"
        )
        _plot_one(df, df_tasks, ssid=diag.ssid, lib_id=diag.lib_id, out_path=out_path)
        rendered.setdefault((diag.ssid, diag.lib_id), []).append(out_path)
        print(f"[mixture_model] Wrote {out_path}")
    return rendered


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("external_repos/sally/out"),
        help="Sally out/ directory (contains prod_* runs).",
    )
    parser.add_argument(
        "--run-glob",
        type=str,
        default="prod_*",
        help="Glob for run directories under out-root (default: prod_*).",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="evaluation",
        help="Subdirectory under each run directory for plots (default: evaluation).",
    )
    parser.add_argument(
        "--latex-copy",
        type=Path,
        default=None,
        help="If set, copy one selected plot to this destination path.",
    )
    parser.add_argument(
        "--latex-ssid",
        type=int,
        default=12307,
        help="SSID to copy into the LaTeX images folder (default: 12307).",
    )
    parser.add_argument(
        "--latex-lib",
        type=int,
        default=208,
        help="Library id to copy into the LaTeX images folder (default: 208).",
    )
    args = parser.parse_args()

    diagnostics = find_mixture_model_diagnostics(args.out_root, run_glob=args.run_glob)
    if not diagnostics:
        raise RuntimeError(
            f"No mixture_model diagnostics found under {args.out_root} with run_glob={args.run_glob!r}"
        )

    rendered = render_all(diagnostics, output_subdir=args.output_subdir)

    if args.latex_copy is not None:
        key = (int(args.latex_ssid), int(args.latex_lib))
        candidates = rendered.get(key, [])
        if not candidates:
            raise RuntimeError(
                f"Requested LaTeX copy for ssid={key[0]} lib={key[1]} but no plot was rendered"
            )

        def _choose_preferred(paths: list[Path]) -> Path:
            ordered = sorted(paths)
            preferred_substrings = ("compassign_pp_ridge", "compassign")
            for substr in preferred_substrings:
                for path in ordered:
                    if substr in path.parent.parent.name:
                        return path
            return ordered[0]

        chosen = _choose_preferred(candidates)

        args.latex_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(chosen, args.latex_copy)
        print(f"[mixture_model] Copied {chosen} -> {args.latex_copy}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
