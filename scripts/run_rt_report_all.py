#!/usr/bin/env python3
"""
Run the full set of experiments used by the RT multilevel pooling report.

This is an orchestration wrapper that calls existing scripts with the report defaults.
It does NOT edit the LaTeX source automatically; instead it writes outputs under `output/`
and prints paths you can use to copy metrics into the report tables.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class UnseenRunSpec:
    lib_id: int
    advi_steps: int
    seed: int
    output_dir: Path


def _run(cmd: list[str], *, cwd: Path = REPO_ROOT) -> None:
    printable = " ".join(cmd)
    print(f"[run] {printable}", flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _parse_libs(libs_csv: str) -> list[int]:
    out: list[int] = []
    for part in libs_csv.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise ValueError("No --libs provided")
    return out


def _cap_label(cap: str) -> str:
    cap = str(cap).strip()
    if cap.isdigit():
        return f"cap{cap}"
    return cap


def _unseen_output_dir(*, lib_id: int, advi_steps: int, seed: int) -> Path:
    return (
        REPO_ROOT
        / f"output/rt_unseen_generalization_chemlinear_lib{lib_id}_advi{advi_steps}_seed{seed}_allmodes"
    )


def _zero_shot_output_dir(*, lib_id: int, seed: int, holdout_n: int) -> Path:
    return (
        REPO_ROOT
        / f"output/rt_zero_shot_chem_models_lib{lib_id}_subset_seed{seed}_holdout{holdout_n}"
    )


def _prod_run_dir() -> Path:
    return REPO_ROOT / "output/rt_prod_report_chemlinear"


def _load_unseen_summary(path: Path) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text())
    if not isinstance(obj, list):
        raise ValueError(f"Expected list summary at {path}")
    return obj


def _fmt_cov(x: float) -> str:
    if x != x:  # nan
        return "nan"
    return f"{x:.3f}"


def _fmt_num(x: float) -> str:
    if x != x:  # nan
        return "nan"
    return f"{x:.6f}"


def _print_unseen_table_rows(*, summary_dir: Path) -> None:
    summary = _load_unseen_summary(summary_dir / "summary.json")
    by_mode = {str(d["mode"]): d for d in summary}

    def row(
        *,
        holdout: str,
        model_label: str,
        metrics: dict[str, Any],
        rows_scored_pct: float,
    ) -> str:
        return (
            f"\\texttt{{{holdout}}} & {model_label} & "
            f"{_fmt_num(float(metrics['rmse']))} & {_fmt_num(float(metrics['mae']))} & "
            f"{_fmt_cov(float(metrics['coverage_95']))} & {rows_scored_pct:.1f} \\\\"
        )

    holdouts = ["species", "species_comp"]
    for i, holdout in enumerate(holdouts):
        row_src = by_mode[holdout]
        print(
            row(
                holdout=holdout,
                model_label="Ridge (partial pooling, chem-linear)",
                metrics=row_src["partial_pool"]["metrics_all"],
                rows_scored_pct=100.0,
            )
        )
        print(
            row(
                holdout=holdout,
                model_label="Ridge (supercategory, sklearn)",
                metrics=row_src["sklearn_supercategory"]["metrics_all"],
                rows_scored_pct=100.0,
            )
        )
        lasso = row_src["lasso_supercategory"]
        lasso_rows = float(lasso["n_rows_evaluated"])
        lasso_total = (
            float(lasso["n_test_rows_seen"]) if float(lasso["n_test_rows_seen"]) > 0 else 1.0
        )
        lasso_pct = 100.0 * lasso_rows / lasso_total
        print(
            row(
                holdout=holdout,
                model_label="Lasso (supercategory, external)",
                metrics=lasso["metrics"],
                rows_scored_pct=lasso_pct,
            )
        )
        if i != len(holdouts) - 1:
            print("\\midrule")


def _sync_prod_plots_into_report(*, prod_run_dir: Path, libs: list[int], cap: str) -> None:
    plots_dir = prod_run_dir / "plots"
    if not plots_dir.is_dir():
        raise FileNotFoundError(f"Missing plots directory: {plots_dir}")

    report_dir = REPO_ROOT / "docs/models/images/rt_pymc_multilevel_pooling_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    names = [
        "global_comparison",
        "by_support_bin",
        "by_species_cluster",
    ]
    tag = "full"

    copied = 0
    for lib_id in libs:
        for stem in names:
            src = plots_dir / f"lib{lib_id}_{stem}_anchor_none_{tag}.png"
            if not src.exists():
                raise FileNotFoundError(f"Missing plot for report sync: {src}")
            dst = report_dir / src.name
            shutil.copy2(src, dst)
            copied += 1
    print(f"[report] synced {copied} plot images into {report_dir} (cap={cap})", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all experiments needed for the RT multilevel pooling report.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--cap", type=str, default="100", help="Cap label (e.g. 100 or cap100).")
    parser.add_argument("--libs", type=str, default="208,209", help="Comma-separated library ids.")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed used across experiments.")
    parser.add_argument(
        "--chem-embeddings-path",
        type=Path,
        default=Path("resources/metabolites/embeddings_chemberta_pca20.parquet"),
        help="ChemBERTa PCA-20 embedding parquet.",
    )
    parser.add_argument(
        "--theta-alpha-prior-sigma",
        type=float,
        default=1.0,
        help="Prior sigma for theta_alpha in the chem-linear compound prior.",
    )
    parser.add_argument(
        "--unseen-advi-steps",
        type=int,
        default=5000,
        help="ADVI steps for the unseen-generalization subset runs (section 4.5).",
    )
    parser.add_argument(
        "--zero-shot-holdout-n",
        type=int,
        default=20,
        help="Held-out chem_id count for zero-shot single-model runs (section 4.6).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Pass --skip-existing to long-running experiment runners where available.",
    )
    parser.add_argument(
        "--skip-prod", action="store_true", help="Skip cap100->realtest training/eval/plots."
    )
    parser.add_argument(
        "--skip-unseen", action="store_true", help="Skip section 4.5 holdout experiments."
    )
    parser.add_argument(
        "--skip-zero-shot", action="store_true", help="Skip section 4.6 zero-shot experiments."
    )
    parser.add_argument(
        "--build-pdf", action="store_true", help="Rebuild the LaTeX PDF at the end."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cap = _cap_label(args.cap)
    libs = _parse_libs(args.libs)
    seed = int(args.seed)
    advi_steps = int(args.unseen_advi_steps)
    holdout_n = int(args.zero_shot_holdout_n)
    chem_embeddings_path = Path(args.chem_embeddings_path)

    if not chem_embeddings_path.is_absolute():
        chem_embeddings_path = (REPO_ROOT / chem_embeddings_path).resolve()

    if not bool(args.skip_prod):
        run_dir = _prod_run_dir()
        train_cmd = [
            str(REPO_ROOT / "scripts/run_rt_prod.sh"),
            "--cap",
            cap,
            "--libs",
            ",".join(str(x) for x in libs),
            "--seed",
            str(seed),
            "--run-dir",
            str(run_dir),
            "--no-eval",
            "--chem-embeddings-path",
            str(chem_embeddings_path),
            "--theta-alpha-prior-sigma",
            str(float(args.theta_alpha_prior_sigma)),
        ]
        if args.skip_existing:
            train_cmd.append("--skip-existing")
        _run(train_cmd)

        eval_cmd = [
            str(REPO_ROOT / "scripts/run_rt_prod_eval.sh"),
            "--run-dir",
            str(run_dir),
            "--cap",
            cap,
            "--libs",
            ",".join(str(x) for x in libs),
        ]
        if args.skip_existing:
            eval_cmd.append("--skip-existing")
        _run(eval_cmd)
        _sync_prod_plots_into_report(prod_run_dir=run_dir, libs=libs, cap=cap)

        print(f"[report] prod run dir (chem-linear) = {run_dir}", flush=True)
        print(f"[report] prod plots dir (chem-linear) = {run_dir / 'plots'}", flush=True)

    unseen_specs: list[UnseenRunSpec] = []
    if not bool(args.skip_unseen):
        for lib_id in libs:
            out_dir = _unseen_output_dir(lib_id=lib_id, advi_steps=advi_steps, seed=seed)
            unseen_specs.append(
                UnseenRunSpec(
                    lib_id=lib_id,
                    advi_steps=advi_steps,
                    seed=seed,
                    output_dir=out_dir,
                )
            )

        for spec in unseen_specs:
            data_csv = (
                REPO_ROOT
                / f"repo_export/lib{spec.lib_id}/{cap}/merged_training_all_lib{spec.lib_id}_{cap}_chemclass_rt_prod.csv"
            )
            cmd = [
                sys.executable,
                "-u",
                str(REPO_ROOT / "scripts/experiments/rt/run_unseen_generalization_prod_subset.py"),
                "--data-csv",
                str(data_csv),
                "--lib-id",
                str(int(spec.lib_id)),
                "--output-dir",
                str(spec.output_dir),
                "--modes",
                "species,species_comp",
                "--train-method",
                "advi",
                "--advi-steps",
                str(int(spec.advi_steps)),
                "--seed",
                str(int(spec.seed)),
                "--holdout-frac",
                "0.2",
                "--clusters",
                "4,5",
                "--species-per-cluster",
                "5",
                "--top-comp-ids",
                "100",
                "--chem-embeddings-path",
                str(chem_embeddings_path),
                "--theta-alpha-prior-sigma",
                str(float(args.theta_alpha_prior_sigma)),
            ]
            if args.skip_existing:
                cmd.append("--skip-existing")
            _run(cmd)

            print(f"[report] unseen summary = {spec.output_dir / 'summary.json'}", flush=True)

        for lib_id in libs:
            out_dir = _unseen_output_dir(lib_id=lib_id, advi_steps=advi_steps, seed=seed)
            if (out_dir / "summary.json").exists():
                print(f"\n[report] LaTeX table rows for section 4.5 (lib{lib_id})", flush=True)
                _print_unseen_table_rows(summary_dir=out_dir)

    if not bool(args.skip_zero_shot):
        for lib_id in libs:
            subset_csv = (
                _unseen_output_dir(lib_id=lib_id, advi_steps=advi_steps, seed=seed)
                / "subset"
                / "subset.csv"
            )
            out_dir = _zero_shot_output_dir(lib_id=lib_id, seed=seed, holdout_n=holdout_n)
            cmd = [
                sys.executable,
                "-u",
                str(REPO_ROOT / "scripts/experiments/rt/run_zero_shot_chem_models.py"),
                "--data-csv",
                str(subset_csv),
                "--chem-embeddings-path",
                str(chem_embeddings_path),
                "--output-dir",
                str(out_dir),
                "--holdout-n",
                str(int(holdout_n)),
                "--holdout-strategy",
                "stratified_rt",
                "--seed",
                str(int(seed)),
                "--report-minimal",
                "--include-pymc",
                "--theta-alpha-prior-sigma",
                str(float(args.theta_alpha_prior_sigma)),
            ]
            if args.skip_existing:
                cmd.append("--skip-existing")
            _run(cmd)
            print(f"[report] zero-shot summary = {out_dir / 'summary.json'}", flush=True)

    if bool(args.build_pdf):
        _run(
            [
                "latexmk",
                "-xelatex",
                "-interaction=nonstopmode",
                "rt_pymc_multilevel_pooling_report.tex",
            ],
            cwd=REPO_ROOT / "docs/models",
        )

    print("[report] Done.", flush=True)


if __name__ == "__main__":
    main()
