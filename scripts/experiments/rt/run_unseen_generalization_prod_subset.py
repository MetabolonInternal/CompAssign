#!/usr/bin/env python3
"""
Run production-data experiments that stress *unseen* species/compound structure.

We use a small, deterministic subset of a production RT CSV and create group-aware splits:

Holdout modes
-------------
1) holdout_mode=species
   - hold out entire species within each species_cluster
   - simulates a new species in an existing supercategory

2) holdout_mode=species_comp
   - hold out (species, comp_id) pairs but keep (species_cluster, comp_id) seen in training
   - simulates an "unseen compound for a specific species" where supercategory has data

For each split we train:
  - PyMC ridge partial pooling model (group=species, comp_id) + PartialPoolBackoffSummaries
  - sklearn ridge supercategory baseline (group=species_cluster, comp_id)

and evaluate:
  - partial pooling with hierarchy backoff (`scripts/pipelines/eval_rt_partial_pool_backoff.py`)
  - supercategory baseline with comp-id backoff (`scripts/pipelines/eval_rt_supercategory_comp_backoff.py`)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import re

import numpy as np
import pandas as pd

DEFAULT_CHEM_EMBEDDINGS_PATH = Path("resources/metabolites/embeddings_chemberta_pca20.parquet")

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.compassign.rt.ridge_stage1 import infer_feature_columns  # noqa: E402
from src.compassign.rt.pymc_partial_pool_ridge import (  # noqa: E402
    train_pymc_partial_pool_ridge_from_csv,
    write_pymc_partial_pool_ridge_artifacts,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run unseen-generalization experiments on a production RT CSV subset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-csv", type=Path, required=True, help="Input RT production CSV.")
    parser.add_argument(
        "--lib-id",
        type=int,
        default=None,
        help="Optional library id (needed for legacy lasso baseline). If omitted, inferred from --data-csv path.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: output/rt_unseen_generalization_<timestamp>).",
    )
    parser.add_argument(
        "--clusters",
        type=str,
        default="4,5",
        help="Comma-separated species_cluster ids to include in the subset.",
    )
    parser.add_argument(
        "--species-per-cluster",
        type=int,
        default=5,
        help="Top-N species per species_cluster (by row count) to include in the subset.",
    )
    parser.add_argument(
        "--top-comp-ids",
        type=int,
        default=100,
        help="Top-N comp_id values (by row count) to include in the subset.",
    )
    parser.add_argument(
        "--holdout-frac",
        type=float,
        default=0.2,
        help="Holdout fraction for the chosen holdout modes.",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for holdout selection.")
    parser.add_argument(
        "--modes",
        type=str,
        default="species,species_comp",
        help="Comma-separated holdout modes: species, species_comp.",
    )
    parser.add_argument(
        "--train-method",
        type=str,
        choices=["map", "advi"],
        default="map",
        help="Inference method for PyMC trainers.",
    )
    parser.add_argument(
        "--map-maxeval",
        type=int,
        default=25_000,
        help="Max evaluations for MAP (if --train-method map).",
    )
    parser.add_argument(
        "--advi-steps",
        type=int,
        default=5000,
        help="ADVI steps (if --train-method advi).",
    )
    parser.add_argument(
        "--partial-alpha-prior-mode",
        type=str,
        choices=["iid", "chem_linear", "chem_interaction"],
        default="chem_linear",
        help="Compound prior mode for the partial pooling model (report default: chem_linear).",
    )
    parser.add_argument(
        "--chem-embeddings-path",
        type=Path,
        default=DEFAULT_CHEM_EMBEDDINGS_PATH,
        help=(
            "ChemBERTa embedding parquet (required when --partial-alpha-prior-mode is chem_linear/"
            "chem_interaction)."
        ),
    )
    parser.add_argument(
        "--theta-alpha-prior-sigma",
        type=float,
        default=1.0,
        help="Prior sigma for theta_alpha (chem_* modes only).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200_000,
        help="Chunk size for streaming the full input CSV when building subsets.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip training/eval when outputs already exist in output-dir.",
    )
    return parser.parse_args()


def _now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def _parse_int_list(s: str) -> list[int]:
    out: list[int] = []
    for part in str(s).split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise SystemExit("Empty integer list")
    return out


def _scan_counts(
    *, data_csv: Path, chunk_size: int
) -> tuple[dict[int, dict[int, int]], dict[int, int]]:
    """Return (cluster->species->count, comp_id->count) by streaming."""
    cluster_species_counts: dict[int, dict[int, int]] = {}
    comp_counts: dict[int, int] = {}
    usecols = ["species_cluster", "species", "comp_id"]
    for chunk in pd.read_csv(data_csv, usecols=usecols, chunksize=int(chunk_size)):
        g = chunk.groupby(["species_cluster", "species"]).size()
        for (cluster_id, species_id), cnt in g.items():
            cluster_id_i = int(cluster_id)
            species_id_i = int(species_id)
            cluster_species_counts.setdefault(cluster_id_i, {})
            cluster_species_counts[cluster_id_i][species_id_i] = cluster_species_counts[
                cluster_id_i
            ].get(species_id_i, 0) + int(cnt)

        vc = chunk["comp_id"].value_counts()
        for comp_id, cnt in vc.items():
            comp_id_i = int(comp_id)
            comp_counts[comp_id_i] = comp_counts.get(comp_id_i, 0) + int(cnt)
    return cluster_species_counts, comp_counts


def _top_n(d: dict[int, int], n: int) -> list[int]:
    return [k for k, _ in sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))[: int(n)]]


def _write_subset_csv(
    *,
    data_csv: Path,
    subset_csv: Path,
    cluster_ids: set[int],
    species_ids: set[int],
    comp_ids: set[int],
    chunk_size: int,
) -> tuple[list[str], int]:
    header = pd.read_csv(data_csv, nrows=0)
    cols = tuple(map(str, header.columns))
    es_candidates = sorted([c for c in cols if c.startswith("ES_")])
    feature_cols = list(infer_feature_columns(cols, es_candidates=es_candidates))
    required = [
        "sampleset_id",
        "rt",
        "compound",
        "comp_id",
        "compound_class",
        "species_cluster",
        "species",
    ]
    usecols = list(dict.fromkeys([*required, *feature_cols]))

    subset_csv.parent.mkdir(parents=True, exist_ok=True)
    wrote_header = False
    rows = 0
    for chunk in pd.read_csv(data_csv, usecols=usecols, chunksize=int(chunk_size)):
        m = (
            chunk["species_cluster"].isin(cluster_ids)
            & chunk["species"].isin(species_ids)
            & chunk["comp_id"].isin(comp_ids)
        )
        if not m.any():
            continue
        sub = chunk.loc[m].copy()
        rows += int(len(sub))
        sub.to_csv(
            subset_csv, mode="w" if not wrote_header else "a", index=False, header=not wrote_header
        )
        wrote_header = True

    if rows <= 0:
        raise SystemExit("Subset selection produced 0 rows; adjust clusters/species/comp_ids")
    return feature_cols, rows


def _choose_holdout_species_comp(
    *, df: pd.DataFrame, holdout_frac: float, seed: int
) -> set[tuple[int, int]]:
    """Hold out (species, comp_id) pairs but keep (species_cluster, comp_id) seen in training."""
    df_pairs = df[["species_cluster", "species", "comp_id"]].drop_duplicates().astype(int)
    # candidate (cluster, comp) groups with >=2 species observed
    grp = df_pairs.groupby(["species_cluster", "comp_id"])["species"].nunique().rename("n_species")
    eligible_cc = grp.loc[grp >= 2].reset_index()[["species_cluster", "comp_id"]]
    if eligible_cc.empty:
        raise SystemExit("No eligible (cluster, comp_id) with >=2 species for species_comp holdout")

    cc = eligible_cc.to_numpy(dtype=np.int64, copy=False)
    n_hold = max(1, int(np.floor(float(holdout_frac) * float(cc.shape[0]))))
    rng = np.random.default_rng(int(seed))
    chosen_idx = rng.choice(np.arange(cc.shape[0]), size=int(n_hold), replace=False)

    hold_pairs: set[tuple[int, int]] = set()
    # For each selected (cluster, comp), hold out exactly one species.
    for i in chosen_idx.tolist():
        cl = int(cc[i, 0])
        cid = int(cc[i, 1])
        species_list = (
            df_pairs.loc[
                (df_pairs["species_cluster"] == cl) & (df_pairs["comp_id"] == cid), "species"
            ]
            .astype(int)
            .unique()
            .tolist()
        )
        if len(species_list) < 2:
            continue
        sp = int(rng.choice(np.asarray(sorted(species_list), dtype=np.int64), size=1)[0])
        hold_pairs.add((sp, cid))
    if not hold_pairs:
        raise SystemExit("Internal error: empty holdout set for species_comp")
    return hold_pairs


def _choose_holdout_species(*, df: pd.DataFrame, holdout_frac: float, seed: int) -> set[int]:
    species_df = df[["species_cluster", "species"]].drop_duplicates().astype(int)
    rng = np.random.default_rng(int(seed))
    holdout_species: set[int] = set()
    for cl, sub in species_df.groupby("species_cluster"):
        sp = sorted(set(map(int, sub["species"].tolist())))
        if len(sp) <= 1:
            continue
        n_hold = max(1, int(np.floor(float(holdout_frac) * float(len(sp)))))
        n_hold = min(n_hold, len(sp) - 1)  # keep at least one species per cluster in training
        chosen = rng.choice(np.asarray(sp, dtype=np.int64), size=int(n_hold), replace=False)
        holdout_species.update(int(x) for x in chosen.tolist())
    if not holdout_species:
        raise SystemExit("Internal error: empty holdout set for species")
    return holdout_species


def _split_train_test(
    *, subset_csv: Path, out_dir: Path, mode: str, holdout_frac: float, seed: int
) -> tuple[Path, Path, dict]:
    df = pd.read_csv(subset_csv)
    if mode == "species_comp":
        hold_pairs = _choose_holdout_species_comp(df=df, holdout_frac=holdout_frac, seed=seed)
        key = list(
            zip(df["species"].astype(int).tolist(), df["comp_id"].astype(int).tolist(), strict=True)
        )
        is_test = np.asarray([(int(a), int(b)) in hold_pairs for a, b in key], dtype=bool)
        holdout_desc = {"mode": mode, "n_holdout_pairs": int(len(hold_pairs))}
    elif mode == "species":
        hold_species = _choose_holdout_species(df=df, holdout_frac=holdout_frac, seed=seed)
        is_test = df["species"].astype(int).isin(hold_species).to_numpy(dtype=bool, copy=False)
        holdout_desc = {"mode": mode, "n_holdout_species": int(len(hold_species))}
    else:
        raise SystemExit(f"Unknown mode (supported: species,species_comp): {mode}")

    train = df.loc[~is_test].copy()
    test = df.loc[is_test].copy()
    if train.empty or test.empty:
        raise SystemExit(f"Bad split for mode={mode}: train={len(train)} test={len(test)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    train_csv = out_dir / "train.csv"
    test_csv = out_dir / "test.csv"
    train.to_csv(train_csv, index=False)
    test.to_csv(test_csv, index=False)

    cfg = {
        "mode": mode,
        "seed": int(seed),
        "holdout_frac": float(holdout_frac),
        "rows_total": int(len(df)),
        "rows_train": int(len(train)),
        "rows_test": int(len(test)),
        **holdout_desc,
    }
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    return train_csv, test_csv, cfg


def _run(cmd: list[str], *, cwd: Path) -> None:
    subprocess.run(cmd, check=True, cwd=str(cwd))


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _maybe_load_existing_paths(
    *, model_dir: Path, required: tuple[str, ...]
) -> dict[str, str] | None:
    cfg = model_dir / "config.json"
    if cfg.exists():
        d = json.loads(cfg.read_text())
        paths = {k: str(d.get(k, "")) for k in required}
        if all(paths.get(k) for k in required):
            return paths
    # Fall back to conventional paths.
    convention = {}
    if "coeff_npz" in required:
        convention["coeff_npz"] = str(model_dir / "models" / "stage1_coeff_summaries_posterior.npz")
    if "backoff_npz" in required:
        convention["backoff_npz"] = str(model_dir / "models" / "partial_pool_backoff_summaries.npz")
    if all(Path(convention[k]).exists() for k in required if k in convention):
        return convention
    return None


def main() -> None:
    args = parse_args()
    data_csv = args.data_csv
    if not data_csv.is_absolute():
        data_csv = (REPO_ROOT / data_csv).resolve()
    if not data_csv.exists():
        raise SystemExit(f"Missing input CSV: {data_csv}")

    lib_id = args.lib_id
    if lib_id is None:
        m = re.search(r"lib(?P<lib_id>\\d+)", str(data_csv))
        if m:
            lib_id = int(m.group("lib_id"))

    clusters = set(_parse_int_list(args.clusters))
    if args.output_dir is None:
        out_dir = REPO_ROOT / "output" / f"rt_unseen_generalization_{_now_ts()}"
    else:
        out_dir = (
            args.output_dir
            if args.output_dir.is_absolute()
            else (REPO_ROOT / args.output_dir).resolve()
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    modes = [m.strip() for m in str(args.modes).split(",") if m.strip()]
    if not modes:
        raise SystemExit("Empty --modes")
    allowed_modes = {"species", "species_comp"}
    invalid_modes = [m for m in modes if m not in allowed_modes]
    if invalid_modes:
        raise SystemExit(
            f"Invalid --modes entries (allowed: species,species_comp): {sorted(invalid_modes)}"
        )

    alpha_prior_mode = str(args.partial_alpha_prior_mode)
    chem_embeddings_path = args.chem_embeddings_path
    if not chem_embeddings_path.is_absolute():
        chem_embeddings_path = (REPO_ROOT / chem_embeddings_path).resolve()
    if alpha_prior_mode != "iid" and not chem_embeddings_path.exists():
        raise SystemExit(f"Missing chem embeddings parquet: {chem_embeddings_path}")

    # 1) Choose a deterministic subset (species + comp_ids).
    print(f"[subset] Scanning counts from {data_csv} ...")
    cluster_species_counts, comp_counts = _scan_counts(
        data_csv=data_csv, chunk_size=int(args.chunk_size)
    )

    species_ids: set[int] = set()
    for cl in sorted(clusters):
        sc = cluster_species_counts.get(int(cl), {})
        if not sc:
            raise SystemExit(f"No rows found for species_cluster={cl}")
        top_species = _top_n(sc, int(args.species_per_cluster))
        species_ids.update(int(s) for s in top_species)
    comp_ids = set(_top_n(comp_counts, int(args.top_comp_ids)))

    subset_dir = out_dir / "subset"
    subset_csv = subset_dir / "subset.csv"
    print(
        f"[subset] Writing subset CSV with clusters={sorted(clusters)}, "
        f"species={len(species_ids)}, comp_ids={len(comp_ids)} ..."
    )
    feature_cols, subset_rows = _write_subset_csv(
        data_csv=data_csv,
        subset_csv=subset_csv,
        cluster_ids=clusters,
        species_ids=species_ids,
        comp_ids=comp_ids,
        chunk_size=int(args.chunk_size),
    )
    subset_cfg = {
        "data_csv": str(data_csv),
        "clusters": sorted(int(x) for x in clusters),
        "species_ids": sorted(int(x) for x in species_ids),
        "comp_ids": sorted(int(x) for x in comp_ids),
        "rows_subset": int(subset_rows),
        "feature_cols": list(feature_cols),
    }
    (subset_dir / "config.json").write_text(json.dumps(subset_cfg, indent=2))
    print(f"[subset] Wrote {subset_csv} rows={subset_rows:,}")

    # 2) For each holdout mode: split, train, evaluate.
    results: list[dict] = []
    for mode in modes:
        print(f"\n[exp] mode={mode}")
        split_dir = out_dir / "splits" / mode
        train_csv, test_csv, split_cfg = _split_train_test(
            subset_csv=subset_csv,
            out_dir=split_dir,
            mode=mode,
            holdout_frac=float(args.holdout_frac),
            seed=int(args.seed),
        )
        print(f"[split] train={train_csv} test={test_csv}")

        model_dir = out_dir / "models" / mode
        partial_dir = model_dir / "partial_pool"
        sklearn_dir = model_dir / "sklearn_ridge_species_cluster"
        method = str(args.train_method)

        partial_paths = None
        if bool(args.skip_existing):
            partial_paths = _maybe_load_existing_paths(
                model_dir=partial_dir, required=("coeff_npz", "backoff_npz")
            )
        if partial_paths is None:
            print(f"[train] partial_pool ({method}; alpha_prior_mode={alpha_prior_mode}) ...")
            partial_art = train_pymc_partial_pool_ridge_from_csv(
                data_csv=train_csv,
                seed=int(args.seed),
                include_es_all=True,
                feature_center="global",
                lambda_slopes=3e-4,
                sigma_y_prior=0.05,
                alpha_prior_mode=alpha_prior_mode,  # type: ignore[arg-type]
                chem_embeddings_path=chem_embeddings_path if alpha_prior_mode != "iid" else None,
                theta_alpha_prior_sigma=float(args.theta_alpha_prior_sigma),
                method=method,  # type: ignore[arg-type]
                advi_steps=int(args.advi_steps),
                advi_draws=30,
                advi_log_every=500,
                map_maxeval=int(args.map_maxeval),
            )
            partial_paths = write_pymc_partial_pool_ridge_artifacts(
                artifacts=partial_art, output_dir=partial_dir
            )

        sklearn_coeff = sklearn_dir / "stage1_coeff_summaries.npz"
        if bool(args.skip_existing) and sklearn_coeff.exists():
            print("[train] Skip sklearn ridge (exists)")
        else:
            print("[train] sklearn ridge supercategory ...")
            _run(
                [
                    sys.executable,
                    "-u",
                    str(REPO_ROOT / "scripts/pipelines/train_rt_stage1_coeff_summaries.py"),
                    "--data-csv",
                    str(train_csv),
                    "--output-dir",
                    str(sklearn_dir),
                    "--seed",
                    str(int(args.seed)),
                    "--lambda-ridge",
                    "3e-4",
                    "--include-es-all",
                    "--feature-center",
                    "global",
                    "--feature-rotation",
                    "none",
                    "--anchor-expansion",
                    "none",
                ],
                cwd=REPO_ROOT,
            )

        eval_dir = out_dir / "eval" / mode
        eval_dir.mkdir(parents=True, exist_ok=True)

        partial_eval_json = eval_dir / "partial_pool.json"
        sklearn_super_eval_json = eval_dir / "sklearn_supercategory.json"

        if bool(args.skip_existing) and partial_eval_json.exists():
            print("[eval] Skip partial_pool backoff-class (exists)")
        else:
            print("[eval] partial_pool with hierarchy backoff (backoff-class) ...")
            _run(
                [
                    sys.executable,
                    "-u",
                    str(REPO_ROOT / "scripts/pipelines/eval_rt_partial_pool_backoff.py"),
                    "--coeff-npz",
                    str(partial_paths["coeff_npz"]),
                    "--backoff-npz",
                    str(partial_paths["backoff_npz"]),
                    "--test-csv",
                    str(test_csv),
                    "--output-dir",
                    str(eval_dir),
                    "--label",
                    "partial_pool_class",
                    "--alpha-backoff-mode",
                    "class",
                    "--log-every-chunks",
                    "0",
                ],
                cwd=REPO_ROOT,
            )
            (eval_dir / "rt_eval_partial_pool_backoff_partial_pool_class.json").rename(
                partial_eval_json
            )

        partial_metrics = _load_json(partial_eval_json)

        if bool(args.skip_existing) and sklearn_super_eval_json.exists():
            print("[eval] Skip sklearn supercategory comp-backoff (exists)")
        else:
            print("[eval] sklearn supercategory with comp-id backoff ...")
            _run(
                [
                    sys.executable,
                    "-u",
                    str(REPO_ROOT / "scripts/pipelines/eval_rt_supercategory_comp_backoff.py"),
                    "--coeff-npz",
                    str(sklearn_dir / "stage1_coeff_summaries.npz"),
                    "--test-csv",
                    str(test_csv),
                    "--output-dir",
                    str(eval_dir),
                    "--label",
                    "sklearn_supercategory",
                    "--log-every-chunks",
                    "0",
                ],
                cwd=REPO_ROOT,
            )
            (eval_dir / "rt_eval_supercategory_comp_backoff_sklearn_supercategory.json").rename(
                sklearn_super_eval_json
            )
        sklearn_super_metrics = _load_json(sklearn_super_eval_json)

        lasso_metrics = None
        if lib_id is not None:
            print("[eval] legacy lasso supercategory (external bundle) ...")
            lasso_out = eval_dir / "lasso"
            _run(
                [
                    sys.executable,
                    "-u",
                    str(
                        REPO_ROOT / "scripts/pipelines/eval_rt_lasso_baseline_by_species_cluster.py"
                    ),
                    "--output-dir",
                    str(lasso_out),
                    "--lib-id",
                    str(int(lib_id)),
                    "--test-csv",
                    str(test_csv),
                    "--chunk-size",
                    "200000",
                    "--label",
                    f"holdout_{mode}",
                ],
                cwd=REPO_ROOT,
            )
            lasso_json = lasso_out / "results" / f"rt_eval_lasso_holdout_{mode}.json"
            if lasso_json.exists():
                lasso_metrics = _load_json(lasso_json)
        else:
            print("[eval] Skip legacy lasso baseline (missing --lib-id and could not infer).")

        row = {
            "mode": mode,
            "split": split_cfg,
            "partial_pool": partial_metrics,
            "sklearn_supercategory": sklearn_super_metrics,
            "lasso_supercategory": lasso_metrics,
            "partial_alpha_prior_mode": alpha_prior_mode,
            "chem_embeddings_path": str(chem_embeddings_path),
        }
        results.append(row)

    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\n[done] Wrote {out_path}")


if __name__ == "__main__":
    main()
