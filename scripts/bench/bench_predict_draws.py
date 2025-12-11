#!/usr/bin/env python
"""
Micro-benchmark: cost of prediction vs number of posterior draws.

This approximates the inner loop of HierarchicalRTModel.predict_new by
constructing random arrays with the same shapes and measuring runtime for
different caps on posterior draws used in prediction.

Usage (defaults are representative but safe to run quickly):

  poetry run python scripts/bench/bench_predict_draws.py \
      --n-draws-total 1500 --n-rows 3000 --n-covariates 8 --n-species 6 --n-compounds 80

To emulate the heavier mix_shift case you observed:

  poetry run python scripts/bench/bench_predict_draws.py \
      --n-draws-total 2500 --n-rows 5600 --n-covariates 8 --n-species 6 --n-compounds 80

"""

from __future__ import annotations

import argparse
import time
from typing import List

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark predict time vs number of posterior draws")
    p.add_argument(
        "--n-draws-total", type=int, default=1500, help="Total posterior draws available"
    )
    p.add_argument(
        "--n-draws-list",
        type=str,
        default="50,100,200,400,800,1500",
        help="Comma-separated list of draw caps to test",
    )
    p.add_argument("--n-rows", type=int, default=3000, help="Number of prediction rows (N)")
    p.add_argument("--n-covariates", type=int, default=8, help="Number of run covariates (D)")
    p.add_argument("--n-species", type=int, default=6, help="Species count")
    p.add_argument("--n-compounds", type=int, default=80, help="Compound count")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--repeats", type=int, default=1, help="Repeat each setting and average time")
    return p.parse_args()


def run_once(
    *,
    n_draws: int,
    mu0: np.ndarray,
    species_eff: np.ndarray,
    compound_eff: np.ndarray,
    gamma: np.ndarray,
    sigma_y: np.ndarray,
    species_idx: np.ndarray,
    compound_idx: np.ndarray,
    cov_std: np.ndarray,
) -> float:
    """Execute the predict loop for n_draws and return seconds elapsed."""
    predictions = np.zeros((n_draws, species_idx.size), dtype=float)
    t0 = time.perf_counter()
    for i in range(n_draws):
        pred = mu0[i] + species_eff[i, species_idx] + compound_eff[i, compound_idx]
        pred = pred + np.sum(cov_std * gamma[i][compound_idx], axis=1)
        predictions[i] = pred
    # Include mean/std aggregation to match predict_new
    _mean = predictions.mean(axis=0)
    _param_var = predictions.var(axis=0)
    _obs_var = float(np.mean(np.square(sigma_y[:n_draws])))
    _ = np.sqrt(np.maximum(_param_var + _obs_var, 0.0))
    t1 = time.perf_counter()
    return t1 - t0


def main() -> None:
    args = parse_args()
    rng = np.random.RandomState(int(args.seed))

    S = int(args.n_draws_total)
    N = int(args.n_rows)
    D = int(args.n_covariates)
    n_species = int(args.n_species)
    n_compounds = int(args.n_compounds)

    # Synthetic posterior arrays (shapes mirror HierarchicalRTModel.predict_new)
    mu0 = rng.normal(9.0, 0.5, size=S).astype(float)
    sigma_y = np.exp(rng.normal(-2.3, 0.2, size=S)).astype(float)
    species_eff = rng.normal(0.0, 0.5, size=(S, n_species)).astype(float)
    compound_eff = rng.normal(0.0, 0.6, size=(S, n_compounds)).astype(float)
    gamma = rng.normal(0.0, 0.2, size=(S, n_compounds, D)).astype(float)

    # Prediction indices and covariates
    species_idx = rng.randint(0, n_species, size=N, dtype=int)
    compound_idx = rng.randint(0, n_compounds, size=N, dtype=int)
    cov_std = rng.normal(0.0, 1.0, size=(N, D)).astype(float)

    # Build list of tested draw caps (respecting total)
    caps: List[int] = [min(int(x), S) for x in args.n_draws_list.split(",") if x.strip()]
    caps = sorted(set(caps))

    # Measure full baseline
    baseline_cap = max(caps)
    t_baseline = 0.0
    for _ in range(int(args.repeats)):
        t_baseline += run_once(
            n_draws=baseline_cap,
            mu0=mu0,
            species_eff=species_eff,
            compound_eff=compound_eff,
            gamma=gamma,
            sigma_y=sigma_y,
            species_idx=species_idx,
            compound_idx=compound_idx,
            cov_std=cov_std,
        )
    t_baseline /= float(max(1, int(args.repeats)))

    print(
        f"Benchmark shapes → draws={S}, rows={N}, covariates={D}, species={n_species}, compounds={n_compounds}"
    )
    print(f"Baseline (cap={baseline_cap}): {t_baseline:.3f}s")
    print("cap_draws  time_s  speedup_vs_full  frac_of_full")

    for cap in caps:
        t = 0.0
        for _ in range(int(args.repeats)):
            t += run_once(
                n_draws=cap,
                mu0=mu0,
                species_eff=species_eff,
                compound_eff=compound_eff,
                gamma=gamma,
                sigma_y=sigma_y,
                species_idx=species_idx,
                compound_idx=compound_idx,
                cov_std=cov_std,
            )
        t /= float(max(1, int(args.repeats)))
        speedup = t_baseline / t if t > 0 else float("inf")
        frac = t / t_baseline if t_baseline > 0 else float("nan")
        print(f"{cap:9d}  {t:6.3f}  {speedup:14.2f}  {frac:11.2%}")


if __name__ == "__main__":
    main()
