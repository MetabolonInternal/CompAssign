#!/usr/bin/env python
"""
Benchmark PyMC sampling on a small hierarchical regression using PyTensor vs JAX/NumPyro.

This is intended to quantify wall time and ESS/sec differences between backends on CPU.
Example usage (run inside the test env to avoid touching your main compassign env):

  conda run -n compassign-jax python scripts/bench/bench_pymc_jax.py --backend both

You can warm up JAX compilation by adding --warmup-jax and repeat to see steady-state:

  conda run -n compassign-jax python scripts/bench/bench_pymc_jax.py \\
      --backend jax --warmup-jax --repeat 2 --chain-method vectorized
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import arviz as az
import numpy as np
import pymc as pm


@dataclass
class SyntheticData:
    x: np.ndarray
    y: np.ndarray
    group_idx: np.ndarray
    n_groups: int


@dataclass
class BenchmarkResult:
    backend: str
    elapsed_s: float
    ess_bulk_min: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare PyMC PyTensor vs JAX sampling on a small hierarchical model"
    )
    parser.add_argument(
        "--backend",
        choices=["pytensor", "jax", "both"],
        default="both",
        help="Backend(s) to run",
    )
    parser.add_argument("--draws", type=int, default=500, help="Posterior draws per chain")
    parser.add_argument("--tune", type=int, default=500, help="Tuning steps per chain")
    parser.add_argument("--chains", type=int, default=2, help="Number of chains")
    parser.add_argument(
        "--cores",
        type=int,
        default=2,
        help="Worker processes for PyTensor sampling (ignored for JAX)",
    )
    parser.add_argument(
        "--chain-method",
        choices=["sequential", "vectorized"],
        default="vectorized",
        help="NumPyro chain execution strategy",
    )
    parser.add_argument("--n-groups", type=int, default=8, help="Number of hierarchical groups")
    parser.add_argument("--n-obs", type=int, default=400, help="Observation count")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat each backend this many times (helps show JIT warm results)",
    )
    parser.add_argument(
        "--warmup-jax",
        action="store_true",
        help="Run an extra un-timed JAX call to trigger compilation before timing",
    )
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--quiet", action="store_true", help="Disable progress bars")
    return parser.parse_args()


def make_data(n_groups: int, n_obs: int, seed: int) -> SyntheticData:
    rng = np.random.default_rng(int(seed))
    group_idx = rng.integers(low=0, high=n_groups, size=n_obs)
    x = rng.normal(loc=0.0, scale=1.0, size=n_obs)
    true_alpha = rng.normal(loc=0.0, scale=1.0, size=n_groups)
    true_beta = 1.25
    true_sigma = 0.6
    y = true_alpha[group_idx] + true_beta * x + rng.normal(scale=true_sigma, size=n_obs)
    return SyntheticData(x=x, y=y, group_idx=group_idx, n_groups=n_groups)


def build_model(data: SyntheticData) -> pm.Model:
    with pm.Model() as model:
        mu_alpha = pm.Normal("mu_alpha", mu=0.0, sigma=1.0)
        sigma_alpha = pm.Exponential("sigma_alpha", lam=1.0)
        alpha = pm.Normal("alpha", mu=mu_alpha, sigma=sigma_alpha, shape=data.n_groups)

        beta = pm.Normal("beta", mu=0.0, sigma=1.0)
        sigma = pm.Exponential("sigma", lam=1.0)

        mu = alpha[data.group_idx] + beta * data.x
        pm.Normal("y", mu=mu, sigma=sigma, observed=data.y)
    return model


def run_pytensor(
    model: pm.Model,
    *,
    draws: int,
    tune: int,
    chains: int,
    cores: int,
    target_accept: float,
    seed: int,
    quiet: bool,
) -> az.InferenceData:
    with model:
        return pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            target_accept=target_accept,
            random_seed=seed,
            progressbar=not quiet,
            return_inferencedata=True,
        )


def run_jax(
    model: pm.Model,
    *,
    draws: int,
    tune: int,
    chains: int,
    target_accept: float,
    seed: int,
    chain_method: str,
    quiet: bool,
) -> az.InferenceData:
    try:
        from pymc.sampling import jax as pmjax
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyMC JAX backend not available; install jax[numpyro]") from exc

    with model:
        return pmjax.sample_numpyro_nuts(
            draws=draws,
            tune=tune,
            chains=chains,
            chain_method=chain_method,
            target_accept=target_accept,
            random_seed=seed,
            postprocessing_backend="cpu",
            progressbar=not quiet,
        )


def time_backend(
    backend: str, *, data: SyntheticData, seed: int, args: argparse.Namespace
) -> BenchmarkResult:
    model = build_model(data)
    t0 = time.perf_counter()
    if backend == "pytensor":
        idata = run_pytensor(
            model,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=args.cores,
            target_accept=args.target_accept,
            seed=seed,
            quiet=args.quiet,
        )
    elif backend == "jax":
        idata = run_jax(
            model,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            target_accept=args.target_accept,
            seed=seed,
            chain_method=args.chain_method,
            quiet=args.quiet,
        )
    else:  # pragma: no cover - guarded by argparse choices
        raise ValueError(f"Unknown backend {backend}")
    elapsed = time.perf_counter() - t0
    ess = float(az.ess(idata, method="bulk").to_array().min().item())
    return BenchmarkResult(backend=backend, elapsed_s=elapsed, ess_bulk_min=ess)


def main(args: argparse.Namespace) -> None:
    backends = ["pytensor", "jax"] if args.backend == "both" else [args.backend]
    data = make_data(n_groups=args.n_groups, n_obs=args.n_obs, seed=args.seed)

    print(
        f"Data: n_groups={data.n_groups}, n_obs={data.y.size}, seed={args.seed}, "
        f"draws={args.draws}, tune={args.tune}, chains={args.chains}"
    )
    print("backend     run  wall_s  ess_bulk_min  ess_per_sec")

    for backend in backends:
        if backend == "jax" and args.warmup_jax:
            _ = time_backend(backend, data=data, seed=args.seed - 1, args=args)
        for run_idx in range(int(args.repeat)):
            run_seed = int(args.seed) + run_idx
            result = time_backend(backend, data=data, seed=run_seed, args=args)
            ess_rate = (
                result.ess_bulk_min / result.elapsed_s if result.elapsed_s > 0 else float("nan")
            )
            print(
                f"{backend:8s}  {run_idx+1:4d}  {result.elapsed_s:6.2f}  "
                f"{result.ess_bulk_min:12.1f}  {ess_rate:11.1f}"
            )


if __name__ == "__main__":
    main(parse_args())
