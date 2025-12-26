"""
Deprecated compatibility wrapper for the RT ridge PyMC models.

Historically, this repository implemented multiple RT ridge model variants (collapsed / explicit
intercepts, multiple priors, optional chemistry features) in this single module.

The production-focused code has since been simplified into two explicit model modules:
  - `src.compassign.rt.pymc_supercategory_ridge` (fully pooled by species_cluster)
  - `src.compassign.rt.pymc_partial_pool_ridge` (partial pooling by species, nested in species_cluster)

Shared artifact definitions and posterior math live in:
  - `src.compassign.rt.ridge_stage1`

This module re-exports the most commonly used names for backward compatibility with older scripts
and tests. New code should import from the modules above directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from .pymc_partial_pool_ridge import (  # noqa: F401
    PartialPoolRidgeTrainArtifacts,
    train_pymc_partial_pool_ridge_from_csv,
    write_pymc_partial_pool_ridge_artifacts,
)
from .pymc_supercategory_ridge import (  # noqa: F401
    SupercategoryRidgeTrainArtifacts,
    train_pymc_supercategory_ridge_from_csv,
    write_pymc_supercategory_ridge_artifacts,
)
from .ridge_stage1 import (  # noqa: F401
    ChemHierBackoffSummaries,
    PartialPoolBackoffSummaries,
    Stage1CoeffSummaries,
    _compute_group_posterior_summaries,
    _compute_group_posterior_summaries_with_b_prior,
    _compute_group_posterior_summaries_with_b_prior_mean,
    _encode_cluster_comp_key,
    _infer_parent_ids,
    encode_group_comp_key,
    infer_feature_columns,
    infer_parent_ids,
)


GroupCol = Literal["species_cluster", "species"]
InterceptMode = Literal["collapsed", "explicit"]
InterceptPrior = Literal["flat", "comp_hier_supercat"]
SlopeHeadMode = Literal["none", "cluster_supercat"]


def train_pymc_collapsed_ridge_from_csv(  # noqa: D401
    *,
    data_csv: Path,
    group_col: GroupCol = "species_cluster",
    intercept_mode: InterceptMode = "collapsed",
    intercept_prior: InterceptPrior = "flat",
    slope_head_mode: SlopeHeadMode = "none",
    seed: int = 42,
    max_train_rows: int = 0,
    include_es_all: bool = False,
    feature_center_mode: Literal["none", "global"] = "global",
    lambda_slopes: float = 3e-4,
    sigma_y_prior: float = 0.05,
    method: Literal["advi", "map"] = "advi",
    advi_steps: int = 5000,
    advi_log_every: int = 1000,
    advi_draws: int = 50,
    map_maxeval: int = 50_000,
) -> SupercategoryRidgeTrainArtifacts | PartialPoolRidgeTrainArtifacts:
    """
    Backward-compatible entrypoint.

    Supported combinations:
      - (group_col=species_cluster, intercept_mode=collapsed) -> supercategory ridge
      - (group_col=species, intercept_mode=explicit, intercept_prior=comp_hier_supercat,
         slope_head_mode=cluster_supercat) -> partial pooling ridge
    """
    if str(feature_center_mode) != "global":
        raise ValueError(
            "Only feature_center_mode='global' is supported in the simplified trainers"
        )

    if group_col == "species_cluster":
        if intercept_mode != "collapsed":
            raise ValueError("supercategory ridge requires intercept_mode='collapsed'")
        if intercept_prior != "flat":
            raise ValueError("supercategory ridge uses intercept_prior='flat'")
        if slope_head_mode != "none":
            raise ValueError("supercategory ridge does not support slope_head_mode")
        return train_pymc_supercategory_ridge_from_csv(
            data_csv=Path(data_csv),
            seed=int(seed),
            max_train_rows=int(max_train_rows),
            include_es_all=bool(include_es_all),
            feature_center="global",
            lambda_slopes=float(lambda_slopes),
            sigma_y_prior=float(sigma_y_prior),
            method=method,
            advi_steps=int(advi_steps),
            advi_log_every=int(advi_log_every),
            advi_draws=int(advi_draws),
            map_maxeval=int(map_maxeval),
        )

    if group_col == "species":
        if intercept_mode != "explicit":
            raise ValueError("partial pooling ridge requires intercept_mode='explicit'")
        if intercept_prior != "comp_hier_supercat":
            raise ValueError("partial pooling ridge requires intercept_prior='comp_hier_supercat'")
        if slope_head_mode != "cluster_supercat":
            raise ValueError("partial pooling ridge requires slope_head_mode='cluster_supercat'")
        return train_pymc_partial_pool_ridge_from_csv(
            data_csv=Path(data_csv),
            seed=int(seed),
            max_train_rows=int(max_train_rows),
            include_es_all=bool(include_es_all),
            feature_center="global",
            lambda_slopes=float(lambda_slopes),
            sigma_y_prior=float(sigma_y_prior),
            method=method,
            advi_steps=int(advi_steps),
            advi_log_every=int(advi_log_every),
            advi_draws=int(advi_draws),
            map_maxeval=int(map_maxeval),
        )

    raise ValueError(f"Unknown group_col: {group_col}")


def write_pymc_collapsed_ridge_artifacts(  # noqa: D401
    *,
    artifacts: SupercategoryRidgeTrainArtifacts | PartialPoolRidgeTrainArtifacts,
    output_dir: Path,
) -> dict[str, str | None]:
    """Backward-compatible artifact writer."""
    if isinstance(artifacts, SupercategoryRidgeTrainArtifacts):
        return write_pymc_supercategory_ridge_artifacts(artifacts=artifacts, output_dir=output_dir)
    return write_pymc_partial_pool_ridge_artifacts(artifacts=artifacts, output_dir=output_dir)
