"""
Retention-time (RT) models and production RT pipeline utilities.

This subpackage consolidates all RT-related model implementations under a single
namespace to keep `src/compassign/` tidy.
"""

from .fast_ridge_prod import (  # noqa: F401
    RidgeGroupCompoundRTModel,
    RidgeProdTrainArtifacts,
    train_ridge_prod_model,
    write_ridge_prod_artifacts,
)
from .pymc_collapsed_ridge import Stage1CoeffSummaries  # noqa: F401

__all__ = [
    "RidgeGroupCompoundRTModel",
    "RidgeProdTrainArtifacts",
    "Stage1CoeffSummaries",
    "train_ridge_prod_model",
    "write_ridge_prod_artifacts",
]
