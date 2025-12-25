"""
CompAssign: Compound Assignment with Bayesian Inference

A Bayesian framework for ultra-high precision compound assignment
in untargeted metabolomics using hierarchical RT modeling and probabilistic matching.
"""

__version__ = "1.0.0"
__author__ = "Metabolon CompAssign Team"

from .utils import load_chemberta_pca20
from .assignment import AssignmentResults, PeakAssignment, PresencePrior
from .active_learning import (
    Oracle,
    OptimalOracle,
    RandomOracle,
    NoisyOracle,
    ConservativeOracle,
    ModelGuidedOracle,
    AdversarialOracle,
    ProbabilisticOracle,
    MixtureOracle,
)
from .active_learning import (
    simulate_annotation_round,
    run_annotation_experiment,
    compare_oracles,
    AnnotationRoundResults,
)

__all__ = [
    "load_chemberta_pca20",
    "PeakAssignment",
    "AssignmentResults",
    "PresencePrior",
    "Oracle",
    "OptimalOracle",
    "RandomOracle",
    "NoisyOracle",
    "ConservativeOracle",
    "ModelGuidedOracle",
    "AdversarialOracle",
    "ProbabilisticOracle",
    "MixtureOracle",
    "simulate_annotation_round",
    "run_annotation_experiment",
    "compare_oracles",
    "AnnotationRoundResults",
]
