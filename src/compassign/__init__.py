"""
CompAssign: Compound Assignment with Bayesian Inference

A Bayesian framework for ultra-high precision compound assignment
in untargeted metabolomics using hierarchical RT modeling and probabilistic matching.
"""

__version__ = "1.0.0"
__author__ = "Metabolon CompAssign Team"

from .rt_hierarchical import HierarchicalRTModel
from .peak_assignment import PeakAssignment, AssignmentResults
from .presence_prior import PresencePrior
from .oracles import (
    Oracle, OptimalOracle, RandomOracle, NoisyOracle,
    ConservativeOracle, ModelGuidedOracle, AdversarialOracle,
    ProbabilisticOracle, MixtureOracle
)
from .eval_loop import (
    simulate_annotation_round, run_annotation_experiment,
    compare_oracles, AnnotationRoundResults
)

__all__ = [
    "HierarchicalRTModel",
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
    "AnnotationRoundResults"
]
