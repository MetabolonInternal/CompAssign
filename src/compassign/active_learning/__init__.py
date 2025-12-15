"""Active learning utilities for peak assignment."""

from __future__ import annotations

from .core import (  # noqa: F401
    ActiveLearner,
    diverse_selection,
    entropy,
    expected_fp_reduction,
    hybrid_acquisition,
    least_confident,
    margin_sampling,
    mutual_information,
    select_batch,
)
from .eval_loop import (  # noqa: F401
    AnnotationRoundResults,
    compare_oracles,
    run_annotation_experiment,
    simulate_annotation_round,
)
from .oracles import (  # noqa: F401
    AdversarialOracle,
    ConservativeOracle,
    MixtureOracle,
    ModelGuidedOracle,
    NoisyOracle,
    OptimalOracle,
    Oracle,
    ProbabilisticOracle,
    RandomOracle,
    SmartNoisyOracle,
)
from .synthetic import MultiCandidateGenerator  # noqa: F401

__all__ = [
    "ActiveLearner",
    "diverse_selection",
    "expected_fp_reduction",
    "entropy",
    "hybrid_acquisition",
    "least_confident",
    "margin_sampling",
    "mutual_information",
    "select_batch",
    "AnnotationRoundResults",
    "simulate_annotation_round",
    "run_annotation_experiment",
    "compare_oracles",
    "Oracle",
    "OptimalOracle",
    "RandomOracle",
    "SmartNoisyOracle",
    "NoisyOracle",
    "ConservativeOracle",
    "ModelGuidedOracle",
    "AdversarialOracle",
    "ProbabilisticOracle",
    "MixtureOracle",
    "MultiCandidateGenerator",
]
