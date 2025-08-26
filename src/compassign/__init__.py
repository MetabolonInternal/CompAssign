"""
CompAssign: Compound Assignment with Bayesian Inference

A Bayesian framework for ultra-high precision compound assignment
in untargeted metabolomics using hierarchical RT modeling and probabilistic matching.
"""

__version__ = "1.0.0"
__author__ = "Metabolon CompAssign Team"

from .synthetic_generator import generate_synthetic_data
from .rt_hierarchical import HierarchicalRTModel
from .peak_assignment import PeakAssignmentModel
from .peak_assignment_enhanced import EnhancedPeakAssignmentModel

__all__ = [
    "generate_synthetic_data",
    "HierarchicalRTModel", 
    "PeakAssignmentModel",
    "EnhancedPeakAssignmentModel"
]