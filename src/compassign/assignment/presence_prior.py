"""
Presence prior module for compound prevalence modeling.

This module implements a Beta-Bernoulli prior system for maintaining beliefs
about compound presence probabilities across species. These priors are updated
online from human annotations and provide log-prior offsets for the softmax
assignment model.
"""

from dataclasses import dataclass
import numpy as np
from typing import Optional, Dict, Tuple


@dataclass
class PresencePrior:
    """
    Maintains Beta-Bernoulli priors for compound presence probabilities.

    For each (species, compound) pair, we maintain Beta(α, β) parameters
    representing our belief about the probability that compound c occurs
    in species s. A separate global prior is maintained for the null class.

    Attributes
    ----------
    alpha : np.ndarray
        Shape (n_species, n_compounds). Alpha parameters of Beta distributions.
    beta : np.ndarray
        Shape (n_species, n_compounds). Beta parameters of Beta distributions.
    alpha_null : float
        Alpha parameter for the null class (global or could be per-species).
    beta_null : float
        Beta parameter for the null class.
    """

    alpha: np.ndarray  # shape (n_species, n_compounds)
    beta: np.ndarray  # shape (n_species, n_compounds)
    alpha_null: float  # for null class (global)
    beta_null: float

    @classmethod
    def init(
        cls,
        n_species: int,
        n_compounds: int,
        smoothing: float = 1.0,
        null_prior: Tuple[float, float] = (1.0, 1.0),
        empirical_counts: Optional[Dict[Tuple[int, int], int]] = None,
    ) -> "PresencePrior":
        """
        Initialize presence priors.

        Parameters
        ----------
        n_species : int
            Number of species
        n_compounds : int
            Number of compounds
        smoothing : float
            Smoothing parameter for Beta priors (default: 1.0 gives uniform prior)
        null_prior : Tuple[float, float]
            (alpha, beta) for null class prior
        empirical_counts : Optional[Dict[Tuple[int, int], int]]
            Dictionary of (species, compound) -> count for empirical initialization

        Returns
        -------
        PresencePrior
            Initialized presence prior object
        """
        # Initialize with smoothing (uniform or Jeffrey's prior)
        alpha = np.full((n_species, n_compounds), smoothing)
        beta = np.full((n_species, n_compounds), smoothing)

        # If empirical counts provided, boost alpha where compounds are observed
        if empirical_counts:
            for (s, c), count in empirical_counts.items():
                if 0 <= s < n_species and 0 <= c < n_compounds:
                    alpha[s, c] += count

        return cls(alpha=alpha, beta=beta, alpha_null=null_prior[0], beta_null=null_prior[1])

    def log_prior_prob(self, species_idx: int) -> np.ndarray:
        """
        Return log probability log(π_sc) for all compounds c in species s.

        Note: This is intentionally log-probability, not log-odds.
        """
        if species_idx < 0 or species_idx >= self.alpha.shape[0]:
            raise ValueError(f"Invalid species index: {species_idx}")

        pi = self.alpha[species_idx] / (self.alpha[species_idx] + self.beta[species_idx])
        return np.log(np.clip(pi, 1e-12, 1.0))

    # Note: `log_prior_odds` removed; use `log_prior_prob` instead.

    def log_prior_null(self) -> float:
        """
        Return log probability of null class.

        Returns
        -------
        float
            Log probability of null assignment
        """
        pi_null = self.alpha_null / (self.alpha_null + self.beta_null)
        return float(np.log(max(pi_null, 1e-12)))

    def update_positive(self, species_idx: int, compound_idx: int, weight: float = 1.0):
        """
        Update prior after observing a positive (present) label.

        Parameters
        ----------
        species_idx : int
            Species index
        compound_idx : int
            Compound index
        weight : float
            Weight for the update (default 1.0)
        """
        if 0 <= species_idx < self.alpha.shape[0] and 0 <= compound_idx < self.alpha.shape[1]:
            self.alpha[species_idx, compound_idx] += float(weight)

    def update_negative(self, species_idx: int, compound_idx: int, weight: float = 1.0):
        """
        Update prior after observing a negative (absent) label.

        Parameters
        ----------
        species_idx : int
            Species index
        compound_idx : int
            Compound index
        weight : float
            Weight for the update (default 1.0)
        """
        if 0 <= species_idx < self.alpha.shape[0] and 0 <= compound_idx < self.alpha.shape[1]:
            self.beta[species_idx, compound_idx] += float(weight)

    def update_null(self, weight: float = 1.0):
        """
        Update prior after observing a null assignment.

        This increments the alpha parameter for null, making null
        assignments slightly more likely in the future. The optional
        ``weight`` parameter allows softened updates (e.g., log-scaled counts).
        """
        self.alpha_null += float(weight)

    def update_not_null(self, weight: float = 1.0):
        """
        Update prior after observing a non-null assignment.

        This increments the beta parameter for null, making null
        assignments slightly less likely in the future. The optional
        ``weight`` parameter mirrors ``update_null`` for balanced updates.
        """
        self.beta_null += float(weight)

    def get_mean_probabilities(self, species_idx: Optional[int] = None) -> np.ndarray:
        """
        Get posterior mean probabilities.

        Parameters
        ----------
        species_idx : Optional[int]
            If provided, return probabilities for this species only.
            Otherwise return for all species.

        Returns
        -------
        np.ndarray
            Shape (n_compounds,) if species_idx provided, else (n_species, n_compounds)
        """
        if species_idx is not None:
            if species_idx < 0 or species_idx >= self.alpha.shape[0]:
                raise ValueError(f"Invalid species index: {species_idx}")
            return self.alpha[species_idx] / (self.alpha[species_idx] + self.beta[species_idx])
        else:
            return self.alpha / (self.alpha + self.beta)

    def get_uncertainty(self, species_idx: Optional[int] = None) -> np.ndarray:
        """
        Get uncertainty (variance) of presence probabilities.

        Higher values indicate less confidence in the presence probability.

        Parameters
        ----------
        species_idx : Optional[int]
            If provided, return uncertainty for this species only.

        Returns
        -------
        np.ndarray
            Variance of Beta distributions
        """
        if species_idx is not None:
            if species_idx < 0 or species_idx >= self.alpha.shape[0]:
                raise ValueError(f"Invalid species index: {species_idx}")
            a = self.alpha[species_idx]
            b = self.beta[species_idx]
            return (a * b) / ((a + b) ** 2 * (a + b + 1))
        else:
            a = self.alpha
            b = self.beta
            return (a * b) / ((a + b) ** 2 * (a + b + 1))

    def save(self, filepath: str):
        """Save presence prior to file."""
        np.savez(
            filepath,
            alpha=self.alpha,
            beta=self.beta,
            alpha_null=self.alpha_null,
            beta_null=self.beta_null,
        )

    @classmethod
    def load(cls, filepath: str) -> "PresencePrior":
        """Load presence prior from file."""
        data = np.load(filepath)
        return cls(
            alpha=data["alpha"],
            beta=data["beta"],
            alpha_null=float(data["alpha_null"]),
            beta_null=float(data["beta_null"]),
        )
