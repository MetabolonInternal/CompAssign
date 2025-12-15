"""
Oracle simulators for human annotation behaviors.

This module provides different oracle implementations that simulate various
human annotator behaviors for active learning experiments. Oracles range from
optimal (always correct) to adversarial (always wrong) with realistic behaviors
in between.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Optional, List


class Oracle(ABC):
    """Abstract base class for annotation oracles."""

    @abstractmethod
    def label_peak(
        self,
        peak_id: int,
        probs: np.ndarray,
        candidate_map: List[Optional[int]],
        true_compound: Optional[int],
    ) -> int:
        """
        Provide a label for a peak.

        Parameters
        ----------
        peak_id : int
            ID of the peak to label
        probs : np.ndarray
            Model's probability distribution over [null, candidates...]
        candidate_map : List[Optional[int]]
            Mapping from slot index to compound ID (None for null at index 0)
        true_compound : Optional[int]
            True compound ID if known, None for null peaks

        Returns
        -------
        int
            Chosen label index (0 for null, 1+ for candidates)
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return oracle name for logging."""
        pass


class OptimalOracle(Oracle):
    """
    Perfect oracle that always provides the correct label.

    This represents an expert annotator with perfect knowledge.
    """

    def label_peak(
        self,
        peak_id: int,
        probs: np.ndarray,
        candidate_map: List[Optional[int]],
        true_compound: Optional[int],
    ) -> int:
        """Return the correct label."""
        if true_compound is None:
            return 0  # Null
        elif true_compound in candidate_map:
            return candidate_map.index(true_compound)
        else:
            # True compound not in candidates (filtered out)
            return 0  # Default to null

    @property
    def name(self) -> str:
        return "Optimal"


class SmartNoisyOracle(Oracle):
    """
    Oracle that uses peak characteristics to make intelligent but imperfect decisions.

    This simulates an expert who uses intensity, RT match, and other features
    to determine if a peak is real or noise. Enhanced with contextual pattern
    recognition for more realistic human-like decisions.
    """

    def __init__(self, base_accuracy: float = 0.7, seed: int = 42, use_context: bool = True):
        """
        Initialize smart oracle.

        Parameters
        ----------
        base_accuracy : float
            Base probability of correct labeling (0.6-0.8 recommended)
        seed : int
            Random seed
        use_context : bool
            Whether to use contextual patterns (isotopes, adducts, etc.)
        """
        self.base_accuracy = base_accuracy
        self.rng = np.random.default_rng(seed)
        self.use_context = use_context
        self.seen_patterns = {}  # Memory for pattern recognition

    def label_peak(
        self,
        peak_id: int,
        probs: np.ndarray,
        candidate_map: List[Optional[int]],
        true_compound: Optional[int],
        peak_features: Optional[Dict] = None,
    ) -> int:
        """
        Make intelligent labeling decision based on peak characteristics.

        Uses intensity and RT confidence to adjust accuracy.
        """
        # Get the correct label
        if true_compound is None:
            correct_label = 0  # Null
        elif true_compound in candidate_map:
            correct_label = candidate_map.index(true_compound)
        else:
            correct_label = 0  # Default to null

        # Calculate accuracy based on peak features if available
        accuracy = self.base_accuracy

        if peak_features is not None:
            # Higher intensity → more confident
            if "intensity" in peak_features:
                intensity_score = min(np.log1p(peak_features["intensity"]) / 15, 1.0)
                accuracy += 0.10 * intensity_score

            # Peak shape cues (allowed): narrow width, modest asymmetry
            # Prefer narrower peaks and moderate asymmetry (not extreme tailing)
            width_val = None
            if "peak_width_rt" in peak_features:
                width_val = peak_features["peak_width_rt"]
            elif "peak_width" in peak_features:
                width_val = peak_features["peak_width"]
            if width_val is not None:
                width_score = max(0.0, 1.0 - float(width_val) / 0.5)
                accuracy += 0.06 * width_score

            if "peak_asymmetry" in peak_features:
                asym = float(peak_features["peak_asymmetry"])
                # Ideal near ~1.2; penalize extremes
                asym_score = max(0.0, 1.0 - abs(asym - 1.2) / 1.2)
                accuracy += 0.04 * asym_score

            # Model's confidence also influences oracle (meta-signal)
            if len(probs) > 0:
                max_prob = float(np.max(probs))
                if max_prob > 0.8:  # High confidence
                    accuracy += 0.05

        # Apply contextual patterns if enabled
        if self.use_context and peak_features is not None:
            accuracy = self._apply_contextual_patterns(accuracy, peak_features, peak_id)

        # Clip to reasonable range
        accuracy = np.clip(accuracy, 0.1, 0.95)

        # Make decision
        if self.rng.random() < accuracy:
            return correct_label
        else:
            # Random incorrect label
            choices = list(range(len(probs)))
            choices.remove(correct_label)
            return self.rng.choice(choices) if choices else correct_label

    def _apply_contextual_patterns(
        self, accuracy: float, peak_features: Dict, peak_id: int
    ) -> float:
        """
        Apply contextual pattern recognition to adjust accuracy.

        Simulates human recognition of:
        - Isotope patterns (M+1, M+2)
        - Adduct series ([M+H]+, [M+Na]+, etc.)
        - Fragment patterns
        - Consistent contamination patterns
        """
        adjusted_accuracy = accuracy

        # Check for isotope pattern (mass difference ~1.003)
        if "mass" in peak_features:
            mass = peak_features["mass"]
            # Store pattern for future reference
            mass_key = f"{mass:.2f}"

            # Check if we've seen a related mass before (potential isotope)
            for stored_mass in self.seen_patterns.keys():
                mass_diff = abs(float(stored_mass) - mass)
                if 0.99 < mass_diff < 1.01:  # Likely isotope
                    adjusted_accuracy += 0.10  # More confident due to pattern
                    break
                elif 21.98 < mass_diff < 22.02:  # Likely Na adduct
                    adjusted_accuracy += 0.08
                    break

            # Remember this mass
            self.seen_patterns[mass_key] = peak_id

        # Simple pattern proxy from allowed features: very weak, broad peaks behave like noise
        if "intensity" in peak_features:
            inten = float(peak_features["intensity"])
            if inten < np.exp(10):  # fairly weak
                adjusted_accuracy += 0.02
        if "peak_width_rt" in peak_features:
            if float(peak_features["peak_width_rt"]) > 0.4:
                adjusted_accuracy += 0.02

        # RT clustering - peaks at similar RT more likely related
        if "rt" in peak_features:
            rt = peak_features["rt"]
            # Simple check for common contaminant RTs
            common_contaminant_rts = [3.0, 5.0, 7.0, 10.0, 12.0]
            for cont_rt in common_contaminant_rts:
                if abs(rt - cont_rt) < 0.5:
                    adjusted_accuracy += 0.03  # Slightly more confident
                    break

        return adjusted_accuracy

    @property
    def name(self) -> str:
        name = f"SmartNoisy_{int(self.base_accuracy*100)}"
        if self.use_context:
            name += "_contextual"
        return name


class RandomOracle(Oracle):
    """
    Random oracle that selects labels uniformly at random.

    This represents an annotator with no domain knowledge.
    """

    def __init__(self, seed: int = 42):
        """Initialize with random seed."""
        self.rng = np.random.default_rng(seed)

    def label_peak(
        self,
        peak_id: int,
        probs: np.ndarray,
        candidate_map: List[Optional[int]],
        true_compound: Optional[int],
    ) -> int:
        """Return a random label."""
        return self.rng.choice(len(probs))

    @property
    def name(self) -> str:
        return "Random"


class NoisyOracle(Oracle):
    """
    Noisy oracle that is mostly correct but makes occasional mistakes.

    This represents a skilled but imperfect annotator.
    """

    def __init__(self, flip_prob: float = 0.1, seed: int = 42):
        """
        Initialize noisy oracle.

        Parameters
        ----------
        flip_prob : float
            Probability of making an error
        seed : int
            Random seed
        """
        self.flip_prob = flip_prob
        self.rng = np.random.default_rng(seed)
        self.optimal = OptimalOracle()

    def label_peak(
        self,
        peak_id: int,
        probs: np.ndarray,
        candidate_map: List[Optional[int]],
        true_compound: Optional[int],
    ) -> int:
        """Return correct label with probability (1 - flip_prob)."""
        correct_label = self.optimal.label_peak(peak_id, probs, candidate_map, true_compound)

        if self.rng.random() < self.flip_prob:
            # Make an error - choose different label
            options = list(range(len(probs)))
            options.remove(correct_label)
            if options:
                return self.rng.choice(options)

        return correct_label

    @property
    def name(self) -> str:
        return f"Noisy(p={self.flip_prob})"


class ConservativeOracle(Oracle):
    """
    Conservative oracle that only labels when highly confident.

    This represents a cautious annotator who prefers null when uncertain.
    """

    def __init__(self, min_prob: float = 0.8, seed: int = 42):
        """
        Initialize conservative oracle.

        Parameters
        ----------
        min_prob : float
            Minimum model probability required to trust the model
        seed : int
            Random seed
        """
        self.min_prob = min_prob
        self.rng = np.random.default_rng(seed)
        self.optimal = OptimalOracle()

    def label_peak(
        self,
        peak_id: int,
        probs: np.ndarray,
        candidate_map: List[Optional[int]],
        true_compound: Optional[int],
    ) -> int:
        """Label correctly if model is confident, otherwise null."""
        max_prob = np.max(probs)

        if max_prob >= self.min_prob:
            # Model is confident - provide correct label
            return self.optimal.label_peak(peak_id, probs, candidate_map, true_compound)
        else:
            # Model is uncertain - default to null
            return 0

    @property
    def name(self) -> str:
        return f"Conservative(τ={self.min_prob})"


class ModelGuidedOracle(Oracle):
    """
    Oracle that tends to agree with the model's predictions.

    This represents an annotator influenced by model suggestions.
    """

    def __init__(self, agreement_prob: float = 0.7, seed: int = 42):
        """
        Initialize model-guided oracle.

        Parameters
        ----------
        agreement_prob : float
            Probability of agreeing with model's top choice
        seed : int
            Random seed
        """
        self.agreement_prob = agreement_prob
        self.rng = np.random.default_rng(seed)
        self.optimal = OptimalOracle()

    def label_peak(
        self,
        peak_id: int,
        probs: np.ndarray,
        candidate_map: List[Optional[int]],
        true_compound: Optional[int],
    ) -> int:
        """Agree with model with some probability, else be optimal."""
        if self.rng.random() < self.agreement_prob:
            # Agree with model's top choice
            return np.argmax(probs)
        else:
            # Provide correct label
            return self.optimal.label_peak(peak_id, probs, candidate_map, true_compound)

    @property
    def name(self) -> str:
        return f"ModelGuided(p={self.agreement_prob})"


class AdversarialOracle(Oracle):
    """
    Adversarial oracle that deliberately provides wrong labels.

    This tests model robustness to malicious or completely wrong feedback.
    """

    def __init__(self, seed: int = 42):
        """Initialize adversarial oracle."""
        self.rng = np.random.default_rng(seed)
        self.optimal = OptimalOracle()

    def label_peak(
        self,
        peak_id: int,
        probs: np.ndarray,
        candidate_map: List[Optional[int]],
        true_compound: Optional[int],
    ) -> int:
        """Return the wrong label with highest model probability."""
        correct_label = self.optimal.label_peak(peak_id, probs, candidate_map, true_compound)

        # Find wrong option with highest probability
        wrong_probs = probs.copy()
        wrong_probs[correct_label] = -1  # Exclude correct answer

        if np.max(wrong_probs) > -1:
            return np.argmax(wrong_probs)
        else:
            # All options are correct (shouldn't happen)
            return 0

    @property
    def name(self) -> str:
        return "Adversarial"


class ProbabilisticOracle(Oracle):
    """
    Oracle that samples from the true posterior distribution.

    This represents uncertainty in human annotations.
    """

    def __init__(self, correctness_rate: float = 0.9, seed: int = 42):
        """
        Initialize probabilistic oracle.

        Parameters
        ----------
        correctness_rate : float
            Base probability of being correct
        seed : int
            Random seed
        """
        self.correctness_rate = correctness_rate
        self.rng = np.random.default_rng(seed)
        self.optimal = OptimalOracle()

    def label_peak(
        self,
        peak_id: int,
        probs: np.ndarray,
        candidate_map: List[Optional[int]],
        true_compound: Optional[int],
    ) -> int:
        """Sample from distribution centered on correct answer."""
        correct_label = self.optimal.label_peak(peak_id, probs, candidate_map, true_compound)

        # Handle edge case where only null class is available
        if len(probs) == 1:
            return 0

        # Build distribution
        label_probs = np.ones(len(probs)) * (1 - self.correctness_rate) / (len(probs) - 1)
        label_probs[correct_label] = self.correctness_rate

        # Sample
        return self.rng.choice(len(probs), p=label_probs)

    @property
    def name(self) -> str:
        return f"Probabilistic(p={self.correctness_rate})"


class MixtureOracle(Oracle):
    """
    Mixture of different oracle types to simulate diverse annotator pool.

    This represents a realistic scenario with annotators of varying expertise.
    """

    def __init__(
        self, oracles: List[Oracle], weights: Optional[List[float]] = None, seed: int = 42
    ):
        """
        Initialize mixture oracle.

        Parameters
        ----------
        oracles : List[Oracle]
            List of oracles to mix
        weights : Optional[List[float]]
            Probability weights for each oracle (uniform if None)
        seed : int
            Random seed
        """
        self.oracles = oracles
        if weights is None:
            weights = [1.0] * len(oracles)
        self.weights = np.array(weights) / np.sum(weights)
        self.rng = np.random.default_rng(seed)

    def label_peak(
        self,
        peak_id: int,
        probs: np.ndarray,
        candidate_map: List[Optional[int]],
        true_compound: Optional[int],
    ) -> int:
        """Sample an oracle and use its label."""
        oracle_idx = self.rng.choice(len(self.oracles), p=self.weights)
        return self.oracles[oracle_idx].label_peak(peak_id, probs, candidate_map, true_compound)

    @property
    def name(self) -> str:
        oracle_names = [o.name for o in self.oracles]
        return f"Mixture({', '.join(oracle_names)})"
