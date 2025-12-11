"""
Active learning strategies for peak selection.

This module implements acquisition functions and selection strategies for
active learning in compound assignment, focusing on precision control and
uncertainty reduction.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.spatial.distance import pdist, squareform


def expected_fp_reduction(probs: np.ndarray, threshold: float) -> float:
    """
    Calculate expected false positive reduction for a peak.

    This acquisition function targets peaks that would be assigned
    (above threshold) but have high uncertainty.

    Parameters
    ----------
    probs : np.ndarray
        Probability distribution over [null, candidates...]
    threshold : float
        Assignment threshold

    Returns
    -------
    float
        Expected FP reduction score
    """
    max_prob = np.max(probs)
    argmax = np.argmax(probs)

    # Only consider non-null assignments above threshold
    if argmax > 0 and max_prob >= threshold:
        return 1.0 - max_prob
    else:
        return 0.0


def entropy(probs: np.ndarray) -> float:
    """
    Calculate Shannon entropy of probability distribution.

    Parameters
    ----------
    probs : np.ndarray
        Probability distribution

    Returns
    -------
    float
        Entropy value
    """
    probs_clipped = np.clip(probs, 1e-12, 1.0)
    return -np.sum(probs_clipped * np.log(probs_clipped))


def mutual_information(prob_samples: np.ndarray) -> float:
    """
    Calculate mutual information (BALD) from posterior samples.

    MI = H[p(y|x)] - E[H[p(y|x,θ)]]

    Parameters
    ----------
    prob_samples : np.ndarray
        Shape (n_samples, n_classes). Probability samples from posterior.

    Returns
    -------
    float
        Mutual information
    """
    # Predictive entropy
    mean_probs = np.mean(prob_samples, axis=0)
    pred_entropy = entropy(mean_probs)

    # Expected entropy under posterior
    sample_entropies = [entropy(prob_samples[i]) for i in range(len(prob_samples))]
    expected_entropy = np.mean(sample_entropies)

    return pred_entropy - expected_entropy


def least_confident(probs: np.ndarray) -> float:
    """
    Least confidence acquisition (1 - max_prob).

    Parameters
    ----------
    probs : np.ndarray
        Probability distribution

    Returns
    -------
    float
        Least confidence score
    """
    return 1.0 - np.max(probs)


def margin_sampling(probs: np.ndarray) -> float:
    """
    Margin between top two probabilities.

    Parameters
    ----------
    probs : np.ndarray
        Probability distribution

    Returns
    -------
    float
        Negative margin (smaller margin = higher score)
    """
    sorted_probs = np.sort(probs)[::-1]
    if len(sorted_probs) >= 2:
        return -(sorted_probs[0] - sorted_probs[1])
    else:
        return 0.0


def hybrid_acquisition(
    probs: np.ndarray, threshold: float, lambda_fp: float = 0.7, lambda_entropy: float = 0.3
) -> float:
    """
    Hybrid acquisition combining FP reduction and entropy.

    Parameters
    ----------
    probs : np.ndarray
        Probability distribution
    threshold : float
        Assignment threshold
    lambda_fp : float
        Weight for FP reduction
    lambda_entropy : float
        Weight for entropy

    Returns
    -------
    float
        Combined acquisition score
    """
    fp_score = expected_fp_reduction(probs, threshold)
    entropy_score = entropy(probs)

    # Normalize and combine
    return lambda_fp * fp_score + lambda_entropy * entropy_score


def select_batch(
    probs_dict: Dict[int, np.ndarray],
    batch_size: int,
    acquisition_fn: str = "hybrid",
    threshold: float = 0.75,
    lambda_fp: float = 0.7,
    diversity_k: Optional[int] = None,
    features_dict: Optional[Dict[int, np.ndarray]] = None,
    prob_samples_dict: Optional[Dict[int, np.ndarray]] = None,
) -> List[int]:
    """
    Select batch of peaks for annotation.

    Parameters
    ----------
    probs_dict : Dict[int, np.ndarray]
        Dictionary of peak_id -> probability distribution
    batch_size : int
        Number of peaks to select
    acquisition_fn : str
        Acquisition function: 'fp', 'entropy', 'mi', 'lc', 'margin', 'hybrid'
    threshold : float
        Assignment threshold (for FP-based acquisition)
    lambda_fp : float
        Weight for FP in hybrid acquisition
    diversity_k : Optional[int]
        If provided, use diverse selection among top-k candidates
    features_dict : Optional[Dict[int, np.ndarray]]
        Peak features for diversity calculation
    prob_samples_dict : Optional[Dict[int, np.ndarray]]
        Posterior probability samples for MI/BALD acquisition

    Returns
    -------
    List[int]
        Selected peak IDs
    """
    # Calculate acquisition scores
    scores = {}

    for peak_id, probs in probs_dict.items():
        if acquisition_fn == "fp":
            score = expected_fp_reduction(probs, threshold)
        elif acquisition_fn == "entropy":
            score = entropy(probs)
        elif acquisition_fn == "lc":
            score = least_confident(probs)
        elif acquisition_fn == "margin":
            score = margin_sampling(probs)
        elif acquisition_fn == "mi":
            if prob_samples_dict is None or peak_id not in prob_samples_dict:
                raise ValueError(
                    "MI acquisition requires prob_samples_dict with posterior samples."
                )
            score = mutual_information(prob_samples_dict[peak_id])
        elif acquisition_fn == "hybrid":
            score = hybrid_acquisition(probs, threshold, lambda_fp, 1 - lambda_fp)
        else:
            raise ValueError(f"Unknown acquisition function: {acquisition_fn}")

        scores[peak_id] = score

    # Sort by score (descending)
    sorted_peaks = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Simple selection or diverse selection
    if diversity_k is None or features_dict is None:
        # Simple top-k selection
        selected = [pid for pid, _ in sorted_peaks[:batch_size]]
    else:
        # Diverse selection from top-k candidates
        selected = diverse_selection(sorted_peaks[:diversity_k], features_dict, batch_size)

    return selected


def diverse_selection(
    candidates: List[Tuple[int, float]], features_dict: Dict[int, np.ndarray], batch_size: int
) -> List[int]:
    """
    Select diverse batch from candidates using farthest point sampling.

    Parameters
    ----------
    candidates : List[Tuple[int, float]]
        List of (peak_id, score) tuples
    features_dict : Dict[int, np.ndarray]
        Feature vectors for each peak
    batch_size : int
        Number to select

    Returns
    -------
    List[int]
        Selected diverse peak IDs
    """
    if len(candidates) <= batch_size:
        return [pid for pid, _ in candidates]

    peak_ids = [pid for pid, _ in candidates]
    feats = np.array([features_dict[pid] for pid in peak_ids])

    distances = squareform(pdist(feats, metric="euclidean"))

    selected_idx = [0]  # start from highest-scoring candidate
    remaining_idx = list(range(1, len(peak_ids)))
    selected_ids = [peak_ids[0]]

    while len(selected_ids) < batch_size and remaining_idx:
        # Get distances from remaining points to all selected points
        d_sel = distances[remaining_idx][:, selected_idx]
        # Find minimum distance to any selected point for each remaining point
        min_dists = d_sel.min(axis=1)
        # Select the point with maximum minimum distance (farthest from all selected)
        next_pos = remaining_idx[int(np.argmax(min_dists))]
        selected_idx.append(next_pos)
        selected_ids.append(peak_ids[next_pos])
        remaining_idx.remove(next_pos)

    return selected_ids


class ActiveLearner:
    """
    Active learning controller for peak assignment.

    This class manages the active learning loop, tracking selected peaks
    and computing acquisition scores.
    """

    def __init__(
        self,
        acquisition_fn: str = "hybrid",
        threshold: float = 0.75,
        lambda_fp: float = 0.7,
        diversity_k: Optional[int] = None,
    ):
        """
        Initialize active learner.

        Parameters
        ----------
        acquisition_fn : str
            Acquisition function to use
        threshold : float
            Assignment threshold
        lambda_fp : float
            Weight for FP reduction in hybrid
        diversity_k : Optional[int]
            Top-k for diverse selection
        """
        self.acquisition_fn = acquisition_fn
        self.threshold = threshold
        self.lambda_fp = lambda_fp
        self.diversity_k = diversity_k
        self.selected_peaks = set()
        self.round_history = []

    def select_next_batch(
        self,
        probs_dict: Dict[int, np.ndarray],
        batch_size: int,
        features_dict: Optional[Dict[int, np.ndarray]] = None,
        prob_samples_dict: Optional[Dict[int, np.ndarray]] = None,
    ) -> List[int]:
        """
        Select next batch of peaks to annotate.

        Parameters
        ----------
        probs_dict : Dict[int, np.ndarray]
            Current probability distributions
        batch_size : int
            Batch size
        features_dict : Optional[Dict[int, np.ndarray]]
            Features for diversity

        Returns
        -------
        List[int]
            Selected peak IDs
        """
        # Filter out already selected peaks
        available_probs = {
            pid: probs for pid, probs in probs_dict.items() if pid not in self.selected_peaks
        }

        if not available_probs:
            return []

        # Select batch
        batch = select_batch(
            available_probs,
            min(batch_size, len(available_probs)),
            self.acquisition_fn,
            self.threshold,
            self.lambda_fp,
            self.diversity_k,
            features_dict,
            prob_samples_dict,
        )

        # Update history
        self.selected_peaks.update(batch)
        self.round_history.append(batch)

        return batch

    def reset(self):
        """Reset the learner state."""
        self.selected_peaks.clear()
        self.round_history.clear()
