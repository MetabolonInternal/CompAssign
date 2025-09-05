"""
Evaluation loop for simulating annotation rounds with oracles.

This module provides functions for simulating the active learning loop where
oracles provide labels, the model is updated, and metrics are tracked to measure
improvement over rounds.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from .peak_assignment_softmax import PeakAssignmentSoftmaxModel
from .presence_prior import PresencePrior
from .oracles import Oracle
from .active_learning import select_batch


@dataclass
class AnnotationRoundResults:
    """Results from a single annotation round."""
    round_num: int
    n_annotations: int
    oracle_name: str
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]
    entropy_before: float
    entropy_after: float
    expected_fp_before: float
    expected_fp_after: float
    labeled_peaks: List[int]
    labels_provided: List[int]


def calculate_entropy(probs_dict: Dict[int, np.ndarray]) -> float:
    """
    Calculate total entropy across all peaks.
    
    Parameters
    ----------
    probs_dict : Dict[int, np.ndarray]
        Dictionary of peak_id -> probability distribution
        
    Returns
    -------
    float
        Total entropy
    """
    total_entropy = 0.0
    for peak_id, probs in probs_dict.items():
        # Shannon entropy
        probs_clipped = np.clip(probs, 1e-12, 1.0)
        entropy = -np.sum(probs_clipped * np.log(probs_clipped))
        total_entropy += entropy
    return total_entropy


def calculate_expected_fp(probs_dict: Dict[int, np.ndarray], 
                         threshold: float) -> float:
    """
    Calculate expected false positives at given threshold.
    
    Expected FP = sum of (1 - p) for all peaks where p >= threshold
    
    Parameters
    ----------
    probs_dict : Dict[int, np.ndarray]
        Dictionary of peak_id -> probability distribution
    threshold : float
        Assignment threshold
        
    Returns
    -------
    float
        Expected number of false positives
    """
    expected_fp = 0.0
    for peak_id, probs in probs_dict.items():
        max_prob = np.max(probs)
        if max_prob >= threshold and np.argmax(probs) > 0:  # Not null
            expected_fp += (1 - max_prob)
    return expected_fp


def simulate_annotation_round(
    softmax_model: PeakAssignmentSoftmaxModel,
    oracle: Oracle,
    batch_peak_ids: List[int],
    peak_df: pd.DataFrame,
    compound_mass: np.ndarray,
    threshold: float = 0.75,
    refit_model: bool = True,
    verbose: bool = True
) -> AnnotationRoundResults:
    """
    Simulate one round of annotation.
    
    Parameters
    ----------
    softmax_model : PeakAssignmentSoftmaxModel
        The softmax assignment model
    oracle : Oracle
        Oracle to provide labels
    batch_peak_ids : List[int]
        Peak IDs to annotate in this round
    peak_df : pd.DataFrame
        Peak data
    compound_mass : np.ndarray
        Compound masses
    threshold : float
        Assignment threshold for metrics
    refit_model : bool
        If True, refit model after annotations
    verbose : bool
        If True, print progress
        
    Returns
    -------
    AnnotationRoundResults
        Results from this annotation round
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Annotation Round with {oracle.name} Oracle")
        print(f"Annotating {len(batch_peak_ids)} peaks")
        print(f"{'='*60}")
    
    # Get initial metrics
    probs_before = softmax_model.predict_probs()
    results_before = softmax_model.assign(prob_threshold=threshold)
    entropy_before = calculate_entropy(probs_before)
    expected_fp_before = calculate_expected_fp(probs_before, threshold)
    
    metrics_before = {
        'precision': results_before.precision,
        'recall': results_before.recall,
        'f1': results_before.f1,
        'ece': results_before.ece,
        'mce': results_before.mce
    }
    
    if verbose:
        print(f"\nMetrics before annotation:")
        print(f"  Entropy: {entropy_before:.3f}")
        print(f"  Expected FP: {expected_fp_before:.3f}")
        print(f"  Precision: {metrics_before['precision']:.3f}")
        print(f"  Recall: {metrics_before['recall']:.3f}")
    
    # Collect oracle labels
    labeled_peaks = []
    labels_provided = []
    rt_observations = []  # For future RT model updates
    
    for peak_id in batch_peak_ids:
        if peak_id not in probs_before:
            continue
            
        # Get peak info
        peak_row = peak_df[peak_df['peak_id'] == peak_id].iloc[0]
        species_idx = int(peak_row['species'])
        true_comp = peak_row['true_compound']
        if not pd.isna(true_comp):
            true_comp = int(true_comp)
        else:
            true_comp = None
        
        # Get model's current beliefs
        probs = probs_before[peak_id]
        
        # Get candidate mapping
        row_idx = softmax_model.train_pack['peak_to_row'][peak_id]
        candidate_map = softmax_model.train_pack['row_to_candidates'][row_idx]
        
        # Get oracle label with peak features for intelligent decisions
        peak_features = {
            'intensity': peak_row['intensity'],
            'rt': peak_row['rt'],
            'mass': peak_row['mass']
        }
        
        # Add discriminative features if available (only non-leaky, observed features)
        if 'peak_quality' in peak_row:
            peak_features['peak_quality'] = peak_row['peak_quality']
        if 'mass_error_ppm' in peak_row:
            peak_features['mass_error_ppm'] = peak_row['mass_error_ppm']
        if 'sn_ratio' in peak_row:
            peak_features['sn_ratio'] = peak_row['sn_ratio']
        if 'peak_width' in peak_row:
            peak_features['peak_width'] = peak_row['peak_width']
        if 'peak_width_rt' in peak_row:
            peak_features['peak_width_rt'] = peak_row['peak_width_rt']
        if 'peak_asymmetry' in peak_row:
            peak_features['peak_asymmetry'] = peak_row['peak_asymmetry']
        
        # Check if oracle supports peak features (SmartNoisyOracle does)
        import inspect
        if 'peak_features' in inspect.signature(oracle.label_peak).parameters:
            label = oracle.label_peak(peak_id, probs, candidate_map, true_comp, peak_features)
        else:
            label = oracle.label_peak(peak_id, probs, candidate_map, true_comp)
        
        labeled_peaks.append(peak_id)
        labels_provided.append(label)
        
        # Update presence prior based on label
        if label == 0:
            # Null assignment
            softmax_model.presence.update_null()
        else:
            # Compound assignment
            compound_idx = candidate_map[label]
            softmax_model.presence.update_positive(species_idx, compound_idx, weight=1.0)
            softmax_model.presence.update_not_null()
            
            # Light negatives for other candidates of this peak
            for k in range(1, len(candidate_map)):
                c_alt = candidate_map[k]
                if c_alt is not None and c_alt != compound_idx:
                    softmax_model.presence.update_negative(species_idx, c_alt, weight=0.25)
            
            # Store RT observation for potential RT model update
            rt_observations.append({
                'species': species_idx,
                'compound': compound_idx,
                'rt': peak_row['rt']
            })
    
    if verbose:
        print(f"\nCollected {len(labeled_peaks)} annotations")
        n_null = sum(1 for l in labels_provided if l == 0)
        print(f"  Null labels: {n_null}")
        print(f"  Non-null labels: {len(labels_provided) - n_null}")
    
    # Refit model if requested
    if refit_model:
        if verbose:
            print("\nRefitting model with updated priors...")
        
        # Update labels in training pack
        for peak_id, label in zip(labeled_peaks, labels_provided):
            row_idx = softmax_model.train_pack['peak_to_row'][peak_id]
            softmax_model.train_pack['labels'][row_idx] = label
        
        # Rebuild and resample model
        softmax_model.build_model()
        softmax_model.sample(draws=1000, tune=1000, chains=4)  # Quick refit
    
    # Get updated metrics
    probs_after = softmax_model.predict_probs()
    results_after = softmax_model.assign(prob_threshold=threshold)
    entropy_after = calculate_entropy(probs_after)
    expected_fp_after = calculate_expected_fp(probs_after, threshold)
    
    metrics_after = {
        'precision': results_after.precision,
        'recall': results_after.recall,
        'f1': results_after.f1,
        'ece': results_after.ece,
        'mce': results_after.mce
    }
    
    if verbose:
        print(f"\nMetrics after annotation:")
        print(f"  Entropy: {entropy_after:.3f} (Δ = {entropy_after - entropy_before:+.3f})")
        print(f"  Expected FP: {expected_fp_after:.3f} (Δ = {expected_fp_after - expected_fp_before:+.3f})")
        print(f"  Precision: {metrics_after['precision']:.3f} (Δ = {metrics_after['precision'] - metrics_before['precision']:+.3f})")
        print(f"  Recall: {metrics_after['recall']:.3f} (Δ = {metrics_after['recall'] - metrics_before['recall']:+.3f})")
    
    return AnnotationRoundResults(
        round_num=1,
        n_annotations=len(labeled_peaks),
        oracle_name=oracle.name,
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        entropy_before=entropy_before,
        entropy_after=entropy_after,
        expected_fp_before=expected_fp_before,
        expected_fp_after=expected_fp_after,
        labeled_peaks=labeled_peaks,
        labels_provided=labels_provided
    )


def run_annotation_experiment(
    softmax_model: PeakAssignmentSoftmaxModel,
    oracle: Oracle,
    peak_df: pd.DataFrame,
    compound_mass: np.ndarray,
    n_rounds: int = 5,
    batch_size: int = 10,
    threshold: float = 0.75,
    selection_method: str = 'random',
    verbose: bool = True
) -> List[AnnotationRoundResults]:
    """
    Run multiple rounds of annotation.
    
    Parameters
    ----------
    softmax_model : PeakAssignmentSoftmaxModel
        The softmax assignment model
    oracle : Oracle
        Oracle to provide labels
    peak_df : pd.DataFrame
        Peak data
    compound_mass : np.ndarray
        Compound masses
    n_rounds : int
        Number of annotation rounds
    batch_size : int
        Number of peaks to annotate per round
    threshold : float
        Assignment threshold
    selection_method : str
        Method for selecting peaks ('random' or 'uncertainty')
    verbose : bool
        If True, print progress
        
    Returns
    -------
    List[AnnotationRoundResults]
        Results from all rounds
    """
    results = []
    annotated_peaks = set()
    
    for round_num in range(1, n_rounds + 1):
        if verbose:
            print(f"\n\n{'#'*60}")
            print(f"ROUND {round_num} / {n_rounds}")
            print(f"{'#'*60}")
        
        # Select peaks for annotation
        available_peaks = [pid for pid in softmax_model.train_pack['peak_ids'] 
                          if pid not in annotated_peaks]
        
        if len(available_peaks) == 0:
            print("No more peaks to annotate!")
            break
        
        # Build features dictionary for diversity-aware selection
        X = softmax_model.train_pack['X']  # (N, Kmax, 9)
        mask = softmax_model.train_pack['mask']  # (N, Kmax)
        peak_ids_array = softmax_model.train_pack['peak_ids']
        
        # Get posterior mean probabilities
        p_mean = softmax_model.trace.posterior['p'].values.mean(axis=(0,1))  # (N, Kmax)
        
        features_dict = {}
        for i, pid in enumerate(peak_ids_array):
            valid_k = np.where(mask[i])[0]
            # Exclude null slot (k=0) from the embedding
            k_nonnull = valid_k[valid_k > 0]
            if len(k_nonnull) == 0:
                features_dict[int(pid)] = np.zeros(X.shape[-1])
                continue
            w = p_mean[i, k_nonnull]
            w = w / (w.sum() + 1e-12)
            # Expected non-null feature vector
            feats = (w[:, None] * X[i, k_nonnull, :]).sum(axis=0)
            features_dict[int(pid)] = feats
        
        # Get probability predictions
        probs_dict = softmax_model.predict_probs()
        available_probs = {pid: probs_dict[pid] for pid in available_peaks}
        
        # Handle special case for random selection
        if selection_method == 'random':
            batch_peaks = np.random.choice(
                available_peaks, 
                size=min(batch_size, len(available_peaks)),
                replace=False
            ).tolist()
        else:
            # Map selection methods to acquisition functions
            acq_map = {
                'uncertainty': 'entropy',
                'entropy': 'entropy',
                'fp': 'fp',
                'mi': 'mi',
                'lc': 'lc',
                'margin': 'margin',
                'hybrid': 'hybrid'
            }
            
            acquisition_fn = acq_map.get(selection_method, 'hybrid')
            
            # Get posterior samples if using MI
            prob_samples_dict = None
            if acquisition_fn == 'mi':
                prob_samples_dict = softmax_model.predict_prob_samples()
                prob_samples_dict = {pid: prob_samples_dict[pid] for pid in available_peaks}
            
            # Use active learning selector
            batch_peaks = select_batch(
                probs_dict=available_probs,
                batch_size=min(batch_size, len(available_peaks)),
                acquisition_fn=acquisition_fn,
                threshold=threshold,
                lambda_fp=0.7,
                diversity_k=max(5, batch_size * 3),  # Small top-k pool for diversity
                features_dict=features_dict,
                prob_samples_dict=prob_samples_dict
            )
        
        # Run annotation round
        round_results = simulate_annotation_round(
            softmax_model=softmax_model,
            oracle=oracle,
            batch_peak_ids=batch_peaks,
            peak_df=peak_df,
            compound_mass=compound_mass,
            threshold=threshold,
            refit_model=True,
            verbose=verbose
        )
        
        round_results.round_num = round_num
        results.append(round_results)
        annotated_peaks.update(batch_peaks)
    
    # Summary
    if verbose:
        print(f"\n\n{'='*60}")
        print("EXPERIMENT SUMMARY")
        print(f"{'='*60}")
        print(f"Total rounds: {len(results)}")
        print(f"Total annotations: {len(annotated_peaks)}")
        print(f"Oracle: {oracle.name}")
        
        if results:
            initial_entropy = results[0].entropy_before
            final_entropy = results[-1].entropy_after
            initial_precision = results[0].metrics_before['precision']
            final_precision = results[-1].metrics_after['precision']
            
            print(f"\nEntropy reduction: {initial_entropy:.3f} -> {final_entropy:.3f} "
                  f"(Δ = {final_entropy - initial_entropy:+.3f})")
            print(f"Precision improvement: {initial_precision:.3f} -> {final_precision:.3f} "
                  f"(Δ = {final_precision - initial_precision:+.3f})")
    
    return results


def compare_oracles(
    softmax_model: PeakAssignmentSoftmaxModel,
    oracles: List[Oracle],
    peak_df: pd.DataFrame,
    compound_mass: np.ndarray,
    n_rounds: int = 5,
    batch_size: int = 10,
    threshold: float = 0.75,
    selection_method: str = 'random'
) -> Dict[str, List[AnnotationRoundResults]]:
    """
    Compare multiple oracles on the same task.
    
    Parameters
    ----------
    softmax_model : PeakAssignmentSoftmaxModel
        Base model (will be copied for each oracle)
    oracles : List[Oracle]
        Oracles to compare
    peak_df : pd.DataFrame
        Peak data
    compound_mass : np.ndarray
        Compound masses
    n_rounds : int
        Number of annotation rounds
    batch_size : int
        Batch size per round
    threshold : float
        Assignment threshold
    selection_method : str
        Peak selection method
        
    Returns
    -------
    Dict[str, List[AnnotationRoundResults]]
        Results for each oracle
    """
    results = {}
    
    for oracle in oracles:
        print(f"\n{'='*80}")
        print(f"Testing Oracle: {oracle.name}")
        print(f"{'='*80}")
        
        # Create a fresh model for each oracle to avoid contamination
        fresh_model = PeakAssignmentSoftmaxModel(
            mass_tolerance=softmax_model.mass_tolerance,
            rt_window_k=softmax_model.rt_window_k,
            use_temperature=softmax_model.use_temperature,
            standardize_features=softmax_model.standardize_features,
            random_seed=softmax_model.random_seed
        )
        
        # Copy RT predictions
        fresh_model.rt_predictions = softmax_model.rt_predictions
        
        # Regenerate training data with fresh presence priors
        n_species = len(np.unique(peak_df['species']))
        n_compounds = len(compound_mass)
        species_cluster = peak_df['species'].values
        
        fresh_model.generate_training_data(
            peak_df=peak_df,
            compound_mass=compound_mass,
            n_compounds=n_compounds,
            species_cluster=species_cluster,
            init_presence=PresencePrior.init(n_species, n_compounds)
        )
        
        # Build and sample the model
        fresh_model.build_model()
        fresh_model.sample(draws=1000, tune=1000, chains=4)  # Quick initial sampling
        
        oracle_results = run_annotation_experiment(
            softmax_model=fresh_model,
            oracle=oracle,
            peak_df=peak_df,
            compound_mass=compound_mass,
            n_rounds=n_rounds,
            batch_size=batch_size,
            threshold=threshold,
            selection_method=selection_method,
            verbose=False
        )
        
        results[oracle.name] = oracle_results
        
        # Print summary
        if oracle_results:
            initial = oracle_results[0]
            final = oracle_results[-1]
            
            print(f"  Entropy: {initial.entropy_before:.3f} -> {final.entropy_after:.3f}")
            print(f"  Precision: {initial.metrics_before['precision']:.3f} -> "
                  f"{final.metrics_after['precision']:.3f}")
            print(f"  Expected FP: {initial.expected_fp_before:.3f} -> "
                  f"{final.expected_fp_after:.3f}")
    
    return results
