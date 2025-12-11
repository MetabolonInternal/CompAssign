"""
Chemical feature computation for distinguishing real metabolite signals from noise.

This module provides functions to compute features based on chemical relationships
between peaks, including isotope patterns, adduct relationships, and RT clustering.
These features help discriminate real metabolite signals from chemical noise.
"""

import numpy as np
import pandas as pd
from typing import Tuple

from .ion_transforms import CHEMICAL_RELATION_MASSES, ISOTOPE_SHIFT


# Constants for chemical relationship detection shared with data generation
ISOTOPE_MASS_DIFF = ISOTOPE_SHIFT  # 13C - 12C mass difference
ADDUCT_MASSES = CHEMICAL_RELATION_MASSES
MASS_TOLERANCE = 0.01  # 10 mDa for relationship detection
RT_TOLERANCE = 0.1  # 0.1 minutes for co-elution
SNR_PERCENTILE = 10  # Empirical baseline percentile for noise estimate


def compute_isotope_features(
    mz: float, rt: float, intensity: float, peak_df: pd.DataFrame, species: int
) -> Tuple[float, float]:
    """
    Detect isotope peaks and compute pattern quality.

    Real metabolites show characteristic isotope patterns from 13C incorporation.
    The M+1 peak intensity depends on the number of carbon atoms.

    Parameters
    ----------
    mz : float
        Mass-to-charge ratio of the peak
    rt : float
        Retention time of the peak
    intensity : float
        Intensity of the peak
    peak_df : pd.DataFrame
        DataFrame containing all peaks
    species : int
        Species/sample ID

    Returns
    -------
    has_isotope : float
        Binary flag (0 or 1) indicating if M+1 isotope peak exists
    isotope_score : float
        Quality score of isotope pattern (0-1), based on intensity ratio
    """
    # Look for M+1 isotope peak
    isotope_mz = mz + ISOTOPE_MASS_DIFF

    # Find peaks in same species within mass and RT tolerance
    species_peaks = peak_df[peak_df["species"] == species]

    isotope_candidates = species_peaks[
        (abs(species_peaks["mass"] - isotope_mz) < MASS_TOLERANCE)
        & (abs(species_peaks["rt"] - rt) < RT_TOLERANCE)
    ]

    has_isotope = float(len(isotope_candidates) > 0)

    # Calculate isotope pattern score based on intensity ratio
    if has_isotope and len(isotope_candidates) > 0:
        # Take the closest mass match if multiple candidates
        mass_errors = abs(isotope_candidates["mass"] - isotope_mz)
        closest_idx = mass_errors.idxmin()
        isotope_intensity = isotope_candidates.loc[closest_idx, "intensity"]

        # Expected intensity ratio: ~1.1% per carbon atom
        # For organic molecules with 5-50 carbons, expect 5.5% to 55% ratio
        intensity_ratio = isotope_intensity / intensity

        # Score based on whether ratio is in reasonable range
        if 0.05 <= intensity_ratio <= 0.55:
            # Good ratio - linearly score based on how typical it is
            # Most metabolites have 10-30 carbons (11-33% ratio)
            if 0.11 <= intensity_ratio <= 0.33:
                isotope_score = 1.0  # Perfect range
            else:
                # Still reasonable but less typical
                isotope_score = 0.7
        else:
            # Ratio outside expected range but isotope present
            isotope_score = 0.3
    else:
        isotope_score = 0.0

    return has_isotope, isotope_score


def compute_adduct_features(mz: float, rt: float, peak_df: pd.DataFrame, species: int) -> float:
    """
    Count related adduct peaks.

    Real metabolites often form multiple adduct ions (Na+, K+, NH4+, etc.)
    that appear at characteristic mass differences.

    Parameters
    ----------
    mz : float
        Mass-to-charge ratio of the peak (assumed to be [M+H]+)
    rt : float
        Retention time of the peak
    peak_df : pd.DataFrame
        DataFrame containing all peaks
    species : int
        Species/sample ID

    Returns
    -------
    n_adducts : float
        Count of related adduct peaks found
    """
    species_peaks = peak_df[peak_df["species"] == species]
    if species_peaks.empty:
        return 0.0

    masses = species_peaks["mass"].to_numpy()
    rts = species_peaks["rt"].to_numpy()

    mass_diffs = masses - mz
    rt_diffs = np.abs(rts - rt)

    n_adducts = 0.0

    for adduct_name, mass_diff in ADDUCT_MASSES.items():
        target_diff = abs(mass_diff)

        # Look for any partner peak whose mass differs by ±target_diff within tolerance
        mass_match = np.abs(np.abs(mass_diffs) - target_diff) <= MASS_TOLERANCE
        rt_match = rt_diffs <= RT_TOLERANCE
        match_mask = mass_match & rt_match

        if np.any(match_mask):
            n_adducts += 1.0

    return n_adducts


def compute_rt_clustering(
    rt: float, intensity: float, peak_df: pd.DataFrame, species: int
) -> Tuple[float, float]:
    """
    Compute RT clustering features.

    Real metabolites often produce multiple ions that co-elute,
    while noise peaks tend to be isolated.

    Parameters
    ----------
    rt : float
        Retention time of the peak
    intensity : float
        Intensity of the peak
    peak_df : pd.DataFrame
        DataFrame containing all peaks
    species : int
        Species/sample ID

    Returns
    -------
    rt_cluster_size : float
        Number of peaks within RT tolerance (excluding self)
    n_correlated : float
        Number of peaks with correlated intensities
    """
    species_peaks = peak_df[peak_df["species"] == species]

    # Find peaks within RT window
    nearby_peaks = species_peaks[(abs(species_peaks["rt"] - rt) <= RT_TOLERANCE)]

    # Exclude the peak itself
    rt_cluster_size = float(len(nearby_peaks) - 1)

    # Check intensity correlations
    if rt_cluster_size > 0:
        # Peaks from same metabolite often have correlated intensities
        # Check if intensities are within reasonable ratio (0.1 to 10x)
        intensity_ratios = nearby_peaks["intensity"] / intensity

        # Count peaks with reasonable intensity ratios
        # Exclude ratio of 1.0 (the peak itself)
        n_correlated = float(
            sum((0.1 <= r <= 10) and abs(r - 1.0) > 0.01 for r in intensity_ratios)
        )
    else:
        n_correlated = 0.0

    return rt_cluster_size, n_correlated


def compute_all_chemical_features(
    mz: float, rt: float, intensity: float, peak_df: pd.DataFrame, species: int
) -> dict:
    """
    Compute all chemical features for a peak.

    Convenience function that computes all chemical relationship features
    and returns them as a dictionary.

    Parameters
    ----------
    mz : float
        Mass-to-charge ratio of the peak
    rt : float
        Retention time of the peak
    intensity : float
        Intensity of the peak
    peak_df : pd.DataFrame
        DataFrame containing all peaks
    species : int
        Species/sample ID

    Returns
    -------
    features : dict
        Dictionary containing all computed features
    """
    has_isotope, isotope_score = compute_isotope_features(mz, rt, intensity, peak_df, species)

    n_adducts = compute_adduct_features(mz, rt, peak_df, species)

    rt_cluster_size, n_correlated = compute_rt_clustering(rt, intensity, peak_df, species)

    return {
        "has_isotope": has_isotope,
        "isotope_score": isotope_score,
        "n_adducts": n_adducts,
        "rt_cluster_size": rt_cluster_size,
        "n_correlated": n_correlated,
    }
