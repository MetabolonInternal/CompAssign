#!/usr/bin/env python3
"""
Multi-Candidate Synthetic Data Generator for Active Learning.

This module creates synthetic metabolomics data with GENUINE assignment challenges:
- Multiple competing candidates per peak (5-20)
- Isomers (same mass, different RT)
- Isobars (similar mass within instrument error)
- Adduct patterns
- Co-eluting compounds

The goal is to transform the problem from binary classification (real vs noise)
to multi-class assignment (choosing among many plausible candidates).
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass

from ..utils import (
    ADDUCT_DEFS,
    FRAGMENT_DEFS,
    ISOTOPE_SHIFT,
    ISOTOPE_INTENSITY_FACTOR,
)


@dataclass
class CompoundGroup:
    """Group of related compounds (isomers, isobars, etc.)."""

    group_type: str  # 'isomer', 'isobar', 'adduct', 'singleton'
    compounds: List[Dict]  # List of compound properties
    base_mass: float
    mass_variance: float
    rt_spread: float


class MultiCandidateGenerator:
    """
    Generate synthetic data with multiple competing candidates per peak.

    Key innovation: Creates overlapping compound masses and RTs to ensure
    multiple candidates survive filtering, creating genuine assignment challenges.
    """

    def __init__(self, seed: int = 42):
        """Initialize generator with a realistic-hard configuration."""
        self.rng = np.random.RandomState(seed)
        self.params = self._get_generator_params()

    def _get_generator_params(self) -> Dict:
        """Return fixed parameters for a challenging, realistic scenario."""
        return {
            "isomer_fraction": 0.35,
            "isobar_fraction": 0.45,
            "adduct_fraction": 0.15,
            "singleton_fraction": 0.05,
            "mass_tolerance": 0.05,
            "rt_uncertainty_min": 1.5,
            "rt_uncertainty_max": 3.5,
            "coelution_rate": 0.5,
            "noise_fraction": 0.5,
            "candidates_target": 12,
        }

    def generate_compound_library(self, n_compounds: int = 60) -> pd.DataFrame:
        """
        Generate a library of compounds with overlapping masses.

        The key innovation: compounds are clustered in mass/RT space to ensure
        multiple candidates compete for each peak assignment.
        """
        compounds = []
        compound_id = 0
        params = self.params

        # Calculate number of compounds of each type
        n_isomers = int(n_compounds * params["isomer_fraction"])
        n_isobars = int(n_compounds * params["isobar_fraction"])
        n_adducts = int(n_compounds * params["adduct_fraction"])
        n_singletons = n_compounds - n_isomers - n_isobars - n_adducts

        # 1. Generate isomer groups (same mass, different RT)
        n_isomer_groups = max(1, n_isomers // 3)  # Average 3 isomers per group

        for group_idx in range(n_isomer_groups):
            base_mass = self.rng.uniform(150, 600)
            base_rt = self.rng.uniform(2, 12)
            n_in_group = min(4, n_isomers - group_idx * 3)

            for iso_idx in range(n_in_group):
                compounds.append(
                    {
                        "compound_id": compound_id,
                        "theoretical_mass": base_mass,  # EXACT same mass
                        "predicted_rt": base_rt + iso_idx * self.rng.uniform(0.5, 2.0),
                        "rt_prediction_std": self.rng.uniform(
                            params["rt_uncertainty_min"], params["rt_uncertainty_max"]
                        ),
                        "compound_type": "isomer",
                        "group_id": f"iso_{group_idx}",
                        "formula": f"C{10+group_idx}H{20-iso_idx}O{5}",  # Different structure
                    }
                )
                compound_id += 1

        # 2. Generate isobar groups (similar mass, may have different RT)
        n_isobar_groups = max(
            1, n_isobars // 8
        )  # INCREASED: Average 8 isobars per group for more overlap

        for group_idx in range(n_isobar_groups):
            base_mass = self.rng.uniform(200, 700)
            base_rt = self.rng.uniform(2, 12)
            n_in_group = min(12, n_isobars - group_idx * 8)  # Allow up to 12 isobars per group

            for iso_idx in range(n_in_group):
                # Mass varies within mass tolerance window for maximum overlap
                max_mass_diff = self.params["mass_tolerance"] * 0.8  # Stay within tolerance
                mass_offset = self.rng.uniform(-max_mass_diff / 2, max_mass_diff / 2)
                rt_offset = self.rng.normal(0, 1.5)  # RTs can overlap or separate

                compounds.append(
                    {
                        "compound_id": compound_id,
                        "theoretical_mass": base_mass + mass_offset,
                        "predicted_rt": base_rt + rt_offset,
                        "rt_prediction_std": self.rng.uniform(
                            params["rt_uncertainty_min"], params["rt_uncertainty_max"]
                        ),
                        "compound_type": "isobar",
                        "group_id": f"isobar_{group_idx}",
                        "formula": f"C{12+iso_idx}H{24-iso_idx*2}N{iso_idx}O{4}",
                    }
                )
                compound_id += 1

        # 3. Generate compounds with adduct patterns
        n_adduct_bases = max(1, n_adducts // 3)  # Each base has ~3 adduct forms

        for base_idx in range(n_adduct_bases):
            base_mass = self.rng.uniform(150, 500)
            base_rt = self.rng.uniform(3, 11)

            # Base compound [M+H]+
            compounds.append(
                {
                    "compound_id": compound_id,
                    "theoretical_mass": base_mass + 1.007,  # Proton adduct
                    "predicted_rt": base_rt,
                    "rt_prediction_std": self.rng.uniform(
                        params["rt_uncertainty_min"], params["rt_uncertainty_max"]
                    ),
                    "compound_type": "adduct_base",
                    "group_id": f"adduct_{base_idx}",
                    "adduct_type": "[M+H]+",
                    "formula": f"C{15+base_idx}H{30}O{5}",
                }
            )
            compound_id += 1

            # Sodium adduct [M+Na]+
            if compound_id < n_compounds:
                compounds.append(
                    {
                        "compound_id": compound_id,
                        "theoretical_mass": base_mass + 22.989,  # Sodium adduct
                        "predicted_rt": base_rt + self.rng.normal(0, 0.05),  # Very similar RT
                        "rt_prediction_std": self.rng.uniform(
                            params["rt_uncertainty_min"], params["rt_uncertainty_max"]
                        ),
                        "compound_type": "adduct_na",
                        "group_id": f"adduct_{base_idx}",
                        "adduct_type": "[M+Na]+",
                        "formula": f"C{15+base_idx}H{30}O{5}",
                    }
                )
                compound_id += 1

            # Ammonium adduct [M+NH4]+
            if compound_id < n_compounds:
                compounds.append(
                    {
                        "compound_id": compound_id,
                        "theoretical_mass": base_mass + 18.034,  # Ammonium adduct
                        "predicted_rt": base_rt + self.rng.normal(0, 0.05),
                        "rt_prediction_std": self.rng.uniform(
                            params["rt_uncertainty_min"], params["rt_uncertainty_max"]
                        ),
                        "compound_type": "adduct_nh4",
                        "group_id": f"adduct_{base_idx}",
                        "adduct_type": "[M+NH4]+",
                        "formula": f"C{15+base_idx}H{30}O{5}",
                    }
                )
                compound_id += 1

        # 4. Generate singleton compounds (isolated, easier)
        for _ in range(min(n_singletons, n_compounds - compound_id)):
            # These are spread out in mass/RT space
            compounds.append(
                {
                    "compound_id": compound_id,
                    "theoretical_mass": self.rng.uniform(100, 800),
                    "predicted_rt": self.rng.uniform(1, 15),
                    "rt_prediction_std": self.rng.uniform(
                        params["rt_uncertainty_min"] * 0.7,  # Slightly better predictions
                        params["rt_uncertainty_max"] * 0.7,
                    ),
                    "compound_type": "singleton",
                    "group_id": f"single_{compound_id}",
                    "formula": (
                        f"C{self.rng.randint(5, 30)}"
                        f"H{self.rng.randint(10, 50)}"
                        f"O{self.rng.randint(1, 10)}"
                    ),
                }
            )
            compound_id += 1

        compounds_df = pd.DataFrame(compounds[:n_compounds])

        # Print library statistics
        print("\n📚 Compound Library Statistics:")
        print(f"  Total compounds: {len(compounds_df)}")
        print(f"  Isomers: {len(compounds_df[compounds_df['compound_type'] == 'isomer'])}")
        print(f"  Isobars: {len(compounds_df[compounds_df['compound_type'] == 'isobar'])}")
        print(
            f"  Adducts: {len(compounds_df[compounds_df['compound_type'].str.contains('adduct')])}"
        )
        print(f"  Singletons: {len(compounds_df[compounds_df['compound_type'] == 'singleton'])}")

        # Analyze mass clustering
        mass_sorted = compounds_df.sort_values("theoretical_mass")
        mass_diffs = np.diff(mass_sorted["theoretical_mass"].values)
        close_pairs = np.sum(mass_diffs < params["mass_tolerance"])
        print("\n  Mass clustering:")
        print(f"    Compounds within {params['mass_tolerance']*1000:.0f} mDa: {close_pairs}")
        print(f"    Average mass spacing: {np.mean(mass_diffs):.3f} Da")

        return compounds_df

    def _generate_related_signals(
        self, peak_id: int, species: int, base_peak: Dict, coelution_rt_offset: float = 0.0
    ) -> Tuple[List[Dict], int]:
        """Emit isotope/adduct/fragment peaks that surround a real signal."""
        new_peaks: List[Dict] = []

        base_mass = base_peak["mass"]
        base_rt = base_peak["rt"] + coelution_rt_offset
        base_intensity = float(base_peak["intensity"])
        base_width = base_peak["peak_width_rt"]
        base_asymmetry = base_peak["peak_asymmetry"]

        # Helper for consistent jitter across synthetic companions
        def _rt_jitter(scale: float = 0.02) -> float:
            return self.rng.normal(0.0, scale)

        def _width_scale() -> float:
            return np.clip(self.rng.normal(1.0, 0.1), 0.7, 1.4)

        def _asym_scale() -> float:
            return np.clip(self.rng.normal(1.0, 0.15), 0.6, 1.8)

        # 1. Isotope (M+1) peak
        if self.rng.random() < 0.85:
            iso_factor = np.clip(self.rng.normal(ISOTOPE_INTENSITY_FACTOR, 0.05), 0.05, 0.7)
            iso_intensity = base_intensity * iso_factor
            new_peaks.append(
                {
                    "peak_id": peak_id,
                    "species": species,
                    "mass": base_mass + ISOTOPE_SHIFT + self.rng.normal(0.0, 0.001),
                    "rt": base_rt + _rt_jitter(base_width * 0.05),
                    "intensity": max(iso_intensity, base_intensity * 0.02),
                    "peak_width_rt": base_width * _width_scale(),
                    "peak_asymmetry": base_asymmetry * _asym_scale(),
                    "true_compound": None,
                }
            )
            peak_id += 1

        # 2. Adduct companions (limited count to keep dataset compact)
        if len(ADDUCT_DEFS) > 0:
            adduct_draws = self.rng.randint(0, 3)
            if adduct_draws > 0:
                adduct_indices = self.rng.choice(len(ADDUCT_DEFS), size=adduct_draws, replace=False)
                for idx in np.atleast_1d(adduct_indices):
                    adduct = ADDUCT_DEFS[int(idx)]
                    if self.rng.random() < 0.55:
                        factor = adduct.get("intensity_factor", 0.4)
                        factor = np.clip(self.rng.normal(factor, 0.1), 0.05, 0.9)
                        adduct_intensity = base_intensity * factor
                        new_peaks.append(
                            {
                                "peak_id": peak_id,
                                "species": species,
                                "mass": base_mass + adduct["delta"] + self.rng.normal(0.0, 0.0015),
                                "rt": base_rt + _rt_jitter(base_width * 0.1),
                                "intensity": max(adduct_intensity, base_intensity * 0.05),
                                "peak_width_rt": base_width * _width_scale(),
                                "peak_asymmetry": base_asymmetry * _asym_scale(),
                                "true_compound": None,
                            }
                        )
                        peak_id += 1

        # 3. Fragment / neutral-loss peaks (rarer)
        frag_draw = self.rng.random()
        if frag_draw < 0.4 and len(FRAGMENT_DEFS) > 0:
            fragment = self.rng.choice(FRAGMENT_DEFS)
            factor = fragment.get("intensity_factor", 0.25)
            factor = np.clip(self.rng.normal(factor, 0.08), 0.02, 0.6)
            new_peaks.append(
                {
                    "peak_id": peak_id,
                    "species": species,
                    "mass": base_mass + fragment["delta"] + self.rng.normal(0.0, 0.002),
                    "rt": base_rt + _rt_jitter(base_width * 0.12),
                    "intensity": max(base_intensity * factor, base_intensity * 0.03),
                    "peak_width_rt": base_width * _width_scale(),
                    "peak_asymmetry": base_asymmetry * _asym_scale(),
                    "true_compound": None,
                }
            )
            peak_id += 1

        return new_peaks, peak_id

    def generate_peaks(
        self, compound_library: pd.DataFrame, n_species: int = 3, n_peaks_per_species: int = 150
    ) -> pd.DataFrame:
        """
        Generate peaks that will have multiple candidate assignments.

        Key: With wider mass tolerance and uncertain RT, each peak will match
        multiple compounds, creating genuine assignment challenges.
        """
        peaks = []
        peak_id = 0
        params = self.params

        for species in range(n_species):
            n_compounds = len(compound_library)

            # Generate real peaks
            n_real = int(n_peaks_per_species * (1 - params["noise_fraction"]))

            for _ in range(n_real):
                # Select a compound (may select same compound multiple times - isotopes, etc.)
                compound_idx = self.rng.choice(n_compounds)
                compound = compound_library.iloc[compound_idx]

                # Generate peak with measurement error
                mass_error = self.rng.normal(0, 0.005)  # 5 mDa measurement error
                rt_error = self.rng.normal(0, 0.1)  # 0.1 min measurement error

                # Intensity based on compound type
                if compound["compound_type"] == "singleton":
                    log_intensity = self.rng.normal(12, 1.5)
                elif "adduct" in compound["compound_type"]:
                    if compound["adduct_type"] == "[M+H]+":
                        log_intensity = self.rng.normal(12, 1.5)
                    else:  # Other adducts typically weaker
                        log_intensity = self.rng.normal(11, 1.5)
                else:  # Isomers and isobars
                    log_intensity = self.rng.normal(11.5, 1.8)

                # Peak shape characteristics
                peak_width = self.rng.gamma(2, 0.05) + 0.05  # 0.1-0.2 min typically
                peak_asymmetry = self.rng.gamma(5, 0.1) + 0.8  # Slight tailing

                peaks.append(
                    {
                        "peak_id": peak_id,
                        "species": species,
                        "mass": compound["theoretical_mass"] + mass_error,
                        "rt": compound["predicted_rt"] + rt_error,
                        "intensity": np.exp(log_intensity),
                        "peak_width_rt": peak_width,
                        "peak_asymmetry": peak_asymmetry,
                        "true_compound": compound["compound_id"],
                        "compound_type": compound["compound_type"],  # For analysis
                    }
                )
                peak_id += 1

                # Inject related chemical signals so downstream features observe them
                related_peaks, peak_id = self._generate_related_signals(
                    peak_id=peak_id,
                    species=species,
                    base_peak=peaks[-1],
                )
                if related_peaks:
                    peaks.extend(related_peaks)

                # Add co-eluting compounds
                if self.rng.random() < params["coelution_rate"]:
                    # Find a nearby compound
                    rt_window = 0.3  # Co-elute within 0.3 minutes
                    nearby = compound_library[
                        np.abs(compound_library["predicted_rt"] - compound["predicted_rt"])
                        < rt_window
                    ]
                    if len(nearby) > 1:
                        coelute_compound = nearby.sample(1, random_state=self.rng).iloc[0]
                        if coelute_compound["compound_id"] != compound["compound_id"]:
                            peaks.append(
                                {
                                    "peak_id": peak_id,
                                    "species": species,
                                    "mass": coelute_compound["theoretical_mass"]
                                    + self.rng.normal(0, 0.005),
                                    "rt": compound["predicted_rt"]
                                    + rt_error
                                    + self.rng.normal(0, 0.02),
                                    "intensity": np.exp(self.rng.normal(11, 1.5)),
                                    "peak_width_rt": peak_width * 1.2,  # Broader due to overlap
                                    "peak_asymmetry": peak_asymmetry * 1.3,
                                    "true_compound": coelute_compound["compound_id"],
                                    "compound_type": "coeluted",
                                }
                            )
                            peak_id += 1

                            # Co-eluting partner also contributes adduct/isotope patterns
                            related_peaks, peak_id = self._generate_related_signals(
                                peak_id=peak_id,
                                species=species,
                                base_peak=peaks[-1],
                            )
                            if related_peaks:
                                peaks.extend(related_peaks)

            # Generate noise peaks
            n_noise = int(n_peaks_per_species * params["noise_fraction"])

            for _ in range(n_noise):
                # Noise can appear anywhere, but often near real compounds
                if self.rng.random() < 0.4:  # 40% chance to be near a compound
                    mimic_compound = compound_library.sample(1, random_state=self.rng).iloc[0]
                    noise_mass = mimic_compound["theoretical_mass"] + self.rng.uniform(-0.1, 0.1)
                    noise_rt = mimic_compound["predicted_rt"] + self.rng.normal(0, 1.0)
                else:
                    noise_mass = self.rng.uniform(100, 800)
                    noise_rt = self.rng.uniform(1, 15)

                # Noise characteristics
                noise_intensity = np.exp(self.rng.uniform(9, 11.5))
                noise_width = self.rng.uniform(0.2, 0.5)
                noise_asymmetry = self.rng.uniform(1.5, 3.0)

                peaks.append(
                    {
                        "peak_id": peak_id,
                        "species": species,
                        "mass": noise_mass,
                        "rt": noise_rt,
                        "intensity": noise_intensity,
                        "peak_width_rt": noise_width,
                        "peak_asymmetry": noise_asymmetry,
                        "true_compound": None,  # Noise has no true compound
                        "compound_type": "noise",
                    }
                )
                peak_id += 1

        peaks_df = pd.DataFrame(peaks)
        peaks_df = peaks_df.sample(frac=1, random_state=self.rng).reset_index(drop=True)

        # Remove analysis-only column before returning
        peaks_df = peaks_df.drop(columns=["compound_type"])

        return peaks_df

    def analyze_candidate_complexity(
        self, peaks_df: pd.DataFrame, compound_library: pd.DataFrame
    ) -> Dict:
        """
        Analyze how many candidates each peak will have.

        This is critical: we need multiple candidates per peak for genuine
        assignment challenges.
        """
        params = self.params
        mass_tolerance = params["mass_tolerance"]

        candidates_per_peak = []

        for _, peak in peaks_df.iterrows():
            # Count how many compounds are within mass tolerance
            mass_matches = (
                np.abs(compound_library["theoretical_mass"] - peak["mass"]) < mass_tolerance
            )

            # For mass-matched candidates, check RT window (2-sigma)
            rt_in_window = []
            for comp_idx in np.where(mass_matches)[0]:
                compound = compound_library.iloc[comp_idx]
                rt_pred = compound["predicted_rt"]
                rt_std = compound["rt_prediction_std"]
                rt_diff = abs(peak["rt"] - rt_pred)
                if rt_diff < 2 * rt_std:  # 2-sigma window
                    rt_in_window.append(comp_idx)

            candidates_per_peak.append(len(rt_in_window))

        stats = {
            "mean_candidates": np.mean(candidates_per_peak),
            "median_candidates": np.median(candidates_per_peak),
            "min_candidates": np.min(candidates_per_peak),
            "max_candidates": np.max(candidates_per_peak),
            "peaks_with_0_candidates": np.sum(np.array(candidates_per_peak) == 0),
            "peaks_with_1_candidate": np.sum(np.array(candidates_per_peak) == 1),
            "peaks_with_2plus_candidates": np.sum(np.array(candidates_per_peak) >= 2),
            "peaks_with_5plus_candidates": np.sum(np.array(candidates_per_peak) >= 5),
            "peaks_with_10plus_candidates": np.sum(np.array(candidates_per_peak) >= 10),
        }

        return stats, candidates_per_peak

    def generate_dataset(
        self, n_compounds: int = 60, n_species: int = 3, n_peaks_per_species: int = 150
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Generate complete dataset with multi-candidate complexity.

        Returns
        -------
        peaks_df : pd.DataFrame
            Peak data with realistic features
        compound_library : pd.DataFrame
            Compound library with overlapping masses
        complexity_stats : Dict
            Statistics on candidate complexity
        """
        # Generate compound library with overlapping masses
        compound_library = self.generate_compound_library(n_compounds)

        # Generate peaks
        peaks_df = self.generate_peaks(compound_library, n_species, n_peaks_per_species)

        # Analyze complexity
        stats, candidates_list = self.analyze_candidate_complexity(peaks_df, compound_library)

        print("\n🎯 Candidate Complexity Analysis:")
        print(f"  Mean candidates per peak: {stats['mean_candidates']:.1f}")
        print(f"  Median candidates: {stats['median_candidates']:.0f}")
        print(f"  Range: {stats['min_candidates']:.0f} - {stats['max_candidates']:.0f}")
        print("\n  Distribution:")
        print(f"    0 candidates: {stats['peaks_with_0_candidates']} peaks")
        print(f"    1 candidate: {stats['peaks_with_1_candidate']} peaks")
        print(f"    2+ candidates: {stats['peaks_with_2plus_candidates']} peaks")
        print(f"    5+ candidates: {stats['peaks_with_5plus_candidates']} peaks")
        print(f"    10+ candidates: {stats['peaks_with_10plus_candidates']} peaks")

        target = self.params["candidates_target"]
        if stats["mean_candidates"] < target * 0.7:
            print(
                f"\n  ⚠️ WARNING: Mean candidates ({stats['mean_candidates']:.1f}) "
                f"below target ({target})"
            )
            print("     Consider increasing mass tolerance or RT uncertainty")
        elif stats["mean_candidates"] > target * 1.3:
            print(
                f"\n  ⚠️ WARNING: Mean candidates ({stats['mean_candidates']:.1f}) "
                f"above target ({target})"
            )
            print("     Consider decreasing mass tolerance or RT uncertainty")
        else:
            print(f"\n  ✅ Good: Mean candidates near target ({target})")

        return peaks_df, compound_library, stats


if __name__ == "__main__":
    # Test the multi-candidate generator
    print("=" * 80)
    print("TESTING MULTI-CANDIDATE GENERATOR")
    print("=" * 80)

    generator = MultiCandidateGenerator()
    peaks, compounds, stats = generator.generate_dataset(
        n_compounds=40, n_species=2, n_peaks_per_species=50
    )

    print("\nDataset summary:")
    print(f"  Total peaks: {len(peaks)}")
    print(f"  Real peaks: {peaks['true_compound'].notna().sum()}")
    print(f"  Noise peaks: {peaks['true_compound'].isna().sum()}")
