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
from typing import Dict, List, Tuple, Optional
from enum import Enum
from dataclasses import dataclass


class DifficultyLevel(Enum):
    """Difficulty levels based on candidate complexity."""
    EASY = "easy"      # 2-3 candidates per peak (75-85% baseline)
    MEDIUM = "medium"  # 5-8 candidates per peak (60-70% baseline)
    HARD = "hard"      # 10-15 candidates per peak (45-55% baseline)


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
    
    def __init__(self, difficulty: DifficultyLevel = DifficultyLevel.MEDIUM, seed: int = 42):
        """Initialize generator with difficulty-based parameters."""
        self.difficulty = difficulty
        self.rng = np.random.RandomState(seed)
        self.params = self._get_difficulty_params()
    
    def _get_difficulty_params(self) -> Dict:
        """Get parameters based on difficulty level."""
        
        if self.difficulty == DifficultyLevel.EASY:
            return {
                'isomer_fraction': 0.2,      # 20% are isomers
                'isobar_fraction': 0.3,      # 30% are isobars
                'adduct_fraction': 0.1,      # 10% have adducts
                'singleton_fraction': 0.4,   # 40% are isolated
                'mass_tolerance': 0.02,      # 20 mDa tolerance
                'rt_uncertainty_min': 0.5,   # RT std dev range
                'rt_uncertainty_max': 1.5,
                'coelution_rate': 0.1,       # 10% co-elution
                'noise_fraction': 0.3,       # 30% noise peaks
                'candidates_target': 3       # Target candidates per peak
            }
        
        elif self.difficulty == DifficultyLevel.MEDIUM:
            return {
                'isomer_fraction': 0.35,     # INCREASED: More isomers
                'isobar_fraction': 0.55,     # INCREASED: Much more isobars
                'adduct_fraction': 0.10,     # Reduced to make room for isobars
                'singleton_fraction': 0.0,   # NO singletons - all overlapping!
                'mass_tolerance': 0.15,      # INCREASED: 150 mDa tolerance
                'rt_uncertainty_min': 3.0,   # INCREASED: Very high uncertainty
                'rt_uncertainty_max': 5.0,   # INCREASED: Extremely uncertain
                'coelution_rate': 0.3,       # 30% co-elution
                'noise_fraction': 0.4,       # 40% noise
                'candidates_target': 7       # Target 5-8 candidates
            }
        
        else:  # HARD
            return {
                'isomer_fraction': 0.35,     # Many isomers
                'isobar_fraction': 0.45,     # Many isobars
                'adduct_fraction': 0.15,     # Adduct complexity
                'singleton_fraction': 0.05,  # Very few isolated
                'mass_tolerance': 0.05,      # 50 mDa tolerance
                'rt_uncertainty_min': 1.5,   # Very uncertain RT
                'rt_uncertainty_max': 3.5,
                'coelution_rate': 0.5,       # 50% co-elution!
                'noise_fraction': 0.5,       # 50% noise
                'candidates_target': 12      # Target 10-15 candidates
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
        n_isomers = int(n_compounds * params['isomer_fraction'])
        n_isobars = int(n_compounds * params['isobar_fraction'])
        n_adducts = int(n_compounds * params['adduct_fraction'])
        n_singletons = n_compounds - n_isomers - n_isobars - n_adducts
        
        # 1. Generate isomer groups (same mass, different RT)
        n_isomer_groups = max(1, n_isomers // 3)  # Average 3 isomers per group
        
        for group_idx in range(n_isomer_groups):
            base_mass = self.rng.uniform(150, 600)
            base_rt = self.rng.uniform(2, 12)
            n_in_group = min(4, n_isomers - group_idx * 3)
            
            for iso_idx in range(n_in_group):
                compounds.append({
                    'compound_id': compound_id,
                    'theoretical_mass': base_mass,  # EXACT same mass
                    'predicted_rt': base_rt + iso_idx * self.rng.uniform(0.5, 2.0),
                    'rt_prediction_std': self.rng.uniform(
                        params['rt_uncertainty_min'], 
                        params['rt_uncertainty_max']
                    ),
                    'compound_type': 'isomer',
                    'group_id': f'iso_{group_idx}',
                    'formula': f'C{10+group_idx}H{20-iso_idx}O{5}'  # Different structure
                })
                compound_id += 1
        
        # 2. Generate isobar groups (similar mass, may have different RT)
        n_isobar_groups = max(1, n_isobars // 8)  # INCREASED: Average 8 isobars per group for more overlap
        
        for group_idx in range(n_isobar_groups):
            base_mass = self.rng.uniform(200, 700)
            base_rt = self.rng.uniform(2, 12)
            n_in_group = min(12, n_isobars - group_idx * 8)  # Allow up to 12 isobars per group
            
            for iso_idx in range(n_in_group):
                # Mass varies within mass tolerance window for maximum overlap
                max_mass_diff = self.params['mass_tolerance'] * 0.8  # Stay within tolerance
                mass_offset = self.rng.uniform(-max_mass_diff/2, max_mass_diff/2)
                rt_offset = self.rng.normal(0, 1.5)  # RTs can overlap or separate
                
                compounds.append({
                    'compound_id': compound_id,
                    'theoretical_mass': base_mass + mass_offset,
                    'predicted_rt': base_rt + rt_offset,
                    'rt_prediction_std': self.rng.uniform(
                        params['rt_uncertainty_min'],
                        params['rt_uncertainty_max']
                    ),
                    'compound_type': 'isobar',
                    'group_id': f'isobar_{group_idx}',
                    'formula': f'C{12+iso_idx}H{24-iso_idx*2}N{iso_idx}O{4}'
                })
                compound_id += 1
        
        # 3. Generate compounds with adduct patterns
        n_adduct_bases = max(1, n_adducts // 3)  # Each base has ~3 adduct forms
        
        for base_idx in range(n_adduct_bases):
            base_mass = self.rng.uniform(150, 500)
            base_rt = self.rng.uniform(3, 11)
            
            # Base compound [M+H]+
            compounds.append({
                'compound_id': compound_id,
                'theoretical_mass': base_mass + 1.007,  # Proton adduct
                'predicted_rt': base_rt,
                'rt_prediction_std': self.rng.uniform(
                    params['rt_uncertainty_min'],
                    params['rt_uncertainty_max']
                ),
                'compound_type': 'adduct_base',
                'group_id': f'adduct_{base_idx}',
                'adduct_type': '[M+H]+',
                'formula': f'C{15+base_idx}H{30}O{5}'
            })
            compound_id += 1
            
            # Sodium adduct [M+Na]+
            if compound_id < n_compounds:
                compounds.append({
                    'compound_id': compound_id,
                    'theoretical_mass': base_mass + 22.989,  # Sodium adduct
                    'predicted_rt': base_rt + self.rng.normal(0, 0.05),  # Very similar RT
                    'rt_prediction_std': self.rng.uniform(
                        params['rt_uncertainty_min'],
                        params['rt_uncertainty_max']
                    ),
                    'compound_type': 'adduct_na',
                    'group_id': f'adduct_{base_idx}',
                    'adduct_type': '[M+Na]+',
                    'formula': f'C{15+base_idx}H{30}O{5}'
                })
                compound_id += 1
            
            # Ammonium adduct [M+NH4]+
            if compound_id < n_compounds:
                compounds.append({
                    'compound_id': compound_id,
                    'theoretical_mass': base_mass + 18.034,  # Ammonium adduct
                    'predicted_rt': base_rt + self.rng.normal(0, 0.05),
                    'rt_prediction_std': self.rng.uniform(
                        params['rt_uncertainty_min'],
                        params['rt_uncertainty_max']
                    ),
                    'compound_type': 'adduct_nh4',
                    'group_id': f'adduct_{base_idx}',
                    'adduct_type': '[M+NH4]+',
                    'formula': f'C{15+base_idx}H{30}O{5}'
                })
                compound_id += 1
        
        # 4. Generate singleton compounds (isolated, easier)
        for _ in range(min(n_singletons, n_compounds - compound_id)):
            # These are spread out in mass/RT space
            compounds.append({
                'compound_id': compound_id,
                'theoretical_mass': self.rng.uniform(100, 800),
                'predicted_rt': self.rng.uniform(1, 15),
                'rt_prediction_std': self.rng.uniform(
                    params['rt_uncertainty_min'] * 0.7,  # Slightly better predictions
                    params['rt_uncertainty_max'] * 0.7
                ),
                'compound_type': 'singleton',
                'group_id': f'single_{compound_id}',
                'formula': f'C{self.rng.randint(5,30)}H{self.rng.randint(10,50)}O{self.rng.randint(1,10)}'
            })
            compound_id += 1
        
        compounds_df = pd.DataFrame(compounds[:n_compounds])
        
        # Print library statistics
        print(f"\n📚 Compound Library Statistics:")
        print(f"  Total compounds: {len(compounds_df)}")
        print(f"  Isomers: {len(compounds_df[compounds_df['compound_type'] == 'isomer'])}")
        print(f"  Isobars: {len(compounds_df[compounds_df['compound_type'] == 'isobar'])}")
        print(f"  Adducts: {len(compounds_df[compounds_df['compound_type'].str.contains('adduct')])}")
        print(f"  Singletons: {len(compounds_df[compounds_df['compound_type'] == 'singleton'])}")
        
        # Analyze mass clustering
        mass_sorted = compounds_df.sort_values('theoretical_mass')
        mass_diffs = np.diff(mass_sorted['theoretical_mass'].values)
        close_pairs = np.sum(mass_diffs < params['mass_tolerance'])
        print(f"\n  Mass clustering:")
        print(f"    Compounds within {params['mass_tolerance']*1000:.0f} mDa: {close_pairs}")
        print(f"    Average mass spacing: {np.mean(mass_diffs):.3f} Da")
        
        return compounds_df
    
    def generate_peaks(self,
                      compound_library: pd.DataFrame,
                      n_species: int = 3,
                      n_peaks_per_species: int = 150) -> pd.DataFrame:
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
            n_real = int(n_peaks_per_species * (1 - params['noise_fraction']))
            
            for _ in range(n_real):
                # Select a compound (may select same compound multiple times - isotopes, etc.)
                compound_idx = self.rng.choice(n_compounds)
                compound = compound_library.iloc[compound_idx]
                
                # Generate peak with measurement error
                mass_error = self.rng.normal(0, 0.005)  # 5 mDa measurement error
                rt_error = self.rng.normal(0, 0.1)  # 0.1 min measurement error
                
                # Intensity based on compound type
                if compound['compound_type'] == 'singleton':
                    log_intensity = self.rng.normal(12, 1.5)
                elif 'adduct' in compound['compound_type']:
                    if compound['adduct_type'] == '[M+H]+':
                        log_intensity = self.rng.normal(12, 1.5)
                    else:  # Other adducts typically weaker
                        log_intensity = self.rng.normal(11, 1.5)
                else:  # Isomers and isobars
                    log_intensity = self.rng.normal(11.5, 1.8)
                
                # Peak shape characteristics
                peak_width = self.rng.gamma(2, 0.05) + 0.05  # 0.1-0.2 min typically
                peak_asymmetry = self.rng.gamma(5, 0.1) + 0.8  # Slight tailing
                
                peaks.append({
                    'peak_id': peak_id,
                    'species': species,
                    'mass': compound['theoretical_mass'] + mass_error,
                    'rt': compound['predicted_rt'] + rt_error,
                    'intensity': np.exp(log_intensity),
                    'peak_width_rt': peak_width,
                    'peak_asymmetry': peak_asymmetry,
                    'true_compound': compound['compound_id'],
                    'compound_type': compound['compound_type']  # For analysis
                })
                peak_id += 1
                
                # Add co-eluting compounds
                if self.rng.random() < params['coelution_rate']:
                    # Find a nearby compound
                    rt_window = 0.3  # Co-elute within 0.3 minutes
                    nearby = compound_library[
                        np.abs(compound_library['predicted_rt'] - compound['predicted_rt']) < rt_window
                    ]
                    if len(nearby) > 1:
                        coelute_compound = nearby.sample(1, random_state=self.rng).iloc[0]
                        if coelute_compound['compound_id'] != compound['compound_id']:
                            peaks.append({
                                'peak_id': peak_id,
                                'species': species,
                                'mass': coelute_compound['theoretical_mass'] + self.rng.normal(0, 0.005),
                                'rt': compound['predicted_rt'] + rt_error + self.rng.normal(0, 0.02),
                                'intensity': np.exp(self.rng.normal(11, 1.5)),
                                'peak_width_rt': peak_width * 1.2,  # Broader due to overlap
                                'peak_asymmetry': peak_asymmetry * 1.3,
                                'true_compound': coelute_compound['compound_id'],
                                'compound_type': 'coeluted'
                            })
                            peak_id += 1
            
            # Generate noise peaks
            n_noise = int(n_peaks_per_species * params['noise_fraction'])
            
            for _ in range(n_noise):
                # Noise can appear anywhere, but often near real compounds
                if self.rng.random() < 0.4:  # 40% chance to be near a compound
                    mimic_compound = compound_library.sample(1, random_state=self.rng).iloc[0]
                    noise_mass = mimic_compound['theoretical_mass'] + self.rng.uniform(-0.1, 0.1)
                    noise_rt = mimic_compound['predicted_rt'] + self.rng.normal(0, 1.0)
                else:
                    noise_mass = self.rng.uniform(100, 800)
                    noise_rt = self.rng.uniform(1, 15)
                
                # Noise characteristics
                noise_intensity = np.exp(self.rng.uniform(9, 11.5))
                noise_width = self.rng.uniform(0.2, 0.5)
                noise_asymmetry = self.rng.uniform(1.5, 3.0)
                
                peaks.append({
                    'peak_id': peak_id,
                    'species': species,
                    'mass': noise_mass,
                    'rt': noise_rt,
                    'intensity': noise_intensity,
                    'peak_width_rt': noise_width,
                    'peak_asymmetry': noise_asymmetry,
                    'true_compound': None,  # Noise has no true compound
                    'compound_type': 'noise'
                })
                peak_id += 1
        
        peaks_df = pd.DataFrame(peaks)
        peaks_df = peaks_df.sample(frac=1, random_state=self.rng).reset_index(drop=True)
        
        # Remove analysis-only column before returning
        peaks_df = peaks_df.drop(columns=['compound_type'])
        
        return peaks_df
    
    def analyze_candidate_complexity(self, 
                                    peaks_df: pd.DataFrame, 
                                    compound_library: pd.DataFrame) -> Dict:
        """
        Analyze how many candidates each peak will have.
        
        This is critical: we need multiple candidates per peak for genuine
        assignment challenges.
        """
        params = self.params
        mass_tolerance = params['mass_tolerance']
        
        candidates_per_peak = []
        
        for _, peak in peaks_df.iterrows():
            # Count how many compounds are within mass tolerance
            mass_matches = np.abs(compound_library['theoretical_mass'] - peak['mass']) < mass_tolerance
            
            # For those within mass tolerance, check RT compatibility
            # Using 2-sigma RT window
            rt_compatible = []
            for comp_idx in np.where(mass_matches)[0]:
                compound = compound_library.iloc[comp_idx]
                rt_pred = compound['predicted_rt']
                rt_std = compound['rt_prediction_std']
                rt_diff = abs(peak['rt'] - rt_pred)
                if rt_diff < 2 * rt_std:  # 2-sigma window
                    rt_compatible.append(comp_idx)
            
            candidates_per_peak.append(len(rt_compatible))
        
        stats = {
            'mean_candidates': np.mean(candidates_per_peak),
            'median_candidates': np.median(candidates_per_peak),
            'min_candidates': np.min(candidates_per_peak),
            'max_candidates': np.max(candidates_per_peak),
            'peaks_with_0_candidates': np.sum(np.array(candidates_per_peak) == 0),
            'peaks_with_1_candidate': np.sum(np.array(candidates_per_peak) == 1),
            'peaks_with_2plus_candidates': np.sum(np.array(candidates_per_peak) >= 2),
            'peaks_with_5plus_candidates': np.sum(np.array(candidates_per_peak) >= 5),
            'peaks_with_10plus_candidates': np.sum(np.array(candidates_per_peak) >= 10)
        }
        
        return stats, candidates_per_peak
    
    def generate_dataset(self,
                        n_compounds: int = 60,
                        n_species: int = 3,
                        n_peaks_per_species: int = 150) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
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
        
        print(f"\n🎯 Candidate Complexity Analysis:")
        print(f"  Mean candidates per peak: {stats['mean_candidates']:.1f}")
        print(f"  Median candidates: {stats['median_candidates']:.0f}")
        print(f"  Range: {stats['min_candidates']:.0f} - {stats['max_candidates']:.0f}")
        print(f"\n  Distribution:")
        print(f"    0 candidates: {stats['peaks_with_0_candidates']} peaks")
        print(f"    1 candidate: {stats['peaks_with_1_candidate']} peaks")
        print(f"    2+ candidates: {stats['peaks_with_2plus_candidates']} peaks")
        print(f"    5+ candidates: {stats['peaks_with_5plus_candidates']} peaks")
        print(f"    10+ candidates: {stats['peaks_with_10plus_candidates']} peaks")
        
        target = self.params['candidates_target']
        if stats['mean_candidates'] < target * 0.7:
            print(f"\n  ⚠️ WARNING: Mean candidates ({stats['mean_candidates']:.1f}) "
                  f"below target ({target})")
            print(f"     Consider increasing mass tolerance or RT uncertainty")
        elif stats['mean_candidates'] > target * 1.3:
            print(f"\n  ⚠️ WARNING: Mean candidates ({stats['mean_candidates']:.1f}) "
                  f"above target ({target})")
            print(f"     Consider decreasing mass tolerance or RT uncertainty")
        else:
            print(f"\n  ✅ Good: Mean candidates near target ({target})")
        
        return peaks_df, compound_library, stats


if __name__ == "__main__":
    # Test the multi-candidate generator
    print("=" * 80)
    print("TESTING MULTI-CANDIDATE GENERATOR")
    print("=" * 80)
    
    for difficulty in [DifficultyLevel.EASY, DifficultyLevel.MEDIUM, DifficultyLevel.HARD]:
        print(f"\n{'='*60}")
        print(f"Testing {difficulty.value.upper()} difficulty")
        print("=" * 60)
        
        generator = MultiCandidateGenerator(difficulty=difficulty)
        peaks, compounds, stats = generator.generate_dataset(
            n_compounds=40,
            n_species=2,
            n_peaks_per_species=50
        )
        
        print(f"\nDataset summary:")
        print(f"  Total peaks: {len(peaks)}")
        print(f"  Real peaks: {peaks['true_compound'].notna().sum()}")
        print(f"  Noise peaks: {peaks['true_compound'].isna().sum()}")
        
        if difficulty == DifficultyLevel.MEDIUM:
            if 5 <= stats['mean_candidates'] <= 8:
                print("\n✅ SUCCESS: MEDIUM difficulty achieves 5-8 candidates per peak!")
            else:
                print(f"\n⚠️ NEEDS TUNING: {stats['mean_candidates']:.1f} candidates (target: 5-8)")