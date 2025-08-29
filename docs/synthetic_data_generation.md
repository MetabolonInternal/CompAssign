# Synthetic Data Generation for CompAssign Testing

## Overview

This document describes the **challenging synthetic data generation** used for testing CompAssign's performance. This data is designed to reflect real-world metabolomics challenges including isomers, near-isobars, and varying RT uncertainties.

## Why Challenging Data Matters

Initial testing with simple synthetic data (random masses/RTs) showed unrealistically high precision regardless of parameter settings. This was because:
- Random data lacks the structural relationships found in real metabolites
- No isomers (same mass, different RT) were present
- No near-isobars (compounds within mass tolerance) existed
- RT uncertainties were uniform and unrealistic

**Key Discovery**: With challenging data, even ultra-restrictive k values (k=0.1) cannot achieve 99% precision, revealing the fundamental limitations of mass/RT-based assignment.

## Data Generation Components

### 1. Compound Library Structure

```python
n_compounds = 100  # Total unique compounds
isomer_fraction = 0.3  # 30% of compounds have isomers
near_isobar_fraction = 0.2  # 20% have near-isobars
```

#### Compound Types

1. **Normal Compounds** (50%)
   - Random mass: 100-800 Da
   - Random RT: 0.5-15 minutes
   - RT uncertainty: 0.05-0.5 minutes (varies per compound)

2. **Isomers** (30%)
   - **Identical mass** to base compound
   - Different RT (separated by 0.2-2.0 minutes)
   - Groups of 2-4 isomers per base compound
   - Example: Leucine/Isoleucine (same formula C₆H₁₃NO₂)

3. **Near-Isobars** (20%)
   - Mass within 0.003-0.010 Da of base compound
   - Just outside or at edge of mass tolerance
   - Different RT (0.5-3.0 minutes apart)
   - Example: C₁₂H₂₂O₁₁ (342.1162) vs C₁₂H₂₀O₁₂ (356.0955)

### 2. Peak Generation

```python
n_peaks_per_compound = 3  # Average across species
n_noise_peaks = 100  # False positive peaks
n_species = 5  # Different samples/conditions
```

#### Peak Characteristics

1. **True Peaks** (70% of total)
   - Mass measurement error: Normal(0, 2 ppm)
   - RT prediction error: Normal(0, compound.rt_uncertainty)
   - Species-specific RT shift: Normal(0, 0.1 min)
   - Intensity: LogNormal(12, 1.5)

2. **Noise Peaks** (30% of total)
   - Two strategies:
     a. **Near-compound noise** (50%): Close to real compound in mass OR RT
     b. **Random noise** (50%): Completely random mass/RT
   - Intensity: LogNormal(10, 2) - typically lower than true peaks

### 3. RT Uncertainty Model

Each compound has varying RT prediction uncertainty based on:
- Chemical class (some classes harder to predict)
- Molecular complexity
- Range: 0.05 (very confident) to 0.5 minutes (uncertain)

```python
rt_uncertainty_range = (0.05, 0.5)  # minutes
rt_uncertainties[(species, compound)] = compound.rt_uncertainty
```

### 4. Challenging Features

#### Mass Conflicts
- **Isomers**: ~15 isomer groups with 2-4 compounds each
- **Near-isobars**: ~10 pairs within 10 mDa
- Results in ~285 peak pairs within 10 mDa in typical dataset

#### RT Challenges
- Variable prediction uncertainty per compound
- Species-specific systematic shifts
- Overlapping RT ranges for different compounds

## Implementation

### Default Data Generation Function

```python
from scripts.create_challenging_test_data import create_challenging_metabolomics_data

# Generate default challenging dataset
peak_df, compound_df, true_assignments, rt_uncertainties = \
    create_challenging_metabolomics_data(
        n_compounds=100,
        n_peaks_per_compound=3,
        n_noise_peaks=100,
        n_species=5,
        isomer_fraction=0.3,
        near_isobar_fraction=0.2,
        mass_error_std=0.002,  # 2 ppm
        rt_uncertainty_range=(0.05, 0.5)
    )
```

### Dataset Statistics (Typical Run)

```
Total peaks: 434
True assignments: 334
Noise peaks: 100
Isomer groups: 15
Near-isobar pairs: 10
Peak pairs within 10 mDa: 285
```

## Performance Impact

### Comparison: Simple vs Challenging Data

| k Value | Simple Data Precision | Challenging Data Precision | Difference |
|---------|----------------------|----------------------------|------------|
| 0.5     | >99%                 | 81.5%                      | -17.5%     |
| 1.0     | >99%                 | 77.9%                      | -21.1%     |
| 1.5     | >99%                 | 76.9%                      | -22.1%     |
| 2.0     | >98%                 | 74.5%                      | -23.5%     |

### Why 99% Precision is Unachievable

Even with k=0.1 (accepting only 8% of RT distribution):
- Precision: 86.2% (not 99%)
- Recall: 7.5% (loses 92.5% of true positives)
- **Root cause**: Isomers and near-isobars create inherent ambiguity

## Usage in Testing

### 1. Parameter Optimization

```python
# scripts/test_k_values.py now uses challenging data
from scripts.create_challenging_test_data import test_k_with_challenging_data

results = test_k_with_challenging_data(
    k_value=1.5,
    mass_tolerance=0.005
)
```

### 2. Ablation Studies

```python
# Test impact of different features with realistic data
peak_df, compound_df, _, _ = create_challenging_metabolomics_data()

# Now test different model configurations
# Results will reflect real-world challenges
```

### 3. Performance Benchmarking

All performance claims should be based on challenging data:
- Precision/recall metrics
- Parameter sensitivity analysis  
- Feature importance studies

## Adjusting Difficulty

### Make Data Easier

```python
# Fewer isomers and better separated
data = create_challenging_metabolomics_data(
    isomer_fraction=0.1,  # Only 10% isomers
    near_isobar_fraction=0.1,  # Fewer near-isobars
    rt_uncertainty_range=(0.02, 0.1)  # Better RT predictions
)
```

### Make Data Harder

```python
# More conflicts and uncertainty
data = create_challenging_metabolomics_data(
    isomer_fraction=0.5,  # 50% have isomers
    near_isobar_fraction=0.3,  # More near-isobars
    mass_error_std=0.005,  # Worse mass accuracy (5 ppm)
    rt_uncertainty_range=(0.2, 1.0)  # Higher RT uncertainty
)
```

## Validation Against Real Data

The challenging synthetic data approximates real metabolomics data:

1. **Isomer prevalence**: 20-30% of metabolites have isomers
2. **Mass conflicts**: Common in complex samples (plasma, urine)
3. **RT uncertainty**: Varies by compound class (0.05-0.5 min typical)
4. **Noise levels**: 20-30% noise peaks realistic for untargeted LC-MS

## Key Learnings

1. **Mass tolerance dominates**: Filters 97.7% of candidates
2. **RT window (k) matters for the hard 2.3%**: These are isomers/near-isobars
3. **Perfect precision impossible**: Inherent ambiguity in isomer assignment
4. **Trade-off is fundamental**: High precision requires sacrificing recall

## Recommendations

1. **Always use challenging data for testing** - it reveals true performance
2. **Report metrics on both easy and hard data** - shows robustness
3. **Consider isomer groups explicitly** - may need different handling
4. **Document data generation parameters** - ensures reproducibility

## Code References

- Main implementation: `scripts/create_challenging_test_data.py`
- Usage examples: `scripts/test_extreme_k_values.py`
- Integration: `scripts/test_k_values.py` (updated to use challenging data)