# NEXT SESSION HANDOVER — Many-to-One Assignment with Decoys

## Status Update (Latest) 
✅ **COMPLETED**: Removed artificial one-to-one matching constraint
✅ **COMPLETED**: Implemented many-to-one evaluation (compound + peak metrics)
✅ **COMPLETED**: Made data genuinely challenging (isomers, near-isobars, RT errors)
✅ **COMPLETED**: Added decoy compounds (50% of library never appears in samples)
⚠️ **BUG FOUND**: Evaluation metrics don't count decoy assignments as false positives!

## Critical Bug to Fix
```python
# Current evaluation only checks "real" compounds:
compounds_total: 15  # Only counting non-decoys!
compound_precision: 1.0  # WRONG - ignores decoy assignments

# Manual check reveals:
Real compounds identified: 9/15
Decoy compounds falsely identified: 7/15  # FALSE POSITIVES!
True compound precision: 9/(9+7) = 56%  # Not 100%!
```

## What Needs to Be Done Next

### 1. Fix Compound-Level Evaluation (URGENT)
The `assign()` method in `peak_assignment.py` needs to:
- Track ALL compounds in library (including decoys)
- Count assignments to decoy compounds as false positives
- Update compound_precision calculation:
  ```python
  # Current (WRONG):
  compound_precision = correct_compounds / pred_compounds  # Only non-decoys
  
  # Should be:
  true_positives = real_compounds_correctly_identified
  false_positives = decoy_compounds_incorrectly_identified
  compound_precision = true_positives / (true_positives + false_positives)
  ```

### 2. Update Metrics Display
- Show decoy statistics in output
- Report true vs false compound identifications
- Add warnings when precision seems unrealistic

### 3. Consider Additional Improvements
- Add per-species evaluation (which samples have better/worse performance)
- Implement confidence calibration for decoy assignments
- Test with different decoy fractions (30%, 70%)
- Add "known unknown" category for peaks that don't match any compound well

### 4. Active Learning Integration (Future)
- Use uncertainty on decoy assignments to detect model confusion
- Prioritize labeling of peaks assigned to decoys with high confidence
- Develop acquisition functions that account for compound-level uncertainty

## What Was Done This Session

### 1. Removed One-to-One Matching Constraint
- Deleted all `--matching` parameters and logic
- Model now allows many peaks → same compound (realistic!)
- Updated evaluation to track compound-level metrics

### 2. Made Synthetic Data Much Harder
```python
# Changes to create_synthetic_data.py:
- Isomers: 40% with RT difference only ±0.3 min (was ±2.0)
- Near-isobars: Mass ±0.005 Da AND similar RT ±0.5 min
- RT errors: Added ±1 min systematic prediction bias
- Presence: Only 25% of compounds per sample (was 65%)
- Mass clustering: Compounds clustered in 3 regions for more overlaps
- DECOYS: 50% of library compounds never appear in samples!
```

### 3. Current Performance (Realistic!)
```
Peak-level:
  Precision: 75-88% (many wrong assignments)
  Recall: 55-70% (conservative)
  
Compound-level (MISLEADING - bug!):
  Shows: Precision 100%, Recall 80%
  Reality: Precision ~56% (assigns decoys!), Recall ~60%
```

## How to Test the Bug

```python
# Run training with decoys:
PYTHONPATH=. python scripts/train.py \
  --n-compounds 30 --n-species 40 \
  --n-samples 100 --n-tune 100 --n-chains 2 \
  --probability-threshold 0.3  # Low threshold to force assignments

# Then check for decoy assignments:
import pandas as pd
import numpy as np
from scripts.create_synthetic_data import create_metabolomics_data

np.random.seed(42)
assignments = pd.read_csv('output/results/peak_assignments.csv')
_, compounds, _, _, _ = create_metabolomics_data(n_compounds=30, decoy_fraction=0.5)

assigned_ids = assignments['assigned_compound'].dropna().unique().astype(int)
decoy_ids = compounds[compounds['is_decoy']]['compound_id'].values
real_ids = compounds[~compounds['is_decoy']]['compound_id'].values

print(f'Real compounds assigned: {len(set(assigned_ids) & set(real_ids))}')
print(f'DECOYS assigned (BUG!): {len(set(assigned_ids) & set(decoy_ids))}')
```

## Key Files to Modify

### 1. `/src/compassign/peak_assignment.py`
- Line ~520-650: The `assign()` method
- Line ~620-640: Compound-level metrics calculation
- Need to track decoy assignments as false positives

### 2. `/scripts/create_synthetic_data.py`
- Line ~145-160: Decoy compound generation
- Currently working correctly - marks compounds as decoys
- Real compounds selected from non-decoy subset

### 3. `/scripts/train.py`  
- Line ~137-146: Data generation with decoy_fraction=0.5
- Line ~315-319: Results display (needs to show decoy stats)

## Why This Matters

### Real Metabolomics Libraries
- Contain 1000s-10,000s of compounds
- Only 100s-1000s present in any sample
- Many compounds have similar masses/RTs
- False compound IDs are a major problem!

### Current Implementation
- 30 compounds: 15 real (can appear), 15 decoys (never appear)
- Model sees all 30 as candidates
- Successfully assigns to decoys (realistic!)
- But metrics don't count this as errors (bug!)

### Expected After Fix
```
Compound-level (REAL):
  Precision: 50-60% (many false IDs from decoys)
  Recall: 60-80% (finds most real compounds)
  F1: ~0.55-0.65 (realistic for hard problem)
```

## Example Fix for Evaluation

```python
# In peak_assignment.py assign() method:

# Get ALL compounds (including decoys)
all_compound_ids = set(range(self.n_compounds))  # 0-29 for 30 compounds

# Track which compounds we assigned peaks to
pred_compounds = set(pred_peaks_by_compound.keys())

# Get ground truth (compounds that ACTUALLY appear)
real_compounds = set()  # Only non-decoy compounds with peaks
for i, label in enumerate(true_labels):
    if label > 0:  # Not null
        compound_id = row_to_candidates[i][label]
        # Check if this is a real compound (not decoy)
        if not compound_info[compound_id].get('is_decoy', False):
            real_compounds.add(compound_id)

# Calculate TRUE metrics
true_positives = pred_compounds & real_compounds
false_positives = pred_compounds - real_compounds  # Includes decoys!

compound_precision = len(true_positives) / len(pred_compounds) if pred_compounds else 0
compound_recall = len(true_positives) / len(real_compounds) if real_compounds else 0
```

## Current Configuration

```python
# Harder defaults (as of this session):
mass_tolerance: 0.01 Da        # Wider tolerance = more candidates
rt_window_k: 2.0σ              # Wider RT window = more confusion
probability_threshold: 0.5     # Lower threshold = more assignments
n_compounds: 30                # More compounds = harder
decoy_fraction: 0.5            # 50% are decoys
isomer_fraction: 0.4           # 40% are isomers  
near_isobar_fraction: 0.3      # 30% are near-isobars
presence_prob: 0.25            # Only 25% compounds per sample
```

## Summary for Next Session

### What Works
✅ Many-to-one assignment (realistic)
✅ Harder synthetic data with isomers/near-isobars
✅ Decoy compounds in library
✅ Model correctly assigns to decoys (false positives)
✅ Clean, simplified codebase

### What's Broken
❌ Evaluation only counts non-decoy compounds
❌ Compound precision shows 100% (should be ~56%)
❌ No tracking of decoy assignments in metrics
❌ Missing decoy statistics in output

### Priority Fix
1. Update `assign()` method to track decoy assignments
2. Fix compound-level precision/recall calculations
3. Add decoy statistics to output
4. Test that metrics reflect realistic performance

Once fixed, this will be a **realistic metabolomics assignment system** with proper many-to-one support and challenging evaluation!