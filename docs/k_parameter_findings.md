# k Parameter (RT Window) Findings

## Executive Summary

After extensive testing with **challenging synthetic data** (including isomers and near-isobars), we've discovered:

1. **Mass tolerance does most of the work** - filters 97.7% of candidates
2. **k parameter is critical for the remaining 2.3%** - these are the hard cases
3. **99% precision is unachievable** even with ultra-restrictive k values
4. **Default k=1.5 is reasonable** - balances precision (77%) and recall (79%)

## Key Discovery: The Precision Ceiling

With challenging data that includes:
- 30% isomers (identical mass, different RT)
- 20% near-isobars (within 10 mDa)
- Variable RT uncertainties (0.05-0.5 min)

**Maximum achievable precision: ~86%** (at k=0.1, losing 92.5% of true positives)

## Performance vs k Value (Challenging Data)

| k Value | Meaning | Statistical Coverage | Precision | Recall | True Positives Lost |
|---------|---------|---------------------|-----------|--------|---------------------|
| 0.1 | Ultra-restrictive | 8% of distribution | 86.2% | 7.5% | 309/334 (92.5%) |
| 0.3 | Very restrictive | 24% of distribution | 85.7% | 21.6% | 262/334 (78.4%) |
| 0.5 | Restrictive | 38% of distribution | 81.5% | 32.9% | 224/334 (67.1%) |
| 1.0 | Conservative | 68% of distribution | 77.9% | 56.0% | 147/334 (44.0%) |
| **1.5** | **Balanced** | **87% of distribution** | **76.9%** | **78.7%** | **71/334 (21.3%)** |
| 2.0 | Permissive | 95% of distribution | 74.5% | 87.4% | 42/334 (12.6%) |
| 3.0 | Very permissive | 99.7% of distribution | 65.9% | 97.3% | 9/334 (2.7%) |

## Why Simple Synthetic Data Misleads

Initial tests with random mass/RT data showed >99% precision regardless of k because:
- No structural relationships between compounds
- No isomers to confuse
- No near-isobars within mass tolerance
- Unrealistic separation in mass/RT space

**Lesson**: Always test with realistic data that includes the challenges your system will face.

## The Two-Stage Filtering Process

```
Input: 434 peaks × 100 compounds = 43,400 candidate pairs

Stage 1: Mass Tolerance (±0.005 Da)
├── Filters: 42,400 candidates (97.7%)
├── Keeps: 1,000 candidates (2.3%)
└── These are the HARD cases (isomers, near-isobars)

Stage 2: RT Window (k × sigma)
├── k=0.5: Filters 863/1000 (86.3%) → 137 remain
├── k=1.0: Filters 757/1000 (75.7%) → 243 remain  
├── k=1.5: Filters 654/1000 (65.4%) → 346 remain
├── k=2.0: Filters 604/1000 (60.4%) → 396 remain
└── k=3.0: Filters 502/1000 (50.2%) → 498 remain

Final: Assignment based on best score among remaining
```

## What k Values Mean Practically

### k < 0.5: "I don't trust my RT predictions"
- Rejects compounds even within expected uncertainty
- Achieves higher precision by being overly conservative
- Massive recall loss makes system impractical

### k = 0.5-1.0: "Conservative assignment"
- Accepts compounds within reasonable uncertainty bounds
- Good for high-stakes applications
- Some recall sacrifice for precision gain

### k = 1.0-2.0: "Balanced approach" ✓
- Accepts most statistically expected matches
- **k=1.5 is the sweet spot** for most applications
- Good precision-recall balance

### k > 2.0: "Permissive assignment"
- Accepts nearly all possibilities
- Maximizes recall at precision cost
- Useful for discovery/screening applications

## Why 99% Precision is Impossible

Even with k=0.1 (rejecting 92% of the normal distribution):

1. **Isomers**: Identical mass, only RT differentiates
   - RT predictions have inherent uncertainty
   - Some isomers too close to distinguish reliably

2. **Near-isobars**: Within mass tolerance by chance
   - Pass mass filter
   - May randomly have similar RT

3. **Noise peaks**: Statistical coincidence
   - Some will randomly fall in correct window
   - No parameter can eliminate all

## Recommendations

### For Different Use Cases

| Use Case | Recommended k | Expected Performance |
|----------|--------------|---------------------|
| Clinical diagnostics | 0.5-0.75 | P: ~80%, R: ~40% |
| Validated metabolites | 1.0-1.5 | P: ~77%, R: ~70% |
| Discovery/untargeted | 1.5-2.0 | P: ~75%, R: ~85% |
| Maximum coverage | 2.5-3.0 | P: ~66%, R: ~97% |

### To Achieve 99% Precision

Since k parameter alone cannot achieve 99% precision, consider:

1. **Tighter mass tolerance**: 0.003 Da instead of 0.005 Da
2. **Additional orthogonal data**: 
   - Ion mobility
   - MS/MS fragmentation
   - Isotope patterns
3. **Compound-specific thresholds**: Stricter for known problematic compounds
4. **Manual review tier**: Flag uncertain assignments for review
5. **Improved RT model**: Reduce prediction uncertainty

## Implementation Guidelines

### Current Settings (Reasonable)
```python
# In peak_assignment.py
mass_tolerance = 0.005  # Da
rt_window_k = 1.5       # Good balance
probability_threshold = 0.9  # Conservative
```

### For Higher Precision (With Recall Cost)
```python
mass_tolerance = 0.003  # Tighter
rt_window_k = 0.75      # More restrictive
probability_threshold = 0.95  # Very conservative
```

### For Higher Recall (With Precision Cost)
```python
mass_tolerance = 0.007  # Looser
rt_window_k = 2.0       # More permissive
probability_threshold = 0.8  # Less conservative
```

## Testing Protocol

All performance claims should be tested with:

```python
from scripts.create_challenging_test_data import (
    create_challenging_metabolomics_data,
    test_k_with_challenging_data
)

# Test with realistic data
results = test_k_with_challenging_data(
    k_value=1.5,
    mass_tolerance=0.005
)
```

## Conclusion

The k parameter optimization revealed that:
- **Perfect precision is impossible** with mass/RT alone when isomers exist
- **k=1.5 is optimal** for balanced performance
- **Mass tolerance is the dominant filter** (97.7% of work)
- **Always test with challenging data** to get realistic performance estimates

The current default settings are well-chosen. Focus optimization efforts on the RT model itself (reducing uncertainty) rather than endlessly tuning k.