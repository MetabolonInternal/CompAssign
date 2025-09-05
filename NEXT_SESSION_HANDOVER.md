# NEXT SESSION HANDOVER - Active Learning Sweet Spot

## 🎯 Current Status

### What Was Accomplished
1. **Identified the core problem**: The synthetic data was using impossible features (mass_error_ppm requiring ground truth) and creating trivial binary classification instead of multi-class assignment
2. **Created the solution**: `multi_candidate_generator.py` that generates 5-8 competing candidates per peak
3. **Achieved complexity target**: 6.2 candidates per peak average with overlapping isomers/isobars

### What Actually Works
- **ONLY KEEP**: `src/compassign/multi_candidate_generator.py` 
- All other synthetic generators were failed attempts and have been deleted
- The key insight: Need multiple compounds competing for each peak, not just feature overlap

## ⚠️ Critical Issues to Fix

### 1. **validate_active_learning_complete.py is BROKEN**
```python
# Line 86: IndexError - trying to access compound 45 when only 45 compounds (0-44)
mass_error_da = abs(mz - compound_mass[m])
```
- The script still uses the OLD data generation with forbidden features
- Needs to be updated to use `multi_candidate_generator.py`

### 2. **Model Must Handle Multiple Candidates**
- Current softmax model filters to 1 candidate per peak (too restrictive)
- Need to ensure mass_tolerance and RT windows allow 5-8 candidates through
- Update model initialization:
```python
# Current (too restrictive):
mass_tolerance=0.01  # Only 10 mDa
rt_window_k=2.5      # 2.5 sigma

# Needed for multi-candidates:
mass_tolerance=0.15  # 150 mDa to match generator
rt_window_k=2.0      # Tighter to force discrimination
```

## 📋 TODO: Next Steps

### Priority 1: Fix and Test the Pipeline
1. **Update validate_active_learning_complete.py**
   - Replace `setup_model_with_data()` to use `multi_candidate_generator`
   - Remove all references to forbidden features (peak_quality, sn_ratio)
   - Fix the indexing error on line 86

2. **Create working test script**
   ```python
   # scripts/test_multi_candidate_baseline.py
   from src.compassign.multi_candidate_generator import MultiCandidateGenerator
   # Test that model gets 60-70% precision with 5-8 candidates
   ```

3. **Verify active learning improvement**
   - Baseline should be 60-70% precision
   - After 150 annotations: 75-85% precision
   - Key: Human helps disambiguate between the 5-8 candidates

### Priority 2: Integration
1. **Update PeakAssignmentSoftmaxModel**
   - Ensure it handles multiple candidates properly
   - No filtering down to 1 candidate!
   - Features computed for ALL candidates

2. **Fix SmartNoisyOracle**
   - Currently uses forbidden features (peak_quality)
   - Should use only: mass, RT, intensity, peak_width, peak_asymmetry
   - Context: isotope patterns, adduct recognition

### Priority 3: Validation
1. **Run full active learning experiment**
   - Use `multi_candidate_generator` with MEDIUM difficulty
   - Test random vs uncertainty vs margin sampling
   - Verify 10-15% improvement from active learning

2. **Create visualization**
   - Learning curves showing improvement
   - Confusion matrices for multi-class assignment
   - Distribution of candidates per peak

## 🚫 What NOT to Do

1. **Don't use synthetic features**: No peak_quality, no mass_error_ppm using true compound
2. **Don't simplify to binary**: The problem MUST be multi-class (5-8 candidates)
3. **Don't trust old test scripts**: Most were testing flawed generators

## 💡 Key Insights to Remember

1. **The real challenge**: Choosing among 5-8 plausible candidates, not filtering to 1
2. **Feature overlap wasn't enough**: Even with 70% overlap, binary classification was trivial
3. **Multi-candidate is the key**: Isomers (same mass), isobars (±0.15 Da), adducts create real ambiguity
4. **Human value**: Pattern recognition across peaks, not just individual features

## 🔧 Technical Parameters

### Working Configuration (MEDIUM difficulty):
```python
{
    'isomer_fraction': 0.35,
    'isobar_fraction': 0.55, 
    'mass_tolerance': 0.15,  # 150 mDa - CRITICAL!
    'rt_uncertainty_min': 3.0,
    'rt_uncertainty_max': 5.0,
    'candidates_target': 7
}
```

### Model Settings Needed:
```python
PeakAssignmentSoftmaxModel(
    mass_tolerance=0.15,  # Must match generator!
    rt_window_k=2.0,      # Tight enough to discriminate
    use_temperature=True,
    standardize_features=True
)
```

## 📊 Success Metrics

✅ **Achieved**: 6.2 candidates per peak average
❌ **Not Yet Tested**: 60-70% baseline precision
❌ **Not Yet Tested**: Active learning improvement to 75-85%

## 🎯 Single Focus for Next Session

**Get ONE complete working demo:**
1. Generate data with `multi_candidate_generator` 
2. Train model with 60-70% baseline
3. Apply active learning
4. Show 10-15% improvement
5. Document in a single clean script

Don't create more generators or variations - make the existing solution work end-to-end!