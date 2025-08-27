# NEXT SESSION HANDOVER - CompAssign Simplification

## Major Discovery (2025-08-27)
**Ablation study proved that the enhanced model is unnecessary.** Simple parameter tuning achieves better performance.

## Ablation Study Results Summary

### Key Finding
**S-Both configuration (standard model with optimized parameters) beats everything:**
- **Precision: 99.5%** (vs 99.3% for enhanced)
- **Recall: 93.9%** (vs 89.9% for enhanced)  
- **False Positives: 7** (vs 9 for enhanced)
- **No code changes needed** - just parameter adjustments!

### Optimal Configuration
```python
# Standard model with:
mass_tolerance = 0.005  # (was 0.01)
threshold = 0.9         # (was 0.5)
```

### What Doesn't Help (Remove These)
- ❌ Enhanced model architecture
- ❌ RT uncertainty features
- ❌ RT absolute difference features
- ❌ Asymmetric loss functions (FP penalty)
- ❌ Probability calibration
- ❌ Staged assignment system

## Work Completed So Far

### ✅ Done
1. Fixed training pipeline hanging issue
2. Created and ran ablation study
3. Identified optimal parameters (mass_tolerance: 0.005, threshold: 0.9)
4. Created comprehensive plan for simplification

### ⚠️ Partially Done (NEEDS COMPLETION)
1. **Started** updating standard model defaults:
   - ✅ Changed default mass_tolerance to 0.005 in `peak_assignment.py`
   - ✅ Changed default threshold to 0.9 in `peak_assignment.py`
   - ❌ Still need to remove enhanced imports/logic from training scripts

2. **Deleted** enhanced model file:
   - ✅ Removed `peak_assignment_enhanced.py`
   - ❌ But all scripts still reference it and will break!

### 🔴 NOT Started Yet
- Training script still has all enhanced model code
- Verification scripts still try to run enhanced model
- Documentation still describes enhanced model
- Imports still reference deleted enhanced model file

## Next Steps (TODO)

### 1. Complete Training Script Simplification
**File: `scripts/train.py`**
- Remove `--model` argument (no longer needed)
- Remove all enhanced-specific arguments:
  - `--fp-penalty`
  - `--test-thresholds`
  - `--high-precision-threshold`
  - `--review-threshold`
- Remove all enhanced model imports and logic
- Simplify to single model path
- Keep only essential parameters:
  - `--mass-tolerance` (default: 0.005)
  - `--probability-threshold` (default: 0.9)

### 2. Update Verification Scripts
**File: `scripts/run_verification.sh`**
- Remove enhanced model configuration
- Change to test parameter variations only:
  - Baseline (old defaults: mass_tol=0.01, threshold=0.5)
  - Optimized (new defaults: mass_tol=0.005, threshold=0.9)
- Update expected results in output messages

### 3. Simplify Benchmark Report Generator
**File: `scripts/generate_benchmark_report.py`**
- Remove enhanced model handling
- Update to compare only parameter configurations
- Simplify visualization code

### 4. Update Documentation
**File: `CLAUDE.md`**
- Remove all enhanced model documentation
- Document the simplified approach
- Emphasize importance of mass_tolerance and threshold
- Add ablation study findings

**File: `docs/precision_optimization.md`**
- Update to reflect that parameter tuning is sufficient
- Remove complex modeling strategies
- Add ablation results

### 5. Clean Up Imports
Search and remove all imports of enhanced model:
```bash
grep -r "peak_assignment_enhanced" --include="*.py"
```

### 6. Update Test Commands
Simplify all test commands to remove enhanced options:
```bash
# Old (remove):
PYTHONPATH=. python scripts/train.py --model enhanced --fp-penalty 5.0 ...

# New (use):
PYTHONPATH=. python scripts/train.py --mass-tolerance 0.005 --threshold 0.9
```

### 7. Create Migration Guide
Create `docs/MIGRATION_TO_SIMPLE.md`:
- Explain why enhanced was removed
- Show ablation study results
- Provide parameter migration guide
- Performance comparison table

## Critical Parameters

### Production Settings
```python
mass_tolerance = 0.005  # 5 ppm - critical for precision
probability_threshold = 0.9  # Conservative decision boundary
```

### Why These Work
1. **Mass tolerance (0.005 Da)**: Tight enough to eliminate most false candidates before modeling
2. **Threshold (0.9)**: High enough to be conservative without destroying recall
3. **Together**: 99.5% precision with 93.9% recall

## Code to Remove (Checklist)

- [x] `src/compassign/peak_assignment_enhanced.py` (DELETED but scripts still reference it!)
- [ ] Enhanced imports in `scripts/train.py` (CRITICAL - will break if not fixed)
- [ ] Enhanced configuration in `scripts/run_verification.sh`
- [ ] Enhanced handling in `scripts/generate_benchmark_report.py`
- [ ] Enhanced references in `scripts/ablation_study.py`
- [ ] Enhanced model tests (if any)
- [ ] Enhanced-specific plotting code
- [ ] Enhanced documentation sections

## ⚠️ CURRENT STATE WARNING
**The codebase is currently BROKEN because:**
1. We deleted `peak_assignment_enhanced.py`
2. But `scripts/train.py` still imports and uses it
3. Running any training with `--model enhanced` will crash

**First priority in next session: Fix the broken imports!**

## Testing After Simplification

Run these to verify everything works:
```bash
# Quick test
PYTHONPATH=. python scripts/train.py --n-samples 100

# Verification
./scripts/run_verification.sh

# Ablation (should show standard params are optimal)
./scripts/run_ablation.sh --quick
```

## Expected Benefits After Simplification

1. **Codebase**: ~1000 lines removed
2. **Performance**: Better (99.5% vs 99.3% precision)
3. **Maintainability**: Single model, clear parameters
4. **Training time**: Faster (no complex features)
5. **Understanding**: Much easier to explain and deploy

## Important Note

The ablation study (`scripts/ablation_study.py`) should be kept as evidence of why we simplified. It proves that complex modeling doesn't beat simple parameter tuning for this problem.

## Contact Info
- Pipeline fix completed: 2025-08-27
- Ablation study completed: 2025-08-27 
- Simplification started: 2025-08-27
- **Status: IN PROGRESS - Continue removing enhanced model code**