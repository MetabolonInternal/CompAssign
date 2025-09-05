# Softmax Peak Assignment Model

## Quick Start

This branch (`feature/softmax-null-assignment`) contains the implementation of a peak-wise softmax assignment model with explicit null class handling and active learning capabilities.

### Installation

```bash
# Activate existing CompAssign environment
conda activate compassign

# No additional dependencies required - uses existing PyMC v5
```

### Basic Usage

```python
from compassign import PeakAssignmentSoftmaxModel, PresencePrior

# Initialize model
model = PeakAssignmentSoftmaxModel(
    mass_tolerance=0.005,
    rt_window_k=2.5,
    use_temperature=True
)

# Train (same workflow as logistic model)
model.compute_rt_predictions(trace_rt, ...)
model.generate_training_data(peak_df, ...)
model.build_model()
model.sample(draws=1000, tune=1000)

# Make assignments with calibrated threshold
results = model.assign(prob_threshold=0.70)  # Note: 0.70 for high precision
```

### Testing

Use the consolidated validation suite:

```bash
./scripts/run_validation.sh -v
```

## Key Improvements Over Baseline (Qualitative)

- Calibration via in-model temperature scaling.
- Explicit null handling reduces spurious assignments.
- Exclusivity enforced directly by softmax.
- Active learning-friendly with posterior probabilities and AL hooks.

## Performance Notes

Performance is dataset‑dependent. Use `scripts/run_validation.sh` and review `validation_results.json` for metrics. Tune mass/RT windows and the assignment threshold to your data:
- For multi‑candidate MEDIUM difficulty: start with `mass_tolerance=0.15`, `rt_window_k=2.0`, and threshold 0.30–0.40 to target a 60–70% baseline precision for AL studies.

## Active Learning Integration

The softmax model supports precision-focused active learning:

```python
from compassign import OptimalOracle, run_annotation_experiment

# Run active learning experiment
results = run_annotation_experiment(
    softmax_model=model,
    oracle=OptimalOracle(),  # Or NoisyOracle, ConservativeOracle, etc.
    peak_df=peak_df,
    compound_mass=compound_mass,
    n_rounds=5,
    batch_size=10,
    threshold=0.95,
    selection_method='uncertainty'
)

# Results show entropy reduction and precision improvement over rounds
```

## Files in This Branch

### Core Implementation
- `src/compassign/peak_assignment_softmax.py` - Main softmax model
- `src/compassign/presence_prior.py` - Compound prevalence priors
- `src/compassign/oracles.py` - Human annotator simulators
- `src/compassign/eval_loop.py` - Evaluation harness
- `src/compassign/active_learning.py` - Acquisition functions

### Documentation
- `docs/softmax_implementation.md` - Technical documentation
- `docs/bayesian_models.md` - Updated with softmax math
- `SOFTMAX_PERFORMANCE_REPORT.md` - Detailed performance analysis
- `README_SOFTMAX.md` - This file

### Validation Entrypoints
- `scripts/run_validation.sh` - Main validation runner
- `scripts/validate_active_learning_complete.py` - Validation experiments

## Migration Path

The softmax model is designed to be a drop-in replacement with minimal code changes:

```python
# Change 1: Import
from compassign import PeakAssignmentSoftmaxModel  # Instead of PeakAssignmentModel

# Change 2: Initialization  
model = PeakAssignmentSoftmaxModel(...)  # Same parameters

# Change 3: Threshold
results = model.assign(prob_threshold=0.70)  # Use 0.70 instead of 0.50
```

## Comparison Guidance

Softmax vs logistic trade‑offs will vary by dataset. In general, softmax offers built‑in exclusivity, an explicit null, and integrated calibration. Use the validation suite for apples‑to‑apples comparisons in your setup.

## Known Limitations

1. **Training Speed**: ~2x slower than logistic model (acceptable trade-off)
2. **Memory Usage**: Slightly higher due to padded tensors
3. **Scale Testing**: Validated up to 75 compounds (larger scales in progress)

## Support

For questions or issues specific to the softmax implementation:
1. Check `docs/softmax_implementation.md` for technical details
2. Review `docs/softmax_implementation.md` for technical details

## Citation

If you use the softmax model in your research, please cite:
```
CompAssign Softmax Model - Peak Assignment with Null Class and Active Learning
Metabolon Internal, 2024
```
