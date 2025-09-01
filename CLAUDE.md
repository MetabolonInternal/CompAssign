# CLAUDE.md

## Project Overview
CompAssign: Bayesian compound assignment for metabolomics using RT prediction and mass tolerance filtering.
**Key principle**: Precision over recall (minimize false positives).
**Optimized**: Now uses only 9 essential features (reduced from 16) for 98.7% F1 score.

## Setup & Run
```bash
conda activate compassign
./scripts/run_training.sh          # Standard training (9 features)
./scripts/run_two_stage.sh         # Two-stage training for debugging
PYTHONPATH=. python scripts/train.py  # Direct Python
```

## Key Files
- `src/compassign/rt_hierarchical.py` - RT prediction (Stage 1)
- `src/compassign/peak_assignment.py` - Peak assignment (Stage 2, 9 features)
- `scripts/train.py` - Main training script
- `scripts/train_two_stage.py` - Two-stage training for debugging
- `FEATURE_ANALYSIS.md` - Feature optimization documentation

## Two-Stage Training (for debugging)
```bash
# Train Stage 1 (RT model) only
./scripts/run_two_stage.sh --stage 1 --quick

# Train Stage 2 (Peak assignment) with saved RT model
./scripts/run_two_stage.sh --stage 2

# Both stages with custom parameters
./scripts/run_two_stage.sh --stage both --output two_stage_output
```

## Features Used (9 Essential)
After optimization, the model uses only 9 features:
- **Core (3)**: mass_err_ppm, rt_z, log_intensity
- **Confidence (3)**: mass_confidence, rt_confidence, combined_confidence  
- **Context (3)**: log_compound_mass, log_rt_uncertainty, log_relative_intensity

See `FEATURE_ANALYSIS.md` for details on the 7 removed redundant features.

## Parameters
- `--mass-tolerance`: Mass window in Da (default: 0.005)
- `--rt-window-k`: RT window multiplier (default: 1.5)  
- `--probability-threshold`: Assignment threshold (default: 0.9)

Lower tolerance and higher thresholds → higher precision, lower recall.