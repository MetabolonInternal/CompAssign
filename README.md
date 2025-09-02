# CompAssign: Bayesian Compound Assignment

CompAssign is a two‑stage Bayesian framework for untargeted metabolomics:
1) hierarchical retention time (RT) prediction, then
2) calibrated peak‑to‑compound assignment.
   
## Install

Requires Python 3.10+. Use the provided Conda environment.

```bash
conda env create -f environment.yml
conda activate compassign
# or: source scripts/setup_environment.sh
```

## Quick Start

```bash
# Fast sanity run (fewer MCMC samples)
./scripts/run_training.sh --quick

# Full run (more samples; slower)
./scripts/run_training.sh --full

# Two-stage control (debugging/experiments)
./scripts/run_two_stage.sh --stage both|1|2 [--quick]

# Direct python entrypoint
python scripts/train.py --help
```

## Project Structure

- `src/compassign/`: Core library (RT model, peak assignment, plots, generators)
- `scripts/`: Entrypoints (`train.py`, `train_two_stage.py`, `run_training.sh`, `run_two_stage.sh`)
- `docs/`: Model specs and references (see `docs/bayesian_models.md`)
- `output/`: Generated artifacts (git‑ignored)
- Root: `environment.yml`, `pyproject.toml`, `AGENTS.md`

## Development

- Format: `black src scripts` (line length 100)
- Lint: `flake8 src scripts`
- Test: `pytest -q` (keep tests fast; use synthetic data and fixed seeds)
- Principle: emphasize precision; include calibration/precision metrics when reporting results

See AGENTS.md (also mirrored in CLAUDE.md) for contributor guidelines.

## Model Overview

[Model details can be found here](docs/bayesian_models.md).

## License

Proprietary to Metabolon. All rights reserved.
