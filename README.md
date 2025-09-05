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
# Active learning validation (recommended entrypoint)
./scripts/run_validation.sh -v   # or omit -v for quiet mode

# Optional: visualize results (uses validation_results.json)
python scripts/visualize_validation_results.py --input validation_results.json
```

## Project Structure

- `src/compassign/`: Core library (RT model, softmax assignment, AL harness, generator)
- `scripts/`: Entrypoints (`run_validation.sh`, `validate_active_learning_complete.py`, `setup_environment.sh`, `visualize_validation_results.py`)
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
