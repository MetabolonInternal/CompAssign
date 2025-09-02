# Repository Guidelines

## Project Structure & Module Organization
- `src/compassign/`: Core library (e.g., `peak_assignment.py`, `rt_hierarchical.py`, plots, generators).
- `scripts/`: Runnable entrypoints (e.g., `train.py`, `train_two_stage.py`, `run_training.sh`, `run_two_stage.sh`, `setup_environment.sh`).
- `docs/`: Reference documentation (e.g., Bayesian model specs).
- `output/`: Generated artifacts; ignored by Git. Keep lightweight summaries only.
- Root config: `environment.yml` (conda), `pyproject.toml` (black/flake8/pytest), `README.md`.

## Build, Test, and Development Commands
- Environment: `source scripts/setup_environment.sh` (create/activate) or `conda env create -f environment.yml && conda activate compassign`.
- Quick training: `./scripts/run_training.sh --quick` (smaller MCMC for fast sanity checks).
- Full training: `./scripts/run_training.sh --full` (more samples; slower, higher fidelity).
- Two‑stage pipeline: `./scripts/run_two_stage.sh --stage both|1|2 [--quick]`.
- Direct usage/help: `python scripts/train.py --help`.
- Format: `black src scripts` (configured to line length 100).
- Lint: `flake8 src scripts`.

## Coding Style & Naming Conventions
- Python ≥3.10; 4‑space indentation; keep functions small and pure where possible.
- Black enforced (line length 100); run before committing.
- Naming: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE` for constants.
- Prefer type hints and docstrings; avoid side effects in import time.

## Testing Guidelines
- Framework: `pytest`.
- Location: place tests under `tests/`, mirroring `src/compassign/` (e.g., `tests/test_peak_assignment.py`).
- Naming: files `test_*.py`; functions `test_*`.
- Run: `pytest -q` (use `-k keyword` to scope). Keep unit tests fast—use small synthetic data and fixed seeds; avoid long MCMC in tests.

## Commit & Pull Request Guidelines
- Commits: imperative, concise, scoped when helpful (e.g., "assigner: calibrate probabilities").
- PRs: include purpose, approach, and risks; link issues; show before/after plots for visualization changes; note performance impacts and reproduction steps (exact commands).
- Checks: code formatted (black), lint‑clean (flake8), tests pass locally.
- Principle: prioritize precision over recall; report calibration/precision metrics when relevant.

## Security & Configuration Tips
- Do not commit datasets, logs, or model artifacts; `output/` is ignored by default.
- No secrets required; avoid adding credentials or tokens.
- Use pinned dependencies in `environment.yml`; prefer repository scripts for reproducibility.
