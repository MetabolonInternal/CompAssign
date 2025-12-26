# Repository Guidelines

## Project Structure & Module Organization
- `src/compassign/`: Core library (e.g., `peak_assignment.py`, `rt/pymc_partial_pool_ridge.py`, plots, generators).
- `src/compassign/rt/`: Production RT pipeline shell entrypoints (`prep.sh`, `train.sh`, `eval.sh`).
- `scripts/`: Auxiliary scripts (e.g., `data_prep/`, `remote/`, `bench/`).
- `docs/`: Reference documentation (e.g., Bayesian model specs).
- `docs/TASKS.md`: Living task plan and next actions for agent work.
- `output/`: Generated artifacts; ignored by Git. Keep lightweight summaries only.
- Root config: `pyproject.toml`, `poetry.lock`, `README.md`.

## Build, Test, and Development Commands
- Environment: `poetry install` and then either `poetry shell` or prefix commands with `poetry run`.
- Production RT ridge: `./src/compassign/rt/train.sh` then `./src/compassign/rt/eval.sh`.
- Ridge trainer help: `poetry run python -m compassign.rt.train_rt_pymc_collapsed_ridge --help`.
- Format: `black src scripts` (configured to line length 100).
- Lint: `flake8 src scripts`.

## Coding Style & Naming Conventions
- Python 3.11 and 4‑space indentation.
- Keep functions small and single‑purpose; prefer pure functions.
- Run Black (line length 100) before committing.
- Naming: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE` for constants.
- Use type hints everywhere and docstrings for public functions.
- No side effects at import time; avoid top‑level execution and global mutation.
- Use `logging` for diagnostics; reserve `print` for CLIs.
- Do not introduce the term “wide” for CSVs; refer to them simply as CSV.

## Testing Guidelines
- Framework: `pytest`.
- Location: under `tests/`, mirroring `src/compassign/`.
- Naming: files `test_*.py`; functions `test_*`.
- Run fast and deterministically: prefer tiny synthetic data and fixed seeds.
- Avoid heavy MCMC in unit tests; if sampling is required, keep draws/tune very small and mark as slow.
- Typical invocation: `pytest -q` (use `-k` to scope). Ensure tests pass locally with `poetry run`.
- CI budget: aim for under a few minutes; each unit test should complete in seconds.

### Testing Do/Don'ts
- Do use fixed seeds (default `42`) and tiny shapes by default (e.g., `n_species<=3`, `n_compounds<=3`, `runs<=2`).
- Do keep sampler settings tiny in tests if needed (`draws<=20`, `tune<=20`, `chains<=1`) and add `@pytest.mark.slow`.
- Do isolate side effects: write only to `tmp_path`, avoid touching `output/` or the repo tree.
- Do stub heavy components with `monkeypatch` and validate shapes/contracts over exact floats when sampling noise is present.
- Do set `np.random.seed(42)` (or generator equivalents) in tests that rely on randomness.
- Don't add long‑running MCMC or large datasets to unit tests.
- Don't rely on external data, network, or non‑deterministic clocks.
- Don't introduce “wide CSV” terminology; refer to CSVs simply as CSV.

## Commit & Pull Request Guidelines
- Commits: write short, plain sentences focused on the purpose (avoid colons or semicolons). Examples: `Add run level inputs across RT pipelines`, `Add production RT loader and example data`.
- PRs: include purpose, approach, and risks; link issues; show before/after plots for visualization changes; note performance impacts and reproduction steps (exact commands).
- Checks: code formatted (black), lint‑clean (flake8), tests pass locally.
- Principle: prioritize precision over recall; report calibration/precision metrics when relevant.

## Data Interfaces
- Synthetic data embeds run‑level covariates directly in `peak_df` as `run_covariate_*` columns. Use `extract_run_metadata` to obtain run matrices and mappings.
- Production CSV defines runs as a composite of `sampleset_id`, `worksheet_id`, and `task_id`, with IS_* numeric covariates. See README for schema.

## Security & Configuration Tips
- Do not commit datasets, logs, or model artefacts; `output/` is ignored by default.
- No secrets required; never add credentials or tokens.
- Dependencies are pinned via `poetry.lock`; use repository scripts for reproducibility and provide `--seed` for determinism (wrappers default to 42).
- Propose new dependencies sparingly and justify them; avoid heavyweight libraries in the hot path and unit tests.
