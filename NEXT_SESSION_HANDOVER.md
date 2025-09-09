# NEXT SESSION HANDOVER — Generative PyMC Assignment

This handover tells Claude exactly how to run the new generative model, validate results, and compare against baselines. It assumes Linux or macOS with Conda and typical dev tools.

## 1) Goals
- Run the end-to-end training pipeline with the new generative assignment (`--model generative`).
- Validate outputs (metrics, artifacts) and sanity‑check responsibilities/calibration.
- Optionally compare against `calibrated` and `softmax` baselines on the same data split.

## 2) Environment Setup
Two options — pick one:

- Option A (Conda, recommended):
  - `conda env create -f environment.yml`
  - `conda activate compassign`
  - If pytest is missing: `pip install pytest`

- Option B (script):
  - `source scripts/setup_environment.sh`

Confirm Python version is ≥3.10 and PyMC imports.

## 3) Quick Experiment (Generative)
This runs a smaller MCMC for a fast sanity check.

- Command:
  - `./scripts/run_training.sh --quick --model generative`

- Expected outputs (under `output/`):
  - `models/assignment_trace.nc` (InferenceData for the generative assignment model)
  - `results/peak_assignments.csv` (peak_id, assigned_compound, probability)
  - `results/assignment_metrics.json` (precision/recall/F1, confusion matrix)

- Sanity checks:
  - Open `assignment_metrics.json` and confirm non‑zero precision and recall.
  - `assignment_trace.nc` has variables: look for `r` (responsibilities) with dims `(chain, draw, N, K_max)`.

## 4) Full(er) Experiment (Generative)
Use more draws for better fidelity (still moderate runtime):

- `./scripts/run_training.sh --samples 1000 --model generative`

If you have time, increase `--samples` to 1500–2000 and `--target-accept` to 0.95.

## 5) Baseline Comparisons (Optional)
Run the same pipeline with the other models. This uses the same synthetic data generation baked into `scripts/train.py`.

- Calibrated (logistic):
  - `./scripts/run_training.sh --quick --model calibrated`

- Softmax:
  - `./scripts/run_training.sh --quick --model softmax`

Collect `assignment_metrics.json` for each run and compare precision/recall/F1.

## 6) Unit Tests (Fast)
Validate core behaviors of the generative model without sampling.

- Install pytest if needed: `pip install pytest`
- Run: `pytest -q -k generative`

These tests cover tensor shapes/masking and prior‑mode predictions on a tiny synthetic set.

## 7) What To Report Back
Provide a short summary with:
- Config used (samples, model type).
- Precision/Recall/F1 from `output/results/assignment_metrics.json`.
- Any warnings (divergences) noted by PyMC.
- Quick comment on runtime.

## 8) Troubleshooting Tips
- If `numpy`/`pytest` not found: confirm the environment is activated and run `pip install numpy pytest`.
- If sampling is slow, use `--quick` first; bump up draws incrementally.
- If `assignment_trace.nc` is missing, check for earlier exceptions in stdout.

## 9) Stretch Tasks (If Time Allows)
- Rerun generative with `--samples 1500 --model generative`; note any calibration improvements.
- Compare calibration quality qualitatively by looking at distribution of max responsibility values (can be inspected with ArviZ/xarray).

## 10) Artifacts to Keep
- `output/models/assignment_trace.nc`
- `output/results/peak_assignments.csv`
- `output/results/assignment_metrics.json`

Keep logs or copy key stdout snippets for later analysis (e.g., sampling diagnostics).

---

Legacy notes about multi‑candidate generators and active‑learning sweet spot were kept in earlier sessions; the plan above is the canonical path for the new generative model. If you need to run active‑learning loops, use `src/compassign/eval_loop.py` with a model instance. It supports both generative (uses `r`) and softmax (uses `p`).
