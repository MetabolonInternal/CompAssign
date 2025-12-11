RT Covariate‑Shift Robustness (`run_covshift_holdout.sh`)

Overview
- Goal: demonstrate that the class-pooled, descriptor-free hierarchical RT model is more robust to covariate shift than baseline Lasso models, and quantify how a small number of anchors per test run further improves performance.
- Approach: fix the training set to in‑support runs and evaluate on out‑of‑support test runs, binned by covariate distance. Aggregate over multiple dataset seeds to obtain mean ± std bands.

Terminology
- Far‑ness (covariate distance): for each run, compute the minimum L2 distance to any other run in standardized IS‑covariate space (zero mean, unit variance). Larger = farther from the rest (more out‑of‑support).
- Bins: mutually exclusive ranges of far‑ness percentiles used for testing: 60–70%, 70–80%, 80–90%, 90–95%, 95–100%.
- Anchors: a small number of labeled compounds per test run used solely for per‑run intercept calibration (test‑time bias correction). Anchors are excluded from evaluation metrics.
- Class‑pooled γ: compound‑specific run‑covariate coefficients shrink toward a class‑level mean (global_gamma=False). Compound baselines (β) also share via chemical classes (include_class_hierarchy=True).

Experimental Design (Fixed‑Train, Binned Test)
1) Dataset: synthetic generator with dense observations per (species×compound) and realistic run‑level IS covariates. Each replicate uses a different RNG seed.
2) Train split: take all runs at or below the 60th percentile of far‑ness (in‑support). Train the hierarchical model once per replicate on this fixed training set.
3) Test split: partition the remaining runs into bins by far‑ness (60–70, 70–80, 80–90, 90–95, 95–100). Evaluate each method on each bin.
4) Methods: Hierarchical (class‑pooled γ; no descriptors), Baseline species×compound Lasso, Baseline cluster×compound Lasso.
5) Anchors: per test run, N∈{0,3,5} anchors are randomly selected and used to compute a per‑run intercept correction Δ_run = mean(y_true_anchor − y_pred_anchor). We add Δ_run to all predictions in that run and exclude anchor rows from the metric.
6) Aggregation: average metrics over replicates; plot mean ± std bands per bin and per anchor budget.

Model Settings
- Hierarchical: include_class_hierarchy=True; global_gamma=False; compound_features=None (no descriptors). This mirrors the historical baseline we cite and isolates class pooling as the chemistry signal.
- Baselines: Lasso/LassoCV with standardized features; solver iteration budgets increased to reduce convergence warnings.
- MCMC profiles:
  - Quick: draws=500, tune=500, chains=4
  - Normal: draws=1000, tune=1000, chains=4

How To Run
- One‑command runner (quick profile default if you pass --quick):
  - Quick (500/500/4, 5 reps): `./scripts/experiments/rt/covariate_shift/run_covshift_holdout.sh --quick`
  - Normal (1000/1000/4, 5 reps): `./scripts/experiments/rt/covariate_shift/run_covshift_holdout.sh`
  - Knobs via CLI/env:
    - `--reps N` to change replicates (default 5)
    - `SEED=42` to change base seed (rep seeds = SEED..SEED+reps-1)
    - `POST_SAMPLES=60` to change posterior draws used at prediction
  - Plot only (no sampling):
    - Rebuild plots from latest JSON in `output/rt_covshift_holdout/`: `./scripts/experiments/rt/covariate_shift/run_covshift_holdout.sh --plot-only`
    - Or specify a particular summary: `./scripts/experiments/rt/covariate_shift/run_covshift_holdout.sh --plot-only --plot-input output/rt_covshift_holdout/covshift_holdout_YYYYMMDD_HHMMSS.json`

Outputs
- Directory: `output/rt_covshift_holdout/`
- Files:
  - `covshift_holdout_<timestamp>.json` — configuration, per‑replicate bin metrics (including hierarchical coverage/z‑std), and aggregate mean ± std series.
  - `covshift_holdout_mae_anchors{N}_<timestamp>.png` — MAE vs distance bins, mean ± std for anchors/run = N.
  - `covshift_holdout_delta_anchors{N}_<timestamp>.png` — ΔMAE (baseline − hier) vs bins, mean ± std for anchors/run = N.
  - `covshift_holdout.log` — run log.

Key Findings (typical)
- Without anchors (N=0): all methods degrade on out‑of‑support bins; hierarchical shows smaller MAE and tighter bands than baselines; ΔMAE (baseline − hier) is positive across bins.
- With small anchors (N=3 or 5): per‑run intercept calibration collapses most of the residual bias. Hierarchical remains best or tied across bins; ΔMAE widens in mid/high shift, indicating more robust extrapolation.
- Calibration: hierarchical coverage remains close to nominal across bins (z‑std near 1), especially with anchors.

Caveats & Extensions
- Absolute MAE depends on bin severity; anchors mitigate this by correcting per‑run bias.
- Descriptor‑based β can be enabled in a future extension to further stabilize intercepts; it’s orthogonal to γ pooling.
- For production datasets, replace the synthetic generator with real runs and keep the same analysis flow.

Position in the Experiment Suite
- `run_descriptor_generalization.sh`: toggles descriptors on/off (with class pooling) to measure chemistry lift on new compounds and shifted class mixes.
- `run_descriptor_shift.sh`: contrasts descriptor-enabled vs chemistry-free hierarchies under run drift; shows how descriptors plus pooling stabilize γ.
- `run_descriptor_ablation.sh`: multi-seed descriptor ablation highlighting rare/unseen lift; uses the same no-descriptor/no-class baseline as the drift study.
This covariate-shift holdout run keeps the historical class-pooled, descriptor-free hierarchy so new reruns remain comparable with previously published robustness curves.
