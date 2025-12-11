# Descriptor Drift Experiment (`run_descriptor_shift.sh`)

This run stresses the hierarchical RT model under worksheet-style drift: we fit on in-support runs, then evaluate on the original covariates and on per-run drifted covariates. The goal is to quantify how descriptors and chemistry pooling stabilize predictions compared with a chemistry-free hierarchy and a Lasso baseline.

## Model Arms
- **Descriptor hierarchy**: descriptors ON (`compound_features` from the generator), class pooling ON (`include_class_hierarchy=True`), `global_gamma=False`.
- **Chemistry-free hierarchy**: descriptors OFF (`compound_features=None`), class pooling OFF (`include_class_hierarchy=False`), `global_gamma=True` to collapse run sensitivities to a single shared mean.
- **Species×compound Lasso**: frequentist baseline on standardized run covariates.

Only the descriptor-enabled arm reuses chemistry in both β (baseline RT) and γ (run sensitivities). The control arm removes all chemistry structure so we can attribute any lift directly to descriptors + pooling.

## Experiment Design
1. **Dataset**: synthetic generator with realistic run covariates and chemistry clusters (same as other RT experiments).
2. **Train/test split**: compute far-ness scores on run covariates; keep the closest `train_quantile` fraction (default 0.60) for training; hold out the rest for evaluation.
3. **Fit once**: train both hierarchical arms on the in-support runs (same random seed for comparability; seed+1 for the chemistry-free arm).
4. **Observed evaluation**: score the held-out runs with their original covariates (`observed` slice).
5. **Drifted evaluation**: perturb every held-out run’s covariates along a chemistry-informed direction plus log-normal noise (scale `--drift`, default 0.5) to mimic worksheet drift; rescore (`drifted` slice).
6. **Metrics**: MAE, RMSE, bias, MAE p95, and coverage for both slices and all arms. Results are printed to stdout and written to JSON.

## Commands
- Direct runner (single seed):  
  ```bash
  poetry run python scripts/experiments/rt/descriptor/assess_descriptor_shift.py --quick --posterior-samples 200
  poetry run python scripts/experiments/rt/descriptor/assess_descriptor_shift.py --seed 42 --draws 1000 --tune 1000 --chains 4 --posterior-samples 400
  ```
  `--quick` flips the sampler to 500/500/4; pass `--posterior-samples` to mirror the wrapper defaults (200 for quick, 400 for full).
- Multi-seed wrapper:  
  ```bash
  ./scripts/experiments/rt/descriptor/run_descriptor_shift.sh --quick
  ```
  Defaults to seeds `42 43 44 45 46`; override with `SEEDS="..."`. Additional CLI args (e.g., `--drift 0.8`) are forwarded to the Python runner.

## Outputs
- Directory: `output/rt_descriptor_shift/<profile>/`
- File: `descriptor_shift_<timestamp>.json`
  - `config`: sampler/dataset settings and drift scale.
  - `split`: run IDs and distance scores for train/test.
  - `metrics.observed`: MAE/RMSE/bias/p95/coverage for each arm on original covariates.
  - `metrics.drifted`: same metrics after drift is applied.
  - `drift_vectors`: per-run drift vectors for reproducibility.

## Typical Findings (quick profile, seeds 42–46)
- Observed covariates: descriptor hierarchy and chemistry-free hierarchy are close (MAE ≈0.45–0.55 min) with modest Lasso lag.
- Drifted covariates: descriptor hierarchy degrades gently (MAE +0.1–0.2 min) while chemistry-free hierarchy and Lasso jump by ≥0.4 min and pick up strong bias; coverage for the descriptor arm stays near nominal.
- Win condition: descriptors + pooling absorb drift because γ shares the chemistry direction whereas the chemistry-free arm cannot align its sensitivity without anchors.

## When to Use This Experiment
- Validating descriptor lift on worksheets with noticeable covariate drift or re-tuned LC methods.
- Stress-testing updated embeddings or γ priors before production rollout.
- Comparing proposed baselines: a new model should beat the descriptor hierarchy on drifted MAE/bias to claim superiority.

## Related Experiments
- `run_covshift_holdout.sh`: historical covariate-shift holdout using the class-pooled, descriptor-free hierarchy (anchors focus).
- `run_descriptor_generalization.sh`: cold-start and mix-shift studies toggling descriptors while retaining class pooling.
- `run_descriptor_ablation.sh`: multi-seed descriptor ablation with rare/unseen focus and the same chemistry-free baseline used here.
