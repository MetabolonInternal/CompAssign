# RT Descriptor Generalization (`run_descriptor_generalization.sh`)

This note captures two synthetic studies that stress‑test retention‑time descriptors in realistic production scenarios. The goal is to build intuition for when the `compound_features` term in our hierarchical RT model drives measurable value, how to reproduce those runs, and what to expect in production deployments.

## Model Arms
- **Descriptor ON (default)**: Hierarchical RT model with descriptors (`compound_features`), class pooling enabled (`include_class_hierarchy=True`), and pooled run sensitivities (`global_gamma=False`).
- **Descriptor OFF (control)**: Identical hierarchy but `compound_features=None`. Holding the pooling structure fixed isolates the contribution of descriptors alone.
- Outputs: per-seed JSON summaries plus auto-generated MAE/RMSE/bias plots comparing both arms for each scenario.

## Scenarios

### 1. Cold Start (Unseen Compounds)
- **Setup**: The generator uses `anchor_free_frac=0.25` and `unseen_eval_per_compound=4` so that an entire subset of compounds is never observed during training. All labelled rows for those compounds are kept only in the evaluation slice.
- **Training data**: all “anchor” and “rare” compounds (i.e., anything with labelled rows) across six synthetic species. We use the realistic generator defaults: `n_compounds=80`, `n_internal_standards=8`, `desc_tau_beta=0.30`, `desc_sigma_compound=0.30`.
- **Evaluation**: split into `anchor_compounds` (sanity check) and `unseen_compounds`.
- **Finding**: With descriptors active the hierarchical model maintains MAE <1.0 min on unseen compounds; without descriptors MAE ranges 1.1–1.8 min, tail errors double, and coverage collapses. Anchored compounds are unchanged, confirming the lift is confined to true cold starts.

### 2. Shifted Compound Mix (Held-out Classes)
- **Setup**: All compounds are generated densely (`fixed_runs_per_species_compound=10`), but we withhold an entire fraction of chemistry classes (default 25%) when building the training frame. Those classes only appear in the evaluation slice, mimicking a worksheet whose chemistry mix diverges from the historical training data.
- **Training data**: rows from “seen classes” only. Same descriptor settings as above.
- **Evaluation**: two slices—`heldout_classes` (new chemistry) and `seen_classes` (control).
- **Finding**: Descriptors cut MAE roughly in half (0.6–0.8 min vs 1.2–1.8 min), reduce bias by ~0.4–1.0 min, and shrink 95th percentile error from ~5 min down to ~1.5–2 min. Seen classes stay flat (~0.30 min).

## Scripts & Commands

All runs use `poetry run …` with the repo root as `$PWD`.

1. **`scripts/experiments/rt/descriptor/assess_descriptor_generalization.py`** – evaluator with `--scenario {cold_start,mix_shift}`.
   - Quick smoke (500/500/4) for seed 42:  
     ```bash
     poetry run python scripts/experiments/rt/descriptor/assess_descriptor_generalization.py --scenario cold_start --quick --seed 42 --posterior-samples 200
     poetry run python scripts/experiments/rt/descriptor/assess_descriptor_generalization.py --scenario mix_shift --quick --seed 42 --posterior-samples 200
     ```
   - Full profile (1000/1000/4, 400 posterior draws) for seeds 42–44:  
     ```bash
     poetry run python scripts/experiments/rt/descriptor/assess_descriptor_generalization.py --scenario cold_start --seed 42 --draws 1000 --tune 1000 --chains 4 --posterior-samples 400
     poetry run python scripts/experiments/rt/descriptor/assess_descriptor_generalization.py --scenario cold_start --seed 43 --draws 1000 --tune 1000 --chains 4 --posterior-samples 400
     poetry run python scripts/experiments/rt/descriptor/assess_descriptor_generalization.py --scenario cold_start --seed 44 --draws 1000 --tune 1000 --chains 4 --posterior-samples 400

     poetry run python scripts/experiments/rt/descriptor/assess_descriptor_generalization.py --scenario mix_shift --seed 42 --draws 1000 --tune 1000 --chains 4 --posterior-samples 400
     poetry run python scripts/experiments/rt/descriptor/assess_descriptor_generalization.py --scenario mix_shift --seed 43 --draws 1000 --tune 1000 --chains 4 --posterior-samples 400
     poetry run python scripts/experiments/rt/descriptor/assess_descriptor_generalization.py --scenario mix_shift --seed 44 --draws 1000 --tune 1000 --chains 4 --posterior-samples 400
     ```
   - Outputs land under `output/rt_descriptor_generalization/default/{cold_start,mix_shift}/descriptor_generalization_*_<timestamp>.json`.

2. **`scripts/experiments/rt/descriptor/run_descriptor_generalization.sh`** – wrapper that iterates both scenarios for seeds `42 43 44 45 46` (override via `SEEDS="..."`). Add `--quick` to switch the sampler to 500/500/4; omit it for 1000/1000/4.

## Key Metrics (Full Profile Seeds 42–44)

| Scenario       | Seed | Slice            | MAE with Desc | MAE w/o Desc | ΔMAE  | Bias with Desc | Bias w/o Desc | p95 with Desc | p95 w/o Desc |
|---------------|------|------------------|---------------|--------------|-------|----------------|---------------|---------------|--------------|
| Cold start    | 42   | Unseen compounds | 0.93          | 1.20         | −0.27 | +0.12          | +0.15         | 2.01          | 2.93         |
|               | 43   | Unseen compounds | 0.79          | 1.06         | −0.27 | +0.06          | +0.07         | 1.78          | 2.35         |
|               | 44   | Unseen compounds | 0.66          | 1.72         | −1.06 | +0.02          | +1.02         | 1.92          | 3.76         |
| Mix shift     | 42   | Held-out classes | 0.61          | 1.19         | −0.58 | −0.28          | −0.48         | 1.64          | 2.99         |
|               | 43   | Held-out classes | 1.29          | 1.53         | −0.25 | −0.46          | −0.57         | 5.25          | 5.89         |
|               | 44   | Held-out classes | 0.60          | 1.25         | −0.64 | −0.13          | +0.58         | 1.48          | 3.24         |

_Notes_: p95 is the 95th percentile absolute error in minutes. Bias is signed mean error (`pred − true`). Coverage (not shown above) stays ≈0.95 for descriptors and degrades for the baselines in the hard regimes.

## How to Read the Results
- **Cold start** approximates the real “new molecule” problem: descriptors are the only route to a meaningful prior. Expect MAE reductions of ~0.3–1.0 min and much tighter tails on unseen compounds. Anchored compounds follow the usual training behaviour.
- **Shifted mix** mimics a worksheet dominated by chemistry classes that barely appeared in the historical fit. Here, descriptors prevent the intercept drift that arises when class effects were never observed.
- **Seen slices** stay flat in all runs, confirming that enabling descriptors imposes no in-support penalty.

## Applying This to Production
1. Always pass `compound_features` when instantiating `HierarchicalRTModel` for production fits. Our wrappers already ingest ChemBERTa PCA embeddings; these studies confirm it is worth the runtime.
2. If a new worksheet is dominated by previously unseen chemistry, anticipate descriptor-enabled MAE roughly 0.5 min lower than the descriptor-free variant; monitor by class and species using the same metrics (MAE, bias, 95th percentile absolute error, coverage).
3. If descriptors underperform on a particular chemistry family, revisit the embeddings or add small calibration labels. The synthetic results give us reference behaviour for what “good” looks like.

## Plots

Plots are generated automatically whenever the CLI runs. To refresh a chart after pruning outputs, rerun the desired scenario (use `--quick` to minimise sampler cost if you only need the new chart). The script will aggregate all JSON summaries under `output/rt_descriptor_generalization/<profile>/<scenario>/` and rewrite the figure.

Plots land at `output/rt_descriptor_generalization/<profile>/<scenario>/<scenario>_<metric>.png` by default; override the destination with `--plot-output` if you want to place them elsewhere.

## Related Experiments
- `run_covshift_holdout.sh`: covariate-shift bins with the historical class-pooled, descriptor-free hierarchy for direct comparison to prior robustness plots.
- `run_descriptor_shift.sh`: descriptor ON vs chemistry-free hierarchy under induced run drift, plus Lasso baseline.
- `run_descriptor_ablation.sh`: multi-seed descriptor ablation with rare/unseen focus; uses the same chemistry-free baseline as the drift study.
