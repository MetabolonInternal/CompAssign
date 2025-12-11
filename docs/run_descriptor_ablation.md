# Descriptor Ablation Experiment (`run_descriptor_ablation.sh`)

This experiment isolates the contribution of chemical descriptors by running the hierarchical RT model with and without descriptors across anchor, rare, and unseen compounds. It is our quantitative reference for descriptor lift on sparse chemistry.

## Model Arms
- **With descriptors**: hierarchical RT model supplied with `compound_features` from the generator, class pooling enabled (`include_class_hierarchy=True`), and standard γ pooling.
- **No descriptors / no classes**: descriptors OFF (`compound_features=None`), collapse to a single compound class (`n_classes=1`), and disable class pooling (`include_class_hierarchy=False`). This ensures the control arm contains no chemistry signal in either β or γ.

## Experiment Design
1. **Synthetic cohort**: 60 compounds, 15 species, 10 internal standards (quick profile). Compounds are stratified into anchors, rare, and unseen groups using descriptor-space clustering.
2. **Label budgets**:
   - Anchors: 3–5 labelled observations, split 80/20 between train/test for stability.
   - Rare compounds: ≤2 labels routed to train, remainder to test with cross-run holdout.
   - Unseen compounds: all labels withheld for testing.
3. **Train/test split**: `split_group_aware` enforces cross-run splits for rare compounds and leaves unseen compounds test-only.
4. **Fitting**: sample both arms with identical sampler settings (quick: 500/500/4; full: 1000/1000/4).
5. **Evaluation**:
   - Overall metrics (MAE/RMSE/R²/median AE) for each arm.
   - Stratified metrics for rare vs non-rare support; group-wise anchor/rare/unseen splits.
   - MAE vs nearest-anchor distance bins.
   - Rare-focused diagnostics (nearest-anchor distance, gamma class support, train/test run counts).
6. **Aggregation**: the wrapper script collates per-seed JSON summaries, computes aggregate deltas, and writes `descriptor_ablation_summary.json`.

## Commands
- Single seed (quick/full):  
  ```bash
  poetry run python scripts/experiments/rt/descriptor/assess_descriptor_ablation.py --quick --seed 42
  poetry run python scripts/experiments/rt/descriptor/assess_descriptor_ablation.py --seed 42 --n-samples 1000 --n-tune 1000 --n-chains 4
  ```
- Multi-seed wrapper (seeds 42–46 by default):  
  ```bash
  ./scripts/experiments/rt/descriptor/run_descriptor_ablation.sh --quick
  ./scripts/experiments/rt/descriptor/run_descriptor_ablation.sh
  ```
  The wrapper creates `output/descriptor_ablation/run_<N>/`, runs each seed, and aggregates into `descriptor_ablation_summary.json`.

## Outputs
Per seed:
- `descriptor_ablation_seed_<seed>.json`: configuration, overall metrics, rare/non-rare stratification, anchor/rare/unseen group metrics, distance bins, rare diagnostics.
- `descriptor_ablation_seed_<seed>.png`: bar chart of MAE by group (descriptor vs no descriptor).

Aggregate (wrapper):
- `descriptor_ablation_summary.json`: number of runs, mean ΔMAE for rare compounds (no-desc − desc), 95% CI, win rate, and overall MAE averages for both arms.

## Typical Findings (quick profile, seeds 42–46)
- Rare compounds: descriptors reduce MAE by ~0.3–0.6 min and improve win rate (>80%).
- Unseen compounds: descriptors provide usable zero-shot baselines; the chemistry-free arm drifts by >1 min.
- Anchors/non-rare: both arms converge; descriptors show no penalty when data is plentiful.

## Sampling & Embedding Reference
The synthetic generator uses 20D SMILES embeddings (ChemBERTa PCA20 or ECFP SVD20) and k-means clustering to stratify chemistry.
- Keep the embedding library cleaned and whitened; version encoder and PCA/SVD artefacts.
- For a cohort of ~60 compounds (quick profile), use ~16 clusters with minimum 2 compounds per cluster; anchors ≈50%, rare ≈25%, unseen remainder.
- Ensure total anchor labels ≳8–10×embedding dimension to learn the descriptor map (`θβ`) robustly.
- Maintain nearest-neighbour pairing inside clusters so every rare/unseen compound has chemically similar anchors.

## Related Experiments
- `run_descriptor_generalization.sh`: descriptor ON vs OFF (with class pooling) for cold-start and mix-shift.
- `run_descriptor_shift.sh`: descriptor hierarchy vs chemistry-free hierarchy under induced run drift.
- `run_covshift_holdout.sh`: historical covariate-shift study (class pooling, no descriptors) for comparison with earlier robustness plots.
