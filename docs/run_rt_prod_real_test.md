## RT production: evaluation on real held-out test data

This document describes how to evaluate the hierarchical RT model on the
“real” held-out test data from Pachyderm, using the existing cap-5-trained
models and the cap-10 evaluation tooling.

The high-level flow (per library, e.g. 208/209) is:

1. Pull raw training exports from Pachyderm.
2. Run the repository merge pipeline to build per-lib merged Parquet.
3. Filter those merged rows using the global train/test split file.
4. Convert the filtered rows into RT production CSVs matching the cap-5/10 schema.
5. Apply the trained RT model and evaluate per-species and per-species-group metrics.

The steps below assume the repository root is the working directory.

---

### 1. Inputs

- **Train/test split map**
  - `data/split_outputs/train_test_split_all.csv`
  - Contains the global split assignment and group information used for
    the original training runs.

- **Pachyderm exports**
  - Raw (unmerged) training data for each library, identified by commit:
    - lib 209: `95661767ab0e4fe28aaeacca8230449b`
    - lib 208: `7210c028155f4241b2ad3132c004fc01`
  - These correspond to the `egress_regression_csvs` commit that should
    match the merged Parquets currently in `repo_export`.

- **Existing RT models**
  - Trained on cap-5 RT production CSVs, e.g.:
    - `output/rt_prod/lib208_cap5`
    - `output/rt_prod/lib209_cap5`
  - Each directory contains `config.json` and `models/rt_trace.nc`.

---

### 2. Pull and merge Pachyderm training outputs

For each library (208, 209):

1. Use `pachctl` (or the relevant client) to pull the `create-training_data`
   step outputs for the commits above into a local directory, e.g.:

   ```bash
   # Pseudocode – adjust repo/branch/pipeline names as appropriate
   pachctl get file egress_regression_csvs@95661767ab0e4fe28aaeacca8230449b:/path/in/pfs/* \
     > data/pachyderm/lib209_raw_*.csv
   pachctl get file egress_regression_csvs@7210c028155f4241b2ad3132c004fc01:/path/in/pfs/* \
     > data/pachyderm/lib208_raw_*.csv
   ```

2. Run the same merge pipeline that produced `merged_training_*_lib<lib>.parquet`
   in `repo_export`. This typically involves:

   - Normalising column names and types.
   - Joining any auxiliary mapping tables (e.g. compound mapping, species mapping).
   - Writing a per-lib `merged_training_<hash>_lib<lib>.parquet` that matches the
     existing schema used by `run_rt_prod_prep.py`.

   If the existing merge script already supports taking a raw directory and
   producing these Parquets, re-use that entrypoint; otherwise, mirror the
   transforms currently used for the cap pipelines.

---

### 3. Filter merged training data using the global split

Once per-lib merged Parquets exist for the real training export:

1. Load `data/split_outputs/train_test_split_all.csv` and identify the join key
   used to match rows to the split. Common choices are:

   - `sample_set_id` (often mapped to `sample_set_id` / `sampleset_id`),
   - optionally `lib_id` if multiple libraries share the same sample set ids.

2. For each library’s merged Parquet:

   - Join with `train_test_split_all.csv` on the chosen key(s).
   - Filter to rows with `split == "test"`.
   - Respect the `group` column when needed (e.g. if multiple test groups
     exist and only specific ones should be used for RT evaluation).

3. Sanity-check the filtered set:

   - Row counts per `species_group_raw` (e.g. “1 human blood”, “4 human / animal urine”).
   - Overlap with the species/compound combinations seen in the cap-5 training
     data – expect some test combinations to be unseen and dropped later.

Write the filtered per-lib test sets back to `repo_export`, e.g.:

```bash
repo_export/merged_training_realtest_lib208.parquet
repo_export/merged_training_realtest_lib209.parquet
```

---

### 4. Build real-test RT production CSVs

The goal is to produce RT CSVs with the same schema as the cap-5/10 RT
production CSVs used by `train_rt_prod.py`:

- Columns:
  - `sampleset_id, worksheet_id, task_id`
  - `species, species_cluster`
  - `compound` (chemical_id), `compound_class`
  - `IS_*` covariates and any other numeric covariates
  - `rt` (from `apex_rt`)

For each lib:

1. Ensure the filtered real-test Parquet includes:

   - `sample_set_id, worksheet_id, task_id`
   - `apex_rt`
   - `chemical_id`
   - `compound_class`
   - IS_* covariates

2. Reuse `scripts/pipelines/make_rt_prod_csv_from_merged.py` on the filtered
   Parquet plus the existing species mapping:

   ```bash
   python scripts/pipelines/make_rt_prod_csv_from_merged.py \
     --input repo_export/merged_training_realtest_lib208.parquet \
     --species-mapping repo_export/merged_training_<hash>_lib208_species_mapping.csv \
     --output repo_export/merged_training_realtest_lib208_chemclass_rt_prod.csv

   python scripts/pipelines/make_rt_prod_csv_from_merged.py \
     --input repo_export/merged_training_realtest_lib209.parquet \
     --species-mapping repo_export/merged_training_<hash>_lib209_species_mapping.csv \
     --output repo_export/merged_training_realtest_lib209_chemclass_rt_prod.csv
   ```

3. Confirm the resulting CSVs have the same column structure as the cap-5/10
   RT production CSVs already used in training (only the row set should differ).

---

### 5. Evaluate RT models on real-test RT CSVs (hierarchical + lasso)

With the cap-5-trained models already saved under `output/rt_prod`, use the
cap-10 evaluation script (which works for any RT production CSV with the
correct schema).

For hierarchical (PyMC) models we use the streaming evaluator, and for
the baseline we use the legacy eslasso models from sally.

Convenience driver:

```bash
# Both models (hierarchical + lasso) on cap-10 and real-test, libs 208/209
scripts/run_rt_prod_eval.sh

# Hierarchical only
scripts/run_rt_prod_eval.sh --only-hierarchical

# Lasso only
scripts/run_rt_prod_eval.sh --only-lasso
```

Hierarchical training includes ES_* covariates for Group 1 (`1 human blood`)
by default when using the `run_rt_prod.sh` wrapper. To enable the same
behaviour when calling the training script directly, pass:

```bash
python scripts/train_rt_prod.py \
  --data-csv repo_export/merged_training_..._lib208_cap10_chemclass_rt_prod.csv \
  --include-es-group1 \
  --quick
```

Only the Group 1 ES columns (per lib) are used; they are zero-filled for other
groups and the feature/group mask is recorded in `config.json` for consistent
masking at evaluation time.

Each hierarchical run writes:

- Global metrics:
  - `output/rt_prod/lib<lib>_cap5/results/rt_eval_cap10_streaming.json`
  - `output/rt_prod/lib<lib>_cap5/results/rt_eval_realtest_streaming.json`
    - RMSE / MAE / 95% coverage on the given dataset.
- Per-species metrics (label suffix `_cap10` or `_realtest`):
  - `rt_eval_streaming_by_species_<label>.csv`
  - `rt_eval_streaming_rmse_by_species_<label>.png`
- Per-species-group (matrix) metrics (label suffix `_cap10` or `_realtest`):
  - `rt_eval_streaming_by_species_group_<label>.csv`
  - `rt_eval_streaming_rmse_by_species_group_<label>.png`

Each lasso run writes:

- Global metrics:
  - `output/rt_prod/lib<lib>_cap5/results/rt_eval_lasso_cap10.json`
  - `output/rt_prod/lib<lib>_cap5/results/rt_eval_lasso_realtest.json`
- Per-species metrics (label suffix `_cap10` or `_realtest`):
  - `rt_eval_lasso_by_species_<label>.csv`
  - `rt_eval_lasso_rmse_by_species_<label>.png`
- Per-species-group metrics (label suffix `_cap10` or `_realtest`):
  - `rt_eval_lasso_by_species_group_<label>.csv`
  - `rt_eval_lasso_rmse_by_species_group_<label>.png`

A comparison helper:

```bash
python scripts/pipelines/compare_rt_models_by_group.py \
  --hier-csv output/rt_prod/lib208_cap5/results/rt_eval_streaming_by_species_group_realtest.csv \
  --lasso-csv output/rt_prod/lib208_cap5/results/rt_eval_lasso_by_species_group_realtest.csv \
  --label realtest
```

produces `rt_eval_compare_<label>_rmse_by_species_group.png` in the same
results directory (and is called automatically by `run_rt_prod_eval.sh`
when both models are evaluated).

These comparisons (cap-10 vs real-test; hierarchical vs lasso) show how
performance changes when moving from capped training data to truly held-out
test runs, and how the new hierarchical model stacks up against the
production eslasso baseline.

---

### 6. Observations: eslasso vs hierarchical RT model

On real-test data (per species_group_raw), the legacy eslasso baseline
currently outperforms the hierarchical model almost everywhere:

- Lib 208 real-test:
  - Hierarchical RMSE per group ≈ 0.015–0.031 minutes.
  - Lasso RMSE per group ≈ 0.012–0.022 minutes (lower for most groups).
- Lib 209 real-test:
  - Hierarchical RMSE per group ≈ 0.022–0.031 minutes.
  - Lasso RMSE per group ≈ 0.008–0.015 minutes (≈2–3× lower).

Possible reasons:

- Feature set:
  - The hierarchical model currently uses only IS* covariates.
  - eslasso uses both IS* and ES_* channels (endogenous surrogates) per
    supercategory, which add a lot of predictive power.
- Granularity:
  - eslasso trains a separate linear model per (supercategory, comp_id),
    with its own feature subset.
  - The hierarchical model shares information aggressively across compounds
    via `compound_class` and builds per-compound covariate effects `gamma`
    shrunk toward class-level means.
- Training data:
  - eslasso was trained on full-scale training exports.
  - The hierarchical model is trained on cap-5 RT data, then evaluated on
    much larger real-test sets.
- Split details:
  - The real-test split comes from `train_test_split_all.csv`.
  - eslasso’s original train/test split was done inside the Hippopotamus
    pipeline; these are intended to align, but any mismatch could make
    eslasso’s numbers slightly optimistic relative to the new model.

Taken together, this means the lasso baseline is a strong and meaningful
benchmark; the current hierarchical model is operating with less input
information (no ES), more structural constraints, and less training data
per compound.

---

### 7. Next steps / ideas for improving the hierarchical model

Potential follow-ups, roughly ordered by impact:

1. **Add ES covariates to the hierarchical model**
   - Extend `train_rt_prod.load_production_csv` and `HierarchicalRTModel`
     to accept a configurable set of ES_* features (per supercategory or
     per lib) instead of IS-only.
   - Mirror the eslasso feature selection by:
     - reusing the IS panel as-is, and
     - including the ES columns that are used in the regression CSVs
       for that supercategory, or a carefully chosen subset.

2. **Relax pooling for richly-observed compounds**
   - For compounds with large numbers of observations, reduce the
     shrinkage toward class-level `gamma_class` so that the hierarchical
     model can approach per-compound flexibility where data supports it.
   - For sparse compounds, keep stronger pooling.

3. **Align training data scale**
   - Train a “full-scale” hierarchical model (or a close approximation)
     using more than cap-5 per pair, and re-evaluate on real-test to
     see how much of the gap is due to training data size vs model form.

4. **Double-check split alignment**
   - Confirm that `train_test_split_all.csv`’s test fold matches the
     intended eslasso test set (by sample_set_id + group).
   - If needed, regenerate a split that is shared by both pipelines.

5. **Richer diagnostics**
   - For a small subset of compounds, plot eslasso vs hierarchical
     predictions directly (per run) to understand where they differ
     (e.g., specific chemistries or retention regions).
   - Look at calibration curves (predicted vs empirical coverage) for
     both models per group.

These tasks give a concrete roadmap for iterating on the hierarchical RT
model while using the eslasso baseline as a strong reference point.
