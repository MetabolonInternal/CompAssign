**RT Production Data Preparation Pipeline**

This document describes how to turn Pachyderm exports plus library metadata into
compact, chemistry‑aware RT training inputs that are compatible with
the production ridge RT pipeline (`./scripts/run_rt_prod.sh`).

The workflow is modular and reproducible; each step is a small script under
`scripts/pipelines` or `scripts/data_prep`.

---

### 1. Merge Pachyderm training shards

Starting point is a Pachyderm export directory that contains:

- `<export>/create_training_data/...`
- `<export>/combine_predictors/...`

Run the merger:

```bash
./scripts/pipelines/merge_pachyderm_training.py -i repo_export/<export_dir>
```

This produces a merged Parquet per export:

- `repo_export/merged_training_<export_hash>.parquet`

Each file contains:

- peak‑level RT and meta (`apex_rt`, `sample_set_id`, `task_id`, `worksheet_id`,
  `sample_type`, `cpd_status`, `comp_id`, `species_matrix_type`, etc.)
- combined predictors (`IS*`, `ES_*`, `RS*`), partitioned by `lib_id`.

---

### 2. Split merged Parquet by library

Large merged files typically contain multiple `lib_id` values. Split them into
per‑library Parquets:

```bash
python scripts/pipelines/split_merged_by_lib.py \
  repo_export/merged_training_<export_hash>.parquet
```

Outputs (examples):

- `repo_export/merged_training_<export_hash>_lib208.parquet`
- `repo_export/merged_training_<export_hash>_lib209.parquet`
- `repo_export/merged_training_<export_hash>_lib305.parquet`
- `repo_export/merged_training_<export_hash>_lib400.parquet`
- `repo_export/merged_training_<export_hash>_lib402.parquet`

Each file contains only a single `lib_id`.

---

### 3. Inspect species × compound coverage (diagnostic)

To understand how many compounds per species matrix type we have, and to plan
sampling, run:

```bash
python scripts/pipelines/summarize_species_compounds.py \
  repo_export/merged_training_*_lib*.parquet
```

For each input this prints:

- a table of `species_matrix_type, n_compounds, n_rows`
- the total number of unique `(species_matrix_type, comp_id)` pairs
- exact total rows retained if we cap to 10/20/50/100 rows per pair

It also writes per‑lib CSVs:

- `repo_export/merged_training_*_lib<lib>_species_compounds.csv`

containing all `(species_matrix_type, comp_id, n_rows)` combinations.

---

### 4. Down‑sample per species×compound (cap per pair)

The raw per‑lib Parquets can contain tens of millions of rows. For hierarchical
NUTS this is too large. We cap the number of rows per
`(species_matrix_type, comp_id)` pair using:

```bash
python scripts/pipelines/sample_per_species_compound.py \
  repo_export/merged_training_<export_hash>_lib<lib>.parquet \
  --cap-per-pair 5 \
  --seed 42
```

and optionally:

```bash
python scripts/pipelines/sample_per_species_compound.py \
  repo_export/merged_training_<export_hash>_lib<lib>.parquet \
  --cap-per-pair 10 \
  --seed 42
```

Outputs:

- `repo_export/merged_training_<export_hash>_lib<lib>_cap5.parquet`
- `repo_export/merged_training_<export_hash>_lib<lib>_cap10.parquet`

Properties:

- preserves all species×compound combinations
- caps to at most N rows per `(species_matrix_type, comp_id)` (N = 5 or 10)
- sampling is random within each pair (seeded for reproducibility)

Typical row counts at cap 5:

- lib 208: ~31k rows
- lib 209: ~65k rows
- lib 305: ~25k rows
- lib 400: ~36k rows
- lib 402: ~21k rows

These are much more manageable for MCMC.

---

### 5. Build global chemistry‑based classes (chem_id → compound_class)

We use 20D ChemBERTa PCA embeddings to define compound classes across all
chemicals:

```bash
python scripts/data_prep/build_chem_classes.py --n-clusters 32
```

Inputs:

- `resources/metabolites/embeddings_chemberta_pca20.parquet`
  (`chem_id, smiles, embedding20`)

Outputs:

- `resources/metabolites/chem_classes_k32.parquet`
- `resources/metabolites/chem_classes_k32.csv`

Both contain:

- `chem_id` (CHEM_ID)
- `compound_class` (0..31, from k‑means in embedding space)

Coverage is high (6,211 chem_ids), but some CHEM_IDs have no embedding (e.g.,
placeholder `0` and certain conjugate species).

---

### 6. Attach CHEM_ID + compound_class and filter unmapped chems

Each lib has a mapping from `comp_id` to CHEM_ID:

- `repo_export/lib_comp_chem_mapping_lib208.csv`
- `repo_export/lib_comp_chem_mapping_lib209.csv`
- `repo_export/lib_comp_chem_mapping_lib305.csv`
- `repo_export/lib_comp_chem_mapping_lib400.csv`
- `repo_export/lib_comp_chem_mapping_lib402.csv`

Columns:

- `lib_id, comp_id, chemical_id`

We enrich the cap‑sampled Parquets with CHEM_ID and `compound_class`, and drop
chemicals that cannot be mapped cleanly:

```bash
python scripts/pipelines/attach_chem_classes_and_filter.py \
  --input repo_export/merged_training_<export_hash>_lib<lib>_cap5.parquet \
  --lib-mapping repo_export/lib_comp_chem_mapping_lib<lib>.csv \
  --classes resources/metabolites/chem_classes_k32.parquet
```

Same for `_cap10.parquet` if desired.

This:

- joins `comp_id` → `chemical_id` via the lib mapping
- joins `chemical_id` → `compound_class` via chem classes
- drops rows where:
  - `comp_id` has no mapping, or
  - `chemical_id` has no class (including CHEM_ID 0 and several high‑ID
    conjugates without SMILES/embeddings)

Outputs:

- `repo_export/merged_training_<export_hash>_lib<lib>_cap5_chemclass.parquet`
- `repo_export/merged_training_<export_hash>_lib<lib>_cap10_chemclass.parquet`

Row counts after this step (cap 5, example):

- lib 208: 31,521 → 27,077 (drop unmapped comp_id) → 25,781 (drop no‑class chem_ids)
- lib 209: 64,861 → 52,436 → 47,266
- lib 305: 24,965 → 24,921 → 24,846
- lib 400: 35,657 → 31,499 → 30,106
- lib 402: 21,021 → 20,371 → 20,301

---

### 7. Build RT production CSVs (RT ridge inputs)

The production RT ridge trainers (driven by `./scripts/run_rt_prod.sh`) expect a CSV with:

- `sampleset_id, worksheet_id, task_id`
- `species, species_cluster`
- `compound, compound_class`
- numeric `IS_*` columns (and other covariates)
- `rt`

We generate these per lib from the filtered Parquets plus species mapping.

Species mapping CSVs (per export/lib) provide:

- `sample_set_id, species, species_cluster`

Examples present in this repo:

- `repo_export/merged_training_5684639a28c04bc5af7c4fd1a75e62b5_lib208_species_mapping.csv`
- `repo_export/merged_training_de194c2cc2114efaa1075ccf7539d0cb_lib209_species_mapping.csv`

Build the RT CSV from a chemclass Parquet:

```bash
python scripts/pipelines/make_rt_prod_csv_from_merged.py \
  --input repo_export/merged_training_de194c2cc2114efaa1075ccf7539d0cb_lib209_cap5_chemclass.parquet \
  --species-mapping repo_export/merged_training_de194c2cc2114efaa1075ccf7539d0cb_lib209_species_mapping.csv
```

This:

- joins `sample_set_id` → `species, species_cluster`
- builds:
  - `sampleset_id = sample_set_id`
  - `worksheet_id = worksheet_id`
  - `task_id = task_id`
  - `species, species_cluster` from species mapping
  - `compound = chemical_id`
  - `compound_class` from chem classes
  - `rt = apex_rt`
- appends numeric covariates (IS_*, ES_*, RS_*, etc.)

Outputs:

- `repo_export/merged_training_de194c2cc2114efaa1075ccf7539d0cb_lib209_cap5_chemclass_rt_prod.csv`
- similarly for libs 208, 305, 400, 402:
  - `*_lib208_cap5_chemclass_rt_prod.csv`
  - `*_lib305_cap5_chemclass_rt_prod.csv`
  - `*_lib400_cap5_chemclass_rt_prod.csv`
  - `*_lib402_cap5_chemclass_rt_prod.csv`

Approximate sizes:

- lib 208 cap5: ~25k rows, ~7 MB
- lib 209 cap5: ~47k rows, ~16 MB
- lib 305 cap5: ~25k rows, ~7.5 MB
- lib 400 cap5: ~30k rows, ~8.9 MB
- lib 402 cap5: ~20k rows, ~5.4 MB

These CSVs are ready for the production ridge training wrapper, e.g.:

```bash
./scripts/run_rt_prod.sh --libs 209 --cap 100 --quick
./scripts/run_rt_prod_eval.sh --libs 209 --cap 100
```

The `--quick` flag reduces ADVI steps for a smoke run.

---

### 8. Summary of end artifacts (cap 5)

Per lib, we end up with:

- cap‑sampled, chemistry‑aware Parquet:
  - `repo_export/merged_training_*_lib<lib>_cap5_chemclass.parquet`
- RT production CSV for the ridge trainers:
  - `repo_export/merged_training_*_lib<lib>_cap5_chemclass_rt_prod.csv`

These preserve:

- All `(species_matrix_type, comp_id)` pairs that have:
  - a lib comp→chem mapping, and
  - a ChemBERTa embedding + compound_class

At training time, `./scripts/run_rt_prod.sh` trains RT ridge models (including a partial pooling
PyMC model) directly from these CSVs and writes stage‑1 coefficient summaries used downstream
for evaluation and peak assignment.
