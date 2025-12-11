# Local eslasso (Sally `local_models`) species mapping

This documents the *exact* mapping logic used to choose a Sally per-matrix eslasso model from a
`species_raw` string when evaluating the **local** lasso baseline.

Source code: `scripts/pipelines/eval_rt_lasso_local_models_by_species_cluster.py` (function
`_infer_species_matrix_from_species_raw`).

## Available local model bundles

The evaluator only knows about the following local bundle keys (directory names under
`external_repos/sally/local_models/eslasso/`):

- `human_plasma`
- `human_urine`
- `human_fecal`
- `human_cells`
- `rat_plasma`
- `rat_liver`

## Mapping algorithm (exact)

Given a `species_raw` value (string), we produce either one of the bundle keys above, or `None`
(meaning “no local model applies”).

1. **Direct key match (lib209 often uses this)**  
   Let `normalized = species_raw.strip().lower()`. If `normalized` is exactly one of the 6 bundle
   keys above, return it.

2. **Ontology parse (lib208 often uses this)**  
   Split the string on `+` into tokens: `parts = species_raw.split('+')`. Let:
   - `p0 = parts[0].upper()` (first token)
   - `p1 = parts[1].upper()` (second token, if present)
   - `upper = species_raw.upper()`
   - `is_human = ('HUMAN' in upper) or ('HOMO SAPIENS' in upper)`
   - `is_rodent = ('RAT' in upper) or ('MOUSE' in upper)`

   Then apply the rules in order:

   - **Plasma/serum**: if `p0` is one of `{PLASMA, SERUM, EDTA PLASMA, EDTA-PLASMA}`:
     - if `is_human` → `human_plasma`
     - else if `is_rodent` → `rat_plasma`
     - else → `None`
   - **Urine**: if `'URINE' in p0` and `is_human` → `human_urine` (otherwise `None`)
   - **Fecal**: if `p0` is one of `{FECES, FECAL}` and `is_human` → `human_fecal` (otherwise `None`)
   - **Cells**: if `'CELL' in p0` and `is_human` → `human_cells` (otherwise `None`)
   - **Rodent liver (explicit tissue token)**: if `('TISSUE' in p0) and ('LIVER' in p1)` and
     `is_rodent` → `rat_liver` (otherwise `None`)
   - **Rodent liver (fallback)**: if `'LIVER' in upper` and `is_rodent` → `rat_liver`
   - Else → `None`

## Lib209: exact `species_raw` → local model key

The following `species_raw` values occur in
`repo_export/lib209/species_mapping/merged_training_all_lib209_species_mapping.csv` and map to a
local model key:

### `human_plasma`

- `human_plasma`

### `human_urine`

- `human_urine`

### `human_fecal`

- `human_fecal`

### `human_cells`

- `human_cell_extract`
- `human_cells`
- `human_cho_cells`
- `human_mammalian_cells_(non-cho)`

### `rat_plasma`

- `rat_plasma`

### `rat_liver`

- `rat_liver`

## Lib208: exact `species_raw` ontology string → local model key

The following `species_raw` values occur in
`repo_export/lib208/species_mapping/merged_training_all_lib208_species_mapping.csv` and map to a
local model key:

### `human_plasma`

- `PLASMA+UNKNOWN COAGULANT+HUMAN+HOMO SAPIENS`
- `SERUM+SERUM+HUMAN+HOMO SAPIENS`

### `human_urine`

- `URINE+DRE SEDIMENT+HUMAN+HOMO SAPIENS`
- `URINE+DRE URINE+HUMAN+HOMO SAPIENS`
- `URINE+URINE+HUMAN+HOMO SAPIENS`

### `human_fecal`

- `FECES+UNSPECIFIED TYPE+HUMAN+HOMO SAPIENS`

### `human_cells`

- `CELL EXTRACT+MIXED+HUMAN+HOMO SAPIENS`
- `CELL EXTRACT+STEM+HUMAN+HOMO SAPIENS`
- `CELL EXTRACT+UNSPECIFIED TYPE+HUMAN+HOMO SAPIENS`
- `CELLS+LEUKOCYTES+HUMAN+HOMO SAPIENS`
- `CELLS+UNSPECIFIED TYPE+HUMAN+HOMO SAPIENS`

### `rat_plasma`

- `PLASMA+UNKNOWN COAGULANT+MOUSE+MOUSE`
- `PLASMA+UNKNOWN COAGULANT+RODENT+RAT`
- `SERUM+SERUM+MOUSE+MOUSE`
- `SERUM+SERUM+RODENT+RAT`

### `rat_liver`

- `TISSUE EXTRACT+LIVER+RODENT+RAT`
- `TISSUE+LIVER+MOUSE+MOUSE`
- `TISSUE+LIVER+RODENT+RAT`

## Notes

- This mapping is used **only** to select a local lasso bundle per `sampleset_id`. Results are
  later aggregated by `species_cluster` for plotting, but `species_cluster` is *not* used to choose
  the local model.
- Any `species_raw` value not listed above maps to `None` (meaning: that row is not scored by the
  local lasso baseline).
- Even if `species_raw` maps to a bundle key, an individual row is only scored if:
  - the bundle has a `regression_<lib>.csv`,
  - the dataset contains all covariates required by that bundle (otherwise the bundle is skipped),
  - and the row’s `comp_id` exists in that bundle’s regression table.
