## RT production: evaluation on realtest (ridge coefficient summaries)

This document describes the current RT production evaluation flow on the real held‑out test set
(`realtest`) using the ridge coefficient‑summary models used for peak assignment.

If you need to regenerate the `repo_export/` RT CSVs (cap datasets and `realtest`), see
`docs/RT_PROD_PIPELINE.md`.

---

### 1. Expected inputs

Per library (e.g. 208/209):

- Training CSV (cap dataset), e.g.:
  - `repo_export/lib208/cap100/merged_training_all_lib208_cap100_chemclass_rt_prod.csv`
  - `repo_export/lib209/cap100/merged_training_all_lib209_cap100_chemclass_rt_prod.csv`
- Real held‑out test CSV:
  - `repo_export/lib208/realtest/merged_training_realtest_lib208_chemclass_rt_prod.csv`
  - `repo_export/lib209/realtest/merged_training_realtest_lib209_chemclass_rt_prod.csv`

---

### 2. Train models (cap → stage‑1 coefficient summaries)

```bash
./scripts/run_rt_prod.sh --libs 208,209 --cap 100
```

Use `--quick` for a smoke run:

```bash
./scripts/run_rt_prod.sh --libs 208,209 --cap 100 --quick
```

This writes a run directory under `output/` and updates:

- `output/rt_prod_latest.txt`

---

### 3. Evaluate on realtest + regenerate plots

```bash
./scripts/run_rt_prod_eval.sh --libs 208,209 --cap 100
```

This evaluates all available models in the run directory on `realtest` and writes:

- per‑model metrics JSON:
  - `.../results/rt_eval_coeff_summaries_by_support_realtest.json`
- plots (global + by support + by species_cluster):
  - `<run-dir>/plots/*.png`

---

### 4. Notes

- The recommended model for peak assignment is the partial pooling PyMC ridge model.
- The sklearn ridge baseline is useful for debugging but typically under‑covers (under‑calibrated)
  at nominal 95% intervals.

