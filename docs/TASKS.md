# Tasks

## RT Ridge Models (cap100)

### Status (partial pooling ridge, 2025-12-25)

- Recommended model: PyMC ridge with partial pooling (exports `Stage1CoeffSummaries`).
- Baselines: PyMC ridge (supercategory) and sklearn ridge (supercategory). Optional legacy lasso baselines exist but are not required.
- Reproduce (cap100 → realtest):
  - `./scripts/run_rt_prod.sh --cap 100 --libs 208,209`
  - `./scripts/run_rt_prod_eval.sh --cap 100 --libs 208,209`
  - `./scripts/plot_rt_multilevel.sh --cap 100 --libs 208,209`
- Report: `docs/models/rt_pymc_multilevel_pooling_report.pdf`

## RT Multi-Level Curate + Supercategory (new)

Goal

- Build a single RT model that can behave like:
  - a curate model (per species/matrix phrase) when data are sufficient, and
  - a supercategory model (pooled across related phrases) when data are sparse,
  without hard relabel-and-merge.

Key observation (repo_export)

- The production CSVs in `repo_export/lib{208,209}/cap*/..._rt_prod.csv` include both `species` and `species_cluster`.
- `repo_export/lib{208,209}/species_mapping/*_species_mapping.csv` clarifies semantics:
  - `species` is an integer encoding of `species_raw` (intended to be a curate-like subgroup id),
  - `species_cluster` is an integer encoding of `species_group_raw` (supercategory group).
- Important: for hierarchical pooling we need `species` to be nested under `species_cluster` (each `species` value belongs to
  exactly one `species_cluster`). lib209 already satisfies this when `species_raw = species_matrix_type`; lib208 needs a better
  `species_raw` definition than organism-only labels.
- Our current `Stage1CoeffSummaries` convention uses `(group_id << 32) + comp_id` keys; historically `group_id` was
  `species_cluster`.

What we should do (fix mapping semantics)

- Regenerate mappings with `scripts/pipelines/check_rt_metadata_mapping.py`:
  - lib209: `species_raw = species_matrix_type`, `species_group_raw = group`.
  - lib208: `species_raw = JCJ_COMBO`, `species_group_raw = group`.
  - The script enforces nesting (each `species_raw` belongs to exactly one `group`) and drops rows with missing group.
- For reproducibility, use the end-to-end prep runner:
  - `bash scripts/pipelines/run_rt_multilevel_data_prep.sh --libs 208,209 --caps 100`

Concrete spec (collapsed ridge; explicit intercept)

- Observations (rows): `rt_i` with run covariates `x_i` (IS/RS/ES + optional poly2 terms).
- Group for coefficient artifact: `g = (group_id, comp_id)` where `group_id` is defined by `--group-col`, e.g.:
  - `species_cluster` (supercategory-only; status quo),
  - `species` (curate-like subgroup nested within supercategory).
- Likelihood: `rt_i ~ Normal(b_g + x_i·w_g, sigma_y)`.
- Slopes: `w_g` are *collapsed* analytically with ridge precision `lambda_diag` (same as existing implementation).
- Intercepts: keep explicit in PyMC so we can add hierarchy, but export as the implied intercept in `Stage1CoeffSummaries`
  (same export path as existing explicit-intercept models).

Supercategory-aware pooling (intended behavior)

- When `group_id` maps deterministically to `species_cluster` (i.e. `group_col=species` with a nested subgroup mapping),
  add a supercategory hierarchy for subgroup-level effects:
  - Subgroup intercept offsets pool within `species_cluster`, with *supercategory-specific shrinkage* so “homogeneous” supercats
    pool strongly while heterogeneous ones pool weakly.
  - Optional: do the same for a slope-head (mean drift response) so sparse subgroups borrow a stable drift response.

Planned implementation (this repo)

- Add `--group-col {species_cluster,species}` to training + evaluators and store it in the coefficient artifact so
  scoring knows how to compute `group_id` from CSV columns.
- Add supercategory-aware priors for species-level effects (new intercept/slope-head variants) that consume the `species_cluster`
  parent id when `group_id` has a unique mapping.
- Keep inference via `--method advi|nuts|map` (NUTS recommended only for small `--max-groups` sanity checks).

Smoke status

- Verified on `lib208/cap5` that `--group-col species --intercept-prior comp_hier_supercat --slope-head-mode cluster_supercat`
  trains and evaluates end-to-end with ADVI on a small subset (`--max-train-rows/--max-groups`).

Next experiments (tail-first; cap5 → realtest)

- Train three variants (same features/cap; compare tail-by-support):
  - baseline grouping: `--group-col species_cluster` (status quo),
  - subgroup grouping only: `--group-col species` (no pooling),
  - subgroup grouping + pooling: `--group-col species --intercept-mode explicit --intercept-prior comp_hier_supercat --slope-head-mode cluster_supercat`.
- Evaluate each with `scripts/pipelines/eval_rt_coeff_summaries_by_support.py` on realtest (streaming; start with `--max-test-rows 200000`).

## Covariate‑Shift: Model Change Discussion

Should we change the model?

- Not before tightening the experiment. It’s better to (1) keep fixed‑K as a fairness control for this benchmark, (2) use inferred (noisy) drift groups for both the hierarchical model and the cluster baseline instead of oracle ids, and (3) enable chemistry descriptors in the hierarchy so it leverages chemistry signal. Optionally add slope heterogeneity in the generator (non‑zero `species_gamma_sd`) to reflect matrix interactions. Then re‑assess.

- If evidence shows species/matrix‑specific slope heterogeneity hurts hierarchical performance, consider extending the model with random slopes:
  1) Species random slopes: γ[compound, :] + δγ_species[species, :] with shrinkage (adds flexibility at the cost of variance/compute).
  2) Cluster random slopes: γ[compound, :] + δγ_cluster[species_cluster, :] (fewer params; aligns with drift groups).
  3) Class×cluster slope pooling: δγ_cluster_class[class, :] for stronger structure, but heavier and riskier to identify.

- Trade‑offs of adding slope heterogeneity
  - Pros: captures matrix‑dependent covariate response; could improve far‑bin generalisation when drift differs across species.
  - Cons: more parameters, slower NUTS, and potential confounding with species intercepts; needs strong shrinkage and careful validation.

- Practical next steps
  - Improve realism first within a single default profile: enable descriptors, keep fixed‑K, infer species drift clusters from IS features for both methods, and introduce modest slope heterogeneity in the generator.
  - Re‑run covariate shift; if hierarchy lags in far bins with heterogeneity on, prototype a cluster‑random‑slope head and compare.
  - Keep the base model as default; gate advanced slope variants behind a flag only if we see consistent lift.

## Action Plan (Single Default Profile)

- Script updates (covariate shift runner)
  - NOTE: covariate shift + descriptor experiments are currently disabled pending a port to the RT ridge pipeline.
  - TODO: re-implement these studies using `Stage1CoeffSummaries` artifacts (PyMC partial pooling + sklearn baseline).
  - Keep `fixed_runs_per_species_compound=10` for fairness and comparability in this benchmark.
  - Infer species clusters from IS features and feed the inferred labels to BOTH:
    - the RT ridge model (for pooling), and
    - the cluster×compound baseline (for its grouping key),
    replacing oracle cluster ids.
    - Approach: aggregate per‑species mean over `run_covariate_*`, standardise, KMeans with K = `hp["n_clusters"]` and fixed seed.
  - Add a small species‑specific slope heterogeneity in the generator (default `species_gamma_sd=0.1`) to reflect realistic matrix interactions while keeping the problem well‑conditioned.

- Experimental settings (default)
  - Anchors: 0, 3, 5 (unchanged).
  - Descriptors: ON (hierarchy receives `compound_features`).
  - Cluster labels: inferred from IS features (for both methods).
  - Slope heterogeneity: `species_gamma_sd = 0.1`.
  - Scarcity: fixed K per (species, compound) retained (`fixed_runs_per_species_compound=10`).
  - Replicates: 5 seeds; runs per species: 8 (unchanged).

- Expected outcomes
  - With descriptors on and inferred clusters, the hierarchy should improve vs the cluster baseline (especially in 80–95% bins), while species×compound remains worst.
  - Retains fairness via fixed K but avoids oracle pooling and enables chemistry, improving realism.

- Metrics and diagnostics
  - Primary: MAE per bin ± 95% CI across replicates.
  - Secondary: coverage and z‑score diagnostics for hierarchy; fraction of NaN predictions for baselines (missing keys).
  - Directional goal: hierarchy achieves a clear margin (≈10–20% lower MAE) over cluster baseline in far bins under the default profile.

- Risks and mitigations
  - If inferred clusters are too strong (near‑oracle), reduce K or cluster on per‑run then map to species by majority vote to inject noise.
  - If MCMC slows with descriptors, keep current `quick` sampler settings and rely on the fixed‑K dataset to remain tractable.

- Timeline
  - Code changes and defaults: ~0.5 day.
  - Quick run to validate: ~15–20 minutes (unchanged order of magnitude).
  - Doc/plot refresh: ~0.5 day.

## Implementation Notes

- Current RT ridge models (Stage-1 coefficient summaries):
  - Partial pooling: `src/compassign/rt/pymc_partial_pool_ridge.py`
  - Supercategory baseline: `src/compassign/rt/pymc_supercategory_ridge.py`
  - Shared collapsed-slope implementation: `src/compassign/rt/ridge_stage1.py`

- Legacy experiment runners under `scripts/experiments/rt/` that depended on the removed hierarchical RT model are now
  disabled stubs and must be ported before use.

## Investigate Species×Chemistry Structure (Future Work)

Motivation

- Our current descriptor usage informs a global compound baseline (β_c ≈ Z_c·θ_β + δ_c) shared across species. Species effects are intercept‑only and independent of descriptors, and γ slopes (run covariates) are per‑compound (optionally class‑pooled) but not species‑specific.
- As a result, descriptors primarily help when compounds are globally scarce or OOD (cold‑start, held‑out classes, drift). They are not designed to help when scarcity is “within a species” because there is no species×chemistry interaction.

Proposal (model variants to explore)

- Species‑specific descriptor coefficients (most direct):
  - β_{c,s} = β_c + Z_c·θ_{β,s} + ε, with θ_{β,s} ~ N(θ_β, Σ) to pool across species.
  - Adds explicit species×chemistry structure so descriptors can help when a compound is rare in species s even if well labeled globally.
- Species‑varying γ (run covariate response) with descriptor gating:
  - γ_{c,s} = γ_c + f_s(Z_c), where f_s is a low‑rank or linear map with shrinkage.
  - Captures species‑specific matrix interactions that depend on chemistry.
- Cluster‑level compromise:
  - Share θ_{β,g} at species‑cluster level (inferred from IS), with species deviations θ_{β,s} ~ N(θ_{β,g}, ·). Fewer parameters than free per‑species heads.

Shortcomings and risks

- Parameter explosion and compute:
  - Per‑species descriptor heads scale as O(n_species × d). With d≥8 and >10 species, this grows quickly; NUTS runtime and memory will increase and can destabilize adaptation.
- Identifiability and regularization:
  - New interactions can confound with species intercepts and compound residuals. Requires strong hierarchical priors (e.g., centered at shared θ_β, tight HalfNormal scales) and possibly dimension‑reduced Z.
- Data requirements:
  - Need sufficient per‑species coverage across chemistry to learn θ_{β,s} meaningfully. Synthetic generator and production regimes may not support this without dedicated scenarios.
- Engineering complexity:
  - Larger model surface (more code paths), trickier PPC and diagnostics, and harder to maintain. Should be gated behind an explicit flag and isolated in experiments first.

Evaluation plan (if pursued)

- Add a species‑rare split: define rarity per (species, compound) by counting train rows per pair and selecting a low threshold (e.g., ≤2). Report metrics on this subset explicitly.
- Compare three arms: full (cov+desc), cov+no‑desc, and desc with species×chemistry head(s). Keep covariate settings identical.
- Start with reduced descriptor dimension (e.g., PCA of Z to d=4–8) and strong shrinkage; run quick sampler first, then confirm with full sampler.

Recommendation

- Do not change the default RT model yet. Prototype species×chemistry variants in a dedicated experiment with clear gating and only adopt if they deliver consistent improvements on the species‑rare slice without unacceptable compute or instability.
