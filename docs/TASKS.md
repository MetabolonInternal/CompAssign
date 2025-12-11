# Tasks

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
  - Always use descriptors for the hierarchical model: pass chemistry features from the generator to `HierarchicalRTModel` instead of `None`.
  - Keep `fixed_runs_per_species_compound=10` for fairness and comparability in this benchmark.
  - Infer species clusters from IS features and feed the inferred labels to BOTH:
    - the hierarchical model (for species‑intercept pooling), and
    - the cluster×compound baseline (for its pooling key),
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

- Hier model uses species clusters for intercept pooling and chemistry for slope pooling:
  - Species intercept prior: `src/compassign/rt_hierarchical.py:297`
  - γ slopes pooled by chemistry: `src/compassign/rt_hierarchical.py:346-355`
  - Mean assembly (what is intercept vs slope): `src/compassign/rt_hierarchical.py:358-363`

- Covariate shift runner changes:
  - Generator call and fixed‑K retained: `scripts/experiments/rt/covariate_shift/assess_covariate_shift_holdout.py:151-160`
  - Hier instantiation and descriptors enabled: `scripts/experiments/rt/covariate_shift/assess_covariate_shift_holdout.py`
  - New species‑cluster inference injected for both methods before training.

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
