# Next Session Handover — Model Rigor Issues

This document tracks open issues found during review of the core pipeline (RT model, peak assignment model, data/feature flow, and evaluation), cross‑checked against `docs/bayesian_models.md` and recent training output. No fixes applied yet.

Format per item: ID, Title, Severity, Status, Summary, Evidence, Proposed Next Step.

## Ranked Issues

1) ID: EVA-001
   - Title: Compound-level metrics include training peaks (optimistic)
   - Severity: High
   - Status: Open
   - Summary: Compound Precision/Recall/F1 and coverage are computed using all peaks, including training-labeled peaks, whereas peak-level metrics correctly exclude training peaks. This likely inflates compound-level metrics (e.g., F1=1.000 in the sample run).
   - Evidence: In `PeakAssignment.assign`, compound mappings (`true_peaks_by_compound`, `pred_peaks_by_compound`) are built over all peaks; only peak-level metrics filter to test by skipping indices with training labels. `scripts/train.py` prints compound metrics without a test-only restriction.
   - Proposed Next Step: Restrict compound-level metrics and coverage to held-out/test peaks (mirror peak-level filtering) and re-report results.

2) ID: RTM-002
   - Title: RT posterior predictive checks (PPC) are skipped
   - Severity: High
   - Status: Open
   - Summary: The training script does not run PPC or residual diagnostics for the RT model, leaving predictive calibration of the RT component unverified.
   - Evidence: `scripts/train.py` passes `{}` to `create_all_diagnostic_plots`; console prints “Skipping PPC plots (no PPC results available)”. `rt_hierarchical.HierarchicalRTModel.posterior_predictive_check` exists but is unused.
   - Proposed Next Step: Run PPC after sampling, compute RMSE/MAE/95% coverage, save residual plots, and surface failures/poor coverage prominently.

3) ID: CFG-003
   - Title: RT `target_accept` CLI overridden to ≥0.99
   - Severity: High
   - Status: Open
   - Summary: The RT sampling uses `max(args.target_accept, 0.99)`, ignoring lower user-specified values and diverging from the command-line shown in logs.
   - Evidence: `scripts/train.py` sets `used_target_accept = max(args.target_accept, 0.99)` before calling `rt_model.sample`.
   - Proposed Next Step: Honor CLI parameter as-is, or clearly document and log the enforced minimum; ensure parity with the assignment model’s behavior.

4) ID: DAT-004
   - Title: RT covariates are random placeholders (docs mismatch)
   - Severity: High
   - Status: Fixed
   - Summary: Synthetic data now generates causal covariates (10‑dim descriptors and IS per species) and RTs consistent with the hierarchical model; train.py consumes these covariates directly. Features are standardized using training-only statistics in the RT model.
   - Evidence: `scripts/create_synthetic_data.py` now returns `(descriptors, internal_std)` and simulates RT via `μ0 + α_s + β_c + D_c·θ + γ·IS_s + ε` with sum‑to‑zero centering. `scripts/train.py` passes these into `HierarchicalRTModel` and `compute_rt_predictions`. `src/compassign/rt_hierarchical.py` standardizes descriptors/IS using only compounds/species present in `obs_df`.
   - Proposed Next Step: Optionally enable RT PPC in training to report RMSE/MAE/95% coverage and monitor posterior recovery of θ and γ; otherwise no action needed.

5) ID: PRI-005
   - Title: Presence priors not updated from labels in training
   - Severity: Medium
   - Status: Open
   - Summary: Presence priors are initialized but not updated using available training labels, whereas docs describe online updates from annotations.
   - Evidence: `PresencePrior` supports updates; `PeakAssignment.generate_training_data/build_model` do not call updates; `train.py` does not update priors during batch training.
   - Proposed Next Step: Decide policy (batch training updates vs. AL-only updates). If batch updates are desired, increment α for observed positives and β for negatives per species-compound during training.

6) ID: MCMC-006
   - Title: No convergence/divergence gating for RT model
   - Severity: Medium
   - Status: Open
   - Summary: Sampling completes without programmatic checks on divergences, R-hat, or ESS; issues are not surfaced as failures/warnings.
   - Evidence: `rt_hierarchical.sample` has a TODO noting disabled checks; diagnostic plots are generated but not enforced.
   - Proposed Next Step: Summarize `az.summary` post-sampling, gate on thresholds (e.g., r_hat ≤ 1.01, no divergences, minimum ESS), and halt or warn if violated.

7) ID: PRI-007
   - Title: Misleading method name `log_prior_odds` (returns log-probability)
   - Severity: Medium
   - Status: Open
   - Summary: Function name suggests log-odds but returns `log(pi)`. The math in code and docs uses `log π` as an additive offset, so behavior is correct; name risks confusion.
   - Evidence: `PresencePrior.log_prior_odds` returns `np.log(pi)`; docs add `log π` to logits.
   - Proposed Next Step: Rename (e.g., `log_prior_prob`) or add explicit docstring clarification.

8) ID: CAL-008
   - Title: ECE computed on top-class confidence only
   - Severity: Medium
   - Status: Open
   - Summary: ECE uses only the maximum probability per peak. This reflects confidence calibration but not full multiclass calibration.
   - Evidence: `PeakAssignment.assign` collects `top_probs` and passes them to `_calculate_ece`.
   - Proposed Next Step: Clarify the ECE definition in docs; optionally add multiclass ECE or one-vs-all variants if broader calibration is desired.

9) ID: UX-009
   - Title: Duplicate RT sampling banners in logs
   - Severity: Low
   - Status: Open
   - Summary: Both `train.py` and `rt_hierarchical.sample` print sampling info, leading to repeated lines.
   - Evidence: Console output shows two “Sampling with …” banners for RT.
   - Proposed Next Step: Print from one place to avoid redundancy.

10) ID: UX-010
    - Title: Step numbering duplication in training output
    - Severity: Low
    - Status: Open
    - Summary: “5. Results” and “5. Training complete” both labeled as step 5.
    - Evidence: `scripts/train.py` final console prints.
    - Proposed Next Step: Renumber for clarity.

## Notes on Code–Docs Alignment
- RT model: Non-centered hierarchies, sum-to-zero constraints, priors, and predictive variance match docs. Covariate generation and standardization now align with the documented approach (see DAT-004 status: Fixed).
- Peak assignment: Minimal 4-feature set, explicit null, presence prior as log-additive offset, hierarchical logit noise, masking, and exchangeability are implemented and tested (see `tests/test_exchangeability.py`). Presence prior “online updates” are not exercised in batch training (see PRI-005).

## Suggested Order of Work
1. EVA-001 (metrics fairness) — unblock trustworthy reporting.
2. RTM-002 (PPC) — validate uncertainty used downstream.
3. CFG-003 (target_accept) — restore CLI fidelity/reproducibility.
4. DAT-004 (covariates) — Fixed; verify via PPC as desired.
5. MCMC-006 (gating) — enforce basic MCMC quality checks.
6. PRI-005 (presence updates) — decide policy and implement/clarify.
7. PRI-007 (naming) — reduce confusion for maintainers.
8. CAL-008 (ECE scope) — document or extend as needed.
9. UX-009/UX-010 — clean logs and numbering.
