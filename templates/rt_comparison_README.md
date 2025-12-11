RT Comparison Outputs

This directory contains artifacts from the 4‑way RT comparison:
- Hierarchical (plain)
- Hierarchical (chemistry: β descriptors + γ class pooling)
- Baseline (per‑cluster Lasso)
- Baseline (per‑cluster × per‑compound Lasso)

Layout
- default/: Default difficulty profile
- hardened/: Hard rare extrapolation profile (cross‑species holdout, 1 train row per rare, far test runs, stronger drift)
- anchored/: Anchored profile (nearest‑anchor pairing; moderate drift)
- natural/: Natural profile (minimal split shaping)

How to run
- Hardened:
  bash scripts/run_rt_comparison.sh --profile hardened
- Quick MCMC (fast smoke):
  bash scripts/run_rt_comparison.sh --quick --profile default
- All profiles:
  bash scripts/run_rt_comparison.sh --profile all

What each file means (per profile folder)
- rt_model_comparison_*.json
  Full report: config (profile, MCMC); metrics_penalized (end‑to‑end); metrics_on_coverage (valid‑only, n per arm);
  metrics_intersection (strict apples‑to‑apples, n); rare/common stratified metrics; coverage (valid vs imputed counts).

- rt_model_comparison_parity_*.png
  Observed RT vs Predicted RT per arm (valid‑only; n per panel). Best for at‑a‑glance fit quality and coverage.

- rt_model_comparison_coverage_*.png
  Valid vs Imputed (penalized) counts per arm. Explains end‑to‑end behavior and missing predictions.

- rt_model_comparison_groups_*.png
  Rare vs Common MAE bars (penalized). Titles include group n.

- rt_model_comparison_deltas_*.png
  ΔMAE relative to Hier (chem): positive bars mean Hier (chem) is better.

- rt_model_comparison.log
  Stable log file with progress and paths. Tail live:
  tail -f output/rt_comparison/<profile>/rt_model_comparison.log

