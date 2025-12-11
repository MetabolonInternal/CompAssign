# Active-Learning MI Acquisition: Quick-Mode Findings

These notes document the first end-to-end run of the mutual-information (MI/BALD) acquisition path enabled in `scripts/experiments/active_learning/assess_active_learning.py`. We executed the quick configuration to keep runtime reasonable and inspected why MI performs similarly to a random-review baseline in this setting.

## Run command (quick preset)

```bash
poetry run python scripts/experiments/active_learning/assess_active_learning.py \
    --quick \
    --acquisition mi \
    --output-dir output/al_assessment_mi_quick
```

Key parameters under `--quick`:

- 15 species × 15 compounds; ~520 peaks after synthetic generation
- RT/assignment NUTS runs: 4 chains × (500 tune + 500 draws)
- Batch size 20, 10 AL rounds, threshold 0.4
- Initial labelled fraction 0.0 (all labels come from the simulated oracle)

## Summary metrics

| Strategy | Annotations to reach naive recall (≈0.962) | Final precision | Final recall | Final F1 | Final ECE |
| -------- | ----------------------------------------- | --------------- | ------------ | -------- | --------- |
| Naive full review | 520 | 0.954 | 0.963 | 0.958 | 0.021 |
| Active learning (MI) | **40** | 0.951 | 0.962 | 0.957 | 0.022 |
| Random review | **40** | 0.945 | 0.969 | 0.957 | 0.028 |

MI matches the naive recall after only two rounds (40 annotations), just like the random baseline. Final precision / recall / F1 all converge to ≈0.95–0.96 once 200 peaks have been labelled.

Artifacts written to `output/al_assessment_mi_quick/`:

- `results/summary.json` — full configuration and metric report (see excerpt below)
- `results/al_progression.csv` — round-by-round history for plotting

```json
{
  "active_learning": {
    "rounds": 10,
    "metrics": {
      "precision": 0.9506,
      "recall": 0.9625,
      "ece": 0.0223
    },
    "clicks_to_target_recall": 40,
    "relative_efficiency_ratio": 13.0
  },
  "random": {
    "metrics": {
      "precision": 0.9451,
      "recall": 0.9688,
      "ece": 0.0281
    }
  }
}
```

## Why MI ≈ random in quick mode

1. **Low candidate ambiguity.** After mass/RT screening the training set averages only ~0.85 non-null candidates per peak. Most peaks are trivial once seen, so any acquisition strategy rapidly uncovers the correct label.
2. **Modest recall target.** The success criterion is “match naive recall,” which is achieved after a single additional round because the naive baseline is already near-perfect once a few challenging peaks are labelled.
3. **Posterior noise from short MCMC runs.** MI/BALD needs reliable posterior variance to spot informative peaks. With 500 tune / 500 draw quick runs the assignment posterior is still noisy (PyMC reports `rhat>1.01` and a few divergences), blunting the signal MI is meant to capture.
4. **Nearly uniform presence priors.** Presence priors start flat and quickly become strong after the first batch. Once priors are updated, the remaining peaks have similar expected benefit, flattening acquisition scores.
5. **Synthetic generator lacks hard clusters.** The quick generator spreads ambiguity evenly. There are no concentrated pockets of genuinely difficult peaks where MI could distinguish itself.

## Recommendations

- **Increase ambiguity**: raise decoy fraction, tighten RT/mass windows, or lower initial labels to create a harder task.
- **Longer MCMC**: move beyond quick settings (e.g., 1,000–1,500 draws, higher `target_accept`) to stabilise posterior entropy for MI.
- **Smaller batches / adaptive batch sizes**: MI gains are more visible with finer-grained selection (batch size 5–10) on ambiguous peaks.
- **Diagnostics**: monitor divergences (`az.summary`/ArviZ) and ensure MI runs on chains that satisfy `rhat ≤ 1.01`.

This document should be revisited when running the full configuration or after modifying the synthetic generator to introduce richer ambiguity profiles.
