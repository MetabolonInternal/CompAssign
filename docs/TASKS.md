# Sally realtest TS improvement (RT + CompAssign partial-pooling ridge)

Objective: increase `mdsutils` TS on Sally realtest SSIDs using the baseline peak-assignment method, without calling
standards or X-compounds (hard requirement). We still report the official `mdsutils` TS (which includes standards and
X-compounds in the evaluation set).

## Scope (from `src/compassign/rt/sally_test.sh`)
- lib209: SSID 20159 (supercat8), SSID 23059 (supercat6)
- lib208: SSID 12307 (supercat6), SSID 12609 (supercat8), SSID 12725 (supercat6)

## What worked (keep)
### RT coherence rejection (q10) for `COMPASSIGN_PP_RIDGE`
This adds a *compound-level* RT-consistency check: for each compound, take the per-task RT log-likelihood values of the
peaks Sally is about to call (restricted to LC sample types), compute a lower-tail quantile (q10 by default), and reject
the compound if that quantile is too low. Operationally this targets the main failure mode we saw in the marginal TS
regime: compounds that accumulate plausible-but-wrong completes across many tasks (often within the RT window), yielding
many FP/FCPD. The rejection preferentially removes those incoherent patterns while keeping compounds whose called peaks are
consistently plausible under the RT model.

This is not “Bayesian-only”. The key requirement is a *continuous* per-candidate RT plausibility score that is at least
roughly calibrated across tasks. A Bayesian posterior predictive density is one principled way to get that, but you could
implement the same idea with any model that outputs a predictive mean plus a usable scale (e.g. a frequentist regression
with an empirical error model, or even a heuristic score if it behaves like a likelihood). The main contrast with the
historical baseline windowing is that the baseline is primarily a *hard filter + nearest-to-centre picker*, which cannot
penalize “barely inside the window” decoys or express cross-task coherence as a single decision.

### Use Sally's default `windowMultiplier=4` for `COMPASSIGN_PP_RIDGE`
Sally has historically used `windowMultiplier` (default 4) as a tuning knob for heuristic RT windows. For the CompAssign
modelType we previously defaulted to multiplier=1 because the exported windows are already intended to be nominal
prediction intervals. Empirically, the strongest TS gains on the realtest SSIDs required using multiplier=4 here as well.
Operationally, this makes the hard RT filter less brittle (fewer true peaks are excluded) while giving the q10 coherence
rejection enough dynamic range to act as the precision lever (net TS gains on the realtest SSIDs).

### Encode the working defaults in `src/compassign/rt/sally_test.sh`
`src/compassign/rt/sally_test.sh` stages the exact RT artifacts used by the report and runs the fixed 5-SSID check in a
single command. The q10 coherence rejection and `windowMultiplier=4` behavior are now defaults in Sally for `COMPASSIGN_PP_RIDGE`, so
the script no longer depends on a large set of `SALLY_RT_*` environment knobs.

### Shelve `lc_column` RT covariate (not validated)
We retrained CompAssign RT artifacts that include `lc_column` as an additional covariate, but we did not validate that
it improves generalization across SSIDs / libs. We are therefore treating `lc_column` as out-of-scope for this baseline
and removing it from the default training/staging paths until there is clear evidence it helps.

## Latest Oracle-backed rerun (VPN on)
Command: `conda activate sally && ./src/compassign/rt/sally_test.sh`

`COMPASSIGN_PP_RIDGE` in this script includes:
- hard RT window filtering + baseline per-task peak selection (closest to corrected expected RT)
- q10 RT coherence rejection across tasks (LC only; q=0.1; min q10 loglik=3.4)
- uses Sally default `windowMultiplier=4`

| lib | supercat | ssid  | model              | precision | recall | ts     |
|---:|---------:|------:|--------------------|----------:|-------:|-------:|
| 208 | 6        | 12307 | ESLASSO            | 0.8436    | 0.7848 | 0.6851 |
| 208 | 6        | 12307 | COMPASSIGN_PP_RIDGE | 0.9390    | 0.8113 | 0.7706 |
| 208 | 8        | 12609 | ESLASSO            | 0.8553    | 0.6513 | 0.5867 |
| 208 | 8        | 12609 | COMPASSIGN_PP_RIDGE | 0.9328    | 0.8031 | 0.7592 |
| 208 | 6        | 12725 | ESLASSO            | 0.6808    | 0.8610 | 0.6134 |
| 208 | 6        | 12725 | COMPASSIGN_PP_RIDGE | 0.9303    | 0.8807 | 0.8262 |
| 209 | 8        | 20159 | ESLASSO            | 0.8865    | 0.8697 | 0.7825 |
| 209 | 8        | 20159 | COMPASSIGN_PP_RIDGE | 0.9871    | 0.8848 | 0.8747 |
| 209 | 6        | 23059 | ESLASSO            | 0.8357    | 0.5940 | 0.5319 |
| 209 | 6        | 23059 | COMPASSIGN_PP_RIDGE | 0.8943    | 0.6737 | 0.6241 |

## Attempted in this session (did not help materially; deprioritize)
- [x] Per-peak RT filtering sweeps (minimal TS movement on lib209):
  - [x] Soft filtering by z-score / log-likelihood thresholds
  - [x] Top-k filtering
  - [x] CF/mode heuristics alone

## Shelved for later (needs separate validation)
- [x] `lc_column` as an RT covariate: retrained artifacts exist, but we are not using them until we see clear gains on both
  libs and are confident it generalizes. This is removed from the current baseline; revisiting it should be done as a
  separate effort with explicit validation.

## Next priority (cleanup, preserve the new baseline)
The TS table in `docs/models/rt_pymc_multilevel_pooling_report.tex` (CompAssign PP ridge + q10 + window multiplier=4) is
the baseline for future work. The next work item is to reduce both repos to the minimal changes required to reproduce
those numbers, and to remove experimental branches/knobs that are not needed.

- [x] Cleanup CompAssign + Sally code to minimal diff while reproducing the LaTeX TS tables exactly.
- [x] Remove the shelved `lc_column` path end-to-end (training/staging/docs).
- [x] Make q10 coherence rejection + `windowMultiplier=4` the default for `COMPASSIGN_PP_RIDGE` in Sally (remove env-gated plumbing).
- [x] Prune artifacts under `out/`, `output/`, and `external_repos/sally/out/` to keep only baseline models + results.
- [x] Re-run `./src/compassign/rt/sally_test.sh` after pruning and confirm the LaTeX TS table is reproduced.

## Follow-ups (after cleanup)
- [x] Quantify the q10 coherence rejection delta (ablation) on the realtest SSIDs (no-q10 vs q10, same window multiplier).
- [ ] Explore continuous RT scoring (soft filter) as a principled next lever: use `rt_loglik` (or z) as the per-task
  selection score and relax the hard window filter; validate carefully against false positives and generalization.
- [ ] Revisit the correction factor (CF) heuristic: test whether CF is still needed under `COMPASSIGN_PP_RIDGE`, and if so
  refactor it into a simpler/more principled representation.
- [ ] If we still need more TS: pursue “cloud/mode coherence” (score/select an RT residual mode across tasks rather than
  making independent per-task picks).

## Repro commands / knobs
- Train CompAssign RT models: `conda activate compassign && ./src/compassign/rt/train.sh`
- Run Sally evaluations: `conda activate sally && ./src/compassign/rt/sally_test.sh`
