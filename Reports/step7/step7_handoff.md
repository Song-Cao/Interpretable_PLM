# Step 7 handoff — IRED width diagnostics, provisional taxonomy, and Gate 2

## Status and scope

Parts A and B are complete. IRED remains exploratory because Gate 1 failed on task gain; none of
these results revise that verdict. The fixed-`k` width sweep, raw-activation rank,
feature-death timing, and one registered dead-feature-resampling run selected `w=1` and `w=2`.
Both widths then completed the null-calibrated provisional taxonomy. Gate 2 reports sparse
features competitive with the dense reference under the rule committed in Step 5.

All 12 death-timing replays exactly reproduced both the authoritative scalar loss traces and
every final state tensor before their epochwise activation counts were accepted. A complete NPZ
exists for every width and seed. No interrupted or partial fit contributes to this report.

## Dictionary-free activation rank

The PCA uses centered covariance over all 259,550 normalized training-pool residue activations;
normalization by a fixed scalar does not change its spectrum or participation ratio.

| model | raw participation ratio | PCs for 50% | PCs for 80% | PCs for 90% | PCs for 95% | PCs for 99% |
|---|---:|---:|---:|---:|---:|---:|
| E_evo | 66.94 | 30 | 98 | 148 | 192 | 256 |
| E_fit | 61.24 | 31 | 106 | 162 | 212 | 309 |

The raw 960-channel activations therefore have a small effective rank before any dictionary is
introduced. For comparison, the prior code-level participation ratios were 122.83, 112.78,
99.12, and 95.02 at `w=1,2,4,8`. Those code values describe the covariance of nonlinear sparse
coordinates and are not PCA dimensions in the input space; the raw spectrum is the relevant
intrinsic-rank constraint. Under v15, overcompleteness is measured against raw effective rank,
not ambient dimension. The 960-feature `w=1` dictionary is already 14.34 times overcomplete
against E_evo and 15.68 times against E_fit; `w=2` is 28.68 and 31.35 times overcomplete.

Exact eigenvalues and cumulative spectra are in
`Reports/step7/part_a/raw_activation_pca_spectrum.csv`; the summary is
`raw_activation_pca_summary.json`.

The matched layer-freeze reserve provides the corresponding shuffled-label control. At depth 26
over the same 17,110 residue observations, PR is 66.93 for E_evo, 60.67 for E_fit, and 60.29 for
the shuffled adapter. The real and shuffled contractions are 9.35% and 9.92%, respectively.
Together with CKA 0.9189 real and 0.9811 shuffled, this shows that rank contraction is generic to
fine-tuning here rather than label-driven. The matched result is in
`Reports/step7/adapter_rank_comparison.json`.

## When features died

Counts below are means across three seeds. “Never active” means zero full-pool activations at
initialization and every epoch. “Crowded out” means dead at epoch 20 but active at least once at
initialization or an earlier epoch.

| w | features | initially dead | never active | crowded out | live at epoch 20 | final dead fraction |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 960 | 30.3 (3.2%) | 14.3 (1.5%) | 299.3 (31.2%) | 646.3 | 0.3267 |
| 2 | 1,920 | 201.7 (10.5%) | 144.7 (7.5%) | 814.0 (42.4%) | 961.3 | 0.4993 |
| 4 | 3,840 | 859.3 (22.4%) | 652.3 (17.0%) | 1,891.0 (49.2%) | 1,296.7 | 0.6623 |
| 8 | 7,680 | 2,877.7 (37.5%) | 2,032.7 (26.5%) | 3,969.3 (51.7%) | 1,678.0 | 0.7815 |

Crowding is the majority source of final death at every width: among final-dead features, the
crowded-out share is 95.4%, 84.9%, 74.4%, and 66.1% for `w=1,2,4,8`. Never-active features also
rise sharply with width. The deaths occur through competition, but random resampling did not
recover them and the raw effective rank is only 61-67. The rising dead fraction is therefore not
treated as a recoverable BatchTopK artifact: the available variation is genuinely concentrated.

Per-run categories, every epoch trajectory, and compressed per-feature counts are under
`Reports/step7/part_a/death_timing/` and the two `death_timing_*.csv` files.

## Dead-feature resampling intervention

The registered intervention used `w=4`, seed 20261311, unchanged reconstruction objective and
fixed `k=32`. Every full-pool-dead feature was randomly reinitialized after epochs 5, 10, and 15;
its Adam moments were cleared while the global optimizer step was preserved. It resampled 2,454,
2,524, and 2,549 features respectively.

The final dead fraction was 0.66797 versus 0.66042 for the matched baseline: an absolute change
of +0.00755, or a pre-registered “reduction” of -0.00755. It fails the 0.10 substantial-recovery
criterion and is also worse than the `w=2` mean of 0.49931. Random resampling therefore did not
rescue `w=4`; it does not justify replacing either default width. The intervention checkpoint is
pod-only at `Models/step7/IRED/diagnostics/w4_seed20261311_resampled_nobias.pt`, SHA-256
`1278de923a2c19bc837ced1abfbf823f8e7a6872cf7b6f90d041c89f91e74433`.

## Width choice

The exploratory Part B dictionaries are `w=1` and `w=2`, both run rather than selecting a single
one. Both are strongly overcomplete relative to the 61-67-dimensional raw activations: `w=1`
supplies the lowest-death 14-16-times-overcomplete reference, while `w=2` tests a 29-31-times
overcomplete dictionary at the edge of what this cache supports. `w=4` and `w=8` are excluded because the
registered intervention did not recover `w=4`, and feature death worsens further at `w=8`.

This supersedes the earlier statement that IRED empirically “cannot support an overcomplete
dictionary.” That statement implemented v13's pre-committed stopping rule. It was not an
impossibility result, and v15 explicitly permits the exploratory comparison.

## Cost

Part A consumed 0.14059 A100 wall-hours (8.44 minutes), including raw PCA, 12 instrumented
replays, exact reproduction checks, and the resampling run. The resampling fit itself used
0.00971 hours. Cumulative documented IRED GPU occupancy through Part A is approximately 13.3
A100 wall-hours; this includes the earlier searches, diagnostic and discarded runs retained in
the project log, the v13 cache/null/width work, and this diagnostic.

## Artifacts and next interface

- `Reports/step7/part_a/part_a_manifest.json`: completed-run manifest and selected widths.
- `Reports/step7/part_a/rank_and_feature_death.png`: raw PCA and live-feature trajectories.
- `Reports/step7/part_a/raw_activation_pca_spectrum.csv`: 960 eigenvalues per model.
- `Reports/step7/part_a/death_timing_runs.csv`: per-width, per-seed death categories and hashes.
- `Reports/step7/part_a/death_timing_epochs.csv`: live counts at initialization and every epoch.
- `Reports/step7/part_a/w4_resampling.json`: intervention decision and checkpoint hash.
- `Models/step7/IRED/w{1,2}_seed{seed}_nobias.pt`: pod-only authoritative Part B inputs.

## Provisional taxonomy

Under v16, every fitted threshold uses the unconditional drift-matched shuffled-label null at
`q=0.05`. Null runs are fitted separately and the conservative cutoff is retained. The artifact
gate is conjunctive: either latent-scaling ratio can exclude a candidate. The decoder norm floor
uses each decoder's own median. Classification follows `dead`, `emergent candidate`,
`evolution-only`, `rotated`, `re-weighted`, then `conserved`. “Emergent candidate” remains
causally unvalidated.

The original step-matched shuffled null produced the following proportions:

| w | dead | emergent candidate | evolution-only | rotated | re-weighted | conserved |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32.67% | 28.16% | 1.63% | 36.04% | 0.00% | 1.49% |
| 2 | 49.93% | 0.00% | 0.94% | 47.81% | 0.05% | 1.27% |

Values are means across three real crosscoder seeds. They are retained as the superseded
step-matched-null result. Recalibrating Delta, both latent-scaling ratios, decoder cosine, and
activation-frequency change against the drift-matched null gives:

| w | dead | emergent candidate | evolution-only | rotated | re-weighted | conserved |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32.67% | 0.00% | 0.03% | 25.00% | 0.07% | 42.22% |
| 2 | 49.93% | 0.00% | 0.14% | 10.76% | 0.05% | 39.11% |

The drift-matched taxonomy has fewer shifted features for every seed and width and is therefore
the v16 headline. At `w=1`, emergent candidates fall from 28.16% to zero. This is not evidence
that emergent features are absent: in two seeds the symmetric Delta threshold is undefined after
one tail has no BH discoveries, and the reconstruction-ratio gate has no BH discoveries in all
three seeds. It is a power-limited classification under the registered conservative rule. No
feature can receive the final `emergent` label without first clearing these gates and later
causal validation.

Class proportions must not be compared directly across widths: `w=2` corrects 1,920 feature
tests while `w=1` corrects 960, so dictionary width changes the multiple-testing burden as well
as the representation. A zero after BH means no feature reached significance at that width; it
does not establish absence. Exact per-feature labels and thresholds are under
`Reports/step6/drift_matched_rotation/`.

### Width-matched reconstruction-ratio audit

Null crosscoders are matched by width: three `w=1` nulls calibrate `w=1`, and three `w=2` nulls
calibrate `w=2`. The v16 effect-size distributions are:

| w | source | median range | 5th percentile range | 95th percentile range |
|---:|---|---:|---:|---:|
| 1 | real pair | 0.7849-0.7942 | 0.3049-0.4002 | 1.0221-1.0465 |
| 1 | drift-matched null | 0.7767-0.7909 | 0.5088-0.5141 | 1.0325-1.0448 |
| 2 | real pair | 0.7797-0.7897 | 0.2377-0.3353 | 1.0574-1.0701 |
| 2 | drift-matched null | 0.7818-0.7896 | 0.5092-0.5231 | 1.0457-1.0663 |

The medians overlap closely at both widths. The real distributions have heavier lower tails,
but after correction no feature reaches reconstruction-ratio significance against every null
run. The resulting `-inf` conservative cutoff is a power limitation. It is especially important
at `w=2`, where the number of simultaneous feature tests doubles. Full quantiles are in
`reconstruction_ratio_effect_sizes.csv`.

### Drift-matched rotation audit

The shuffled adapter used registered rank 16, learning rate 0.0001, weight decay 0.01, and
shuffled labels. It reached relative drift 0.6484 after 575 optimizer steps, within 2.3% of real
drift 0.6338. Recalibrating cosine alone reduced rotated fractions to 7.12% and 10.45%, but that
v15 intermediate is not a full taxonomy: under v16, removing former emergent candidates lets
some features fall through the precedence order into `rotated`, yielding the headline 25.00%
and 10.76%.

Residual rotation does not disappear above the decoder-norm medians. Restricting to features at
or above each decoder's own median leaves 382-406 features at `w=1`, of which 27.3-32.2% are
rotated, and 824-836 at `w=2`, of which 17.4-18.5% are rotated. Activation-frequency
concentration also survives: above the medians, rotation-flag Spearman is 0.40-0.54 at `w=1`
and 0.23-0.36 at `w=2`; the highest-frequency quintile has 35-73% rotation versus 2-11% in the
lowest quintile. Low-norm cosine noise therefore cannot explain the surviving rotation signal.

Reconstruction remains high but is reported separately by model. Across real seeds, `w=1`
explains 99.85% (E_evo) and 99.59% (E_fit) of variance; `w=2` explains 99.86% and 99.68%.
These values do not rescue dead or artifact-failing features.

## Gate 2

The best dense reference selected only by five-fold validation Spearman was `dense_fit_mean`;
the best sparse arm by the same rule was `phi_w1_max_all_live`. Their pg_kfold test Spearman
values were 0.5007 and 0.5495. Sparse is ahead by 0.0488 relative to this pre-committed reference;
the paired bootstrap interval for dense minus sparse is `[-0.0956, -0.00153]`, barely excluding
zero. Gate 2 therefore returns `sparse_competitive`.

All dense references are shown to separate the pre-committed selection rule from a post-hoc
test ranking:

| dense reference | validation Spearman | pg_kfold test Spearman |
|---|---:|---:|
| h_fit mean | 0.6231 | 0.5007 |
| h_evo mean | 0.6077 | 0.5543 |
| PCs of h_fit - h_evo | 0.3948 | 0.4882 |

`h_evo` is strongest on test at 0.5543, leaving sparse 0.0048 behind it. Selecting `h_evo` after
reading test would violate the registered protocol, but this sensitivity bounds the scientific
claim: sparse features are competitive with dense probes, not superior to dense probes. The
accepted Gate 2 decision still compares validation-selected `h_fit` against validation-selected
sparse features; the full audit is
`Reports/step7/gate2_dense_reference_audit.csv`.

Step 9 was run regardless under the pre-recorded IRED-only resource-allocation override. This
does not weaken Gate 1 or Gate 3 and does not apply to GB1.

## Step 9 interpretation records

The headline expressivity gap is `R2(L4)-R2(L0)=0.152725` on pg_kfold test. Fitness prediction
therefore needs materially nonlinear feature logic on IRED. The additive readout retained for
Steps 8 and 11 is weaker than this ceiling, so its attributions describe that weaker additive
model rather than the full predictive logic.

The validation-selected sparse arm is `phi_w1_max_all_live`: it uses max pooling over residue
codes, whereas the plan's displayed notation defines `phi_j` using a residue sum. The computed
result is internally consistent and downstream code is unaffected, but the notation must be
corrected to distinguish the selected max-pooled arm from sum-pooled arms.

## Incomplete or uncertain

All class proportions remain provisional. No emergent candidate survives v16 calibration at
`w=1`, so the user-specified condition for Part C enrichment and ablation planning is not met;
no enrichment, patching-budget calculation, patching, or oracle validation was run. The zeros at
both widths are power-limited BH outcomes and do not establish absence, while cross-width class
proportions are not directly comparable because the number of tested features differs. The
resampling result tests one registered random-reset intervention, not every anti-death optimizer.
Gate 1 remains failed, and no Step 7 result is confirmatory evidence for the central claim.
