# IRED v16 matched-drift recalibration

## Decision

The drift-matched null is the headline calibration at both widths because it yields fewer shifted
features for every real crosscoder seed. No emergent candidate survives at `w=1`, so the
user-specified condition for proceeding to Step 8 enrichment is not met. No enrichment,
patching-budget calculation, ablation, or oracle validation was run.

## Old and new taxonomy

Means across three real crosscoder seeds:

| null | w | dead | emergent candidate | evolution-only | rotated | re-weighted | conserved |
|---|---:|---:|---:|---:|---:|---:|---:|
| step-matched | 1 | 32.67% | 28.16% | 1.63% | 36.04% | 0.00% | 1.49% |
| drift-matched | 1 | 32.67% | 0.00% | 0.03% | 25.00% | 0.07% | 42.22% |
| step-matched | 2 | 49.93% | 0.00% | 0.94% | 47.81% | 0.05% | 1.27% |
| drift-matched | 2 | 49.93% | 0.00% | 0.14% | 10.76% | 0.05% | 39.11% |

The zero emergent-candidate counts are power limitations, not evidence of absence. At `w=1`,
two seeds have no conservative symmetric Delta cutoff after one tail has no BH discoveries, and
all three seeds have no reconstruction-ratio discoveries against every null run. At `w=2`, the
same reconstruction gate has no discoveries while 1,920 simultaneous feature tests are
corrected instead of 960. Class proportions are therefore not directly comparable across widths.

## Reconstruction-ratio effect size

| w | source | median across-seed range | 5th percentile range | 95th percentile range |
|---:|---|---:|---:|---:|
| 1 | real pair | 0.7849-0.7942 | 0.3049-0.4002 | 1.0221-1.0465 |
| 1 | drift-matched null | 0.7767-0.7909 | 0.5088-0.5141 | 1.0325-1.0448 |
| 2 | real pair | 0.7797-0.7897 | 0.2377-0.3353 | 1.0574-1.0701 |
| 2 | drift-matched null | 0.7818-0.7896 | 0.5092-0.5231 | 1.0457-1.0663 |

The observed and null medians overlap at both widths. The real pairs have heavier lower tails,
but no feature remains significant against every matched null after BH correction.

## Rotation above median decoder norms

Rotation survives the stricter norm restriction. At `w=1`, 27.3-32.2% of the 382-406 features
clearing both decoder medians are rotated. At `w=2`, the fraction is 17.4-18.5% among 824-836
features. Activation-frequency concentration also remains: Spearman correlation between the
rotation flag and activation frequency is 0.40-0.54 at `w=1` and 0.23-0.36 at `w=2`. The
highest-frequency quintile has 35-73% rotation, compared with 2-11% in the lowest quintile.
Residual rotation therefore does not disappear when low-norm decoder columns are excluded.

## Gate 2 sensitivity

Sparse scores 0.5495 on pg_kfold test. It is 0.0488 ahead of the validation-selected dense
`h_fit` reference (0.5007), with the registered dense-minus-sparse interval
`[-0.0956, -0.00153]`. The strongest dense arm on test is `h_evo` at 0.5543, leaving sparse
0.0048 behind. The defensible claim is that sparse features are competitive with dense probes,
not that sparse features beat dense probes.

Exact thresholds, per-feature classifications, effect-size quantiles, and above-median audits
are under `Reports/step6/drift_matched_rotation/`.
