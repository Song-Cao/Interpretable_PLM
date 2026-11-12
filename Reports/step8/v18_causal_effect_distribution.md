# IRED causal patching effect distribution and spatial linkage

Date: 2026-11-12
Representation: ESM-C hidden depth 26, primary w=1 crosscoder seed 20261311

## Distribution

The full 116-feature table remains `v18_causal_patching_feature_effects.csv`; the full
29,696 feature-variant table is `v18_causal_patching_variant_effects.csv`. The table below is
the distribution across features of each feature's mean absolute standardized prediction
effect over the same 256 variants.

| quantile | effect (SD) |
|---:|---:|
| 0% | 0.000508 |
| 5% | 0.001123 |
| 10% | 0.001483 |
| 25% | 0.002467 |
| 50% | 0.003832 |
| 75% | 0.006583 |
| 90% | 0.012916 |
| 95% | 0.015320 |
| 99% | 0.033275 |
| 100% | 0.035150 |

The distribution is broad and right-tailed, not a few effects on an otherwise-null panel.
The random-live control's mean absolute effect is 0.001575 SD; 103/116 (88.8%) rotated
features exceed it. All 116 exceed the dead-feature mean absolute effect of 0.000484 SD.

For an equal sampling-axis comparison, the absolute per-variant effect quantiles are:

| population | 5% | 25% | 50% | 75% | 95% | maximum |
|---|---:|---:|---:|---:|---:|---:|
| rotated feature-variant effects (n=29,696) | 0.000219 | 0.001173 | 0.002925 | 0.006495 | 0.019426 | 0.326381 |
| dead control (n=256) | 0.000043 | 0.000161 | 0.000353 | 0.000683 | 0.001377 | 0.002299 |
| random-live control (n=256) | 0.000279 | 0.000866 | 0.001426 | 0.002017 | 0.003667 | 0.005883 |

All registered signed and absolute quantiles are in
`v18_causal_patching_effect_quantiles.csv`. The 768 control effects were backfilled without
rerunning any candidate ablation and are persisted in `v18_causal_patching_control_effects.csv`.

## Ten largest feature-level mean absolute effects

| feature | mean signed effect | mean absolute effect |
|---:|---:|---:|
| 378 | -0.023521 | 0.035150 |
| 950 | -0.033843 | 0.033843 |
| 492 | +0.030060 | 0.030060 |
| 600 | +0.020170 | 0.020170 |
| 675 | -0.020135 | 0.020135 |
| 287 | +0.017681 | 0.017724 |
| 525 | -0.004488 | 0.014518 |
| 607 | -0.013844 | 0.013844 |
| 929 | -0.013565 | 0.013565 |
| 470 | -0.013316 | 0.013522 |

The complete top-ten rows, including SD and bootstrap intervals, are in
`v18_causal_patching_top_features.csv`.

## Spatial-causal linkage

For each approved rotated feature, mean absolute patching effect was correlated with its mean
static activation in each shell. Distal combines distal-functional-network and generic-distal
positions. Two-sided feature-label permutation tests used 10,000 draws; BH correction covers
the four shells.

| shell | positions | Spearman rho | permutation p | FDR q |
|---|---:|---:|---:|---:|
| catalytic | 3 | +0.0209 | 0.8203 | 0.8203 |
| first shell | 37 | +0.1873 | 0.0441 | 0.0588 |
| second shell | 36 | +0.2782 | 0.0028 | 0.0112 |
| distal | 214 | +0.1979 | 0.0331 | 0.0588 |

Only the second-shell association survives FDR. Thus the strongest class-level enrichment is
linked to prediction-changing features specifically in the second shell; there is no evidence
that activation at the three catalytic positions predicts causal effect. First-shell and
distal relationships remain nominal and are not called significant.

`v18_causal_feature_shell_profiles.csv` contains all 116 feature-level effects and shell means;
`v18_causal_spatial_effect_correlations.csv` contains the registered tests.
