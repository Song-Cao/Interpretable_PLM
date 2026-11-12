# IRED causal-effect distributions and spatial linkage

## Rotated-feature effects

Feature-level mean absolute standardized effects across 256 variants:

| quantile | effect (SD) |
|---:|---:|
| 0.05 | 0.001123 |
| 0.25 | 0.002467 |
| 0.50 | 0.003832 |
| 0.75 | 0.006583 |
| 0.95 | 0.015320 |
| 1.00 | 0.035150 |

The ten largest features by mean absolute standardized effect are:

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

Control absolute-effect quantiles across the same 256 variant backgrounds:

| control | q05 | q25 | q50 | q75 | q95 | maximum |
|---|---:|---:|---:|---:|---:|---:|
| random-live feature | 0.000279 | 0.000866 | 0.001426 | 0.002017 | 0.003667 | 0.005883 |
| dead feature | 0.000043 | 0.000161 | 0.000353 | 0.000683 | 0.001377 | 0.002299 |

Mann-Whitney comparisons use all 29,696 rotated feature-variant absolute effects and the 256
control effects. Rotated exceeds random-live with U=5,347,415, p=3.06e-29 and rank-biserial
0.4068 (hierarchical-bootstrap 95% interval [0.3211, 0.4894]). Rotated exceeds dead control
with U=6,797,533, p=6.55e-105 and rank-biserial 0.7883 (interval [0.7461, 0.8275]). The
bootstrap resamples rotated features and shared variant backgrounds.

## Fraction-based spatial-causal linkage

Each shell value is the fraction of a feature's total static activation mass assigned to that
shell. This replaces and discards the absolute-activation analysis.

| shell | Spearman rho | permutation p | BH q |
|---|---:|---:|---:|
| catalytic | +0.0216 | 0.8139 | 0.8139 |
| first shell | +0.1198 | 0.2000 | 0.4000 |
| second shell | +0.1713 | 0.0668 | 0.2672 |
| distal | -0.0717 | 0.4463 | 0.5950 |

Second shell has the strongest positive association, but no shell clears FDR. The
class-level generic-distal depletion and the within-rotated distal fraction correlation are
different estimands and do not conflict.
