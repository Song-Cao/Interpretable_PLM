# Step 8 handoff: IRED depth-26 semantics and causal patching

Date: 2026-11-12
Status: final; no further IRED Step 8 analyses are proposed

## Scope and interpretation

IRED remains exploratory because Gate 1 failed on task gain. All active Step 8 results use
depth-26 codes from the provenance-verified cache. Rotated features are the target class and
conserved features are the comparison. Static codes are aggregated to residues before
class-level inference, and five dictionary seeds are replicate blocks. Claim-bearing families
share FDR correction; the campaign-derived evolutionary-peak track is a separate positive
control.

## Class-level semantic tests

| track | positions | rotated - conserved (SD) | 95% interval | q |
|---|---:|---:|---:|---:|
| catalytic | 3 | +0.041 | [-0.064, +0.145] | 0.286 |
| first shell | 37 | +0.287 | [+0.197, +0.371] | 0.000129 |
| second shell | 36 | +0.688 | [+0.608, +0.773] | 0.000129 |
| distal functional network | 3 | -0.196 | [-0.273, -0.120] | 0.000129 |
| generic distal | 211 | -0.311 | [-0.383, -0.241] | 0.000129 |
| substrate direct contact | 1 | +0.319 | [+0.216, +0.425] | 0.000129 |
| NADPH direct contact | 1 | -0.238 | [-0.314, -0.159] | 0.000129 |
| NADPH proximal/coupled | 4 | -0.222 | [-0.295, -0.152] | 0.000129 |
| crystallographic ketone proxy | 36 | +0.038 | [-0.032, +0.107] | 0.286 |

The robust headline is first- and second-shell enrichment with generic-distal depletion.
Single-position substrate and NADPH-direct tracks cannot carry standalone claims. The
positive-control effect is +0.444 SD (permutation p=0.00010). Top-position shell distributions
differ between rotated and conserved features (total-variation distance 0.09645, permutation
p=0.00010).

## Causal patching

Patching targets 116 rotated features above both decoder medians on 256 training-pool variants
stratified by clipped mutation count and fitness quartile. It performs 29,696 candidate and 768
control partial forward passes (30,464 total), inside E_fit at depth 26, using
`D_fit(z_ablated) + original_fit_residual`.

All hygiene checks passed before candidate effects were read. The no-op maximum was
0.00034395 raw prediction units or 0.002299 SD; the dead-feature maximum was identical. The
random-live control mean absolute effect was 0.001575 SD. All are below the common 0.05-SD
tolerance. The fresh-versus-float16-cache maximum raw activation difference was 21, replacing
the wrong-depth discrepancy of 2,284.

Across 116 features, mean absolute standardized effect was 0.005796 (median 0.003832); the
largest feature-level mean absolute effect was 0.035150. Bootstrap intervals on mean signed
effect excluded zero for 111 features (53 positive, 58 negative). The feature-level
mean-absolute 5th/25th/50th/75th/95th percentiles were
0.001123/0.002467/0.003832/0.006583/0.015320 SD. These are effects on the registered additive
attribution readout, not the nonlinear prediction ceiling.

The distribution comparison replaces the superseded count relative to a control mean.
Using all 29,696 rotated feature-variant absolute effects and the 256 effects for each control,
the rotated distribution exceeds random-live (Mann-Whitney U=5,347,415, p=3.06e-29;
rank-biserial=0.4068, hierarchical-bootstrap 95% interval [0.3211, 0.4894]) and dead control
(U=6,797,533, p=6.55e-105; rank-biserial=0.7883, interval [0.7461, 0.8275]). The bootstrap
resamples rotated features and the shared variant backgrounds. Complete quantiles and the ten
largest features are in `v19_causal_effect_distribution.md` and its companion CSV files.

## Spatial-causal linkage

The final linkage statistic uses, for each feature, the fraction of its total static activation
mass assigned to each mutually exclusive shell. This measures location rather than activation
magnitude. Correlations use the feature's mean absolute standardized patching effect and a
10,000-draw feature-label permutation test, with BH correction over four shells.

| shell | positions | Spearman rho | permutation p | q |
|---|---:|---:|---:|---:|
| catalytic | 3 | +0.0216 | 0.8139 | 0.8139 |
| first shell | 37 | +0.1198 | 0.2000 | 0.4000 |
| second shell | 36 | +0.1713 | 0.0668 | 0.2672 |
| distal | 214 | -0.0717 | 0.4463 | 0.5950 |

Second shell is the strongest positive association, first shell is weaker, catalytic is null,
and distal is weakly negative. No fraction-based linkage clears FDR. The class-level result
that rotated features are depleted at generic-distal positions and the within-rotated result
that distal activation fraction does not predict causal effect answer different questions and
do not conflict. The earlier absolute-activation linkage is discarded because it conflated
location with feature magnitude; its artifacts are quarantined under
`Reports/archive/ired_step8_absolute_linkage_v18/`.

## Active artifacts and interface

- `v18_class_semantic_tests.csv`: depth-26 class-level semantic effects and position counts.
- `v18_mutation_responsive_tests.csv`: depth-26 mutation-responsive tests.
- `v18_spatial_structure_test.json`: class-level spatial-structure permutation result.
- `v18_causal_patching_hygiene.json`: registered hygiene checks and tolerances.
- `v18_causal_patching_feature_effects.csv`: one row per rotated feature.
- `v18_causal_patching_variant_effects.csv`: one row per feature-variant ablation.
- `v18_causal_patching_control_effects.csv`: no-op, dead, and random-live controls.
- `v19_causal_patching_effect_quantiles.csv`: rotated and control effect quantiles.
- `v19_causal_patching_top_features.csv`: ten largest feature-level absolute effects.
- `v19_causal_effect_distribution_tests.csv`: Mann-Whitney and rank-biserial results.
- `v19_causal_feature_shell_fractions.csv`: feature-level shell fractions and causal effects.
- `v19_causal_spatial_fraction_correlations.csv`: final spatial-causal linkage tests.

## Incomplete or uncertain

No oracle validation was requested in this run. The 116-feature inference has limited power
for shell-effect correlations, especially for the three-position catalytic shell. Step 8 is
closed without further IRED analyses.
