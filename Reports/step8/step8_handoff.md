# Step 8 handoff: IRED depth-26 semantics and causal patching

Date: 2026-11-12
Status: registered enrichment and approved patching complete; oracle validation not run

## Semantic tests

All tests use depth-26 codes derived from the provenance-verified cache. Rotated features are
the target and conserved features the comparison; static codes are aggregated to residues
before class-level inference. Five dictionary seeds are the replicate blocks. Claim-bearing
families share FDR correction; the campaign-derived evolutionary-peak track is a separate
positive control.

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

The robust headline is the shell gradient, especially first- and second-shell enrichment and
generic-distal depletion. Single-position substrate and NADPH-direct tracks cannot carry a
standalone claim. Unlike the withdrawn wrong-depth result, NADPH tracks are depleted rather
than null; this corrected outcome is retained without post-hoc adjustment. Top-position shell
distributions differ (total-variation distance 0.09645, permutation p=0.00010). The positive
control effect is +0.444 SD (p=0.00010).

## Causal patching

Patching targets 116 rotated features above both decoder medians on 256 training-pool variants
stratified by clipped mutation count and fitness quartile. It performs 29,696 candidate and 768
control partial forward passes (30,464 total), inside E_fit at depth 26, using
`D_fit(z_ablated) + original_fit_residual`.

All hygiene checks passed before candidate effects were read: the no-op maximum was
0.00034395 raw prediction units or 0.002299 SD; the dead-feature maximum was identical; the
random-live control mean absolute effect was 0.001575 SD. Each is below the common 0.05-SD
tolerance. Fresh-versus-float16-cache maximum raw activation difference was 21, replacing the
wrong-depth discrepancy of 2,284.

Across 116 features, mean absolute standardized effect was 0.005796 (median 0.003832); the
largest feature-level mean absolute effect was 0.03515. Bootstrap intervals excluded zero for
111 features (53 positive, 58 negative). These are effects on the registered additive
attribution readout, not the nonlinear prediction ceiling.

The feature-level mean-absolute distribution has 5th/25th/50th/75th/95th percentiles
0.001123/0.002467/0.003832/0.006583/0.015320 SD. It is broad rather than driven only by a few
features: 103/116 feature means exceed the random-live control mean absolute effect of
0.001575 SD. The complete distribution, ten largest features, and matched control quantiles are
reported in `v18_causal_effect_distribution.md`.

Spatial and causal results meet at the second shell. Across the 116 features, absolute patching
effect correlates with mean second-shell activation at Spearman rho 0.2782 (10,000-permutation
p=0.0028, four-shell FDR q=0.0112). Catalytic activation is unrelated (rho 0.0209, q=0.8203).
First-shell and distal correlations are nominal but do not survive FDR (both q=0.0588). The
result therefore supports a second-shell link, not a general claim that every active-site-
proximal or distal activation is causal.

Files: `v18_class_semantic_tests.csv`, `v18_mutation_responsive_tests.csv`,
`v18_spatial_structure_test.json`, `v18_causal_patching_hygiene.json`, and
`v18_causal_patching_{feature,variant}_effects.csv` in `Reports/step8/`. The full linked table is
`v18_causal_feature_shell_profiles.csv`.

## Incomplete or uncertain

No oracle validation was requested in this v18 run. IRED remains exploratory because Gate 1
failed; these results do not enter a confirmatory central-claim table.
