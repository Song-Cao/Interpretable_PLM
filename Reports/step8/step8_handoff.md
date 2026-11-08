# Step 8 handoff — IRED semantic tests and causal budget

## Status

The v17 class-level semantic, spatial-structure, and mutation-responsive analyses are complete.
Causal patching has not begun: this handoff is the required approval boundary. Oracle validation
also has not begun. IRED remains an exploratory pipeline shakedown because Gate 1 failed on task
gain; none of these results supplies confirmatory evidence for CROSSFIT's central claim.

The five-seed pooled-null taxonomy selects `rotated` as the target because it is the only shifted
class populated in all five `w=1` dictionaries (mean 25.23%, SD 1.55%). `conserved` is the
pre-registered comparison. Evolution-only and re-weighted classes contain fewer than one feature
on average and cannot support a class-level semantic test. Emergent candidates remain zero.

## Static residue-level semantic test

For each dictionary seed, the analysis computes `A_jp = E_x[z_jp(x)]` over all 1,182 curated
campaign variants before testing annotations. Feature labels are permuted within each independent
dictionary seed; effects are standardized within seed and averaged over five seeds. Bootstrap
intervals resample both dictionary seeds and features. FDR is controlled across the nine
claim-bearing annotation families. The campaign-derived evolutionary-peak track is tested
separately as a positive control and is excluded from mechanistic FDR and claims.

| annotation family | rotated minus conserved (SD units) | 95% bootstrap interval | FDR q |
|---|---:|---:|---:|
| catalytic shell | 0.084 | [0.004, 0.165] | 0.0364 |
| first shell | 0.152 | [0.067, 0.236] | 0.000225 |
| second shell | 0.399 | [0.319, 0.482] | 0.000225 |
| distal functional network | 0.010 | [-0.082, 0.105] | 0.786 |
| generic distal | -0.267 | [-0.357, -0.172] | 0.000225 |
| substrate direct contact | 0.198 | [0.089, 0.309] | 0.000225 |
| NADPH direct contact | 0.055 | [-0.023, 0.137] | 0.186 |
| NADPH proximal/coupled | -0.028 | [-0.144, 0.083] | 0.502 |
| crystallographic ketone-analogue proxy | -0.030 | [-0.140, 0.079] | 0.502 |

The replicated profile is enrichment of rotated features at first- and second-shell positions and
the supplied substrate-contact position, with depletion in the generic-distal shell. The weaker
catalytic-shell effect clears FDR narrowly. Cofactor and crystallographic proxy tracks do not.
These are class-level associations, not per-feature semantic labels or causal effects.

The evolutionary-peak positive control behaves as required: effect 0.272 SD units, permutation
`p=0.00010`, positive in every seed. Because those residues were derived from campaign outcomes,
this is a pipeline check only and supports no biological claim.

## Spatial structure and mutation response

The shell distribution of each feature's top-activating residue differs between rotated and
conserved features. The mean across-seed total-variation distance is 0.0808 with blocked
permutation `p=0.000500`. Rotated top positions are relatively shifted toward second-shell
residues and away from generic-distal positions. Exact per-seed counts are in
`v17_top_position_shell_counts.csv`.

The mutation-responsive panel uses the previously cached exact
`Delta z_j(x) = z_j(x_mutant) - z_j(x_WT)` for the primary dictionary seed. It restricts the
perturbation set to 289 variants whose mutations all lie in distal-network or generic-distal
shells. Resampling keeps all variants with the same exact mutated-position set together, yielding
257 background groups. The rotated-versus-conserved response profile follows the static result:
larger absolute responses at catalytic, first, second and substrate-contact positions and smaller
responses at generic-distal positions. This panel is single-dictionary-seed and exploratory;
`v17_mutation_responsive_tests.csv` contains the full grouped intervals and FDR values.

Only prepared, provenance-checked functional-geometry tracks were tested. No access-tunnel,
conservation/DCA, energetic, docking, or oracle track was inferred to fill missing headline
blocks. The 5OCM ketone-analogue contact remains labelled a proxy rather than a campaign-substrate
contact.

## Causal patching budget — approval required

The primary `w=1`, seed 20261311 taxonomy contains 248 rotated features. Restricting to features
at or above both their own decoder median norms leaves 116 causal candidates. The proposed
variant panel contains 64 pg_kfold fold-0 training-pool variants, selected with fixed seed by
clipped mutation count and fitness quartile. No test, reserve, round-held-out, or plate variant is
used.

| component | partial forward passes |
|---|---:|
| 116 candidate features x 64 variants | 7,424 |
| no-op, dead-feature, random-live controls x 64 | 192 |
| total partial forward passes | 7,616 |

There are 64 separate baseline forward passes, not counted as partial passes above. The future
implementation must patch inside `E_fit`, splice back the crosscoder residual, verify numerical
no-op behavior first, and reject the analysis if dead/random controls are not near zero. The exact
candidate feature IDs and fixed variant sample are in `v17_causal_patching_budget.json` and
`v17_causal_variant_subsample.csv`. No patch was executed and no effect was inspected.

## Artifacts

- `v17_class_semantic_tests.csv`: five-seed effects, intervals, permutation p-values, and FDR.
- `v17_positive_control_test.csv`: separately reported campaign-derived positive control.
- `v17_spatial_structure_test.json` and `v17_top_position_shell_counts.csv`: shell concentration.
- `v17_mutation_responsive_tests.csv`: grouped distal-mutation response panel.
- `v17_semantic_manifest.json`: checkpoint hashes, sample count, resampling unit, and cost.
- `v17_causal_patching_budget.json`: planned-only causal scope and pass count.

Five-seed profile extraction and all semantic statistics consumed 0.01748 A100 wall-hours.
Approximate cumulative documented IRED GPU occupancy through this stopping point is 14.32 A100
wall-hours, including retained superseded and diagnostic runs. No application-workstream file was
read or modified.

## Incomplete and uncertain

All feature labels remain provisional until causal validation. The static associations average
over campaign variants and cannot establish that a feature causes fitness. Mutation-responsive
results use one crosscoder seed because the precomputed Delta-z cache predates the five-seed
extension. The catalytic and contact assignments retain their supplied/preprint provenance, and
9RH is a proxy ligand. Causal patching and every oracle remain pending explicit approval.
