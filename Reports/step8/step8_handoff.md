# Step 8 handoff: IRED feature atlas and causal validation

Date: 2026-11-20

Status: complete under CROSSFIT v20

Interpretation: exploratory because IRED failed Gate 1 on task gain

## Result in one paragraph

At ESM-C depth 26, the populated shifted class is `rotated`. Relative to conserved features,
rotated features concentrate in the substrate/catalytic structural domain and at the dimer
interface, and are depleted from the cofactor-binding domain. This domain result is stronger and
more interpretable than the secondary distance-shell result: it reproduces the first/second-shell
enrichment but locates it chiefly in the substrate/catalytic half of the dimeric enzyme. Patching
establishes that rotated features affect the registered additive fitness readout, but the fraction
of a feature's activation in any domain or shell does not predict its causal-effect magnitude
after FDR correction. Distal mutations do not produce an aggregate increase in active-site/cofactor
response: mean absolute delta-z is lower at the nine functional positions than at 211 generic-
distal positions. These are exploratory IRED findings and do not revise the failed Gate 1 verdict.

## Provenance and registered inputs

The analysis uses the v17 pooled-null taxonomy at width 1, five real crosscoder seeds, and the
authoritative depth-26 cache. Before final reporting, a fresh wild-type forward pass on the
restarted RTX PRO 4500 Blackwell pod reproduced both canonical reference blocks exactly: base
Pearson 1.0000, relative RMSE 0, scale 1; adapted Pearson 1.0000, relative RMSE 0, scale 1.
`v20_cache_session_verification.json` holds the complete check. The copied 120-MB delta-z and
residue-profile cache remains gitignored.

The primary dictionary seed contains 642 live features: 393 conserved, 248 rotated, and one
re-weighted. One conserved feature was active once during crosscoder training but has zero
activation mass under registered per-sequence semantic coding; it remains in the atlas with an
explicit undefined-profile flag rather than being silently removed.

## Structural partition

The 290 positions comprise 159 cofactor-binding-domain residues, 130 substrate/catalytic-region
residues, and one structurally unassigned N-terminal residue. Orthogonally, 105 residues are at
the dimer interface, 184 are core, and 106 are surface. Assignments come from CATH topology,
deposited 5OCM ligand contacts, all six chain copies in the three biological dimers, and assembly-
level RSA. The exact chain provenance and the overlap with distance shells are in
`v20_domain_partition.md` and the `ired_domain_*` CSV files.

## Class-level domain enrichment

All effects are rotated minus conserved in within-seed SD units. Tests permute class labels
within each of five dictionary seeds; intervals bootstrap seed blocks and features. The per-seed
effects are columns in `v20_domain_class_enrichment.csv`.

| structural axis | positions | effect | 95% interval | q |
|---|---:|---:|---:|---:|
| substrate/catalytic domain | 130 | +0.998 | [+0.917, +1.072] | 0.00010 |
| cofactor-binding domain | 159 | -0.990 | [-1.067, -0.906] | 0.00010 |
| dimer interface | 105 | +0.608 | [+0.540, +0.677] | 0.00010 |
| core | 184 | -0.281 | [-0.356, -0.205] | 0.00010 |
| surface | 106 | +0.281 | [+0.205, +0.356] | 0.00010 |

The one-position unassigned category also rejects the null but cannot support a biological claim.
The domain result leads Step 8. Distance shells remain secondary: rotated features are enriched
in first and second shells and depleted at generic-distal positions, as reported in
`v18_class_semantic_tests.csv`.

## Causal patching and spatial linkage

Patching targeted 116 rotated features above both decoder medians on 256 registered training-
pool variants. The no-op, dead-feature, and random-live controls passed before candidate effects
were read. Rotated absolute effects exceed both control distributions; the median feature effect
is 0.003832 SD and the 95th percentile is 0.015320 SD. Full feature-variant effects, controls,
quantiles, and distribution tests remain in the v18/v19 patching artifacts.

No activation-location fraction predicts effect magnitude at q <= 0.05. The strongest domain
associations are substrate/catalytic fraction (rho +0.198, q=0.114) and cofactor-domain fraction
(rho -0.207, q=0.114); core fraction is rho +0.171 (q=0.121). The secondary shell analysis is
also null after FDR. These are statements about average spatial occupancy, not evidence that
individual features lack spatial meaning. Per-feature domain fractions and effects are committed
in `v20_causal_feature_domain_fractions.csv`.

## Long-range mutation response

The registered functional region is the union of proposed catalytic, substrate-contact,
NADPH-contact, and NADPH-proximal positions (nine unique residues). The comparison region is the
211-position generic-distal shell. There are 289 variants whose substitutions all lie in the
distal-functional or generic-distal shells.

| feature set | functional mean abs delta-z | distal mean abs delta-z | paired difference | 95% interval | q |
|---|---:|---:|---:|---:|---:|
| all live | 5.373 | 6.097 | -0.724 | [-0.727, -0.719] | 0.00010 |
| rotated | 4.852 | 7.139 | -2.287 | [-2.291, -2.282] | 0.00010 |
| conserved | 5.714 | 5.414 | +0.300 | [+0.296, +0.304] | 0.00010 |

Thus the aggregate representation does not show the hypothesized distal-mutation amplification
inside the functional region. The opposing class-specific pattern is descriptive: conserved
features respond slightly more inside, while rotated features respond much more strongly at
distal positions. `v20_long_range_response_by_variant.csv` contains every paired observation.

## Stability and solubility oracles

The official ThermoMPNN point-mutant model was run in CPU-only mode on 5OCM chain A. Its sign
agrees with all six registered experimental anchors: T241A is stabilizing; S68P, P203Q, H244Y,
A72T, and G37C are destabilizing. Campaign multi-mutant scores are the explicit additive sum of
the official point-mutant ddG predictions. This covers 1,181/1,182 variants; M1I-S68P-T241A is
unsupported because 5OCM has no coordinates for campaign residue 1, and no value is imputed.

SoluProt 1.0.1.0 was run in its authors' pinned CPU environment with the official no-TMHMM model.
It covers all 1,182 variants but is labelled exploratory: IRED has no solubility anchor, and the
authors state that SoluProt is not designed to measure point-mutation effects. ThermoMPNN and
SoluProt agree only weakly across variants (Spearman 0.110). Per-feature correlations and
500-draw rank-pair bootstrap intervals are columns in `v20_ired_feature_atlas.csv`; they are
descriptors, not causal evidence.

## Feature atlas and rotated sub-types

`v20_ired_feature_atlas.csv` has one row per live width-1 feature. It includes taxonomy
statistics, decoder norms, Delta, cosine, activation frequency, residue-profile-matched seed
replication, five top positions with domain and shell labels, activation-weighted values and
global-FDR q values for 26 annotation tracks, causal effects where measured, and both oracle
correlations. The per-feature annotation tests have no q <= 0.05 after correcting the full
feature-by-track family; biological inference therefore remains class-level.

Five descriptive rotated sub-types maximize the pre-registered silhouette candidates, but the
silhouette is low (0.202) and one sub-type has only two features:

| sub-type | features | profile summary | patched features | mean abs effect (SD) |
|---:|---:|---|---:|---:|
| 0 | 112 | substrate/catalytic, interface, core, helical | 60 | 0.00630 |
| 1 | 69 | surface, non-interface, generic distal | 34 | 0.00618 |
| 2 | 54 | cofactor domain, core, sheet/NADP context | 13 | 0.00401 |
| 3 | 11 | first shell, NADP/substrate-contact context | 9 | 0.00355 |
| 4 | 2 | catalytic/ketone-proxy positions | 0 | not measured |

Causal magnitude does not differ across the four sub-types represented in the 116-feature
patching set (permutation p=0.378). The sub-types describe distinct position profiles but do not
explain the pooled causal-effect distribution on IRED. Assignment, selection diagnostics,
enrichment rows, and causal test are committed in the `v20_rotated_subtype_*` artifacts.

## Audit, limitations, and interface

The previously missing mutated-position-group records were recovered without inference and are
now in `v20_mutation_responsive_group_effects.csv`. `v20_per_run_record_audit.csv` now marks all
reported Step 8 means, ranges, and intervals as backed by committed observations or named
per-seed values.

Limitations are explicit: IRED failed Gate 1; all feature results are hypothesis-generating.
Class labels depend on the registered additive attribution readout, which is weaker than the
nonlinear prediction ceiling. The domain sub-types are low-separation descriptive clusters.
ThermoMPNN multi-mutant scores assume additivity, SoluProt has no assay-specific validation, and
neither oracle enters the patching result.

This closes IRED Step 8. No further IRED analysis is proposed. The next dataset runs the frozen
v20 specification without per-dataset adjustment.

## Compute accounting

This continuation occupied the restarted pod for approximately 0.67 wall-clock hours. GPU use
was approximately 0.004 GPU-hours, solely for the mandatory two-reference cache verification;
ThermoMPNN was forced to CPU mode, and all other analyses were local CPU work. Cumulative
documented IRED use is approximately 0.397 RTX PRO 4500 Blackwell GPU-hours, 14.92 A100
GPU-hours, and 0.03 RTX 6000 Ada GPU-hours.
