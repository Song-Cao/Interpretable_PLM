# Step 7 handoff: IRED taxonomy and recomputed Gate 2

Date: 2026-11-12
Status: taxonomy retained; Gate 2 recomputed at depth 26

## Taxonomy

Steps 6 and 7 consumed the authoritative training-only depth-26 cache and were not rerun under
v18. The five-real-seed, pooled drift-matched-null taxonomy therefore stands. At w=1 the mean
provisional proportions are: dead 32.67%, emergent candidate 0.00%, evolution-only 0.03%,
rotated 25.23%, re-weighted 0.07%, and conserved 42.22%. Rotation is the only shifted class
populated in all five real dictionaries. The expanded 34-null resolution audit supports zero
emergent candidates; it is not a three-null p-value-resolution artifact.

The Step 4 crosscoder cache is depth 26, training-only, normalized by one frozen scalar per
model, and contains 259,550 residue positions. Its committed manifest is
`Reports/step4/cache_manifest.json`.

## Gate 2

The dense probes and phi predictors were recomputed from the new provenance-verified depth-26
cache. Selection remained validation-only. The validation-selected dense reference is
`dense_fit_mean` (validation Spearman 0.628324; pg_kfold test 0.551305). The validation-selected
sparse arm is `phi_w2_max_all_live` (validation 0.636133; test 0.550402).

Paired bootstrap dense-minus-sparse on the 228-row pg_kfold test set is 0.000903 with 95%
interval [-0.038226, 0.040975]. The pre-committed verdict remains `sparse_competitive`; the
interval does not support improvement in either direction. The machine-readable decision is
`Reports/step7/gate2_result.json`.

Gate 2 is a routing gate. Its IRED-specific resource override still permits Step 9, but the
substantive result is competitiveness, not superiority.

## Incomplete or uncertain

The taxonomy remains exploratory because IRED Gate 1 failed on task gain. No confirmatory
central-claim evidence is inferred from it.
