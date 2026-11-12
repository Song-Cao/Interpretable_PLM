# Step 9 handoff: IRED phi prediction at depth 26

Date: 2026-11-12
Status: feature-set arms and registered expressivity ladder complete

## Inputs and selection

All phi inputs are pooled residue codes from the provenance-verified depth-26 campaign and
plate activations. Widths 1 and 2 and sum, max, and binarized pooling were evaluated with the
registered feature sets. Hyperparameters were selected only by five-fold validation inside the
pg_kfold training pool. Test, round-held-out, reserve, and plate labels were reporting-only.
Every CV held-out and final-condition prediction is committed in `Reports/step9/`.

The architecture shortlist validation Spearman values are MLP 0.642659, random forest
0.634123, and ridge 0.616636. The selected family is MLP. The selected feature arm is
`phi_w2_max_all_live`: validation 0.636133; pg_kfold test 0.550402 (95% bootstrap interval
[0.432634, 0.649810]); round-held-out 0.524105 [0.434199, 0.605063]; plate 0.463007
[0.215360, 0.659043]. The plate interval is wide at n=54 and does not establish superiority
over classical encodings.

The full arm table is `ired_phi_feature_set_results.csv`; the selected predictions are
`phi_w2_max_all_live_predictions.csv`.

## Expressivity ladder

On the headline all-live max-pooled w=2 arm, L0 test R2 is 0.058432 and L4 test R2 is
0.120331, giving R2(L4)-R2(L0) = 0.061899. This replaces the withdrawn wrong-depth gap of
0.152725. The additive attribution readout remains weaker than the nonlinear ceiling, but the
gap is materially smaller after correction. The registered interaction gate did not pass
(permutation p=1.0), and no analysis was altered in response.

## Compute and artifacts

The feature-set run used 0.174302 GPU-hours and the ladder 0.122662 GPU-hours. Search and final
checkpoints remain on the pod network volume under `Models/step9/IRED/`; every report-side
search table records checkpoint hashes. `ired_phi_cache_manifest.json` records the depth-26
source and exact wild-type reference verification.

## Incomplete or uncertain

Sample-efficiency curves were not rerun because v18 requested Step 5, Gate 2, Step 9 phi arms,
and the expressivity ladder. The selected-arm manifest's historical `sample_efficiency_pending`
field records that omission. No partial training result is reported.
