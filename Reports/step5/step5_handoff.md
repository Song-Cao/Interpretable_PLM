# Step 5 handoff: IRED dense-probe reference at depth 26

Date: 2026-11-12
Status: complete after v18 depth correction

## Inputs and provenance

The campaign and external-plate tensors were re-extracted at hidden-state depth 26 from
`biohub/ESMC-300M` revision `e4bac860f0c502cd80f3aeac3d0ce6524c6627cb` and the hash-verified
Step 3 adapter. The cache contains 1,182 campaign and 54 plate sequences, each with 290 x 960
raw activations. Float16 tensor payload is 1,376,410,112 bytes. The network-volume path is
`Cache/step5/IRED/analysis_layer_26/`; its committed provenance copy is
`Reports/step5/ired_analysis_cache_manifest.json`. Fresh wild-type forward verification was
exact for both base and adapted reference blocks before every downstream reader ran.

The producing runtime was an NVIDIA RTX PRO 4500 Blackwell, Torch 2.11.0+cu130, bfloat16
compute, float16 storage, matmul TF32 off and cuDNN TF32 on. ESM-C fell back to PyTorch SDPA,
pure-PyTorch rotary embeddings, and unfused Transformer Engine operations.

## Dense probes

Selection used five-fold validation within the 895-row pg_kfold fold-0 training pool. No test,
round-held-out, reserve, or plate label was used for selection.

| arm (depth 26) | validation Spearman | pg_kfold test | round-held-out | plate |
|---|---:|---:|---:|---:|
| adapted mean | 0.628324 | 0.551305 | 0.533867 | 0.612045 |
| base mean | 0.574769 | 0.561556 | 0.527706 | 0.604422 |
| PCs of adapted-base delta | 0.373585 | 0.459725 | 0.379748 | 0.552049 |

`dense_fit_mean` remains the validation-selected Gate 2 reference. Held-out predictions are
persisted in `Reports/step5/<arm>_predictions.csv`; every CV held-out prediction is in
`<arm>_cv_predictions.csv`; searches and checkpoint hashes are in `<arm>_search.csv`.

## Superseded result

The prior nominal depth-26 dense tables actually loaded quarantined depth-29 tensors and are
archived under `Reports/archive/ired_depth29_v17/`. They are not active results. The v18 values
above are the outcome-neutral recomputation and replace them.

## Incomplete or uncertain

None for the registered dense-probe reference. The historical A100 producer settings for older
caches were unavailable; that limitation does not apply to this newly extracted cache.
