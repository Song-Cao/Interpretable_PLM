# IRED activation-cache diagnosis

## Verdict

The `2284` discrepancy is a storage/depth mismatch, not normalization and not a different
numerical model. `Cache/step3/IRED/full_selected_layer_{base,adapted}.npy` stores the original
depth-29 extraction. Those files were explicitly declared superseded when depth 26 was accepted,
but Step 8 and Step 9 later consumed them as if they were depth 26.

The authoritative Step 4 training-only shards really are depth 26 and agree with current
single-sequence re-extraction. Therefore the Step 6/7 crosscoders and the five-seed taxonomy,
including the 25.23% rotated fraction, remain supported. Later analyses that encoded the
superseded Step 3 full-campaign files do not.

## Direct comparisons

No wild-type depth-26 tensor was persisted. The available wild-type cache is the base model's
final normalized embedding. That exact wild-type comparison and direct selected-layer checks on
aligned campaign variants were used instead.

| comparison | Pearson | Spearman | through-origin scale | max absolute difference |
|---|---:|---:|---:|---:|
| WT final-layer base cache vs current | 0.999738 | 0.999555 | 0.999782 | 0.015625 |
| superseded Step 3 base file vs current depth 26 | 0.446689 | 0.311353 | 0.147252 | 2248 |
| superseded Step 3 adapted file vs current depth 26 | 0.175909 | 0.128945 | 0.034917 | 2198 |
| superseded Step 3 base file vs current depth 29 | 0.999685 | 0.999558 | 0.999504 | 64 |
| superseded Step 3 adapted file vs current depth 29 | 0.999788 | 0.999762 | 0.997500 | 48 |
| authoritative Step 4 base depth 26, fixed scalar reversed | 0.999788 | 0.999726 | 1.000248 | 14.02 |
| authoritative Step 4 adapted depth 26, fixed scalar reversed | 0.999605 | 0.999529 | 0.999853 | 11.52 |

The superseded-file/current-depth-26 ratios are not constant: median ratios are 0.107 for base
and 0.0255 for adapted, with sign changes and wide tails. Absolute error is strongly related to
cached magnitude (Pearson 0.900 and 0.955). These are consequences of comparing different depths,
not fp16 scatter. Against current depth 29, both files recover unit scale and correlations above
0.9996. The all-layer reserve independently confirms correct layer indexing: every same-depth
base comparison is above Pearson 0.99968, and cached depth 26 matches current depth 26 best.

For the authoritative depth-26 shards, reversing the registered fixed normalization scalars
recovers unit scale. Median absolute differences are 0.499 raw units for base and 0.621 for
adapted; correlations exceed 0.9996 and error has little association with activation magnitude
(0.112 and 0.047). That is the expected fp16-storage signature.

## Precision and normalization provenance

The cache-producing run used an NVIDIA A100, ESM-C 3.4.1.post1 at revision
`e4bac860f0c502cd80f3aeac3d0ce6524c6627cb`, bfloat16 model parameters, float32 extraction, and
float16 persistence. It requested ESM-C's default `flash_attention_2` path. The historical Torch,
CUDA, TF32 flags, realized attention kernel, and Transformer Engine availability were not
persisted and are reported as unknown rather than reconstructed from memory.

The diagnostic used an NVIDIA RTX PRO 4500 Blackwell, Torch 2.11.0+cu130, CUDA 13.0, the same
ESM-C version and model revision, bfloat16 parameters, float32 returned activations, no autocast,
matmul TF32 off, cuDNN TF32 on, and `highest` float32 matmul precision. Flash-Attention, xFormers,
and Transformer Engine were absent, so ESM-C used its PyTorch fallbacks. The earlier hygiene run
used an RTX 6000 Ada. Device/runtime differences are not the source of the large discrepancy:
the correctly identified layers agree closely across runs.

The Step 3 files are raw, unnormalized float16 activations. Step 4 multiplies them by one fixed
scalar per model (`0.0272291482` base; `0.0296749163` adapted) and persists normalized float16
shards. Current re-extraction was raw and was compared either directly to raw Step 3 data or to
Step 4 data after reversing those exact scalars.

## Scope

- Valid: Gate 1 reserve curve and depth-26 selection; authoritative Step 4 cache; Step 6 nulls;
  Step 7 width diagnostics, crosscoders, thresholds, and taxonomy.
- Superseded pending recomputation: Step 5 dense probes, Gate 2, Step 9 phi predictors and
  expressivity results, and Step 8 static/mutation-responsive enrichment. Their campaign inputs
  came from the superseded depth-29 Step 3 files while reports/configuration called them depth 26.
- Not run: candidate causal ablations. Hygiene passes after correcting its units, but patching
  remains blocked until the cache source is corrected and reviewed.

Machine-readable distributions, magnitude bins, per-position affine diagnostics, and all-depth
comparisons are in `v17_activation_cache_diagnosis.json`.
