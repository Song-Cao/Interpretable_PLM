# Step 3 handoff: IRED adaptation through Gate 1

Date: 2026-10-22
Status: Gate 1 passed; stopped before Step 4 caching.

## Measurement and stopping contract

The backbone is ESM-C 300M at pinned revision
`e4bac860f0c502cd80f3aeac3d0ce6524c6627cb`. Adaptation uses rank-16 LoRA on all
linear layers, alpha 32, dropout 0.05, frozen embeddings, and exactly `Linear(960,1)`
as the fitness head. Task gain is measured by five-fold cross-validation inside the 895-row
pg_kfold training pool. The 59-row layer-freeze reserve is used only for activation drift,
shuffled-label separation, and depth selection.

Every real fold early-stops on validation Spearman and restores its best state. The
ReduceLROnPlateau scheduler also monitors validation Spearman. Config-fixed fold 0 supplies
the final real adapter. Its best checkpoint occurs at epoch 9 and 207 optimizer steps;
training executed 483 steps before the patience rule fired and the best state was restored.
The shuffled-label control uses the same fold-0 training rows, initialization seed, optimizer,
rank, and exact learning-rate sequence for exactly 207 steps. It has no independent stopping
rule. The auditable counts are in `Reports/step3/gate1_training_complete.json`.

## Task gain

`Reports/step3/gate1_task_gain_cv.csv` contains only fully completed folds:

| fold | frozen Spearman | adapted Spearman | gain | best epoch | best steps | executed steps |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.54138 | 0.55794 | 0.01656 | 9 | 207 | 483 |
| 1 | 0.65519 | 0.69543 | 0.04025 | 17 | 391 | 667 |
| 2 | 0.59067 | 0.56151 | -0.02916 | 9 | 207 | 483 |
| 3 | 0.60042 | 0.68128 | 0.08086 | 17 | 391 | 667 |
| 4 | 0.60623 | 0.59302 | -0.01321 | 13 | 299 | 575 |

Mean frozen and adapted Spearman are 0.59878 and 0.61784, respectively, for mean task
gain +0.01906. Two of five fold gains are negative; the positive mean is not uniform
improvement.

## Drift, separation, and Gate 1

Depth 29 maximizes mean real-minus-shuffled relative drift on the reserve. Real drift is
1.29809, shuffled drift is 0.84576, and the paired bootstrap 95% interval for the gap is
[0.38609, 0.52066]. Linear CKA is 0.06415 for the real adapter and 0.81942 for the
shuffled control. All 30 depths have a positive bootstrap lower bound for the drift gap.
The numerical activation-change assertion passes: maximum change is 7475.10 against a
floor of 3.69e-5. Token alignment also passes.

Gate 1 passes because mean task gain is positive and the selected-depth separation interval
is strictly above zero. The full curve and plot are
`Reports/step3/gate1_by_layer.csv` and `Reports/step3/gate1_drift_by_depth.png`.

Depth 29 is the final available depth, so the registered +2/+4 sensitivity offsets do not
exist. The available -2/-4 depths remain positive: gaps are 0.32644 at depth 27 and 0.24897
at depth 25. This boundary optimum leaves no later ESM-C block after the analysis layer and
must be considered when the common-layer agreement check is performed on the second dataset.
Depth 29 is IRED-only for now; no common cross-dataset layer has been frozen.

## Checkpoints and cache interface

Pod-only checkpoints are:

- `Models/step3/IRED/real_adapter.pt`: 66 tensors, 3,903,813 bytes, SHA-256
  `38da4c4aec3cbec958e866679be078c24948cd434be0d65b50ba2a337a02e4b4`.
- `Models/step3/IRED/shuffled_adapter.pt`: 66 tensors, 3,904,101 bytes, SHA-256
  `287db0b1206fa2190af0814e3a314f87966c4cf396e833239a9e7b0f9511219b`.

Each contains 64 LoRA parameter tensors plus the readout weight and bias. Load only into the
pinned backbone and matching rank-16 adapter with
`crossfit.adaptation.load_trainable_state`. Checkpoint provenance is committed in
`Reports/step3/gate1_checkpoint_verification.json`.

`Cache/step3/IRED/` remains on the pod and contains:

- `reserve_all_layers_adapted.npy`: float16 `[59, 30, 290, 960]`.
- `reserve_all_layers_shuffled.npy`: float16 `[59, 30, 290, 960]`.
- `full_selected_layer_base.npy`: float16 `[1182, 290, 960]`, depth 29.
- `full_selected_layer_adapted.npy`: float16 `[1182, 290, 960]`, depth 29.
- `full_variant_rows.npy`: int64 `[1182]`, row alignment for both full tensors.

The directory occupies 3.1 GB. Every representation extraction forces eval mode, and
full-corpus extraction retains only depth 29 batch-by-batch. No full-dataset all-depth tensor
was written.

## Incomplete, uncertain, or deviating

The first completed Gate attempt used a fixed 80 epochs, contrary to the clarified stopping
rule, and is archived rather than reported. Its generic PEFT export was empty; the replacement
explicit state format is loadable and hash-recorded. A later checkpoint recovery exposed
initialization-before-seeding, which is now fixed. No partial fold or interrupted model state
was reused.

The first analysis of the authoritative 207-step checkpoints extracted the shuffled model
while it remained in training mode, so dropout contaminated that cache. Training itself was
complete and unaffected. The faulty report/cache is preserved under
`Reports/archive/step3_gate1_trainmode_shuffled/` and
`Cache/step3/IRED_trainmode_shuffled_20261022/`. The active analysis was recomputed from the
complete checkpoints with forced eval mode. The archived result is not a Gate result.

Step 4 cache bounds, without writing a Step 4 cache, are in
`Reports/step3/step4_budget_ired.json`: 0.996672 GB for all training residues, 0.897005 GB
for the mutable-position union, and 0.009492 GB for mutations only. v12 does not define the
window radius or distant-position draw count, so no final subsampled budget is claimed.
