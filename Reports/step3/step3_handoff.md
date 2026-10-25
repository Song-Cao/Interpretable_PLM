# Step 3 handoff: IRED adaptation through Gate 1

Date: 2026-10-24
Status: Gate 1 failed; stopped before Step 4 caching.

## Measurement and stopping contract

The backbone is ESM-C 300M at pinned revision
`e4bac860f0c502cd80f3aeac3d0ce6524c6627cb`. The registered adapter uses rank-16
LoRA on all linear layers, alpha 32, dropout 0.05, learning rate 1e-4, weight decay
0.01, frozen embeddings, and exactly `Linear(960,1)` as the fitness head. Task gain is
measured by five-fold cross-validation inside the 895-row pg_kfold training pool. The
59-row layer-freeze reserve is used only for activation statistics and layer selection.

Every real fold early-stops on validation Spearman and restores its best state. The
ReduceLROnPlateau scheduler also monitors validation Spearman. Config-fixed fold 0
supplies the final real adapter. Its best checkpoint occurs at epoch 9 and 207 optimizer
steps; training executed 483 steps before patience fired and restored that state. The
shuffled-label control uses the same fold-0 rows, initialization seed, optimizer, rank,
and exact learning-rate sequence for exactly 207 optimizer steps. It has no independent
stopping rule. The counts are in `Reports/step3/gate1_training_complete.json`.

## Task gain and Gate verdict

`Reports/step3/gate1_task_gain_cv.csv` contains only completed folds:

| fold | frozen Spearman | adapted Spearman | gain | best epoch | best steps | executed steps |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.54138 | 0.55794 | 0.01656 | 9 | 207 | 483 |
| 1 | 0.65519 | 0.69543 | 0.04025 | 17 | 391 | 667 |
| 2 | 0.59067 | 0.56151 | -0.02916 | 9 | 207 | 483 |
| 3 | 0.60042 | 0.68128 | 0.08086 | 17 | 391 | 667 |
| 4 | 0.60623 | 0.59302 | -0.01321 | 13 | 299 | 575 |

Mean frozen and adapted Spearman are 0.59878 and 0.61784. The five paired gains have
mean 0.01906 and standard error 0.01955. A two-sided paired t test gives
`t(4)=0.97506`, `p=0.38474`; two folds are negative. Task gain is therefore not
established at alpha 0.05, and `gate1_pass` is false. The machine-readable verdict is
`Reports/step3/gate1_report.json`.

The per-fold gain standard deviation is 0.04371, giving Cohen's d = 0.436. A two-sided
paired test at alpha 0.05 needs approximately 42.5 folds for 80% power at d = 0.44.
Even treating ten folds as independent realizations would give only `t(9)=1.379`,
`p=0.201`. A variant-level bootstrap of fixed out-of-fold predictions is not substituted:
it conditions away model-fit variance and can therefore make an underpowered model-level
comparison look artificially decisive. Task gain is small and positive in all four tested
configurations and increases monotonically across the tested capacity settings, but it is not
establishable from 895 training variants. The criterion is underpowered, not established as
absent, and Gate 1 remains failed.

## Corrected reserve-only layer selection

The original argmax rule selected collapsed final depth 29: real CKA 0.06415,
shuffled CKA 0.81942, real relative drift 1.29809, and maximum activation change
7475.10. This is not evidence of useful reorganization and would make downstream
patching equivalent to readout ablation. Its complete report is retained under
`Reports/archive/step3_gate1_depth29_pre_cka_floor/`.

The replacement rule excludes the first and final three depths, requires real CKA at
least 0.5, and selects the earliest eligible depth whose real-minus-shuffled drift gap
is at least 90% of the eligible maximum. The maximum eligible gap is 0.29208 and the
threshold is 0.26287. Depth 26 is the only qualifying depth and is frozen for the exploratory
IRED Steps 4-8 pipeline.

At depth 26, real and shuffled relative drift are 0.63382 and 0.34174, a ratio of
1.85470. The gap is 0.29208 with paired bootstrap 95% interval [0.28320, 0.30120].
Real and shuffled CKA are 0.91895 and 0.98112. Mean activation norms are 1137.86 for
the base model and 1042.84 for the adapted model, so
`||h_fit||/||h_evo||=0.91650`. The drift ratio is inside [0.5, 2], so no matched-drift
null is required.

Thus the drift and shuffled-label separation criteria are met cleanly at depth 26: real drift
is 0.63382 against shuffled drift 0.34174, with gap interval [0.28320, 0.30120]. The
representational precondition holds; only the task-gain criterion fails.

The complete 30-depth scalar curve is `Reports/step3/gate1_by_layer.csv`; it reports
both drifts, both CKA values, the gap interval, both mean activation norms, and their
ratio at every depth. Reporting sensitivity around depth 26 is:

| offset | depth | drift gap | 95% interval | real CKA | norm ratio |
|---:|---:|---:|---:|---:|---:|
| -4 | 22 | 0.17599 | [0.16837, 0.18392] | 0.98039 | 0.91709 |
| -2 | 24 | 0.21504 | [0.20645, 0.22396] | 0.97057 | 0.92076 |
| +2 | 28 | 0.37331 | [0.36353, 0.38337] | 0.62427 | 0.90284 |
| +4 | unavailable | — | — | — | — |

## Bounded task-gain sweep

The registered rank-16, learning-rate-1e-4 result above remains the Gate measurement.
A bounded diagnostic sweep changes only rank and learning rate upward while retaining
the same five folds, stopping metric, best-state restoration, head, alpha, dropout,
weight decay, optimizer, and data. Its three configurations are `(16,3e-4)`,
`(32,1e-4)`, and `(32,3e-4)`. Only fully completed five-fold configurations are
reported in `Reports/step3/task_gain_sweep/summary.csv`:

| configuration | mean gain | SE | t(4) | two-sided p | established |
|---|---:|---:|---:|---:|---:|
| rank 16, lr 1e-4 (registered) | 0.01906 | 0.01955 | 0.97506 | 0.38474 | no |
| rank 16, lr 3e-4 | 0.02314 | 0.01879 | 1.23124 | 0.28568 | no |
| rank 32, lr 1e-4 | 0.03086 | 0.02263 | 1.36380 | 0.24433 | no |
| rank 32, lr 3e-4 | 0.03104 | 0.01959 | 1.58446 | 0.18826 | no |

No tested configuration establishes positive task gain at alpha 0.05. Increasing rank
and learning rate raises the sample mean but not enough relative to between-fold
variation. This is a negative result about what can be established from IRED's 895-row
training pool, not a reason to change the Gate threshold or analysis layer. The three
new configurations consumed 3.02390 A100 GPU-hours in total.

## Checkpoints and cache interface

Pod-only registered checkpoints are:

- `Models/step3/IRED/real_adapter.pt`: 66 tensors, 3,903,813 bytes, SHA-256
  `38da4c4aec3cbec958e866679be078c24948cd434be0d65b50ba2a337a02e4b4`.
- `Models/step3/IRED/shuffled_adapter.pt`: 66 tensors, 3,904,101 bytes, SHA-256
  `287db0b1206fa2190af0814e3a314f87966c4cf396e833239a9e7b0f9511219b`.

Each contains 64 LoRA tensors plus readout weight and bias. Load only into the pinned
backbone and matching rank-16 adapter with `crossfit.adaptation.load_trainable_state`.
Checkpoint provenance is in `Reports/step3/gate1_checkpoint_verification.json`.

`Cache/step3/IRED/` remains on the pod. The all-depth reserve tensors have shape
`[59,30,290,960]`. The existing full-corpus selected-layer tensors have shape
`[1182,290,960]` but are depth 29 and explicitly superseded; they must not be consumed
as depth-26 data. The separate authoritative Step 4 depth-26 training-only cache is documented
in `Reports/step4/step4_handoff.md`; it does not include test or reserve variants.

## Deferred and incomplete

Depth 26 is frozen, but headline drift on the 228-row pg_kfold test split and the
reserve/train/test stability profiles still require GPU re-extraction. The reserve selected
the layer, the test split reports the headline drift after selection, and the training profile
is diagnostic only. Neither later profile may change the selection or the failed Gate verdict.

The first fixed-80-epoch Gate attempt and the train-mode shuffled extraction are
archived and never used. No incomplete fold or interrupted training result is reported.
All bounded task-gain configurations completed, and none passed. The preliminary Step 4
bounds are superseded by the realized training-only table at
`Reports/step4/step4_budget_ired.md`; the approved cache has now completed.

Historical out-of-fold predictions were not persisted. Fold 0's registered real and shuffled
checkpoint predictions have now been backfilled by inference at
`Reports/step3/rank16_lr1e-4_fold0_checkpoint_backfill_predictions.csv`; the other four folds
and all bounded-sweep fits remain unrecoverable without retraining, which is explicitly not
authorized. The persistence audit and future enforcement contract are in
`Reports/step3/prediction_persistence_audit.md`.
