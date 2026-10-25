# Prediction persistence audit

Date: 2026-10-25

This audit describes the completed historical IRED runs. It does not imply that missing
predictions were reconstructed. The replacement contract applies to every future fit and is
implemented in `crossfit.persistence`.

| step | arm | per-fold held-out predictions | per-condition predictions | individual members | checkpoints | recovery |
|---:|---|---|---|---|---|---|
| 1 | data curation | not applicable | not applicable | not applicable | no fitted models | not applicable |
| 2 | one-hot ridge, unaugmented | missing | pg test, round, plate: exact prediction available | seed rows not expanded; exact fit is seed-invariant | selected pg model copies, pod-only | CV requires refit |
| 2 | one-hot ridge + dPLL | missing | pg test, round, plate: exact prediction available | seed rows not expanded; exact fit is seed-invariant | selected pg model copies, pod-only | CV requires refit |
| 2 | AAIndex ridge, unaugmented | missing | pg test, round, plate: exact prediction available | seed rows not expanded; exact fit is seed-invariant | selected pg model copies, pod-only | CV requires refit |
| 2 | AAIndex ridge + dPLL | missing | pg test, round, plate: exact prediction available | seed rows not expanded; exact fit is seed-invariant | selected pg model copies, pod-only | CV requires refit |
| 2 | ESM mean ridge | missing | pg test, round, plate: ensemble mean available | missing | five selected pg members, pod-only | members can be inferred when pod returns; CV requires refit |
| 2 | ESM mean MLP | missing | pg test, round, plate: ensemble mean available | missing | five selected pg members, pod-only | members can be inferred when pod returns; CV requires refit |
| 2 | ESM mean random forest | missing | pg test, round, plate: ensemble mean available | missing | five selected pg members, pod-only | members can be inferred when pod returns; CV requires refit |
| 2 | ESM mutated-position ridge | missing | pg test, round, plate: ensemble mean available | missing | five selected pg members, pod-only | members can be inferred when pod returns; CV requires refit |
| 2 | ESM residue convolution | missing | pg test, round, plate: ensemble mean available | missing | five selected pg members, pod-only | members can be inferred when pod returns; CV requires refit |
| 2 | ESM residue-delta convolution | missing | pg test, round, plate: ensemble mean available | missing | five selected pg members, pod-only | members can be inferred when pod returns; CV requires refit |
| 2 | one-hot positional convolution | missing | pg test, round, plate: ensemble mean available | missing | five selected pg members, pod-only | members can be inferred when pod returns; CV requires refit |
| 3 | registered rank 16, lr 1e-4 | fold 0 real and shuffled available; four folds missing | not applicable: Step 3 measured CV task gain | not applicable | fold 0 real and shuffled only, pod-only | fold 0 backfilled by inference; other folds require refit |
| 3 | rank 16, lr 3e-4 | missing | not applicable | not applicable | none | requires refit |
| 3 | rank 32, lr 1e-4 | missing | not applicable | not applicable | none | requires refit |
| 3 | rank 32, lr 3e-4 | missing | not applicable | not applicable | none | requires refit |
| 4 | activation caching | not applicable | not applicable | not applicable | complete pod-only cache | no prediction model |
| 5 | dense probes | not started | not started | not started | not started | pending |
| 6 | self-pair null | not applicable | not applicable | not applicable | complete pod-only checkpoint | no fitness predictions |
| 7 | width diagnostic | not applicable | not applicable | not applicable | 12 complete pod-only checkpoints | no phi predictor fitted |
| 8 | causal patching | not started | not started | not started | no fitted models yet | not applicable |

Step 2's `<arm>_search.csv` files contain configuration-fold metrics, not predictions.
Its historical `<arm>_predictions.csv` files contain one ensemble mean per condition and
variant; they do not contain member rows. Step 3 reduced every held-out fold to Spearman and
discarded the predictions. The sweep discarded every adapter; the registered run retained
only fold 0's real adapter for drift and its matched shuffled control. Frozen heads and the
other four registered adapters were not retained.

`Code/scripts/backfill_ired_gate1_fold0_predictions.py` performed the inference-only backfill
for the two surviving Step 3 checkpoints. It verified their recorded SHA-256 hashes and wrote
179 held-out rows for each checkpoint without fitting. The result is
`Reports/step3/rank16_lr1e-4_fold0_checkpoint_backfill_predictions.csv`. The other four folds
remain unrecoverable without retraining.

For future runs, the common committed schema is
`dataset,step,arm,config_id,fold,seed,condition,variant_row,target,prediction`, plus
`uncertainty` when produced. Full-search prediction partitions are pod-only parquet under
`Cache/`; selected CV and all per-member evaluation-condition predictions are committed under
`Reports/`. Checkpoint manifests record a SHA-256 hash for every retained model. All checkpoint
classes are enabled by default in the step YAML; disabling one requires an explicit flag.
