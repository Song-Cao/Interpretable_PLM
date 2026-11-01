# Step 9 handoff — exploratory IRED sparse-feature prediction

## Status

The IRED feature-set comparison, expressivity ladder, and reporting-only sample-efficiency curve
are complete. Selection used five-fold validation Spearman inside the pg_kfold training pool.
Neither pg_kfold test, round-held-out, nor plate labels influenced selection. Every selected fit
has member-level predictions and a persisted checkpoint; full-search predictions remain
pod-only parquet where registered.

Gate 1 remains failed, so these are exploratory pipeline-shakedown results and cannot support a
confirmatory central-claim table.

## Feature-set result

The best arm by the registered selection metric is `phi_w1_max_all_live`: width 1, max pooling,
all 642 live features, MLP, eight final-refit epochs. Bootstrap intervals below are 95%.

| condition | Spearman | bootstrap interval |
|---|---:|---:|
| pg_kfold test | 0.5495 | [0.4337, 0.6465] |
| round-held-out | 0.5652 | [0.4869, 0.6348] |
| external plate | 0.5002 | [0.2442, 0.7094] |

All 18 width/pooling/feature-set arms, with Spearman and NDCG bootstrap intervals for all three
conditions, are in `ired_phi_feature_set_results.csv`. The largest plate point estimate was
0.5729 for `phi_w2_max_objective_shifted`, with a wide interval [0.3571, 0.7506], but that arm
was not selected by validation performance. It is therefore a post-hoc diagnostic, not the
headline result.

The pre-registered transfer expectation broadly holds for the selected arm: its plate Spearman
0.5002 improves on the earlier dense ESM arms (0.23-0.44) but remains below the one-hot and
AAIndex range (0.53-0.57). Some non-selected narrow `w=2` arms overlap or slightly exceed that
classical range; their intervals are wide and their taxonomy is width-unstable, so this is not
treated as an unqualified sparse-feature success. Max pooling is critical: several sum-pooled
arms reverse rank on plate despite similar validation performance.

## Expressivity ladder

The full ladder was run only for the headline arm, as registered. Other feature sets received
L0 and L1.5 only. The interaction gate failed (`H^2=0.9357`, permutation `p=1.0`, 200
permutations), so L2 was not fitted.

| level | validation Spearman | pg test Spearman | pg test R2 | round Spearman | plate Spearman |
|---|---:|---:|---:|---:|---:|
| L0 ridge | 0.5797 | 0.5021 | -0.0481 | 0.3842 | -0.5289 |
| L1.5 forest | 0.6435 | 0.5741 | 0.1646 | 0.5826 | 0.3957 |
| L1 splines | 0.4982 | 0.4206 | -0.1604 | 0.3615 | -0.6229 |
| L4 selected MLP | 0.6548 | 0.5495 | 0.1046 | 0.5652 | 0.5002 |

The registered expressivity gap is `R2(L4)-R2(L0) = 0.152725`. L1.5 has the strongest test
point estimates but was not allowed to replace the validation-selected L4 headline after test
inspection. Exact bootstrap intervals and every auxiliary arm are in
`ired_expressivity_ladder.csv`.

## Sample efficiency and budget

The reporting-only curve used the already selected arm and fixed eight-epoch policy, with ten
seeds per size. Mean pg_kfold test Spearman rose from 0.406 at N=16 to 0.539 at N=512. It did
not select any model or hyperparameter.

Ten representative ladder fits took 4.09 seconds. The projected 925-fit grid was 0.105 GPU-hours,
so no search reduction was needed. Measured GPU wall time was 0.3573 hours for dense/feature-set
prediction, 0.0948 for the ladder, and 0.0129 for sample efficiency: 0.4650 hours total for the
prediction portion, well below the four-hour Part B cap. Completed checkpoint/prediction pairs
were resumed after interruptions; no partial fit contributed a result.

## Artifacts

- `Reports/step9/ired_phi_feature_set_results.csv`: all feature-set arms and three conditions.
- `Reports/step9/ired_expressivity_ladder.csv`: complete registered ladder outputs.
- `Reports/step9/sample_efficiency_{runs,summary,predictions}.csv`: reporting-only curve.
- `Reports/step9/*_predictions.csv`: every ensemble member and ensemble mean by condition.
- `Models/step9/IRED/`: pod-only loadable checkpoints; hashes are recorded in search/run files.

## Incomplete or uncertain

No causal semantics follow from predictive performance. Plate estimates use only 54 variants
and have wide intervals. The strong dependence on pooling and provisional width-specific
taxonomy must be retested on GB1. No enrichment, patching, or oracle work was run here.
