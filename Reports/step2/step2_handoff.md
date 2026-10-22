# Step 2 handoff: IRED predictor comparison

Date: 2026-10-22
Status: pooled and classical arms complete; convolutional arms stopped by the v12 compute rule.

## Scope and selection contract

All results are for IRED alone. Primary fold 0 contains 895 training-pool and 228 test variants after excluding the 59-row layer-freeze reserve. Every hyperparameter was selected by five-fold cross-validation inside those 895 rows using validation Spearman. The test fold, round-5 holdout, and 54-row plate set were never used for fitting, tuning, calibration, or selection. The plate table was evaluated only by Spearman and NDCG.

Random-search seeds and spaces are in `Code/config/step2.yaml`. Continuous classical searches ran 60 configurations. The finite ESM ridge space contains 48 unique configurations; MLP and random forest ran 60. Every reported stochastic result is a five-member ensemble with seeds 0-4. The exact convex classical fits are seed-invariant and therefore have ensemble SD zero.

## Extraction and cache interface

The pinned backbone is `biohub/ESMC-300M` revision `e4bac860f0c502cd80f3aeac3d0ce6524c6627cb` under `esm==3.4.1.post1`. It returned 960 channels and 30 hidden-state depths.

`Cache/step2/IRED/representations/` remains on the pod and contains:

- `variant_residue.npy`: float16 `[1182, 290, 960]`, aligned to `variant_rows.npy`.
- `plate_residue.npy`: float16 `[54, 290, 960]`, plate CSV order.
- `wild_type_residue.npy`: float16 `[290, 960]`.
- `reserve_all_layers_base.npy`: float16 `[59, 30, 290, 960]`, aligned to `reserve_rows.npy`.
- `metadata.json`: model, shapes, task counts, elapsed time, and byte count.

The realized cache is 1,674,691,432 bytes before the small metadata file. Extraction took 90.62 seconds of model work. It stores all depths only for the reserve, never for the full campaign.

`Cache/step2/IRED/dpll.csv` has unique columns `aa_seq,dpll` and 1,237 rows: wild type, all campaign sequences, and all plate sequences. It used 3,441 variant-context masked tasks and 277 wild-type position tasks. Every scalar is finite. The cache is consumed only on the pod.

## Training and artifacts

Library interfaces:

- `Code/crossfit/losses.py`: direct-score composite loss, in-batch ranking pairs, standard per-variant Tobit, learned residual scale, and L2 term.
- `Code/crossfit/representations.py`: dense whole-sequence one-hot and AAIndex, full/mean/mutated-pool ESM states, and residue deltas.
- `Code/crossfit/models.py`: exact diagonal ridge, linear head, MLP, random forest, and residual 1-D convolution.
- `Code/crossfit/training.py`: training-only normalization, validation-Spearman early stopping, and validation-free fixed-epoch refits.
- `Code/crossfit/evaluation.py`: direct-score metrics, five-member ensembles, bootstrap intervals, and Mondrian conformal intervals.

Models are loadable on the pod from `Models/step2/IRED/<arm>/seed_<0-4>.*`. Neural `.pt` files contain the state dict, learned residual scale, frozen normalizer, selected configuration, selected epoch count, and input shape. Random forests are `.joblib`. Classical `.npz` files contain coefficients, intercept, penalties, and fitting variant rows.

Neural search early-stopped on validation Spearman and restored the best checkpoint. Each
five-seed final refit used the rounded median best epoch across the selected configuration's
five CV folds: 16 epochs for ESM-mean ridge, 7 for ESM-mean MLP, and 9 for
ESM-mutated-position ridge. Thus final refit duration was selected without consulting test,
round-held-out, calibration, or plate labels.

Each arm has compact committed files under `Reports/step2/`:

- `<arm>_search.csv`: one row per configuration and inner fold.
- `<arm>_timing.json`: ten-fit timing audit, count, projection, and sampling seed where stochastic.
- `<arm>_result.json`: selected configuration, boundary audit, three evaluation conditions, ensemble variability, bootstrap intervals, and conformal strata.
- `<arm>_predictions.csv`: `condition,variant_row,target,prediction` and `epistemic_sd` for stochastic arms.

## Results read from disk

The three complete tables are:

- `Reports/step2/ired_pg_kfold_test_results.csv`
- `Reports/step2/ired_round_held_out_results.csv`
- `Reports/step2/ired_plate_results.csv`

Spearman point estimates are:

| arm | inner CV | pg test | round holdout | plate |
|---|---:|---:|---:|---:|
| onehot ridge, unaugmented | 0.6151 | 0.5320 | 0.5291 | 0.5637 |
| AAIndex ridge, unaugmented | 0.6150 | 0.5322 | 0.5291 | 0.5634 |
| onehot ridge + dPLL | 0.6117 | 0.5441 | 0.5394 | 0.5300 |
| AAIndex ridge + dPLL | 0.6114 | 0.5442 | 0.5371 | 0.5296 |
| ESM mean ridge | 0.6076 | 0.5224 | 0.4987 | 0.3368 |
| ESM mean MLP | 0.6058 | 0.4397 | 0.4433 | 0.2392 |
| ESM mean random forest | 0.5396 | 0.4320 | 0.4455 | 0.2317 |
| ESM mutated-position ridge | 0.5501 | 0.5452 | 0.4483 | 0.4443 |

Bootstrap confidence intervals and every other metric are in the three tables. IRED fold 0's observed hit rate at the standardized wild-type cutoff 0.3217626895 is 0.4956140351.

The architecture shortlist and exact selected configurations are in `Reports/step2/ired_architecture_shortlist.csv`. Selected boundaries that require reporting are:

- ESM mean ridge: minimum weight decay and minimum batch size.
- ESM mean MLP: maximum dropout, maximum learning rate, minimum batch size.
- Random forest: maximum depth, maximum `min_samples_leaf`, and maximum feature fraction.

The fixed `1e-8` dPLL coefficient penalty is not a searched boundary. Classical site penalties did not land exactly at `1e-4` or `1e4`.

## Pooling-domain answer

`Reports/step2/ired_pooling_domain.json` selects all-residue mean pooling because validation Spearman is 0.60765 versus 0.55007 for mutated-position pooling under matched ridge heads. This selection uses validation only. The pg-test point estimates reverse weakly (0.52243 versus 0.54519) with overlapping intervals, and plate estimates are 0.33684 versus 0.44425. Those reversals are reported but cannot override the selection protocol.

## Conformal calibration

The 134-row calibration split is excluded from each final pg model fit. Quantiles are fit separately by mutation count and position novelty. All observed test strata were represented in calibration, but several novelty strata contain only one to three calibration rows; their finite-sample quantiles are infinite and coverage is consequently 1.0. This is valid but uninformative uncertainty, not evidence of exceptional calibration.

## Incomplete, uncertain, or deviating

Ten representative `esm_residue -> conv residual` fits averaged 137.72 seconds. This projects to 11.48 GPU-hours at 60 configurations and 5.74 hours at the required minimum 30 for one arm. The two required ESM convolutional arms therefore cannot both fit under the nine-hour session ceiling after Step 2 pooled arms and Step 3; the optional one-hot convolution was also omitted. Timing evidence is committed as `Reports/step2/esm_residue_conv_timing.json`. No arm is reported from fewer than 30 configurations, and no ensemble seed or measurement was cut.

The exact published classical normal-equation baselines optimize diagonally penalized squared error, and scikit-learn random forests optimize squared-error tree splits. Those solvers cannot optimize the differentiable composite Tobit/ranking objective used by the neural arms. This is the literal published-form implementation in sections 2f-2g, but it is an objective mismatch relative to section 2a's statement that every model uses one identical objective. Results preserve both facts rather than hiding the mismatch.

The completed neural searches used validation Spearman for early stopping, best-checkpoint
selection, and ReduceLROnPlateau. This is the standing policy for every dataset: all three
training decisions use the same pre-committed quantity, preserving cross-dataset
comparability. Training loss remains a diagnostic and an optimization objective, not a
scheduler monitor.

Step 4 budget bounds are in `Reports/step3/step4_budget_ired.json`. All 895 training variants at all 290 residues would occupy 0.996672 GB for two fp16 models; mutation-only is a 0.009492 GB lower bound. v12 does not fix the window radius or distant-position count, so a final realized subsampled budget cannot be named without those Step 4 choices. No Step 4 cache was written.
