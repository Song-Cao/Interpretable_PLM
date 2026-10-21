# Step 2 handoff: v10 migration and IRED curation

Date: 2026-10-21
Status: stopped at the required post-curation review boundary; no model training or ESM-C inference has run.

## Plan and standing contract

`CROSSFIT_Implementation_Plan_v10.docx` is the sole plan. Root `AGENTS.md` carries its standing invariants. Each dataset now has an independent direct-fitness pipeline; nothing pools labels, models, gradients, or results across datasets. `pg_kfold` is the only model-selection split. The layer-freeze reserve, round-held-out split, and external plate set cannot enter fitting or tuning.

`Code/tests/test_invariants.py` fails if the removed assay sampler, cross-dataset optimizer, grouped folds, pairwise score aggregation, difference-censoring path, pair-dataset constructors, prohibited config keys, or superseded mutation-indicator symbol return. It also verifies the frozen Georgiev table hash.

## Step 2 modules

### Objective

`Code/crossfit/losses.py` exposes:

- `in_batch_rank_indices(target, margin) -> (left, right)`: forms strict `|y_i-y_j| > c` upper-triangle pairs inside one batch.
- `CompositeFitnessLoss(...)`: accepts aligned per-variant `scores`, `targets`, and censor labels. It returns one weighted scalar plus ranking, uncensored magnitude, per-variant Tobit, L2, and count diagnostics.
- `LearnedResidualScale`: one positive scale for one dataset pipeline, learned jointly with the predictor.

The floor Tobit term is `-log Phi((floor-score)/sigma)`; the symmetric ceiling branch is retained and tested. The config fixes `c=0.25` within-dataset SD and `T=1.0`; neither is swept. Magnitude excludes censored rows. Regularization is L2 and receives the intended parameter iterable explicitly.

### Representations and density feature

`Code/crossfit/representations.py` provides one named interface for:

- `onehot`: dense `N x (20L)` whole-sequence encoding, including all positions and all 20 columns.
- `onehot_positional`: the identical information as `N x L x 20`.
- `aaindex_flat`: dense `N x (19L)` absolute Georgiev values at every position.
- `esm_mean`, `esm_residue`, and `esm_residue_delta`.
- `esm_mutpool`, labelled only as the Step 7 pooling-domain diagnostic.

Non-canonical residues and unaligned sequence lengths raise. `aaindex_positional`, sparse difference encodings, observed-vocabulary restriction, and early subtraction no longer exist.

`Code/crossfit/density.py` exposes `DPLLDensityFeature`. For each substituted position it requests the variant residue probability in the variant's masked context, subtracts the wild-type residue probability in the wild-type masked context, and sums only those positions. Wild-type terms are cached by position and each completed scalar is cached by variant.

### Models

`Code/crossfit/models.py` exposes `build_model(name, input_shape, parameters)` for ridge, MLP, random forest, gradient-boosted regression trees, residual 1-D convolution, and attention pooling. The tree arm is included because published supervised protein-fitness work uses GBT regression. Neural arms return one score per variant and preserve sequence shape for convolution/attention.

`DiagonalRidgeRegressor` is the exact augmented classical solver. It constructs the normal equations with an explicit diagonal penalty: intercept 0, each site feature lambda, and dPLL `1e-8`. It does not use a scalar-alpha approximation or dPLL rescaling.

### Evaluation

`Code/crossfit/evaluation.py` provides direct-score Spearman, Pearson, MSE, NDCG, wild-type-cutoff top-k precision, test-set hit rate, exactly five-member ensemble aggregation, and split Mondrian conformal intervals by `(mutation_count, unseen_position)`. Supplying top-k budgets without a wild-type cutoff raises. This is how the plate set is prevented from receiving an invented cutoff.

`Code/config/step2.yaml` names only the five v10 datasets, registers the representation/model pairings and tuning grids, fixes objective constants, and records every IRED curation seed, split, path, and threshold. It contains no balancing, pair-distance, low-N selection, or uncertainty-weighting controls.

## IRED curated data

The detailed audit is `Reports/step2/ired_curation_summary.md`; the machine-readable counts are `Reports/step2/ired_curation_summary.json`.

### Campaign table

`Data/processed/ired_variants.parquet` has 1,182 rows and columns:

`assay_id, variant_row, mutations, n_mutations, mutated_positions, y_raw, y, sigma_v, sigma_source, n_measurements, censored, dup_disagrees, is_layer_freeze_reserve, start_round`

The source contained 1,183 rows including one wild type. Sequence length is 290. Every mutation and sequence reconstructs, all residues are canonical, and all non-WT sequences are unique. The raw-label mean and population SD used for standardization are -1.3385716276552053 and 4.049363506220418. There are 173 floor-censored and zero ceiling-censored variants. The wild-type raw fitness is -0.035637535033723; `wild_type_fitness("IRED")` returns its standardized cutoff from the curated table.

The mutation-count distribution is 189, 318, 375, 204, 72, 22, and 2 for one through seven mutations. Start-round counts excluding WT are 185, 149, 298, 165, and 385.

### Splits and reserve

- `Data/processed/splits/ired_pg_kfold.csv`: 1,182 rows. Full fold sizes are 237, 237, 236, 236, 236; usable sizes after reserve exclusion are 228, 226, 225, 224, 220. Fold 0 is the primary test fold, with 895 usable rows in its training pool.
- `Data/processed/splits/ired_round_held_out.csv`: 1,182 rows. Rounds 1-4 versus round 5 contain 797/385 rows before reserve exclusion and 758/365 afterward.
- The layer-freeze reserve contains 59 rows. `load_assay("IRED")` excludes them by default, leaving 1,123.

### External plate

`Data/processed/ired_plate.csv` has exactly `aa_seq, mut, 1/s`. One duplicate source measurement was averaged, leaving 54 unique sequences; campaign overlap is zero. `load_plate_dataset("IRED")` is separate from every split loader. Only Spearman and NDCG are permitted because the plate contains no wild-type kcat.

### Loader calls

```python
from crossfit.datasets import get_split, load_assay, load_plate_dataset, wild_type_fitness

campaign = load_assay("IRED", with_sequences=True)
train_idx, test_idx = get_split("IRED", "pg_kfold", fold=0)
round_train_idx, round_test_idx = get_split("IRED", "round_held_out")
plate = load_plate_dataset("IRED")
wt_cutoff = wild_type_fitness("IRED")
```

Each selected dataset is checked at load time for zero ceiling-censored rows. IRED, GB1, GFP, KRAS, and DAAO all pass with count zero.

## Superseded artifacts

Invalid pre-v10 GB1 fits are isolated at `Reports/archive/step2_pre_v10_gb1_additive/`. Old pair/Hamming diagnostics are isolated at `Reports/archive/step2_pre_v10_pair_diagnostics/`. No v10 config or loader points to either directory. The obsolete modules, scripts, tests, grouped split table, and prior handoff were deleted or replaced; exact names are recorded in `PROJECT_LOG.md`.

## Verification

- Curation rerun: `python Code/scripts/curate_ired.py` completed and reproduced the recorded counts and hashes.
- Full suite: 186 tests passed with `python -m pytest Code/tests -q` after adding the two tree-factory cases.
- `python -m compileall -q Code/crossfit Code/scripts` passed.
- `git diff --check` passed.
- GPU cost: 0 hours.

## Incomplete, uncertain, or deviating

No deviation from the user-overridden v10 design is known. The one plate duplicate is intentionally averaged and explicitly reported rather than counted twice in evaluation.

The following are deliberately incomplete pending review: ESM-C residue extraction, wild-type masked-term caching, variant dPLL computation, hyperparameter search, five-seed ensembles, conformal calibration, the four classical augmented/unaugmented fits, architecture comparison, three result tables, pooling-domain answer, shortlist, and saved tuned artifacts. Consequently there are no Step 2 performance results yet. Training on IRED must not begin until this curation handoff is accepted.
