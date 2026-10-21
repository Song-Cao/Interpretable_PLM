# IRED curation summary

Generated from the artifacts on disk on 2026-10-21. No model was trained.

## Inputs and provenance

- Campaign: `Data/processed/IRED_data/data_processed.csv`, SHA-256 `e97e180738c14ecb33da6538b84a05c35346a60c579c73f47a6a5ed577d582fd`.
- External plate: `Data/processed/IRED_data/plate_dataset.csv`, SHA-256 `53e493d783c9f161013547dc9c1bf10685dd49e1d7fb427b51624dc29fea13ff`.
- Mutation strings were parsed as `-`-separated substitutions and stored canonically with `:` separators. `WT` identified the single wild-type row.
- Wild-type sequence length is 290 and its campaign fitness is -0.035637535033723 in raw units.

## Campaign QC

The source has 1,183 rows: one wild type and 1,182 unique non-WT variants. All sequences are length 290, use only the 20 canonical amino acids, and reconstruct exactly from the wild type and mutation string. No label is missing, no mutant sequence is duplicated, and no mutation token is unparsed or inconsistent with the reference.

The non-WT mutation-count distribution is 189 singles, 318 doubles, 375 triples, 204 quadruples, 72 five-mutants, 22 six-mutants, and 2 seven-mutants. Round counts after excluding WT are 185, 149, 298, 165, and 385 for start rounds 1 through 5.

Within-dataset standardization used all 1,182 curated variants, including the reserve: raw mean -1.3385716276552053 and population SD 4.049363506220418. The Step 1 censoring rule marks 173 floor-censored and zero ceiling-censored variants. Per-variant measurement error is unavailable.

## Reservations and splits

- Layer-freeze reserve: 59 variants. The loader excludes them by default, leaving 1,123 usable variants.
- `pg_kfold`: deterministic five-fold sizes before reserve exclusion are 237, 237, 236, 236, and 236. Usable test sizes are 228, 226, 225, 224, and 220. Fold 0 is the registered primary test fold, leaving 895 usable training-pool variants.
- `round_held_out`: rounds 1-4 contain 797 variants before reserve exclusion and 758 after it; round 5 contains 385 before and 365 after. This split is evaluation-only.

## External plate set

The source contains 55 measurements. One sequence occurs twice with the same mutation string; its two `1/s` values were averaged, leaving 54 unique variants. All reconstruct from the campaign wild type, and none overlaps a campaign sequence. The curated loader exposes exactly `aa_seq`, `mut`, and `1/s`. This set is never fitted, tuned, calibrated, or selected on. Because it has no wild-type kcat, only Spearman and NDCG are valid.

## Outputs

- `Data/processed/ired_variants.parquet`: 1,182 rows; SHA-256 `eaa664029ea82767e2b45c9a1fb288722eba771f725207a8e3073dced964954c`.
- `Data/processed/ired_plate.csv`: 54 rows; SHA-256 `5551a5d5614ffd289e8e6b7de6464bee740551779bb0c6f990e9e43a147e0244`.
- `Data/processed/splits/ired_pg_kfold.csv`: 1,182 assignments.
- `Data/processed/splits/ired_round_held_out.csv`: 1,182 assignments.
- `Data/processed/assay_manifest.csv` and `Data/processed/qc_report.csv`: one IRED row appended to each.

## Incomplete or uncertain

No ESM-C activations or dPLL values were computed, and no Step 2 predictor was trained. The explicit review boundary after curation was honored. The external duplicate pooling is recorded above; no other row was pooled or dropped beyond excluding the wild-type reference from the supervised variant table.
