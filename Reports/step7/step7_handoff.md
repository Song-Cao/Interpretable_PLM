# Step 7 handoff — IRED Part A width diagnostics

## Status and scope

Part A is complete and Parts B and C have not started. IRED remains exploratory because Gate 1
failed on task gain; these diagnostics do not revise that verdict. The fixed-`k` width sweep,
raw-activation rank, feature-death timing, and one registered dead-feature-resampling run select
`w=1` and `w=2` as the two working dictionaries for Part B.

All 12 death-timing replays exactly reproduced both the authoritative scalar loss traces and
every final state tensor before their epochwise activation counts were accepted. A complete NPZ
exists for every width and seed. No interrupted or partial fit contributes to this report.

## Dictionary-free activation rank

The PCA uses centered covariance over all 259,550 normalized training-pool residue activations;
normalization by a fixed scalar does not change its spectrum or participation ratio.

| model | raw participation ratio | PCs for 50% | PCs for 80% | PCs for 90% | PCs for 95% | PCs for 99% |
|---|---:|---:|---:|---:|---:|---:|
| E_evo | 66.94 | 30 | 98 | 148 | 192 | 256 |
| E_fit | 61.24 | 31 | 106 | 162 | 212 | 309 |

The raw 960-channel activations therefore have a small effective rank before any dictionary is
introduced. For comparison, the prior code-level participation ratios were 122.83, 112.78,
99.12, and 95.02 at `w=1,2,4,8`. Those code values describe the covariance of nonlinear sparse
coordinates and are not PCA dimensions in the input space; the raw spectrum is the relevant
intrinsic-rank constraint. Under v15, overcompleteness is measured against raw effective rank,
not ambient dimension. The 960-feature `w=1` dictionary is already 14.34 times overcomplete
against E_evo and 15.68 times against E_fit; `w=2` is 28.68 and 31.35 times overcomplete.

Exact eigenvalues and cumulative spectra are in
`Reports/step7/part_a/raw_activation_pca_spectrum.csv`; the summary is
`raw_activation_pca_summary.json`.

The matched layer-freeze reserve provides the corresponding shuffled-label control. At depth 26
over the same 17,110 residue observations, PR is 66.93 for E_evo, 60.67 for E_fit, and 60.29 for
the shuffled adapter. The real and shuffled contractions are 9.35% and 9.92%, respectively.
Together with CKA 0.9189 real and 0.9811 shuffled, this shows that rank contraction is generic to
fine-tuning here rather than label-driven. The matched result is in
`Reports/step7/adapter_rank_comparison.json`.

## When features died

Counts below are means across three seeds. “Never active” means zero full-pool activations at
initialization and every epoch. “Crowded out” means dead at epoch 20 but active at least once at
initialization or an earlier epoch.

| w | features | initially dead | never active | crowded out | live at epoch 20 | final dead fraction |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 960 | 30.3 (3.2%) | 14.3 (1.5%) | 299.3 (31.2%) | 646.3 | 0.3267 |
| 2 | 1,920 | 201.7 (10.5%) | 144.7 (7.5%) | 814.0 (42.4%) | 961.3 | 0.4993 |
| 4 | 3,840 | 859.3 (22.4%) | 652.3 (17.0%) | 1,891.0 (49.2%) | 1,296.7 | 0.6623 |
| 8 | 7,680 | 2,877.7 (37.5%) | 2,032.7 (26.5%) | 3,969.3 (51.7%) | 1,678.0 | 0.7815 |

Crowding is the majority source of final death at every width: among final-dead features, the
crowded-out share is 95.4%, 84.9%, 74.4%, and 66.1% for `w=1,2,4,8`. Never-active features also
rise sharply with width. The deaths occur through competition, but random resampling did not
recover them and the raw effective rank is only 61-67. The rising dead fraction is therefore not
treated as a recoverable BatchTopK artifact: the available variation is genuinely concentrated.

Per-run categories, every epoch trajectory, and compressed per-feature counts are under
`Reports/step7/part_a/death_timing/` and the two `death_timing_*.csv` files.

## Dead-feature resampling intervention

The registered intervention used `w=4`, seed 20261311, unchanged reconstruction objective and
fixed `k=32`. Every full-pool-dead feature was randomly reinitialized after epochs 5, 10, and 15;
its Adam moments were cleared while the global optimizer step was preserved. It resampled 2,454,
2,524, and 2,549 features respectively.

The final dead fraction was 0.66797 versus 0.66042 for the matched baseline: an absolute change
of +0.00755, or a pre-registered “reduction” of -0.00755. It fails the 0.10 substantial-recovery
criterion and is also worse than the `w=2` mean of 0.49931. Random resampling therefore did not
rescue `w=4`; it does not justify replacing either default width. The intervention checkpoint is
pod-only at `Models/step7/IRED/diagnostics/w4_seed20261311_resampled_nobias.pt`, SHA-256
`1278de923a2c19bc837ced1abfbf823f8e7a6872cf7b6f90d041c89f91e74433`.

## Width choice

The exploratory Part B dictionaries are `w=1` and `w=2`, both run rather than selecting a single
one. Both are strongly overcomplete relative to the 61-67-dimensional raw activations: `w=1`
supplies the lowest-death 14-16-times-overcomplete reference, while `w=2` tests a 29-31-times
overcomplete dictionary at the edge of what this cache supports. `w=4` and `w=8` are excluded because the
registered intervention did not recover `w=4`, and feature death worsens further at `w=8`.

This supersedes the earlier statement that IRED empirically “cannot support an overcomplete
dictionary.” That statement implemented v13's pre-committed stopping rule. It was not an
impossibility result, and v15 explicitly permits the exploratory comparison.

## Cost

Part A consumed 0.14059 A100 wall-hours (8.44 minutes), including raw PCA, 12 instrumented
replays, exact reproduction checks, and the resampling run. The resampling fit itself used
0.00971 hours. Cumulative documented IRED GPU occupancy through Part A is approximately 13.3
A100 wall-hours; this includes the earlier searches, diagnostic and discarded runs retained in
the project log, the v13 cache/null/width work, and this diagnostic.

## Artifacts and next interface

- `Reports/step7/part_a/part_a_manifest.json`: completed-run manifest and selected widths.
- `Reports/step7/part_a/rank_and_feature_death.png`: raw PCA and live-feature trajectories.
- `Reports/step7/part_a/raw_activation_pca_spectrum.csv`: 960 eigenvalues per model.
- `Reports/step7/part_a/death_timing_runs.csv`: per-width, per-seed death categories and hashes.
- `Reports/step7/part_a/death_timing_epochs.csv`: live counts at initialization and every epoch.
- `Reports/step7/part_a/w4_resampling.json`: intervention decision and checkpoint hash.
- `Models/step7/IRED/w{1,2}_seed{seed}_nobias.pt`: pod-only authoritative Part B inputs.

Part B must run the complete taxonomy, latent-scaling artifact gate, each decoder's own-median
norm floor, exact classification precedence, Gate 2, and Step 9 at both working widths. It has
not started.

## Blocked Step 8 inputs

Part C remains blocked and has not started. Supply:

1. A WT IRED structure as PDB or mmCIF, with the relevant chain identified, all possible 290
   sequence positions mapped, and substrate/cofactor atoms retained. If structure numbering is
   not identical to 1-based curated sequence positions, include a mapping CSV with
   `sequence_position,chain_id,structure_resseq,insertion_code`.
2. A curated residue-panel CSV with one row per annotation and columns
   `sequence_position,wild_type_residue,site_class,evidence,source`. `site_class` must use
   `catalytic`, `substrate_contact`, `cofactor_contact`, or `access_pathway`. Multiple classes per
   residue may use multiple rows. Optional structure identifiers and `ligand_id` may be included.
3. Provenance for the structure and every residue designation, including the contact definition
   used for substrate/cofactor residues and how access-pathway residues were established.

No residues, contacts, or generic distance tracks will be inferred or substituted.

## Incomplete or uncertain

Part B and all of Step 8 are incomplete by design. The resampling result tests one registered
random-reset intervention, not every possible anti-death optimizer. No architecture or loss was
tuned to reduce death. Gate 1 remains failed, and no Part A result is confirmatory evidence for
the central claim.
