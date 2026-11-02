# SrIRED in-silico directed-evolution application

> Application workstream only. No result here is evidence for a CROSSFIT gate,
> table, or scientific claim.

## Interpretation

These are model proposals and a retrospective audit against measured trajectories,
not a confident ranking. Every model was trained on campaign fitness, a selection
readout. On the independent plate kcat set, classical encodings reached Spearman
0.53-0.57 while every ESM arm reached 0.23-0.44; the classical arms therefore have
the stronger external-transfer evidence when catalytic rate is the goal.

## Protocol and budget

- 13 frozen model arms, retrained without re-searching on all 1,182 curated non-WT
  campaign variants.
- Four founders, four one-mutation rounds, 5,510 complete WT single-mutant scans,
  300 reusable winner mutations per model, and
  30 survivors per founder and round.
- Representative projection: 2.030 GPU-hours.
- Search wall time: 3.152 GPU-hours; accounted application
  total including generation fits and failed pre-result launches: 5.719
  GPU-hours.

## Consensus candidates

Consensus is the number of model-specific final top lists containing a variant; mean
rank breaks ties. High rank spread is disagreement, not uncertainty to average away.
The maximum consensus is only 2 of 13 models,
and every displayed candidate is supported only by the paired one-hot/AAIndex ridge
arms. Those encodings span nearly the same linear space, so this is not independent
cross-family agreement. Eight candidates are additionally flagged for high rank
dispersion across all models. All 29 of
30 candidates contain at least one known dead-end control mutation. None
should be treated as a confident lab recommendation.

| mutation_set_canonical | consensus_count | mean_rank | cross_model_rank_sd | high_consensus_high_disagreement | known_dead_end_mutations |
| --- | --- | --- | --- | --- | --- |
| G37S-K38N-S68P-A72T-P203Q-V216I-T241A | 2 | 1.00 | 5.59 | False | A72T |
| G37S-K38N-S68P-A72T-P203Q-V216I-T241A-H244Y | 2 | 1.00 | 6.68 | False | A72T |
| G37S-S68P-A72T-K195G-V216I-T241A | 2 | 1.00 | 8.18 | False | A72T |
| K38N-S68P-A72T-P203Q-V216I-T241A | 2 | 1.00 | 8.28 | False | A72T |
| R2T-G37S-S68P-A72T-P203Q-V216I-T241A | 2 | 1.00 | 6.76 | False | A72T |
| R2T-G37S-S68P-A72T-P203Q-V216I-T241A-H244Y | 2 | 1.00 | 8.71 | True | A72T |
| R2T-G37S-S68P-A72T-V216I-T241A-H244Y | 2 | 1.00 | 6.58 | False | A72T |
| G37C-K38N-S68P-A72T-P203Q-V216I-T241A | 2 | 2.00 | 8.65 | True | A72T;G37C |
| G37C-K38N-S68P-A72T-P203Q-V216I-T241A-H244Y | 2 | 2.00 | 8.31 | True | A72T;G37C |
| G37S-S68P-A72T-P203Q-V216I-T241A | 2 | 2.00 | 7.05 | False | A72T |
| K38N-S68P-A72T-P203Q-V216I-T241A-H244Y | 2 | 2.00 | 7.92 | False | A72T |
| R2S-G37S-S68P-A72T-P203Q-V216I-T241A | 2 | 2.00 | 7.62 | False | A72T |
| R2S-G37S-S68P-A72T-P203Q-V216I-T241A-H244Y | 2 | 2.00 | 9.37 | True | A72T |
| R2T-G37S-S68P-A72T-V216I-T241A | 2 | 2.00 | 5.44 | False | A72T |
| G37S-S68P-A72T-K195G-V216I-T241A-H244Y | 2 | 2.50 | 8.54 | True | A72T |
| R2S-G37S-S68P-A72T-V216I-T241A-H244Y | 2 | 2.50 | 5.76 | False | A72T |
| G37S-K38N-S68P-P203Q-V216I-T241A | 2 | 3.00 | 6.93 | False |  |
| G37S-S68P-A72T-K195G-P203Q-V216I-T241A | 2 | 3.00 | 7.05 | False | A72T |
| G37S-S68P-A72T-K195G-P203Q-V216I-T241A-H244Y | 2 | 3.00 | 7.30 | False | A72T |
| K38N-S68P-A72T-E197D-P203Q-V216I-T241A-H244Y | 2 | 3.00 | 6.90 | False | A72T |
| R2S-G37S-S68P-A72T-V216I-T241A | 2 | 3.00 | 3.85 | False | A72T |
| G37C-S68P-A72T-P203Q-V216I-T241A | 2 | 4.00 | 9.74 | True | A72T;G37C |
| G37S-D55E-S68P-A72T-P203Q-V216I-T241A | 2 | 4.00 | 4.95 | False | A72T |
| G37S-D55E-S68P-A72T-P203Q-V216I-T241A-H244Y | 2 | 4.00 | 5.02 | False | A72T |
| G37S-S68P-A72T-P203Q-V216I-T241A-H244Y | 2 | 4.00 | 6.38 | False | A72T |
| K38N-S68P-A72T-E197D-P203Q-V216I-T241A | 2 | 4.00 | 7.22 | False | A72T |
| K38N-S68P-A72T-E197K-P203Q-V216I-T241A-H244Y | 2 | 4.00 | 9.24 | True | A72T |
| R2T-S68P-A72T-K195G-V216I-T241A | 2 | 4.00 | 4.41 | False | A72T |
| R2T-S68P-A72T-K195G-V216I-T241A-H244Y | 2 | 4.00 | 5.96 | False | A72T |
| G37C-K38N-S68P-P203Q-V216I-T241A-H244Y | 2 | 5.00 | 10.28 | True | G37C |

## Known-control ranks

Ranks are among all valid single additions to S68P-T241A. A72T and G37C are known
dead ends; H244Y is the known positive control. Lower is better.

| model | A72T | G37C | H244Y |
| --- | --- | --- | --- |
| aaindex_flat_ridge_augmented | 2 | 298 | 236 |
| aaindex_flat_ridge_unaugmented | 3 | 6 | 8 |
| adapter_linear | 394 | 1699 | 3898 |
| esm_mean_mlp | 404 | 2837 | 37 |
| esm_mean_random_forest | 765 | 2623 | 86 |
| esm_mean_ridge | 784 | 100 | 1010 |
| esm_mutpool_ridge | 19 | 148 | 331 |
| esm_residue_conv | 663 | 20 | 967 |
| esm_residue_delta_conv | 341 | 159 | 48 |
| onehot_positional_conv | 125 | 99 | 560 |
| onehot_ridge_augmented | 2 | 327 | 238 |
| onehot_ridge_unaugmented | 3 | 6 | 8 |
| phi_w1_max_all_live | 4565 | 160 | 605 |

## Could the models recover the campaign trajectory?

| model | target | passes_through | first_selected_round |
| --- | --- | --- | --- |
| onehot_ridge_unaugmented | S68P-P203Q-T241A | True | 1.0 |
| onehot_ridge_unaugmented | S68P-T241A-H244Y | True | 1.0 |
| onehot_ridge_unaugmented | S68P-P203Q-T241A-H244Y | True | 2.0 |
| onehot_ridge_augmented | S68P-P203Q-T241A | False |  |
| onehot_ridge_augmented | S68P-T241A-H244Y | False |  |
| onehot_ridge_augmented | S68P-P203Q-T241A-H244Y | False |  |
| aaindex_flat_ridge_unaugmented | S68P-P203Q-T241A | True | 1.0 |
| aaindex_flat_ridge_unaugmented | S68P-T241A-H244Y | True | 1.0 |
| aaindex_flat_ridge_unaugmented | S68P-P203Q-T241A-H244Y | True | 2.0 |
| aaindex_flat_ridge_augmented | S68P-P203Q-T241A | False |  |
| aaindex_flat_ridge_augmented | S68P-T241A-H244Y | False |  |
| aaindex_flat_ridge_augmented | S68P-P203Q-T241A-H244Y | False |  |
| esm_mean_ridge | S68P-P203Q-T241A | False |  |
| esm_mean_ridge | S68P-T241A-H244Y | False |  |
| esm_mean_ridge | S68P-P203Q-T241A-H244Y | False |  |
| esm_mean_mlp | S68P-P203Q-T241A | False |  |
| esm_mean_mlp | S68P-T241A-H244Y | False |  |
| esm_mean_mlp | S68P-P203Q-T241A-H244Y | False |  |
| esm_mean_random_forest | S68P-P203Q-T241A | True | 1.0 |
| esm_mean_random_forest | S68P-T241A-H244Y | False |  |
| esm_mean_random_forest | S68P-P203Q-T241A-H244Y | False |  |
| esm_mutpool_ridge | S68P-P203Q-T241A | False |  |
| esm_mutpool_ridge | S68P-T241A-H244Y | False |  |
| esm_mutpool_ridge | S68P-P203Q-T241A-H244Y | False |  |
| esm_residue_conv | S68P-P203Q-T241A | False |  |
| esm_residue_conv | S68P-T241A-H244Y | False |  |
| esm_residue_conv | S68P-P203Q-T241A-H244Y | False |  |
| esm_residue_delta_conv | S68P-P203Q-T241A | False |  |
| esm_residue_delta_conv | S68P-T241A-H244Y | False |  |
| esm_residue_delta_conv | S68P-P203Q-T241A-H244Y | False |  |
| onehot_positional_conv | S68P-P203Q-T241A | False |  |
| onehot_positional_conv | S68P-T241A-H244Y | False |  |
| onehot_positional_conv | S68P-P203Q-T241A-H244Y | False |  |
| phi_w1_max_all_live | S68P-P203Q-T241A | False |  |
| phi_w1_max_all_live | S68P-T241A-H244Y | False |  |
| phi_w1_max_all_live | S68P-P203Q-T241A-H244Y | False |  |
| adapter_linear | S68P-P203Q-T241A | False |  |
| adapter_linear | S68P-T241A-H244Y | False |  |
| adapter_linear | S68P-P203Q-T241A-H244Y | False |  |

Round-level recovery totals across the four founder searches:

| model | selected_rows | campaign_recoveries |
| --- | --- | --- |
| aaindex_flat_ridge_augmented | 480 | 3 |
| aaindex_flat_ridge_unaugmented | 480 | 11 |
| adapter_linear | 480 | 0 |
| esm_mean_mlp | 480 | 1 |
| esm_mean_random_forest | 480 | 8 |
| esm_mean_ridge | 480 | 0 |
| esm_mutpool_ridge | 480 | 0 |
| esm_residue_conv | 480 | 0 |
| esm_residue_delta_conv | 480 | 4 |
| onehot_positional_conv | 480 | 3 |
| onehot_ridge_augmented | 480 | 4 |
| onehot_ridge_unaugmented | 480 | 11 |
| phi_w1_max_all_live | 480 | 0 |

## Uncertainty

Every candidate has five-member within-model spread and cross-model rank spread.
The existing Step 2 Mondrian calibration contains no informative final-design stratum:
six-mutation, seen-position variants have only one calibration row and therefore an
unbounded finite-sample 90% interval; seven- and eight-mutation strata are absent.
Phi and the adapter have no registered Step 2 conformal calibrator. These limitations
are recorded per model and candidate rather than replaced by a pooled interval.

## Biophysical consistency checks

ThermoMPNN passed all 6 required sign checks on an exact
290/290 structure-sequence match:

| mutation | experimental_anchor | thermompnn_ddg_kcal_mol | predicted_sign | sign_matches |
| --- | --- | --- | --- | --- |
| T241A | stabilising | -0.113253116607666 | stabilising | True |
| S68P | destabilising | 0.0343245863914489 | destabilising | True |
| P203Q | destabilising | 1.2316198348999023 | destabilising | True |
| H244Y | destabilising | 0.1373022198677063 | destabilising | True |
| A72T | destabilising | 0.596011221408844 | destabilising | True |
| G37C | destabilising | 0.7194771766662598 | destabilising | True |

Candidate values below sum the validated single-substitution estimates. They are an
additive consistency check, not a direct multi-mutant or epistatic prediction.

| mutation_set_canonical | thermompnn_additive_ddg_kcal_mol | stability_consistency | functional_shells_touched |
| --- | --- | --- | --- |
| G37S-K38N-S68P-A72T-P203Q-V216I-T241A | 3.684 | predicted_net_destabilising | distal_functional_network;first_shell;generic_distal |
| G37S-K38N-S68P-A72T-P203Q-V216I-T241A-H244Y | 3.821 | predicted_net_destabilising | distal_functional_network;first_shell;generic_distal |
| G37S-S68P-A72T-K195G-V216I-T241A | 1.183 | predicted_net_destabilising | distal_functional_network;first_shell;generic_distal |
| K38N-S68P-A72T-P203Q-V216I-T241A | 2.472 | predicted_net_destabilising | distal_functional_network;first_shell;generic_distal |
| R2T-G37S-S68P-A72T-P203Q-V216I-T241A | 3.263 | predicted_net_destabilising | distal_functional_network;first_shell;generic_distal |
| R2T-G37S-S68P-A72T-P203Q-V216I-T241A-H244Y | 3.401 | predicted_net_destabilising | distal_functional_network;first_shell;generic_distal |
| R2T-G37S-S68P-A72T-V216I-T241A-H244Y | 2.169 | predicted_net_destabilising | distal_functional_network;first_shell;generic_distal |
| G37C-K38N-S68P-A72T-P203Q-V216I-T241A | 3.192 | predicted_net_destabilising | distal_functional_network;first_shell;generic_distal |
| G37C-K38N-S68P-A72T-P203Q-V216I-T241A-H244Y | 3.329 | predicted_net_destabilising | distal_functional_network;first_shell;generic_distal |
| G37S-S68P-A72T-P203Q-V216I-T241A | 2.932 | predicted_net_destabilising | distal_functional_network;first_shell;generic_distal |
| K38N-S68P-A72T-P203Q-V216I-T241A-H244Y | 2.610 | predicted_net_destabilising | distal_functional_network;first_shell;generic_distal |
| R2S-G37S-S68P-A72T-P203Q-V216I-T241A | 3.371 | predicted_net_destabilising | distal_functional_network;first_shell;generic_distal |
| R2S-G37S-S68P-A72T-P203Q-V216I-T241A-H244Y | 3.509 | predicted_net_destabilising | distal_functional_network;first_shell;generic_distal |
| R2T-G37S-S68P-A72T-V216I-T241A | 2.032 | predicted_net_destabilising | distal_functional_network;first_shell;generic_distal |
| G37S-S68P-A72T-K195G-V216I-T241A-H244Y | 1.321 | predicted_net_destabilising | distal_functional_network;first_shell;generic_distal |
| R2S-G37S-S68P-A72T-V216I-T241A-H244Y | 2.277 | predicted_net_destabilising | distal_functional_network;first_shell;generic_distal |
| G37S-K38N-S68P-P203Q-V216I-T241A | 3.088 | predicted_net_destabilising | distal_functional_network;first_shell;generic_distal |
| G37S-S68P-A72T-K195G-P203Q-V216I-T241A | 2.415 | predicted_net_destabilising | distal_functional_network;first_shell;generic_distal |
| G37S-S68P-A72T-K195G-P203Q-V216I-T241A-H244Y | 2.552 | predicted_net_destabilising | distal_functional_network;first_shell;generic_distal |
| K38N-S68P-A72T-E197D-P203Q-V216I-T241A-H244Y | 3.034 | predicted_net_destabilising | distal_functional_network;first_shell;generic_distal |
| R2S-G37S-S68P-A72T-V216I-T241A | 2.140 | predicted_net_destabilising | distal_functional_network;first_shell;generic_distal |
| G37C-S68P-A72T-P203Q-V216I-T241A | 2.440 | predicted_net_destabilising | distal_functional_network;first_shell;generic_distal |
| G37S-D55E-S68P-A72T-P203Q-V216I-T241A | 2.886 | predicted_net_destabilising | distal_functional_network;first_shell;generic_distal |
| G37S-D55E-S68P-A72T-P203Q-V216I-T241A-H244Y | 3.023 | predicted_net_destabilising | distal_functional_network;first_shell;generic_distal |
| G37S-S68P-A72T-P203Q-V216I-T241A-H244Y | 3.069 | predicted_net_destabilising | distal_functional_network;first_shell;generic_distal |
| K38N-S68P-A72T-E197D-P203Q-V216I-T241A | 2.896 | predicted_net_destabilising | distal_functional_network;first_shell;generic_distal |
| K38N-S68P-A72T-E197K-P203Q-V216I-T241A-H244Y | 2.961 | predicted_net_destabilising | distal_functional_network;first_shell;generic_distal |
| R2T-S68P-A72T-K195G-V216I-T241A | 0.304 | predicted_net_destabilising | distal_functional_network;first_shell;generic_distal |
| R2T-S68P-A72T-K195G-V216I-T241A-H244Y | 0.441 | predicted_net_destabilising | distal_functional_network;first_shell;generic_distal |
| G37C-K38N-S68P-P203Q-V216I-T241A-H244Y | 2.733 | predicted_net_destabilising | distal_functional_network;first_shell;generic_distal |

Active-site geometry was not scored because no validated mutant-structure generator
was available; reporting unchanged wild-type coordinates would be misleading.
SoluProt was unreachable and has no IRED anchor, FoldX was unavailable, and docking
was the first registered budget cut. No causal attribution is made.

## Primary artifacts

- `controls_ranks.csv`: mandatory known false-positive and positive-control ranks.
- `trajectory_overlap.csv` and `trajectory_passage_controls.csv`: measured recovery
  and winning-path audit.
- `consensus_candidates.csv` and `final_candidate_model_scores.csv`: consensus,
  five-member spread, cross-model disagreement, and conformal support status.
- `<arm>_round1_single_scan.csv`, `<arm>_winner_pool.csv`, `<arm>_top20.csv`, and
  `round_top/<arm>/*.csv`: committed model-specific proposals.
- `full_lineage/<arm>/*.parquet`: pod-only evaluated and rejected candidates with
  parent links; intentionally gitignored.
- `trajectory_graphs/`: one graph per model and one ensemble graph.
