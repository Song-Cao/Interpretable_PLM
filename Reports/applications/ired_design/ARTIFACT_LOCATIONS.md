# IRED artifact locations

Snapshot taken before stopping the RunPod pod. The pod project root `/workspace/Interpretable_PLM` is mounted from RunPod network volume `8pxh42cxur` (`mfs#us-ks-2.runpod.net:9421[/networkvolumes/8pxh42cxur]`, FUSE). The container root is a separate overlay filesystem. A root-filesystem search found no IRED-design, Step 2, Step 3, or Step 7 artifacts outside `/workspace`.

This is an application workstream inventory. None of its outputs are CROSSFIT evidence.

## Storage verdict

| Asset | Count | Size | Location | Git |
|---|---:|---:|---|---|
| Application generation checkpoints | 65, all hash-verified | 314 MB allocated (307 MB file bytes) | Network volume | Ignored; matching SHA-256 values committed in generation manifests |
| Full evaluated lineage parquets | 208 / 1,147,143 rows | 76 MB allocated (50.8 MB file bytes) | Network volume | Ignored; file/row manifest committed |
| Committed round-top detail | 6,240 rows | See index | Local, network volume after pull, and Git | Committed |
| Step 2 checkpoints / cache | 45 / 11 files | 259 MB / 1.675 GB | Network volume | Ignored |
| Step 3 checkpoints / cache | 12 / 15 files | 15.6 MB / 9.862 GB | Network volume | Ignored |
| Step 7 checkpoints | 25 files | 1.387 GB | Network volume | Ignored |
| All main-track model checkpoints / caches | 1,927 / 1,305 files | 5.254 GB / 13.824 GB | Network volume | Ignored |

The detailed file-level answer is in `storage_artifact_audit.csv`. It has one row per artifact, including size, Git status, workstation/network location, and recoverability. `pod_full_lineage_manifest.csv` lists every one of the 208 lineage parquets and its row count.

## What survives

- **Stopping the pod:** everything. Application checkpoints, full lineage, main-track checkpoints, and activation caches remain on network volume `8pxh42cxur`; committed reports also remain locally and in Git.
- **Deleting the pod while retaining the network volume:** the same artifacts survive. A replacement pod must attach network volume `8pxh42cxur` at `/workspace`.
- **Deleting the network volume:** committed code, tables, figures, manifests, and the meeting-ready extraction survive locally and in Git. The 65 application checkpoints, 208 full-lineage parquets, all `Models/step*`, and all `Cache/step*` would be lost.
- **Container disk:** no relevant artifact was found there.

## Recompute exposure if the network volume were lost

| Track | Lost material | Recorded GPU exposure |
|---|---|---:|
| IRED design | 65 generation checkpoints | 0.253 GPU-hours for the completed non-adapter and adapter refits |
| IRED design | 208 full-lineage parquets | 3.152 GPU-hours for the valid reduced 300/30 search, assuming models and extraction caches must also be restored |
| IRED design | Entire exact workstream | 5.719 GPU-hours accounted, including archived/failed launches; productive rerun would omit those failed launches |
| CROSSFIT Step 2 | ESM caches and baseline checkpoints | About 7.8 A100 wall-hours recorded across the initial suite and convolutional completion; this includes discarded diagnostics |
| CROSSFIT Step 3 | Adapter checkpoints and reserve/full activation caches | About 2.0 A100 hours for the authoritative registered run; the bounded task-gain sweep added 3.024 hours but its discarded-fold models are not all persisted |
| CROSSFIT Steps 4/6/7/9 | Training-pool activations, crosscoders, nulls, dense/sparse predictors | Recorded current-stage components total under 1 GPU-hour, but exact regeneration depends on the surviving upstream Step 3 adapter and Step 4 cache |

These are observed accounting figures, not new estimates. Loss of the network volume would also remove the only copies of large ignored artifacts, so it should not be deleted without a separate backup.

## Application tables and figures

The registered budget rule reduced the winner pool/survivors from 500/50 to 300/30 before the valid run. Therefore the complete committed retained-round table is 13 models x 4 founders x 4 rounds x 30 = 6,240 rows, not 10,400. FoldX, mutant geometry, SoluProt, and docking were not run on round tops. The only per-round biophysical value is additive ThermoMPNN ddG from the validated 290 x 19 non-identity single-substitution matrix.

| Path | Rows | Contents | Supports |
|---|---:|---|---|
| `Reports/applications/ired_design/aaindex_flat_ridge_augmented_round1_single_scan.csv` | 5510 | Complete 5,510 single-substitution scan for one model | Round-1 halo comparison |
| `Reports/applications/ired_design/aaindex_flat_ridge_augmented_top20.csv` | 20 | Top 20 final suggestions from one model | Per-model recommendations |
| `Reports/applications/ired_design/aaindex_flat_ridge_augmented_winner_pool.csv` | 300 | Budget-adjusted top-300 reusable winner mutation pool | Four-round candidate generation |
| `Reports/applications/ired_design/aaindex_flat_ridge_unaugmented_round1_single_scan.csv` | 5510 | Complete 5,510 single-substitution scan for one model | Round-1 halo comparison |
| `Reports/applications/ired_design/aaindex_flat_ridge_unaugmented_top20.csv` | 20 | Top 20 final suggestions from one model | Per-model recommendations |
| `Reports/applications/ired_design/aaindex_flat_ridge_unaugmented_winner_pool.csv` | 300 | Budget-adjusted top-300 reusable winner mutation pool | Four-round candidate generation |
| `Reports/applications/ired_design/adapter_linear_round1_single_scan.csv` | 5510 | Complete 5,510 single-substitution scan for one model | Round-1 halo comparison |
| `Reports/applications/ired_design/adapter_linear_top20.csv` | 20 | Top 20 final suggestions from one model | Per-model recommendations |
| `Reports/applications/ired_design/adapter_linear_winner_pool.csv` | 300 | Budget-adjusted top-300 reusable winner mutation pool | Four-round candidate generation |
| `Reports/applications/ired_design/consensus_all_final_candidates.csv` | 1336 | All distinct final candidates before consensus truncation | Consensus denominator and candidate audit |
| `Reports/applications/ired_design/consensus_candidates.csv` | 30 | Final 30-candidate consensus with cross-model disagreement | Meeting shortlist |
| `Reports/applications/ired_design/controls_ranks.csv` | 39 | Ranks of A72T, G37C, and H244Y controls for every model | Known-answer false-positive/positive-control audit |
| `Reports/applications/ired_design/esm_mean_mlp_round1_single_scan.csv` | 5510 | Complete 5,510 single-substitution scan for one model | Round-1 halo comparison |
| `Reports/applications/ired_design/esm_mean_mlp_top20.csv` | 20 | Top 20 final suggestions from one model | Per-model recommendations |
| `Reports/applications/ired_design/esm_mean_mlp_winner_pool.csv` | 300 | Budget-adjusted top-300 reusable winner mutation pool | Four-round candidate generation |
| `Reports/applications/ired_design/esm_mean_random_forest_round1_single_scan.csv` | 5510 | Complete 5,510 single-substitution scan for one model | Round-1 halo comparison |
| `Reports/applications/ired_design/esm_mean_random_forest_top20.csv` | 20 | Top 20 final suggestions from one model | Per-model recommendations |
| `Reports/applications/ired_design/esm_mean_random_forest_winner_pool.csv` | 300 | Budget-adjusted top-300 reusable winner mutation pool | Four-round candidate generation |
| `Reports/applications/ired_design/esm_mean_ridge_round1_single_scan.csv` | 5510 | Complete 5,510 single-substitution scan for one model | Round-1 halo comparison |
| `Reports/applications/ired_design/esm_mean_ridge_top20.csv` | 20 | Top 20 final suggestions from one model | Per-model recommendations |
| `Reports/applications/ired_design/esm_mean_ridge_winner_pool.csv` | 300 | Budget-adjusted top-300 reusable winner mutation pool | Four-round candidate generation |
| `Reports/applications/ired_design/esm_mutpool_ridge_round1_single_scan.csv` | 5510 | Complete 5,510 single-substitution scan for one model | Round-1 halo comparison |
| `Reports/applications/ired_design/esm_mutpool_ridge_top20.csv` | 20 | Top 20 final suggestions from one model | Per-model recommendations |
| `Reports/applications/ired_design/esm_mutpool_ridge_winner_pool.csv` | 300 | Budget-adjusted top-300 reusable winner mutation pool | Four-round candidate generation |
| `Reports/applications/ired_design/esm_residue_conv_round1_single_scan.csv` | 5510 | Complete 5,510 single-substitution scan for one model | Round-1 halo comparison |
| `Reports/applications/ired_design/esm_residue_conv_top20.csv` | 20 | Top 20 final suggestions from one model | Per-model recommendations |
| `Reports/applications/ired_design/esm_residue_conv_winner_pool.csv` | 300 | Budget-adjusted top-300 reusable winner mutation pool | Four-round candidate generation |
| `Reports/applications/ired_design/esm_residue_delta_conv_round1_single_scan.csv` | 5510 | Complete 5,510 single-substitution scan for one model | Round-1 halo comparison |
| `Reports/applications/ired_design/esm_residue_delta_conv_top20.csv` | 20 | Top 20 final suggestions from one model | Per-model recommendations |
| `Reports/applications/ired_design/esm_residue_delta_conv_winner_pool.csv` | 300 | Budget-adjusted top-300 reusable winner mutation pool | Four-round candidate generation |
| `Reports/applications/ired_design/final_candidate_model_scores.csv` | 390 | Every final candidate scored by every model with uncertainty status | Model disagreement and uncertainty audit |
| `Reports/applications/ired_design/final_candidate_stability.csv` | 30 | Validated ThermoMPNN additive stability panel | Biophysical consistency check |
| `Reports/applications/ired_design/final_consensus_biophysical_panel.csv` | 30 | Consensus shortlist joined to the complete available biophysical panel and explicit unavailable-oracle statuses | Single meeting-ready candidate table |
| `Reports/applications/ired_design/onehot_positional_conv_round1_single_scan.csv` | 5510 | Complete 5,510 single-substitution scan for one model | Round-1 halo comparison |
| `Reports/applications/ired_design/onehot_positional_conv_top20.csv` | 20 | Top 20 final suggestions from one model | Per-model recommendations |
| `Reports/applications/ired_design/onehot_positional_conv_winner_pool.csv` | 300 | Budget-adjusted top-300 reusable winner mutation pool | Four-round candidate generation |
| `Reports/applications/ired_design/onehot_ridge_augmented_round1_single_scan.csv` | 5510 | Complete 5,510 single-substitution scan for one model | Round-1 halo comparison |
| `Reports/applications/ired_design/onehot_ridge_augmented_top20.csv` | 20 | Top 20 final suggestions from one model | Per-model recommendations |
| `Reports/applications/ired_design/onehot_ridge_augmented_winner_pool.csv` | 300 | Budget-adjusted top-300 reusable winner mutation pool | Four-round candidate generation |
| `Reports/applications/ired_design/onehot_ridge_unaugmented_round1_single_scan.csv` | 5510 | Complete 5,510 single-substitution scan for one model | Round-1 halo comparison |
| `Reports/applications/ired_design/onehot_ridge_unaugmented_top20.csv` | 20 | Top 20 final suggestions from one model | Per-model recommendations |
| `Reports/applications/ired_design/onehot_ridge_unaugmented_winner_pool.csv` | 300 | Budget-adjusted top-300 reusable winner mutation pool | Four-round candidate generation |
| `Reports/applications/ired_design/oracles/thermompnn/INVALID_chain_A_numbering_shift.csv` | 5800 | Raw ThermoMPNN 290 x 20 output including identity rows | Oracle provenance; noncanonical shifted-chain file is explicitly invalid |
| `Reports/applications/ired_design/oracles/thermompnn/ThermoMPNN_inference_5ocm_deposited.csv` | 5800 | Raw ThermoMPNN 290 x 20 output including identity rows | Oracle provenance; noncanonical shifted-chain file is explicitly invalid |
| `Reports/applications/ired_design/oracles/thermompnn/thermompnn_single_substitution_5510.csv` | 5510 | Validated 290 x 19 non-identity ThermoMPNN single-substitution matrix | Additive ddG lookup for round-top and final candidates |
| `Reports/applications/ired_design/per_round_top_variants.csv` | 6240 | All retained per-round candidates with lineage, uncertainty status, additive ThermoMPNN ddG, and controls | Raw round-by-round review and trajectory reconstruction |
| `Reports/applications/ired_design/phi_w1_max_all_live_round1_single_scan.csv` | 5510 | Complete 5,510 single-substitution scan for one model | Round-1 halo comparison |
| `Reports/applications/ired_design/phi_w1_max_all_live_top20.csv` | 20 | Top 20 final suggestions from one model | Per-model recommendations |
| `Reports/applications/ired_design/phi_w1_max_all_live_winner_pool.csv` | 300 | Budget-adjusted top-300 reusable winner mutation pool | Four-round candidate generation |
| `Reports/applications/ired_design/pod_full_lineage_manifest.csv` | 208 | Path, row count, and size for every pod-only lineage parquet | Network-volume completeness audit |
| `Reports/applications/ired_design/round_top/aaindex_flat_ridge_augmented/S68P-P203Q-T241A-H244Y_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/aaindex_flat_ridge_augmented/S68P-P203Q-T241A-H244Y_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/aaindex_flat_ridge_augmented/S68P-P203Q-T241A-H244Y_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/aaindex_flat_ridge_augmented/S68P-P203Q-T241A-H244Y_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/aaindex_flat_ridge_augmented/S68P-P203Q-T241A_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/aaindex_flat_ridge_augmented/S68P-P203Q-T241A_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/aaindex_flat_ridge_augmented/S68P-P203Q-T241A_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/aaindex_flat_ridge_augmented/S68P-P203Q-T241A_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/aaindex_flat_ridge_augmented/S68P-T241A-H244Y_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/aaindex_flat_ridge_augmented/S68P-T241A-H244Y_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/aaindex_flat_ridge_augmented/S68P-T241A-H244Y_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/aaindex_flat_ridge_augmented/S68P-T241A-H244Y_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/aaindex_flat_ridge_augmented/S68P-T241A_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/aaindex_flat_ridge_augmented/S68P-T241A_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/aaindex_flat_ridge_augmented/S68P-T241A_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/aaindex_flat_ridge_augmented/S68P-T241A_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/aaindex_flat_ridge_unaugmented/S68P-P203Q-T241A-H244Y_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/aaindex_flat_ridge_unaugmented/S68P-P203Q-T241A-H244Y_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/aaindex_flat_ridge_unaugmented/S68P-P203Q-T241A-H244Y_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/aaindex_flat_ridge_unaugmented/S68P-P203Q-T241A-H244Y_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/aaindex_flat_ridge_unaugmented/S68P-P203Q-T241A_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/aaindex_flat_ridge_unaugmented/S68P-P203Q-T241A_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/aaindex_flat_ridge_unaugmented/S68P-P203Q-T241A_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/aaindex_flat_ridge_unaugmented/S68P-P203Q-T241A_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/aaindex_flat_ridge_unaugmented/S68P-T241A-H244Y_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/aaindex_flat_ridge_unaugmented/S68P-T241A-H244Y_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/aaindex_flat_ridge_unaugmented/S68P-T241A-H244Y_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/aaindex_flat_ridge_unaugmented/S68P-T241A-H244Y_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/aaindex_flat_ridge_unaugmented/S68P-T241A_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/aaindex_flat_ridge_unaugmented/S68P-T241A_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/aaindex_flat_ridge_unaugmented/S68P-T241A_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/aaindex_flat_ridge_unaugmented/S68P-T241A_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/adapter_linear/S68P-P203Q-T241A-H244Y_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/adapter_linear/S68P-P203Q-T241A-H244Y_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/adapter_linear/S68P-P203Q-T241A-H244Y_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/adapter_linear/S68P-P203Q-T241A-H244Y_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/adapter_linear/S68P-P203Q-T241A_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/adapter_linear/S68P-P203Q-T241A_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/adapter_linear/S68P-P203Q-T241A_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/adapter_linear/S68P-P203Q-T241A_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/adapter_linear/S68P-T241A-H244Y_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/adapter_linear/S68P-T241A-H244Y_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/adapter_linear/S68P-T241A-H244Y_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/adapter_linear/S68P-T241A-H244Y_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/adapter_linear/S68P-T241A_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/adapter_linear/S68P-T241A_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/adapter_linear/S68P-T241A_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/adapter_linear/S68P-T241A_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_mlp/S68P-P203Q-T241A-H244Y_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_mlp/S68P-P203Q-T241A-H244Y_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_mlp/S68P-P203Q-T241A-H244Y_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_mlp/S68P-P203Q-T241A-H244Y_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_mlp/S68P-P203Q-T241A_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_mlp/S68P-P203Q-T241A_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_mlp/S68P-P203Q-T241A_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_mlp/S68P-P203Q-T241A_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_mlp/S68P-T241A-H244Y_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_mlp/S68P-T241A-H244Y_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_mlp/S68P-T241A-H244Y_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_mlp/S68P-T241A-H244Y_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_mlp/S68P-T241A_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_mlp/S68P-T241A_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_mlp/S68P-T241A_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_mlp/S68P-T241A_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_random_forest/S68P-P203Q-T241A-H244Y_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_random_forest/S68P-P203Q-T241A-H244Y_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_random_forest/S68P-P203Q-T241A-H244Y_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_random_forest/S68P-P203Q-T241A-H244Y_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_random_forest/S68P-P203Q-T241A_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_random_forest/S68P-P203Q-T241A_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_random_forest/S68P-P203Q-T241A_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_random_forest/S68P-P203Q-T241A_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_random_forest/S68P-T241A-H244Y_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_random_forest/S68P-T241A-H244Y_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_random_forest/S68P-T241A-H244Y_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_random_forest/S68P-T241A-H244Y_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_random_forest/S68P-T241A_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_random_forest/S68P-T241A_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_random_forest/S68P-T241A_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_random_forest/S68P-T241A_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_ridge/S68P-P203Q-T241A-H244Y_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_ridge/S68P-P203Q-T241A-H244Y_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_ridge/S68P-P203Q-T241A-H244Y_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_ridge/S68P-P203Q-T241A-H244Y_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_ridge/S68P-P203Q-T241A_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_ridge/S68P-P203Q-T241A_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_ridge/S68P-P203Q-T241A_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_ridge/S68P-P203Q-T241A_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_ridge/S68P-T241A-H244Y_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_ridge/S68P-T241A-H244Y_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_ridge/S68P-T241A-H244Y_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_ridge/S68P-T241A-H244Y_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_ridge/S68P-T241A_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_ridge/S68P-T241A_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_ridge/S68P-T241A_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mean_ridge/S68P-T241A_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mutpool_ridge/S68P-P203Q-T241A-H244Y_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mutpool_ridge/S68P-P203Q-T241A-H244Y_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mutpool_ridge/S68P-P203Q-T241A-H244Y_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mutpool_ridge/S68P-P203Q-T241A-H244Y_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mutpool_ridge/S68P-P203Q-T241A_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mutpool_ridge/S68P-P203Q-T241A_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mutpool_ridge/S68P-P203Q-T241A_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mutpool_ridge/S68P-P203Q-T241A_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mutpool_ridge/S68P-T241A-H244Y_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mutpool_ridge/S68P-T241A-H244Y_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mutpool_ridge/S68P-T241A-H244Y_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mutpool_ridge/S68P-T241A-H244Y_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mutpool_ridge/S68P-T241A_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mutpool_ridge/S68P-T241A_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mutpool_ridge/S68P-T241A_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_mutpool_ridge/S68P-T241A_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_residue_conv/S68P-P203Q-T241A-H244Y_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_residue_conv/S68P-P203Q-T241A-H244Y_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_residue_conv/S68P-P203Q-T241A-H244Y_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_residue_conv/S68P-P203Q-T241A-H244Y_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_residue_conv/S68P-P203Q-T241A_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_residue_conv/S68P-P203Q-T241A_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_residue_conv/S68P-P203Q-T241A_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_residue_conv/S68P-P203Q-T241A_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_residue_conv/S68P-T241A-H244Y_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_residue_conv/S68P-T241A-H244Y_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_residue_conv/S68P-T241A-H244Y_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_residue_conv/S68P-T241A-H244Y_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_residue_conv/S68P-T241A_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_residue_conv/S68P-T241A_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_residue_conv/S68P-T241A_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_residue_conv/S68P-T241A_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_residue_delta_conv/S68P-P203Q-T241A-H244Y_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_residue_delta_conv/S68P-P203Q-T241A-H244Y_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_residue_delta_conv/S68P-P203Q-T241A-H244Y_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_residue_delta_conv/S68P-P203Q-T241A-H244Y_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_residue_delta_conv/S68P-P203Q-T241A_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_residue_delta_conv/S68P-P203Q-T241A_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_residue_delta_conv/S68P-P203Q-T241A_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_residue_delta_conv/S68P-P203Q-T241A_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_residue_delta_conv/S68P-T241A-H244Y_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_residue_delta_conv/S68P-T241A-H244Y_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_residue_delta_conv/S68P-T241A-H244Y_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_residue_delta_conv/S68P-T241A-H244Y_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_residue_delta_conv/S68P-T241A_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_residue_delta_conv/S68P-T241A_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_residue_delta_conv/S68P-T241A_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/esm_residue_delta_conv/S68P-T241A_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_positional_conv/S68P-P203Q-T241A-H244Y_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_positional_conv/S68P-P203Q-T241A-H244Y_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_positional_conv/S68P-P203Q-T241A-H244Y_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_positional_conv/S68P-P203Q-T241A-H244Y_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_positional_conv/S68P-P203Q-T241A_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_positional_conv/S68P-P203Q-T241A_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_positional_conv/S68P-P203Q-T241A_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_positional_conv/S68P-P203Q-T241A_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_positional_conv/S68P-T241A-H244Y_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_positional_conv/S68P-T241A-H244Y_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_positional_conv/S68P-T241A-H244Y_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_positional_conv/S68P-T241A-H244Y_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_positional_conv/S68P-T241A_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_positional_conv/S68P-T241A_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_positional_conv/S68P-T241A_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_positional_conv/S68P-T241A_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_ridge_augmented/S68P-P203Q-T241A-H244Y_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_ridge_augmented/S68P-P203Q-T241A-H244Y_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_ridge_augmented/S68P-P203Q-T241A-H244Y_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_ridge_augmented/S68P-P203Q-T241A-H244Y_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_ridge_augmented/S68P-P203Q-T241A_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_ridge_augmented/S68P-P203Q-T241A_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_ridge_augmented/S68P-P203Q-T241A_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_ridge_augmented/S68P-P203Q-T241A_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_ridge_augmented/S68P-T241A-H244Y_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_ridge_augmented/S68P-T241A-H244Y_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_ridge_augmented/S68P-T241A-H244Y_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_ridge_augmented/S68P-T241A-H244Y_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_ridge_augmented/S68P-T241A_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_ridge_augmented/S68P-T241A_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_ridge_augmented/S68P-T241A_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_ridge_augmented/S68P-T241A_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_ridge_unaugmented/S68P-P203Q-T241A-H244Y_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_ridge_unaugmented/S68P-P203Q-T241A-H244Y_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_ridge_unaugmented/S68P-P203Q-T241A-H244Y_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_ridge_unaugmented/S68P-P203Q-T241A-H244Y_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_ridge_unaugmented/S68P-P203Q-T241A_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_ridge_unaugmented/S68P-P203Q-T241A_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_ridge_unaugmented/S68P-P203Q-T241A_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_ridge_unaugmented/S68P-P203Q-T241A_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_ridge_unaugmented/S68P-T241A-H244Y_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_ridge_unaugmented/S68P-T241A-H244Y_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_ridge_unaugmented/S68P-T241A-H244Y_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_ridge_unaugmented/S68P-T241A-H244Y_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_ridge_unaugmented/S68P-T241A_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_ridge_unaugmented/S68P-T241A_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_ridge_unaugmented/S68P-T241A_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/onehot_ridge_unaugmented/S68P-T241A_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/phi_w1_max_all_live/S68P-P203Q-T241A-H244Y_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/phi_w1_max_all_live/S68P-P203Q-T241A-H244Y_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/phi_w1_max_all_live/S68P-P203Q-T241A-H244Y_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/phi_w1_max_all_live/S68P-P203Q-T241A-H244Y_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/phi_w1_max_all_live/S68P-P203Q-T241A_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/phi_w1_max_all_live/S68P-P203Q-T241A_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/phi_w1_max_all_live/S68P-P203Q-T241A_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/phi_w1_max_all_live/S68P-P203Q-T241A_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/phi_w1_max_all_live/S68P-T241A-H244Y_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/phi_w1_max_all_live/S68P-T241A-H244Y_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/phi_w1_max_all_live/S68P-T241A-H244Y_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/phi_w1_max_all_live/S68P-T241A-H244Y_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/phi_w1_max_all_live/S68P-T241A_round_1.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/phi_w1_max_all_live/S68P-T241A_round_2.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/phi_w1_max_all_live/S68P-T241A_round_3.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/round_top/phi_w1_max_all_live/S68P-T241A_round_4.csv` | 30 | Top 30 retained candidates for one model/founder/round | Per-file lineage and round ranking |
| `Reports/applications/ired_design/storage_artifact_audit.csv` | 4227 | File-level storage, git, size, and recovery snapshot | Pod-stop safety audit |
| `Reports/applications/ired_design/thermompnn_anchor_validation.csv` | 6 | Oracle validation against six measured stability anchors | Permission to use ThermoMPNN |
| `Reports/applications/ired_design/trajectory_graphs/aaindex_flat_ridge_augmented.png` | — | Node-and-edge in-silico trajectory graph | Visual trajectory comparison |
| `Reports/applications/ired_design/trajectory_graphs/aaindex_flat_ridge_unaugmented.png` | — | Node-and-edge in-silico trajectory graph | Visual trajectory comparison |
| `Reports/applications/ired_design/trajectory_graphs/adapter_linear.png` | — | Node-and-edge in-silico trajectory graph | Visual trajectory comparison |
| `Reports/applications/ired_design/trajectory_graphs/ensemble.png` | — | Node-and-edge in-silico trajectory graph | Visual trajectory comparison |
| `Reports/applications/ired_design/trajectory_graphs/esm_mean_mlp.png` | — | Node-and-edge in-silico trajectory graph | Visual trajectory comparison |
| `Reports/applications/ired_design/trajectory_graphs/esm_mean_random_forest.png` | — | Node-and-edge in-silico trajectory graph | Visual trajectory comparison |
| `Reports/applications/ired_design/trajectory_graphs/esm_mean_ridge.png` | — | Node-and-edge in-silico trajectory graph | Visual trajectory comparison |
| `Reports/applications/ired_design/trajectory_graphs/esm_mutpool_ridge.png` | — | Node-and-edge in-silico trajectory graph | Visual trajectory comparison |
| `Reports/applications/ired_design/trajectory_graphs/esm_residue_conv.png` | — | Node-and-edge in-silico trajectory graph | Visual trajectory comparison |
| `Reports/applications/ired_design/trajectory_graphs/esm_residue_delta_conv.png` | — | Node-and-edge in-silico trajectory graph | Visual trajectory comparison |
| `Reports/applications/ired_design/trajectory_graphs/onehot_positional_conv.png` | — | Node-and-edge in-silico trajectory graph | Visual trajectory comparison |
| `Reports/applications/ired_design/trajectory_graphs/onehot_ridge_augmented.png` | — | Node-and-edge in-silico trajectory graph | Visual trajectory comparison |
| `Reports/applications/ired_design/trajectory_graphs/onehot_ridge_unaugmented.png` | — | Node-and-edge in-silico trajectory graph | Visual trajectory comparison |
| `Reports/applications/ired_design/trajectory_graphs/phi_w1_max_all_live.png` | — | Node-and-edge in-silico trajectory graph | Visual trajectory comparison |
| `Reports/applications/ired_design/trajectory_overlap.csv` | 208 | Observed-campaign overlap and measured fitness by model, founder, and round | Trajectory recovery analysis |
| `Reports/applications/ired_design/trajectory_passage_controls.csv` | 65 | First selected round for winning intermediates, peak, and dead ends | Could the model find the measured trajectory? |


## Pod-only application tables

The 208 full-lineage parquets under `Reports/applications/ired_design/full_lineage/<model>/<founder>_round_<n>.parquet` contain all 1,147,143 evaluated candidates, including rejected candidates and parent edges. They are pod/network-volume only. Their sizes are committed in `pod_full_lineage_manifest.csv`. Restarting or replacing the pod is needed only to query those full rejected-candidate tables or to use the 65 generation checkpoints; it is not needed for the meeting tables and plots.

| Path | Rows | Contents | Supports |
|---|---:|---|---|
| `Reports/applications/ired_design/full_lineage/aaindex_flat_ridge_augmented/S68P-P203Q-T241A-H244Y_round_1.parquet` | 259 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/aaindex_flat_ridge_augmented/S68P-P203Q-T241A-H244Y_round_2.parquet` | 7,131 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/aaindex_flat_ridge_augmented/S68P-P203Q-T241A-H244Y_round_3.parquet` | 6,987 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/aaindex_flat_ridge_augmented/S68P-P203Q-T241A-H244Y_round_4.parquet` | 6,849 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/aaindex_flat_ridge_augmented/S68P-P203Q-T241A_round_1.parquet` | 260 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/aaindex_flat_ridge_augmented/S68P-P203Q-T241A_round_2.parquet` | 7,161 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/aaindex_flat_ridge_augmented/S68P-P203Q-T241A_round_3.parquet` | 7,013 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/aaindex_flat_ridge_augmented/S68P-P203Q-T241A_round_4.parquet` | 6,876 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/aaindex_flat_ridge_augmented/S68P-T241A-H244Y_round_1.parquet` | 262 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/aaindex_flat_ridge_augmented/S68P-T241A-H244Y_round_2.parquet` | 7,219 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/aaindex_flat_ridge_augmented/S68P-T241A-H244Y_round_3.parquet` | 7,121 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/aaindex_flat_ridge_augmented/S68P-T241A-H244Y_round_4.parquet` | 6,953 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/aaindex_flat_ridge_augmented/S68P-T241A_round_1.parquet` | 263 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/aaindex_flat_ridge_augmented/S68P-T241A_round_2.parquet` | 7,251 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/aaindex_flat_ridge_augmented/S68P-T241A_round_3.parquet` | 7,133 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/aaindex_flat_ridge_augmented/S68P-T241A_round_4.parquet` | 6,981 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/aaindex_flat_ridge_unaugmented/S68P-P203Q-T241A-H244Y_round_1.parquet` | 224 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/aaindex_flat_ridge_unaugmented/S68P-P203Q-T241A-H244Y_round_2.parquet` | 5,902 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/aaindex_flat_ridge_unaugmented/S68P-P203Q-T241A-H244Y_round_3.parquet` | 5,443 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/aaindex_flat_ridge_unaugmented/S68P-P203Q-T241A-H244Y_round_4.parquet` | 4,896 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/aaindex_flat_ridge_unaugmented/S68P-P203Q-T241A_round_1.parquet` | 243 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/aaindex_flat_ridge_unaugmented/S68P-P203Q-T241A_round_2.parquet` | 6,467 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/aaindex_flat_ridge_unaugmented/S68P-P203Q-T241A_round_3.parquet` | 6,015 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/aaindex_flat_ridge_unaugmented/S68P-P203Q-T241A_round_4.parquet` | 5,466 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/aaindex_flat_ridge_unaugmented/S68P-T241A-H244Y_round_1.parquet` | 243 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/aaindex_flat_ridge_unaugmented/S68P-T241A-H244Y_round_2.parquet` | 6,472 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/aaindex_flat_ridge_unaugmented/S68P-T241A-H244Y_round_3.parquet` | 6,032 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/aaindex_flat_ridge_unaugmented/S68P-T241A-H244Y_round_4.parquet` | 5,489 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/aaindex_flat_ridge_unaugmented/S68P-T241A_round_1.parquet` | 262 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/aaindex_flat_ridge_unaugmented/S68P-T241A_round_2.parquet` | 7,037 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/aaindex_flat_ridge_unaugmented/S68P-T241A_round_3.parquet` | 6,585 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/aaindex_flat_ridge_unaugmented/S68P-T241A_round_4.parquet` | 6,046 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/adapter_linear/S68P-P203Q-T241A-H244Y_round_1.parquet` | 271 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/adapter_linear/S68P-P203Q-T241A-H244Y_round_2.parquet` | 7,500 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/adapter_linear/S68P-P203Q-T241A-H244Y_round_3.parquet` | 7,608 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/adapter_linear/S68P-P203Q-T241A-H244Y_round_4.parquet` | 7,448 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/adapter_linear/S68P-P203Q-T241A_round_1.parquet` | 280 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/adapter_linear/S68P-P203Q-T241A_round_2.parquet` | 7,776 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/adapter_linear/S68P-P203Q-T241A_round_3.parquet` | 7,934 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/adapter_linear/S68P-P203Q-T241A_round_4.parquet` | 7,681 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/adapter_linear/S68P-T241A-H244Y_round_1.parquet` | 271 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/adapter_linear/S68P-T241A-H244Y_round_2.parquet` | 7,543 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/adapter_linear/S68P-T241A-H244Y_round_3.parquet` | 7,653 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/adapter_linear/S68P-T241A-H244Y_round_4.parquet` | 7,449 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/adapter_linear/S68P-T241A_round_1.parquet` | 280 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/adapter_linear/S68P-T241A_round_2.parquet` | 7,805 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/adapter_linear/S68P-T241A_round_3.parquet` | 7,819 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/adapter_linear/S68P-T241A_round_4.parquet` | 7,754 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_mlp/S68P-P203Q-T241A-H244Y_round_1.parquet` | 284 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_mlp/S68P-P203Q-T241A-H244Y_round_2.parquet` | 7,980 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_mlp/S68P-P203Q-T241A-H244Y_round_3.parquet` | 8,196 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_mlp/S68P-P203Q-T241A-H244Y_round_4.parquet` | 8,160 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_mlp/S68P-P203Q-T241A_round_1.parquet` | 285 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_mlp/S68P-P203Q-T241A_round_2.parquet` | 8,009 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_mlp/S68P-P203Q-T241A_round_3.parquet` | 8,100 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_mlp/S68P-P203Q-T241A_round_4.parquet` | 8,189 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_mlp/S68P-T241A-H244Y_round_1.parquet` | 287 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_mlp/S68P-T241A-H244Y_round_2.parquet` | 8,101 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_mlp/S68P-T241A-H244Y_round_3.parquet` | 8,370 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_mlp/S68P-T241A-H244Y_round_4.parquet` | 8,339 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_mlp/S68P-T241A_round_1.parquet` | 288 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_mlp/S68P-T241A_round_2.parquet` | 8,124 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_mlp/S68P-T241A_round_3.parquet` | 8,397 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_mlp/S68P-T241A_round_4.parquet` | 8,382 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_random_forest/S68P-P203Q-T241A-H244Y_round_1.parquet` | 282 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_random_forest/S68P-P203Q-T241A-H244Y_round_2.parquet` | 7,948 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_random_forest/S68P-P203Q-T241A-H244Y_round_3.parquet` | 8,011 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_random_forest/S68P-P203Q-T241A-H244Y_round_4.parquet` | 8,095 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_random_forest/S68P-P203Q-T241A_round_1.parquet` | 282 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_random_forest/S68P-P203Q-T241A_round_2.parquet` | 7,919 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_random_forest/S68P-P203Q-T241A_round_3.parquet` | 8,131 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_random_forest/S68P-P203Q-T241A_round_4.parquet` | 8,136 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_random_forest/S68P-T241A-H244Y_round_1.parquet` | 291 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_random_forest/S68P-T241A-H244Y_round_2.parquet` | 8,183 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_random_forest/S68P-T241A-H244Y_round_3.parquet` | 8,514 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_random_forest/S68P-T241A-H244Y_round_4.parquet` | 8,521 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_random_forest/S68P-T241A_round_1.parquet` | 291 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_random_forest/S68P-T241A_round_2.parquet` | 8,201 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_random_forest/S68P-T241A_round_3.parquet` | 8,501 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_random_forest/S68P-T241A_round_4.parquet` | 8,523 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_ridge/S68P-P203Q-T241A-H244Y_round_1.parquet` | 284 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_ridge/S68P-P203Q-T241A-H244Y_round_2.parquet` | 7,814 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_ridge/S68P-P203Q-T241A-H244Y_round_3.parquet` | 7,538 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_ridge/S68P-P203Q-T241A-H244Y_round_4.parquet` | 7,469 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_ridge/S68P-P203Q-T241A_round_1.parquet` | 284 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_ridge/S68P-P203Q-T241A_round_2.parquet` | 7,838 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_ridge/S68P-P203Q-T241A_round_3.parquet` | 7,509 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_ridge/S68P-P203Q-T241A_round_4.parquet` | 7,386 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_ridge/S68P-T241A-H244Y_round_1.parquet` | 284 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_ridge/S68P-T241A-H244Y_round_2.parquet` | 7,835 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_ridge/S68P-T241A-H244Y_round_3.parquet` | 7,513 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_ridge/S68P-T241A-H244Y_round_4.parquet` | 7,477 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_ridge/S68P-T241A_round_1.parquet` | 284 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_ridge/S68P-T241A_round_2.parquet` | 7,824 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_ridge/S68P-T241A_round_3.parquet` | 7,533 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mean_ridge/S68P-T241A_round_4.parquet` | 7,467 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mutpool_ridge/S68P-P203Q-T241A-H244Y_round_1.parquet` | 255 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mutpool_ridge/S68P-P203Q-T241A-H244Y_round_2.parquet` | 6,872 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mutpool_ridge/S68P-P203Q-T241A-H244Y_round_3.parquet` | 6,779 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mutpool_ridge/S68P-P203Q-T241A-H244Y_round_4.parquet` | 6,530 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mutpool_ridge/S68P-P203Q-T241A_round_1.parquet` | 255 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mutpool_ridge/S68P-P203Q-T241A_round_2.parquet` | 6,881 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mutpool_ridge/S68P-P203Q-T241A_round_3.parquet` | 6,791 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mutpool_ridge/S68P-P203Q-T241A_round_4.parquet` | 6,557 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mutpool_ridge/S68P-T241A-H244Y_round_1.parquet` | 262 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mutpool_ridge/S68P-T241A-H244Y_round_2.parquet` | 7,085 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mutpool_ridge/S68P-T241A-H244Y_round_3.parquet` | 6,933 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mutpool_ridge/S68P-T241A-H244Y_round_4.parquet` | 6,674 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mutpool_ridge/S68P-T241A_round_1.parquet` | 262 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mutpool_ridge/S68P-T241A_round_2.parquet` | 7,077 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mutpool_ridge/S68P-T241A_round_3.parquet` | 6,942 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_mutpool_ridge/S68P-T241A_round_4.parquet` | 6,662 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_residue_conv/S68P-P203Q-T241A-H244Y_round_1.parquet` | 277 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_residue_conv/S68P-P203Q-T241A-H244Y_round_2.parquet` | 7,603 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_residue_conv/S68P-P203Q-T241A-H244Y_round_3.parquet` | 7,405 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_residue_conv/S68P-P203Q-T241A-H244Y_round_4.parquet` | 7,432 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_residue_conv/S68P-P203Q-T241A_round_1.parquet` | 278 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_residue_conv/S68P-P203Q-T241A_round_2.parquet` | 7,654 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_residue_conv/S68P-P203Q-T241A_round_3.parquet` | 7,409 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_residue_conv/S68P-P203Q-T241A_round_4.parquet` | 7,451 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_residue_conv/S68P-T241A-H244Y_round_1.parquet` | 278 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_residue_conv/S68P-T241A-H244Y_round_2.parquet` | 7,649 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_residue_conv/S68P-T241A-H244Y_round_3.parquet` | 7,323 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_residue_conv/S68P-T241A-H244Y_round_4.parquet` | 7,455 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_residue_conv/S68P-T241A_round_1.parquet` | 279 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_residue_conv/S68P-T241A_round_2.parquet` | 7,664 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_residue_conv/S68P-T241A_round_3.parquet` | 7,460 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_residue_conv/S68P-T241A_round_4.parquet` | 7,469 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_residue_delta_conv/S68P-P203Q-T241A-H244Y_round_1.parquet` | 288 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_residue_delta_conv/S68P-P203Q-T241A-H244Y_round_2.parquet` | 7,880 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_residue_delta_conv/S68P-P203Q-T241A-H244Y_round_3.parquet` | 7,834 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_residue_delta_conv/S68P-P203Q-T241A-H244Y_round_4.parquet` | 7,690 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_residue_delta_conv/S68P-P203Q-T241A_round_1.parquet` | 289 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_residue_delta_conv/S68P-P203Q-T241A_round_2.parquet` | 7,919 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_residue_delta_conv/S68P-P203Q-T241A_round_3.parquet` | 7,838 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_residue_delta_conv/S68P-P203Q-T241A_round_4.parquet` | 7,726 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_residue_delta_conv/S68P-T241A-H244Y_round_1.parquet` | 290 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_residue_delta_conv/S68P-T241A-H244Y_round_2.parquet` | 7,962 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_residue_delta_conv/S68P-T241A-H244Y_round_3.parquet` | 7,909 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_residue_delta_conv/S68P-T241A-H244Y_round_4.parquet` | 7,818 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_residue_delta_conv/S68P-T241A_round_1.parquet` | 291 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_residue_delta_conv/S68P-T241A_round_2.parquet` | 8,004 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_residue_delta_conv/S68P-T241A_round_3.parquet` | 8,061 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/esm_residue_delta_conv/S68P-T241A_round_4.parquet` | 7,897 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_positional_conv/S68P-P203Q-T241A-H244Y_round_1.parquet` | 246 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_positional_conv/S68P-P203Q-T241A-H244Y_round_2.parquet` | 6,584 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_positional_conv/S68P-P203Q-T241A-H244Y_round_3.parquet` | 6,412 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_positional_conv/S68P-P203Q-T241A-H244Y_round_4.parquet` | 6,306 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_positional_conv/S68P-P203Q-T241A_round_1.parquet` | 257 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_positional_conv/S68P-P203Q-T241A_round_2.parquet` | 6,920 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_positional_conv/S68P-P203Q-T241A_round_3.parquet` | 6,607 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_positional_conv/S68P-P203Q-T241A_round_4.parquet` | 6,711 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_positional_conv/S68P-T241A-H244Y_round_1.parquet` | 251 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_positional_conv/S68P-T241A-H244Y_round_2.parquet` | 6,742 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_positional_conv/S68P-T241A-H244Y_round_3.parquet` | 6,655 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_positional_conv/S68P-T241A-H244Y_round_4.parquet` | 6,708 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_positional_conv/S68P-T241A_round_1.parquet` | 262 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_positional_conv/S68P-T241A_round_2.parquet` | 7,068 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_positional_conv/S68P-T241A_round_3.parquet` | 6,980 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_positional_conv/S68P-T241A_round_4.parquet` | 6,999 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_ridge_augmented/S68P-P203Q-T241A-H244Y_round_1.parquet` | 259 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_ridge_augmented/S68P-P203Q-T241A-H244Y_round_2.parquet` | 7,126 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_ridge_augmented/S68P-P203Q-T241A-H244Y_round_3.parquet` | 6,988 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_ridge_augmented/S68P-P203Q-T241A-H244Y_round_4.parquet` | 6,847 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_ridge_augmented/S68P-P203Q-T241A_round_1.parquet` | 260 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_ridge_augmented/S68P-P203Q-T241A_round_2.parquet` | 7,166 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_ridge_augmented/S68P-P203Q-T241A_round_3.parquet` | 6,987 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_ridge_augmented/S68P-P203Q-T241A_round_4.parquet` | 6,856 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_ridge_augmented/S68P-T241A-H244Y_round_1.parquet` | 262 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_ridge_augmented/S68P-T241A-H244Y_round_2.parquet` | 7,227 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_ridge_augmented/S68P-T241A-H244Y_round_3.parquet` | 7,099 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_ridge_augmented/S68P-T241A-H244Y_round_4.parquet` | 6,937 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_ridge_augmented/S68P-T241A_round_1.parquet` | 263 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_ridge_augmented/S68P-T241A_round_2.parquet` | 7,256 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_ridge_augmented/S68P-T241A_round_3.parquet` | 7,133 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_ridge_augmented/S68P-T241A_round_4.parquet` | 6,971 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_ridge_unaugmented/S68P-P203Q-T241A-H244Y_round_1.parquet` | 224 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_ridge_unaugmented/S68P-P203Q-T241A-H244Y_round_2.parquet` | 5,902 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_ridge_unaugmented/S68P-P203Q-T241A-H244Y_round_3.parquet` | 5,461 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_ridge_unaugmented/S68P-P203Q-T241A-H244Y_round_4.parquet` | 4,914 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_ridge_unaugmented/S68P-P203Q-T241A_round_1.parquet` | 243 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_ridge_unaugmented/S68P-P203Q-T241A_round_2.parquet` | 6,467 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_ridge_unaugmented/S68P-P203Q-T241A_round_3.parquet` | 6,011 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_ridge_unaugmented/S68P-P203Q-T241A_round_4.parquet` | 5,462 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_ridge_unaugmented/S68P-T241A-H244Y_round_1.parquet` | 243 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_ridge_unaugmented/S68P-T241A-H244Y_round_2.parquet` | 6,472 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_ridge_unaugmented/S68P-T241A-H244Y_round_3.parquet` | 6,032 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_ridge_unaugmented/S68P-T241A-H244Y_round_4.parquet` | 5,490 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_ridge_unaugmented/S68P-T241A_round_1.parquet` | 262 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_ridge_unaugmented/S68P-T241A_round_2.parquet` | 7,037 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_ridge_unaugmented/S68P-T241A_round_3.parquet` | 6,582 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/onehot_ridge_unaugmented/S68P-T241A_round_4.parquet` | 6,046 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/phi_w1_max_all_live/S68P-P203Q-T241A-H244Y_round_1.parquet` | 295 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/phi_w1_max_all_live/S68P-P203Q-T241A-H244Y_round_2.parquet` | 8,059 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/phi_w1_max_all_live/S68P-P203Q-T241A-H244Y_round_3.parquet` | 7,696 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/phi_w1_max_all_live/S68P-P203Q-T241A-H244Y_round_4.parquet` | 7,380 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/phi_w1_max_all_live/S68P-P203Q-T241A_round_1.parquet` | 297 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/phi_w1_max_all_live/S68P-P203Q-T241A_round_2.parquet` | 8,119 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/phi_w1_max_all_live/S68P-P203Q-T241A_round_3.parquet` | 7,738 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/phi_w1_max_all_live/S68P-P203Q-T241A_round_4.parquet` | 7,440 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/phi_w1_max_all_live/S68P-T241A-H244Y_round_1.parquet` | 295 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/phi_w1_max_all_live/S68P-T241A-H244Y_round_2.parquet` | 8,064 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/phi_w1_max_all_live/S68P-T241A-H244Y_round_3.parquet` | 7,680 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/phi_w1_max_all_live/S68P-T241A-H244Y_round_4.parquet` | 7,409 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/phi_w1_max_all_live/S68P-T241A_round_1.parquet` | 297 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/phi_w1_max_all_live/S68P-T241A_round_2.parquet` | 8,119 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/phi_w1_max_all_live/S68P-T241A_round_3.parquet` | 7,731 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |
| `Reports/applications/ired_design/full_lineage/phi_w1_max_all_live/S68P-T241A_round_4.parquet` | 7,460 | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |


## CROSSFIT main-track artifacts

The main track was audited read-only and was not modified. Reports are committed; models and caches are Git-ignored and live only on the RunPod network volume.

| Directory | Files | Size | Storage |
|---|---:|---:|---|
| `Cache/step2/` | 11 | 1.675 GB | network volume only; gitignored |
| `Cache/step3/` | 15 | 9.862 GB | network volume only; gitignored |
| `Cache/step4/` | 30 | 996.689 MB | network volume only; gitignored |
| `Cache/step6/` | 32 | 996.696 MB | network volume only; gitignored |
| `Cache/step8/` | 8 | 178.680 MB | network volume only; gitignored |
| `Cache/step9/` | 1,209 | 115.133 MB | network volume only; gitignored |
| `Models/step2/` | 45 | 259.318 MB | network volume only; gitignored |
| `Models/step3/` | 12 | 15.640 MB | network volume only; gitignored |
| `Models/step6/` | 15 | 298.947 MB | network volume only; gitignored |
| `Models/step7/` | 25 | 1.387 GB | network volume only; gitignored |
| `Models/step9/` | 1,830 | 3.293 GB | network volume only; gitignored |
| `Reports/step1/` | 5 | 623.186 KB | local + network + git |
| `Reports/step2/` | 54 | 768.571 KB | local + network + git |
| `Reports/step3/` | 18 | 206.039 KB | local + network + git |
| `Reports/step4/` | 5 | 10.290 KB | local + network + git |
| `Reports/step5/` | 11 | 1.456 MB | local + network + git |
| `Reports/step6/` | 42 | 4.376 MB | local + network + git |
| `Reports/step7/` | 38 | 3.070 MB | local + network + git |
| `Reports/step8/` | 8 | 63.963 KB | local + network + git |
| `Reports/step9/` | 146 | 28.964 MB | local + network + git |


For file-level paths and recovery status, filter `storage_artifact_audit.csv` to `track == "CROSSFIT main"`. The especially costly assets are `Cache/step3/` (9.862 GB), `Models/step7/` (1.387 GB), `Models/step9/` (3.293 GB), and the Step 2/4/6 activation caches. All are on the network volume, not container disk.
