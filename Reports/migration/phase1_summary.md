# RunPod to Modal migration: phase-one inventory

Inventory date: 2026-11-21. This phase performed classification and reference-block preservation only. It did not transfer checkpoints or cache payloads, configure Modal, or delete, move, or rename anything on the RunPod network volume.

## Scope and method

The mounted network-volume root is `/workspace` (`mfs#us-ks-2.runpod.net:9421[/networkvolumes/8pxh42cxur]`). Every regular file under that root was read and hashed with SHA-256. Non-directory symlinks are recorded with `file_type=symlink`; their hash is over the link target text. Directory symlinks are not counted as payload files. In particular, the four historical `Interpretable_PLM_v17*` trees point `Cache` and `Models` to the active project directories, so those payloads are counted once.

The provider's `df` output reports service-wide capacity and is not a tenant-usage measurement. The byte totals below are sums of the inventoried files. The full per-file record is in `volume_inventory.csv`; every `regenerable` row includes its producing command and estimated GPU-hours.

Classification was conservative. Files outside CROSSFIT were marked `irreplaceable` unless an upstream-backed clean git checkout, a locked environment, or a laptop copy established recoverability. During review, `fmts-workshop-venv` was correctly reclassified as regenerable; its 48 installed package pins are committed in `fmts_workshop_requirements.txt`.

## Totals

| Classification | Files | Bytes | Decimal GB |
|---|---:|---:|---:|
| committed | 5,798 | 794,095,329 | 0.794 |
| local_uncommitted | 20 | 122,858,241 | 0.123 |
| regenerable | 214,034 | 41,692,063,787 | 41.692 |
| irreplaceable | 24,712 | 27,596,816,301 | 27.597 |
| **Total** | **244,564** | **70,205,833,658** | **70.206** |

The reviewed CROSSFIT transfer set is **9,221,227,791 bytes** across 7,621 files. It deliberately includes repository histories and the requested Step 8 cache as well as irreplaceable files; it is therefore not identical to the classification totals. Unrelated FMTS and meta-flow-matching assets are excluded.

Laptop free space measured at 2026-11-21 19:05:27 UTC, after committing this inventory and the reference blocks, was **44,808,388,608 bytes**. A full 32,981,435,987-byte approved transfer requires **39,577,723,185 bytes** with 20% headroom, so it fit at that snapshot by **5,230,665,423 bytes**. Free space must be checked again immediately before Part 2.

## CROSSFIT-specific findings

The active `/workspace/Interpretable_PLM` tree contains 15.498 GB under `Cache`, 8.813 GB under `Models`, and 6.258 GB in `.venv`. Activation payloads and the environment are classified as regenerable. All model and training checkpoints are classified as irreplaceable unless an identical laptop file already exists. This includes adapter, crosscoder, Step 2/9 predictor, and the 65 application checkpoints.

Within `Models`, the largest groups are:

| Group | Bytes |
|---|---:|
| archive | 3,292,588,495 |
| step9 | 2,706,602,169 |
| step7 | 1,475,794,673 |
| step6 | 756,279,382 |
| applications | 306,915,081 |
| step2 | 259,317,737 |
| step3 | 15,640,016 |

Recreating every distinct current and historical CROSSFIT activation-cache job represented on the volume is estimated at **6.154 GPU-hours**. That total deduplicates commands; it is not a sum over files. Historical pre-deterministic, train-mode, and depth-29 caches account for 4.03 of those hours. Commands and per-job estimates are recorded directly on the corresponding CSV rows.

## Canonical reference blocks

Fourteen cache reference blocks were copied to `Reports/migration/reference_blocks/`, preserving cache-relative paths. They contain ten distinct SHA-256 values and occupy 112,475,392 bytes. All copied hashes match both the source files and the checksums recorded in their cache manifests. `reference_blocks_manifest.csv` records the source, local copy, layer, units, and all hash comparisons.

The copied manifests and blocks cover the active Step 2, 3, 4, 5, 6, 8, and 9 cache families. Historical trees expose these same files through directory symlinks; they are aliases, not additional caches.

## Verification and stopping point

The inventory pass reported zero read failures. The repository suite passes with the project environment: `270 passed, 2 warnings` using `.venv/bin/python -m pytest Code/tests -q`.

Part 2 subsequently copied the approved set to `/Users/songcao/workdir/Interpretable_PLM_RunPod_Phase2_2026-11-21`. All 7,621 local SHA-256 values match the refreshed pod manifest. See `transfer_verification.csv` and `part2_transfer_summary.md`. No source payload was deleted, moved, or renamed.
