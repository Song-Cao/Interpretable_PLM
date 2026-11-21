# RunPod to Modal migration: phase-one inventory

Inventory date: 2026-11-21. This phase performed classification and reference-block preservation only. It did not transfer checkpoints or cache payloads, configure Modal, or delete, move, or rename anything on the RunPod network volume.

## Scope and method

The mounted network-volume root is `/workspace` (`mfs#us-ks-2.runpod.net:9421[/networkvolumes/8pxh42cxur]`). Every regular file under that root was read and hashed with SHA-256. Non-directory symlinks are recorded with `file_type=symlink`; their hash is over the link target text. Directory symlinks are not counted as payload files. In particular, the four historical `Interpretable_PLM_v17*` trees point `Cache` and `Models` to the active project directories, so those payloads are counted once.

The provider's `df` output reports service-wide capacity and is not a tenant-usage measurement. The byte totals below are sums of the inventoried files. The full per-file record is in `volume_inventory.csv`; every `regenerable` row includes its producing command and estimated GPU-hours.

Classification was conservative. Files outside CROSSFIT were marked `irreplaceable` unless an upstream-backed clean git checkout, a locked environment, or a laptop copy established recoverability. This is why the transfer candidate total includes unrelated FMTS assets on the shared volume.

## Totals

| Classification | Files | Bytes | Decimal GB |
|---|---:|---:|---:|
| committed | 5,792 | 794,054,491 | 0.794 |
| local_uncommitted | 20 | 122,858,241 | 0.123 |
| regenerable | 192,228 | 36,244,602,538 | 36.245 |
| irreplaceable | 45,779 | 32,981,435,987 | 32.981 |
| **Total** | **243,819** | **70,142,951,257** | **70.143** |

Pending pod-to-laptop transfer, if every `irreplaceable` row is approved, is **32,981,435,987 bytes**. The `local_uncommitted` files are already byte-identical on the laptop and therefore are not included in that transfer total. The CROSSFIT-related portion of the pending transfer is **9,035,068,239 bytes**; the other **23,946,367,748 bytes** belong to unrelated volume content and should be reviewed separately.

Laptop free space after writing this inventory and preserving the reference blocks is **44,874,211,328 bytes**. A full 32,981,435,987-byte approved transfer requires **39,577,723,185 bytes** with 20% headroom, so it fits at this snapshot by **5,296,488,143 bytes**. Free space must be checked again immediately before Part 2.

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

No Part 2 artifact transfer has started. No destination directory for Part 2 has been selected, and no volume content has been changed.
