#!/usr/bin/env python3
"""Write the application and main-track artifact location document."""
from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "Reports/applications/ired_design"


DESCRIPTIONS = {
    "per_round_top_variants.csv": ("All retained per-round candidates with lineage, uncertainty status, additive ThermoMPNN ddG, and controls", "Raw round-by-round review and trajectory reconstruction"),
    "controls_ranks.csv": ("Ranks of A72T, G37C, and H244Y controls for every model", "Known-answer false-positive/positive-control audit"),
    "trajectory_overlap.csv": ("Observed-campaign overlap and measured fitness by model, founder, and round", "Trajectory recovery analysis"),
    "trajectory_passage_controls.csv": ("First selected round for winning intermediates, peak, and dead ends", "Could the model find the measured trajectory?"),
    "consensus_candidates.csv": ("Final 30-candidate consensus with cross-model disagreement", "Meeting shortlist"),
    "consensus_all_final_candidates.csv": ("All distinct final candidates before consensus truncation", "Consensus denominator and candidate audit"),
    "final_candidate_model_scores.csv": ("Every final candidate scored by every model with uncertainty status", "Model disagreement and uncertainty audit"),
    "final_candidate_stability.csv": ("Validated ThermoMPNN additive stability panel", "Biophysical consistency check"),
    "final_consensus_biophysical_panel.csv": ("Consensus shortlist joined to the complete available biophysical panel and explicit unavailable-oracle statuses", "Single meeting-ready candidate table"),
    "thermompnn_anchor_validation.csv": ("Oracle validation against six measured stability anchors", "Permission to use ThermoMPNN"),
    "pod_full_lineage_manifest.csv": ("Path, row count, and size for every pod-only lineage parquet", "Network-volume completeness audit"),
    "storage_artifact_audit.csv": ("File-level storage, git, size, and recovery snapshot", "Pod-stop safety audit"),
}


def _rows(path: Path) -> str:
    if path.suffix == ".csv":
        with path.open(newline="") as handle:
            return str(max(sum(1 for _ in csv.reader(handle)) - 1, 0))
    return "—"


def _describe(path: Path) -> tuple[str, str]:
    name = path.name
    if name in DESCRIPTIONS:
        return DESCRIPTIONS[name]
    if "round_top" in path.parts:
        return ("Top 30 retained candidates for one model/founder/round", "Per-file lineage and round ranking")
    if name.endswith("_round1_single_scan.csv"):
        return ("Complete 5,510 single-substitution scan for one model", "Round-1 halo comparison")
    if name.endswith("_winner_pool.csv"):
        return ("Budget-adjusted top-300 reusable winner mutation pool", "Four-round candidate generation")
    if name.endswith("_top20.csv"):
        return ("Top 20 final suggestions from one model", "Per-model recommendations")
    if path.parent.name == "trajectory_graphs":
        return ("Node-and-edge in-silico trajectory graph", "Visual trajectory comparison")
    if "thermompnn" in path.parts and name.endswith(".csv"):
        if name == "thermompnn_single_substitution_5510.csv":
            return ("Validated 290 x 19 non-identity ThermoMPNN single-substitution matrix", "Additive ddG lookup for round-top and final candidates")
        return ("Raw ThermoMPNN 290 x 20 output including identity rows", "Oracle provenance; noncanonical shifted-chain file is explicitly invalid")
    return ("Application result table or figure", "Application audit/support")


def _application_index() -> list[tuple[str, str, str, str]]:
    paths = sorted(
        path for path in REPORT.rglob("*")
        if path.is_file()
        and path.suffix in {".csv", ".png", ".pdf", ".svg"}
        and "full_lineage" not in path.parts
    )
    rows = []
    for path in paths:
        description, support = _describe(path)
        rows.append((str(path.relative_to(ROOT)), _rows(path), description, support))
    return rows


def _main_summary() -> list[tuple[str, int, int, str]]:
    audit = pd.read_csv(REPORT / "storage_artifact_audit.csv")
    main = audit[audit["track"] == "CROSSFIT main"].copy()
    main["directory"] = main["path"].str.extract(r"^((?:Reports|Models|Cache)/step[^/]+)")[0]
    summary = main.dropna(subset=["directory"]).groupby("directory", sort=True).agg(
        files=("path", "size"), size_bytes=("size_bytes", "sum")
    )
    rows = []
    for directory, item in summary.iterrows():
        storage = "local + network + git" if directory.startswith("Reports/") else "network volume only; gitignored"
        rows.append((directory, int(item.files), int(item.size_bytes), storage))
    return rows


def _lineage_index() -> list[tuple[str, int]]:
    table = pd.read_csv(REPORT / "pod_full_lineage_manifest.csv")
    return [(str(row.path), int(row.rows)) for row in table.itertuples()]


def _human(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1000 or unit == "TB":
            return f"{value:.3f} {unit}"
        value /= 1000
    raise AssertionError


def main() -> int:
    app_rows = _application_index()
    main_rows = _main_summary()
    lineage_rows = _lineage_index()
    text = """# IRED artifact locations

Snapshot taken before stopping the RunPod pod. The pod project root `/workspace/Interpretable_PLM` is mounted from RunPod network volume `8pxh42cxur` (`mfs#us-ks-2.runpod.net:9421[/networkvolumes/8pxh42cxur]`, FUSE). The container root is a separate overlay filesystem. A root-filesystem search found no IRED-design, Step 2, Step 3, or Step 7 artifacts outside `/workspace`.

This is an application workstream inventory. None of its outputs are CROSSFIT evidence.

## Storage verdict

| Asset | Count | Size | Location | Git |
|---|---:|---:|---|---|
| Application generation checkpoints | 65 | 314 MB allocated (307 MB file bytes) | Network volume | Ignored; hashes committed in generation manifests |
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
"""
    for path, rows, description, support in app_rows:
        text += f"| `{path}` | {rows} | {description} | {support} |\n"
    text += """

## Pod-only application tables

The 208 full-lineage parquets under `Reports/applications/ired_design/full_lineage/<model>/<founder>_round_<n>.parquet` contain all 1,147,143 evaluated candidates, including rejected candidates and parent edges. They are pod/network-volume only. Their sizes are committed in `pod_full_lineage_manifest.csv`. Restarting or replacing the pod is needed only to query those full rejected-candidate tables or to use the 65 generation checkpoints; it is not needed for the meeting tables and plots.

| Path | Rows | Contents | Supports |
|---|---:|---|---|
"""
    for path, rows in lineage_rows:
        text += f"| `{path}` | {rows:,} | Every evaluated candidate, including rejected candidates and parent edges | Full trajectory reconstruction and post-hoc queries |\n"
    text += """

## CROSSFIT main-track artifacts

The main track was audited read-only and was not modified. Reports are committed; models and caches are Git-ignored and live only on the RunPod network volume.

| Directory | Files | Size | Storage |
|---|---:|---:|---|
"""
    for directory, files, size, storage in main_rows:
        text += f"| `{directory}/` | {files:,} | {_human(size)} | {storage} |\n"
    text += """

For file-level paths and recovery status, filter `storage_artifact_audit.csv` to `track == "CROSSFIT main"`. The especially costly assets are `Cache/step3/` (9.862 GB), `Models/step7/` (1.387 GB), `Models/step9/` (3.293 GB), and the Step 2/4/6 activation caches. All are on the network volume, not container disk.
"""
    (REPORT / "ARTIFACT_LOCATIONS.md").write_text(text)
    print(
        f"wrote {len(app_rows)} local table/figure entries, {len(lineage_rows)} "
        f"pod-only lineage entries, and {len(main_rows)} main summaries"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
