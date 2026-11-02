#!/usr/bin/env python3
"""Snapshot local and RunPod artifact locations before the pod is stopped."""
from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "Reports/applications/ired_design"
POD = "runpod-interpretable-plm"
POD_ROOT = "/workspace/Interpretable_PLM"


def _run(*args: str) -> str:
    return subprocess.run(args, check=True, text=True, capture_output=True).stdout


def _pod_files() -> dict[str, int]:
    command = """cd /workspace/Interpretable_PLM && find \
Reports/applications/ired_design Models/applications/ired_design \
Reports/step* Models/step* Cache/step* -type f -printf '%p\\t%s\\n' 2>/dev/null"""
    output = _run("ssh", POD, command)
    files = {}
    for line in output.splitlines():
        path, size = line.rsplit("\t", 1)
        files[path] = int(size)
    return files


def _pod_lineage_manifest() -> list[dict[str, object]]:
    program = r'''import glob, json, os
import pyarrow.parquet as pq
for path in sorted(glob.glob("Reports/applications/ired_design/full_lineage/**/*.parquet", recursive=True)):
    print(json.dumps({"path": path, "rows": pq.ParquetFile(path).metadata.num_rows, "size_bytes": os.path.getsize(path)}))
'''
    command = f"cd {POD_ROOT} && .venv/bin/python - <<'PY'\n{program}PY"
    output = _run("ssh", POD, command)
    return [json.loads(line) for line in output.splitlines() if line]


def _local_files() -> dict[str, int]:
    paths: dict[str, int] = {}
    roots = [
        ROOT / "Reports/applications/ired_design",
        *sorted((ROOT / "Reports").glob("step*")),
    ]
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file():
                paths[str(path.relative_to(ROOT))] = path.stat().st_size
    return paths


def _human(size: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(size)
    for unit in units:
        if value < 1000 or unit == units[-1]:
            return f"{value:.3f} {unit}"
        value /= 1000
    raise AssertionError


def main() -> int:
    pod = _pod_files()
    local = _local_files()
    tracked = set(_run("git", "-C", str(ROOT), "ls-files").splitlines())
    snapshot = datetime.now(timezone.utc).isoformat()
    rows = []
    for path in sorted(set(pod) | set(local)):
        on_pod = path in pod
        on_local = path in local
        size = local.get(path, pod.get(path, 0))
        committed = path in tracked
        track = "IRED design application" if "applications/ired_design" in path else "CROSSFIT main"
        if path.startswith("Models/"):
            kind = "model checkpoint"
        elif path.startswith("Cache/"):
            kind = "activation/runtime cache"
        elif path.endswith((".png", ".pdf", ".svg")):
            kind = "figure"
        else:
            kind = "table/report/manifest"
        if on_local and on_pod:
            location = "local workstation + RunPod network volume"
        elif on_local:
            location = "local workstation"
        else:
            location = "RunPod network volume only"
        if committed:
            recoverable = "yes: committed to git"
        elif on_local:
            recoverable = "yes: local workstation copy"
        elif kind in {"model checkpoint", "activation/runtime cache"} or "full_lineage/" in path:
            recoverable = "no if network volume is lost; GPU recomputation required"
        else:
            recoverable = "network-volume sole copy; inspect dependencies before regeneration"
        rows.append({
            "track": track,
            "path": path,
            "artifact_kind": kind,
            "size_bytes": size,
            "size_human": _human(size),
            "git_committed": committed,
            "storage_location": location,
            "pod_storage_class": "network volume" if on_pod else "not on pod",
            "recoverable_without_gpu_if_network_volume_lost": recoverable,
            "snapshot_utc": snapshot,
        })
    output = REPORT / "storage_artifact_audit.csv"
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    lineage = _pod_lineage_manifest()
    lineage_output = REPORT / "pod_full_lineage_manifest.csv"
    with lineage_output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("path", "rows", "size_bytes"))
        writer.writeheader()
        writer.writerows(lineage)
    print(f"wrote {len(rows)} rows to {output}")
    print(f"wrote {len(lineage)} rows to {lineage_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
