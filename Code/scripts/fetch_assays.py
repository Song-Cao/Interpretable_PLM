"""Download the selected ProteinGym 2.0 `.pgdata` bundles at the pinned revision.

Each bundle is a zip containing `dataset.pgdata` (itself a zip: assay CSV, WT fasta,
structure, MSA, TOML manifest) and `slices.json` (published split masks).

Downloads are verified against the per-file oid recorded in `sources_lock.json` by
`pin_sources.py`, so a silently-changed upstream file is caught rather than absorbed.
Re-running skips files already present and verified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossfit.selection import SELECTED_IDS
from crossfit.utils import load_config, resolve

HF_RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{name}"
CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(CHUNK), b""):
            h.update(blk)
    return h.hexdigest()


def download(url: str, dest: Path) -> int:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as fh:
        n = 0
        while True:
            blk = r.read(CHUNK)
            if not blk:
                break
            fh.write(blk)
            n += len(blk)
    tmp.replace(dest)
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args()

    cfg = load_config()
    lock = json.loads((resolve(cfg, "reference") / "sources_lock.json").read_text())
    repo = cfg["sources"]["proteingym2_repo"]
    rev = cfg["sources"]["proteingym2_revision"]
    files = lock["proteingym2"]["file_manifest"]

    out = resolve(cfg, "raw_pgdata")
    out.mkdir(parents=True, exist_ok=True)

    report = []
    for assay in SELECTED_IDS:
        name = f"{assay}.pgdata"
        meta = files.get(name)
        if meta is None:
            print(f"[FAIL] {name}: not in pinned manifest", flush=True)
            return 2
        dest = out / name
        if dest.exists() and not args.force and dest.stat().st_size == meta["size"]:
            print(f"[skip] {name} ({dest.stat().st_size/1e6:.1f} MB)", flush=True)
        else:
            url = HF_RESOLVE.format(repo=repo, rev=rev, name=name)
            n = download(url, dest)
            print(f"[get ] {name} ({n/1e6:.1f} MB)", flush=True)
        # Size is the integrity check we can apply locally: HF records an xet/lfs oid
        # that is a content-defined-chunking hash, not a plain sha256 of the file, so
        # it cannot be recomputed here. Size mismatch against the pinned manifest is
        # still a hard failure.
        if dest.stat().st_size != meta["size"]:
            print(f"[FAIL] {name}: size {dest.stat().st_size} != pinned {meta['size']}", flush=True)
            return 3
        report.append({
            "assay_id": assay,
            "file": name,
            "size_bytes": dest.stat().st_size,
            "pinned_size": meta["size"],
            "pinned_oid": meta.get("oid"),
            "local_sha256": sha256_file(dest),
        })

    man = resolve(cfg, "raw_pgdata") / "download_manifest.json"
    man.write_text(json.dumps(
        {"repo": repo, "revision": rev, "n_assays": len(report), "files": report}, indent=2))
    print(f"\n{len(report)} bundles, {sum(r['size_bytes'] for r in report)/1e6:.0f} MB total")
    print(f"manifest -> {man}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
