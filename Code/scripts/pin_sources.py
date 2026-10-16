"""Pin the exact revisions of every external data source used in Step 1.

Writes Data/raw/reference/SOURCES.md and sources_lock.json, and stamps the
revision hashes back into Code/config/step1.yaml. Run once; re-run only when
deliberately moving to a newer upstream revision.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crossfit.utils import REPO_ROOT, load_config, sha256_file  # noqa: E402

HF_API = "https://huggingface.co/api/datasets"
GH_API = "https://api.github.com/repos/OATML-Markslab/ProteinGym/commits"


def main() -> None:
    cfg = load_config()
    ref_dir = REPO_ROOT / cfg["paths"]["reference"]
    ref_dir.mkdir(parents=True, exist_ok=True)
    repo = cfg["sources"]["proteingym2_repo"]
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    info = requests.get(f"{HF_API}/{repo}", timeout=60).json()
    hf_sha, hf_mod = info["sha"], info["lastModified"]

    tree = requests.get(f"{HF_API}/{repo}/tree/main?recursive=1", timeout=120).json()
    files = [f for f in tree if f["type"] == "file" and f["path"].endswith(".pgdata")]
    # HF exposes the git-lfs/xet oid per file; record it as the content pin.
    manifest = {
        f["path"]: {"size": f["size"], "oid": (f.get("lfs") or {}).get("oid") or f.get("oid")}
        for f in files
    }

    ref_csv = ref_dir / "DMS_substitutions.csv"
    gh_sha = None
    try:
        commits = requests.get(
            GH_API, params={"path": "reference_files/DMS_substitutions.csv", "per_page": 1},
            timeout=60,
        ).json()
        gh_sha = commits[0]["sha"] if isinstance(commits, list) and commits else None
    except Exception as exc:  # network-restricted environments
        print(f"[warn] GitHub commit lookup failed: {exc}", file=sys.stderr)

    lock = {
        "pinned_at_utc": now,
        "proteingym2": {
            "hf_repo": repo,
            "revision_sha": hf_sha,
            "last_modified": hf_mod,
            "n_pgdata_files": len(files),
            "total_bytes": sum(f["size"] for f in files),
            "file_manifest": manifest,
        },
        "reference_v1": {
            "url": cfg["sources"]["reference_v1_url"],
            "github_commit": gh_sha,
            "local_sha256": sha256_file(ref_csv) if ref_csv.exists() else None,
        },
    }
    (ref_dir / "sources_lock.json").write_text(json.dumps(lock, indent=1))

    md = f"""# Step 1 data sources (pinned {now})

## Primary — ProteinGym 2.0
- HuggingFace dataset: `{repo}`
- Revision SHA: `{hf_sha}`
- Upstream last modified: {hf_mod}
- Files: {len(files)} `.pgdata` bundles, {sum(f["size"] for f in files) / 1e9:.2f} GB total

Each `.pgdata` is a zip containing `dataset.pgdata` (a nested zip) and
`slices.json`. The nested zip holds:

| member | content |
|---|---|
| `manifest.lock` | TOML provenance: phenotype, readout, library method, replicate count, `has_uncertainty`, UniProt/Pfam ids, organism, publication DOI, structure pLDDT |
| `assays/<id>.csv` | per-variant table: `sequence`, `DMS Score`, `mutant` |
| `sequences/<id>.fasta` | wild-type reference sequence |
| `structures/<id>.pdb` | experimental or AlphaFold structure |
| `msas/<id>.a3m` | multiple sequence alignment (not used in Step 1) |

`slices.json` carries the published cross-validation scheme as boolean masks
aligned to CSV row order (`kfold`: 5 folds).

**Why this source.** The brief requires data already quality-controlled and
split by a published benchmark, not re-derived from sequencing counts.
ProteinGym 2.0 satisfies both and additionally ships structures and provenance
in the same bundle, which the annotation work in later steps depends on.

## Selection metadata — ProteinGym v1 reference file
- URL: `{cfg["sources"]["reference_v1_url"]}`
- GitHub commit: `{gh_sha}`
- Local sha256: `{lock["reference_v1"]["local_sha256"]}`

Used for assay-level metadata absent from the 2.0 bundles:
`coarse_selection_type`, `DMS_number_multiple_mutants`, `raw_DMS_directionality`,
`seq_len`, `region_mutated`, MSA depth statistics. All 211 `.pgdata` filenames
join to this table on `DMS_id`.

## Considered and not used
- **FLIP** — GB1 and AAV tasks overlap ProteinGym assays already selected here;
  meltome is a cross-protein Tm regression, not a single-protein DMS, so it does
  not fit the residue-level activation-caching design. GB1 enters via
  `SPG1_STRSG_*` with the ProteinGym split.
- **Schmirler, Heinzinger & Rost 2024 (Nat Commun 15:7407)** — the fine-tuning
  study cited in the brief. Its DMS tasks are drawn from ProteinGym/FLIP, so
  reusing ProteinGym directly avoids a second, redundant provenance chain.
- **MaveDB** — held in reserve to fill a diversity gap; not needed if the
  ProteinGym selection already spans the required phenotype/family space.

## Reproduction
```bash
python Code/scripts/pin_sources.py     # refresh this file
python Code/scripts/fetch_assays.py    # download selected bundles, verify sha256
```
"""
    (ref_dir / "SOURCES.md").write_text(md)

    cfg_path = REPO_ROOT / "Code" / "config" / "step1.yaml"
    text = cfg_path.read_text()
    text = text.replace(
        "  proteingym2_revision: null        # filled by pin_sources.py",
        f'  proteingym2_revision: "{hf_sha}"',
    )
    if gh_sha:
        text = text.replace(
            "  reference_v1_commit: null         # filled by pin_sources.py",
            f'  reference_v1_commit: "{gh_sha}"',
        )
    cfg_path.write_text(text)
    print(json.dumps({"hf_sha": hf_sha, "gh_sha": gh_sha, "n_files": len(files)}, indent=1))


if __name__ == "__main__":
    main()
