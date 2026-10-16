# CROSSFIT

CROSSFIT measures how a protein language model's internal representation changes when its objective shifts from evolutionary sequence likelihood to measured engineered fitness. A shared BatchTopK crosscoder encodes aligned residue activations from a frozen base model and a LoRA-adapted copy, while separate decoders expose features that are emergent, evolution-only, rotated, re-weighted, or conserved. Representational change is the primary measurement; whether changed features improve few-shot fitness prediction is tested separately.

Step 1 curated 633,984 variants from ten ProteinGym 2.0 DMS assays. The default loader exposes 602,284 variants and excludes the 31,700-row layer-freeze reserve. Training code must use `crossfit.datasets` rather than reading split files directly, and must never train on the reserve or on an active split's test indices.

## Reproduce Step 1

Python 3.10 or newer is required. The source revisions are already fixed in `Data/raw/reference/sources_lock.json` and `Code/config/step1.yaml`.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python Code/scripts/fetch_assays.py
.venv/bin/python Code/scripts/build_dataset.py
.venv/bin/python -m pytest Code/tests -q
```

`fetch_assays.py` downloads and verifies the ten pinned `.pgdata` bundles (about 195 MB). `build_dataset.py` reconstructs the unified tables, QC artifacts, reservations, and split assignments. `pin_sources.py` deliberately refreshes upstream pins and is therefore not part of exact reproduction; `build_candidates.py` rebuilds the assay-selection audit when needed.

## Directory map

```text
Code/
  crossfit/          Library modules: loading, QC, labels, splits, selection
  scripts/           Reproducible command-line entry points
  tests/             Pytest suite
  config/step1.yaml  Step 1 paths, thresholds, and seeds
Data/
  raw/reference/     Pinned source metadata and reference table
  raw/proteingym2/   Downloaded assay bundles (gitignored)
  interim/           Candidate-selection audit table
  processed/         Curated manifests, QC, reports, and split metadata
Reports/step1/       Step 1 report, figures, rationale, and handoff
Cache/               Activation caches (gitignored)
Models/              Model checkpoints (gitignored)
PROJECT_LOG.md       Decisions, adjustments, findings, and compute costs
```

The large variant tables and large regenerable split CSVs are gitignored; small result artifacts and provenance files are versioned.
