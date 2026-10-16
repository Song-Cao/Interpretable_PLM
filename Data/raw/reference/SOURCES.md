# Step 1 data sources (pinned 2026-08-27T17:30:53+00:00)

## Primary — ProteinGym 2.0
- HuggingFace dataset: `ProteinGym/ProteinGym2.0`
- Revision SHA: `2e5c7b10c3dda55ca7295c5f1334887801294131`
- Upstream last modified: 2026-08-25T18:49:13.000Z
- Files: 211 `.pgdata` bundles, 4.85 GB total

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
- URL: `https://raw.githubusercontent.com/OATML-Markslab/ProteinGym/main/reference_files/DMS_substitutions.csv`
- GitHub commit: `62eab5b3c83a3727b5b57bfbe377b24ef7545ccc`
- Local sha256: `a8f498011532a74aa9fe556a50555a75e928c5837d19c06a87592ae04049b308`

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
