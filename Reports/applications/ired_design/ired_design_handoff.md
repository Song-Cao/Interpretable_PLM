# IRED design application handoff

Status: complete. This is an application workstream and is not CROSSFIT evidence.

## Completed

- Trained and hashed 60 non-adapter model members plus five adapter
  members on all 1,182 curated non-WT variants.
- Completed 13 model searches, four founders, and four rounds with full
  parent-linked lineage persistence.
- Produced the controls and trajectory-overlap audits before biophysical work.
- Produced 30 consensus candidates, top 20 per model, five-member spread, cross-model
  rank disagreement, trajectory graphs, and the validated ThermoMPNN check.

## Budget

Reduced-search projection: 2.030 GPU-hours. Search:
3.152. Accounted total: 5.719 GPU-hours, below the
six-hour ceiling.

## Limitations

- Rankings target campaign fitness, not kcat; external transfer favours classical
  encodings.
- Existing Mondrian strata do not provide finite intervals for these 6-8-mutation
  designs.
- 29 of 30 consensus
  proposals contain A72T or G37C, the registered known dead ends; the sole remaining
  proposal still has only 2/13 support from the equivalent unaugmented classical pair.
- ThermoMPNN multi-mutant values are additive single-mutation checks. Geometry,
  SoluProt, FoldX, and docking were excluded rather than reported without validation.
- Complete parquet lineages and all application checkpoints remain pod-only. Their
  committed manifests contain hashes; no main-project model or result directory was
  modified.

See `ired_design_report.md` for the result tables and artifact map.
