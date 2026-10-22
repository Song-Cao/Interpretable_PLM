# CROSSFIT

CROSSFIT measures the feature-level structure of representation change when a protein language model is adapted from evolutionary sequence modelling to measured fitness. Each dataset has an independent pipeline. A shared BatchTopK dictionary within that pipeline identifies evolutionary features the fitness model conserves, re-weights, rotates, abandons, or creates. Gate 1 drift is a precondition for this measurement, not the result itself.

`CROSSFIT_Implementation_Plan_v12.docx` is the only implementation plan. IRED runs first, followed by GB1, GFP, KRAS, and DAAO. Models predict per-variant fitness directly; ranking comparisons exist only transiently inside a training batch. Step 2 searches the registered arms by random search with five-fold validation inside the pg_kfold training pool; the test fold is reporting-only. IRED Gate 1 is being rerun under the pre-committed validation-Spearman stopping rule; its preliminary fixed-epoch run is archived, not reported as final.

## Reproduce curation

Python 3.10 or newer is required. ProteinGym source revisions are pinned in `Data/raw/reference/sources_lock.json`; IRED source tables live in `Data/processed/IRED_data/`.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/python Code/scripts/fetch_assays.py
.venv/bin/python Code/scripts/build_dataset.py
.venv/bin/python Code/scripts/curate_ired.py
.venv/bin/python -m pytest Code/tests -q
```

The default loader excludes every dataset's layer-freeze reserve. Use `crossfit.datasets.load_assay`, `get_split`, and `load_plate_dataset`; do not read split tables directly in training code. `pg_kfold` is the only selection protocol. IRED's round-held-out and plate sets are evaluation-only, and the plate set exposes only `aa_seq`, `mut`, and `1/s`.

## Directory map

```text
Code/crossfit/       Library modules: curation, direct-score loss, encodings, models, evaluation
Code/scripts/        Reproducible entry points
Code/tests/          Unit, leakage, and design-invariant tests
Code/config/         Step YAML and frozen Georgiev descriptor table
Data/raw/            Pinned ProteinGym sources (large bundles gitignored)
Data/processed/      Curated tables, manifests, QC, and split metadata
Reports/step1/       ProteinGym and IRED curation reports
Reports/step2/       Step 2 reports; pre-v10 fits are isolated under Reports/archive/
Cache/               Activation and dPLL caches (gitignored)
Models/              Tuned artifacts and checkpoints (gitignored)
PROJECT_LOG.md       Decisions, adjustments, findings, and compute costs
```

Large or regenerable tables, caches, and checkpoints remain outside git; compact provenance, splits, tests, and reports are versioned.

## Reproduce the completed IRED run

GPU work runs from the repository root on the pod. It requires the pinned ESM-C snapshot
and leaves all tensors and checkpoints under the gitignored `Cache/` and `Models/` trees.

```bash
.venv/bin/python Code/scripts/extract_ired_esmc.py --device cuda
.venv/bin/python Code/scripts/run_classical_step2.py --representation onehot --augmentation unaugmented
.venv/bin/python Code/scripts/run_classical_step2.py --representation onehot --augmentation augmented
.venv/bin/python Code/scripts/run_classical_step2.py --representation aaindex_flat --augmentation unaugmented
.venv/bin/python Code/scripts/run_classical_step2.py --representation aaindex_flat --augmentation augmented
.venv/bin/python Code/scripts/run_neural_step2.py --arm esm_mean_ridge --device cuda
.venv/bin/python Code/scripts/run_neural_step2.py --arm esm_mean_mlp --device cuda
.venv/bin/python Code/scripts/run_neural_step2.py --arm esm_mean_random_forest --device cpu
.venv/bin/python Code/scripts/run_neural_step2.py --arm esm_mutpool_ridge --device cuda
.venv/bin/python Code/scripts/summarize_step2.py
.venv/bin/python Code/scripts/run_ired_gate1.py --device cuda
```

Committed results are summarized in `Reports/step2/step2_handoff.md` and
`Reports/step3/step3_handoff.md`. The convolutional Step 2 arms were not launched after the
required ten-fit timing audit projected 5.74 GPU-hours per arm even at the minimum search
count; this budget-limited incompleteness is part of the result.
