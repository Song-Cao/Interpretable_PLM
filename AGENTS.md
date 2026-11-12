# CROSSFIT repository rules

`CROSSFIT_Implementation_Plan_v19.docx` is the only implementation plan. Earlier plans and handoffs are superseded.

- Run one independent pipeline per dataset and report every result per dataset. Never pool fitness values, training data, or metrics across datasets.
- Use `pg_kfold` as the only model-selection split. Low-N, unseen-position, mutation-count, round-held-out, and external data are evaluation conditions only.
- Never train, tune, calibrate, or select on a test split, the layer-freeze reserve, or an external validation set. The crosscoder activation pool comes only from the dataset's training split.
- Use direct per-variant fitness scores. Form ranking pairs within a batch from those scores; do not construct pair datasets or aggregate pairwise differences.
- Use BatchTopK in the crosscoder. No L1 penalty is allowed anywhere in the crosscoder objective; predictor regularization is separate.
- Normalize activations before crosscoding with one fixed scalar per dataset, model, and layer so that the training-pool mean norm is `sqrt(d)`. Freeze that scalar and report the raw adapted/base gain ratio.
- Interpret crosscoder latents per residue. Pooling is allowed only in downstream predictors.
- Fit every threshold from its own shuffled-label null for the same dataset and backbone.
- Pool eligible drift-matched null samples across runs into one null per dataset, width,
  and statistic. Treat leave-one-seed-out threshold variation as uncertainty; never make
  significance stricter merely by adding null seeds.
- Use the backbone final output for embedding baselines. Use the frozen selected depth for
  dense probes, crosscoder inputs, sparse features, semantic tests, and patching. Name the
  depth beside every reported number.
- Give every activation-derived cache a provenance manifest and canonical wild-type reference
  block. Verify the reference with a fresh forward pass before reading the cache downstream.
- Quarantine tensors, checkpoints, and intermediates in the same action that reverses the
  decision that produced them. Active loaders must never resolve archived artifacts.
- Compare numerical tolerances in the units of the quantity being judged.
- Answer strange observations with more replicates and a reported spread, never with a
  dataset-specific rule change.
- Compare baselines with the same backbone, data, and splits. Report per-dataset results; never pool them.
- Treat a failed gate as a result. Do not change an architecture, statistic, threshold, or grid to make a gate pass.
- Keep all thresholds, seeds, paths, and hyperparameters in the step YAML. Put library logic in `Code/crossfit/`, entry points in `Code/scripts/`, tests in `Code/tests/`, and result artifacts in `Reports/stepN/`.
- Append decisions, adjustments, findings, and GPU cost to `PROJECT_LOG.md` as they occur. Preserve inconvenient or negative results in each step handoff.
- Respect the compute budget: random-search 30-60 configurations per Step 2 arm, tune architectures once on IRED, and report GPU-hours against each step estimate.

## Separate application workstreams

`Code/applications/ired_design/`, `Code/config/ired_design.yaml`,
`Reports/applications/ired_design/`, and `Models/applications/ired_design/` form an
isolated application workstream. Its outputs are design proposals, not CROSSFIT
evidence, and must never enter a CROSSFIT gate, result table, or claim.
