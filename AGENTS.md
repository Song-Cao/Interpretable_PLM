# CROSSFIT repository rules

`CROSSFIT_Implementation_Plan_v10.docx` is the only implementation plan. Earlier plans and handoffs are superseded.

- Run one independent pipeline per dataset and report every result per dataset. Never pool fitness values, training data, or metrics across datasets.
- Use `pg_kfold` as the only model-selection split. Low-N, unseen-position, mutation-count, round-held-out, and external data are evaluation conditions only.
- Never train, tune, calibrate, or select on a test split, the layer-freeze reserve, or an external validation set. The crosscoder activation pool comes only from the dataset's training split.
- Use direct per-variant fitness scores. Form ranking pairs within a batch from those scores; do not construct pair datasets or aggregate pairwise differences.
- Use BatchTopK in the crosscoder. No L1 penalty is allowed anywhere in the crosscoder objective; predictor regularization is separate.
- Normalize activations before crosscoding with one fixed scalar per dataset, model, and layer so that the training-pool mean norm is `sqrt(d)`. Freeze that scalar and report the raw adapted/base gain ratio.
- Interpret crosscoder latents per residue. Pooling is allowed only in downstream predictors.
- Fit every threshold from its own shuffled-label null for the same dataset and backbone.
- Compare baselines with the same backbone, data, and splits. Report per-dataset results; never pool them.
- Treat a failed gate as a result. Do not change an architecture, statistic, threshold, or grid to make a gate pass.
- Keep all thresholds, seeds, paths, and hyperparameters in the step YAML. Put library logic in `Code/crossfit/`, entry points in `Code/scripts/`, tests in `Code/tests/`, and result artifacts in `Reports/stepN/`.
- Append decisions, adjustments, findings, and GPU cost to `PROJECT_LOG.md` as they occur. Preserve inconvenient or negative results in each step handoff.
