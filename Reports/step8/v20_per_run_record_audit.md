# IRED per-run record audit

Date: 2026-11-19

The four records named in the v20 request are auditable from committed files:

- Taxonomy proportions: five seed-level rows per class and width are in
  `Reports/step6/v17_pooled_null/class_proportions_by_seed.csv`.
- Norm-restricted rotation: five seed-level values per width are in
  `Reports/step6/v17_pooled_null/rotation_above_median_by_seed.csv`.
- Shell distribution: class-by-shell counts for each of five seed indices are in
  `v18_top_position_shell_counts.csv`; the index-to-seed ordering is committed in
  `v18_semantic_manifest.json`.
- Enrichment effects: each row in `v18_class_semantic_tests.csv` contains the five named
  seed-level effects from which the mean and interval were formed.

The wider audit is in `v20_per_run_record_audit.csv`. One gap remains: the grouped
mutated-position effects underlying `v18_mutation_responsive_tests.csv` were reduced before
being written. The exact matrix is recoverable from the registered delta-z cache without model
inference, but that cache is not present locally, so it was not backfilled in this CPU-only
session.
