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

The wider audit is in `v20_per_run_record_audit.csv`. The previously missing grouped
mutated-position effects underlying `v18_mutation_responsive_tests.csv` have now been recovered
from the registered delta-z cache and committed in `v20_mutation_responsive_group_effects.csv`.
This was arithmetic only; no model was refitted or re-extracted. All reported Step 8 means,
ranges, and intervals now have committed underlying observations or named per-seed values.
