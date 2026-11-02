# IRED design application log

This log is separate from `PROJECT_LOG.md`; nothing recorded here changes CROSSFIT.

- [DECISION] The application reuses frozen architecture and hyperparameter selections but retrains generation models on all 1,182 curated non-WT variants. No application result is a CROSSFIT gate, table, or claim.
- [DECISION] Complete stage manifests are written only after expected artifacts validate. Interrupted shards resume only when their checkpoint and row-count contract both pass; otherwise that shard restarts from its beginning.
- [CAVEAT] Models predict campaign fitness, a selection readout. On external plate kcat, classical encodings transferred at Spearman 0.53-0.57 while ESM arms transferred at 0.23-0.44; classical arms therefore have stronger evidence when catalytic rate is the goal.
