# IRED design application log

This log is separate from `PROJECT_LOG.md`; nothing recorded here changes CROSSFIT.

- [DECISION] The application reuses frozen architecture and hyperparameter selections but retrains generation models on all 1,182 curated non-WT variants. No application result is a CROSSFIT gate, table, or claim.
- [DECISION] Complete stage manifests are written only after expected artifacts validate. Interrupted shards resume only when their checkpoint and row-count contract both pass; otherwise that shard restarts from its beginning.
- [CAVEAT] Models predict campaign fitness, a selection readout. On external plate kcat, classical encodings transferred at Spearman 0.53-0.57 while ESM arms transferred at 0.23-0.44; classical arms therefore have stronger evidence when catalytic rate is the goal.
- [DECISION] The representative embedding benchmark put the unreduced 500-winner/50-survivor search above the six GPU-hour ceiling. The registered budget rule was applied before search: 300 winner mutations and 30 survivors per round, with four rounds unchanged.
- [ADJUST] The first search launch stopped before producing any prediction artifact because the application loader treated the Step 4 JSON manifest as a YAML project config. The loader now reads JSON directly, and the search restarted from the beginning.
- [DECISION] ThermoMPNN is the stability oracle. Its deposited-chain sequence matches the 290-residue campaign wild type exactly after accounting for the oracle's zero-based position column. Candidate multi-mutant values are explicitly additive compositions of single-substitution estimates, not direct epistatic predictions.
