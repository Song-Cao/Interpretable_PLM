# Step 7 handoff — IRED dictionary-width diagnostic

## Result

All four registered width multipliers completed for three seeds with fixed `k=32`, bias-free
decoders, 20 epochs, and no L1 or fitness supervision. The table is mean ± sample SD across
seeds; participation ratio is `trace(cov(z))^2 / trace(cov(z)^2)` over the full activation pool.

| w | features | samples/feature | dead fraction | effective rank | E_evo FVE | E_fit FVE | realized L0 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 960 | 270.36 | 0.3267 ± 0.0126 | 122.83 ± 1.67 | 0.99850 ± 0.00005 | 0.99591 ± 0.00010 | 32.000 ± 0.000 |
| 2 | 1,920 | 135.18 | 0.4993 ± 0.0063 | 112.78 ± 7.64 | 0.99858 ± 0.00014 | 0.99679 ± 0.00015 | 32.000 ± 0.000 |
| 4 | 3,840 | 67.59 | 0.6623 ± 0.0018 | 99.12 ± 4.29 | 0.99866 ± 0.00009 | 0.99715 ± 0.00010 | 31.999 ± 0.002 |
| 8 | 7,680 | 33.80 | 0.7815 ± 0.0032 | 95.02 ± 9.30 | 0.99869 ± 0.00007 | 0.99730 ± 0.00008 | 31.914 ± 0.038 |

The full seed rows and checkpoint hashes are in `Reports/step7/width_sweep_runs.csv`; aggregate
values are in `width_sweep_summary.csv`, and the requested curve is
`dead_fraction_by_samples_per_feature.png`. The authoritative sweep consumed 0.10493 A100
wall-hours and reported nothing until all 12 fits and diagnostics completed.

## Width decision

No width is selected. `w=1` is not overcomplete and still leaves 32.7% of features dead. Every
overcomplete setting is mostly or nearly mostly dead: 49.9% at `w=2`, 66.2% at `w=4`, and
78.2% at `w=8`. Effective rank stays near 95–123 while nominal feature count grows eightfold.
High FVE therefore does not rescue the sparse-decomposition premise; it is achieved with a
small active subspace. Under v13's no-working-width rule, IRED cannot support an overcomplete
dictionary at this data volume. This is a consequence of 259,550 position samples, not a claim
that the crosscoder method generally fails.

## Checkpoint interface

The 12 complete checkpoints remain pod-only under `Models/step7/IRED/` with names
`w{1,2,4,8}_seed{seed}_nobias.pt`. Each contains its state, seed, width, fixed k, decoder-bias
flag, and full loss trace. The committed run table records every SHA-256 hash. An interrupted
run has no final checkpoint and restarts from its beginning; only a complete identity-matched
checkpoint can be resumed.

## Incomplete or uncertain

The taxonomy, latent-scaling filter, norm floor, classification precedence, phi predictors,
and Gate 2 comparison were not run. The shuffled-label tau required by taxonomy is also absent.
Proceeding with any of those would require choosing a mostly-dead or non-overcomplete
dictionary contrary to v13. Gate 2 remains pre-committed but unevaluated; the dense Step 5
reference and sparse predictor arms have not been fitted in this pass.
