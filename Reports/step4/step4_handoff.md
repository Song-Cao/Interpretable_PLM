# Step 4 handoff — IRED activation cache

## Result

The approved cache completed for the 895 variants in the pg_kfold fold-0 training pool, at all
290 residue positions and frozen analysis depth 26. It contains 259,550 aligned position rows
per model. Test, layer-freeze reserve, round-held-out test, and plate variants are absent.

The tensor payload is 996,672,000 bytes (0.996672 decimal GB); the realized NPY files occupy
996,675,584 bytes including headers. Extraction consumed 0.02682 A100 wall-hours.

## Interface

The cache remains pod-only at `Cache/step4/IRED/`:

- `variant_rows.npy`: 895 int64 curated row identifiers in extraction order.
- `evo/shard_0000.npy` through `shard_0013.npy`: normalized base activations.
- `fit/shard_0000.npy` through `shard_0013.npy`: normalized adapted activations.
- `manifest.json`: shapes, scalars, hashes, exclusions, checkpoint hash, and cost.

Each shard is fp16 with shape `[variants, 290, 960]`; the first 13 contain 64 variants and the
last contains 63. Shard names and shapes are identical between models. Load only through
`crossfit.activation_cache.iter_cached_pairs` or `load_cached_pair`, which require a complete
manifest and assert alignment. The compact committed copy is
`Reports/step4/cache_manifest.json`.

One scalar per model was fitted on this training pool and then frozen:

| model | raw mean norm | scalar | realized normalized mean norm |
|---|---:|---:|---:|
| E_evo | 1137.893354 | 0.0272291482 | 30.983911 |
| E_fit | 1044.109659 | 0.0296749163 | 30.984206 |

The target is `sqrt(960) = 30.983867`; the small discrepancy is fp16 rounding. The raw adapted
to base mean-norm ratio is 0.917581. Token alignment was exact, and the real adapter checkpoint
hash matched the Gate 1 verification record before extraction.

## Incomplete or uncertain

Nothing in the approved base/real training-pool cache is incomplete. A training-pool cache for
the shuffled adapter was not approved or extracted, so the Step 6 shuffled-label crosscoder and
tau calibration remain pending. No test or reserve activation was added to this store.
