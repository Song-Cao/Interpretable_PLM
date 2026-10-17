# CROSSFIT Step 2 — Local CPU Handoff

**Scope.** This handoff covers the local composite objective, assay-balanced sampler,
representation module, rank aggregation, Mondrian conformal calibration, cross-assay gradient
accumulation contract, real-data diagnostics, and tests. No predictor model, embedding
extraction, hyperparameter search, or bake-off was run.

**Repo root.** `/Users/songcao/workdir/Interpretable_PLM`

**Environment.** Python 3.11; local CPU PyTorch 2.2.2; NumPy 1.26.4.

**Test status.** `python3 -m pytest Code/tests -q` -> **191 passed in 83.82 s** on 2026-10-17.

## 1. Files built

| Path | Contents |
|---|---|
| `Code/config/step2.yaml` | Pair thresholds, loss weights and sweeps, residual-scale settings, sampler settings, representation metadata, optimizer contract |
| `Code/config/georgiev_2009_standardized.csv` | Frozen standardized 20 amino acid x 19 descriptor table |
| `Code/crossfit/losses.py` | Pair construction, censoring rules, learned per-assay residual scales, one composite objective |
| `Code/crossfit/sampling.py` | Capped square-root allocation, without-replacement epoch draws, single-assay batches, realized-share records |
| `Code/crossfit/representations.py` | Mutation indicator, positional Georgiev/AAIndex, and adapters for precomputed ESM activations |
| `Code/crossfit/evaluation.py` | Anchored graph-Laplacian rank aggregation and Mondrian conformal calibration |
| `Code/crossfit/optimization.py` | One optimizer step after cross-assay gradient accumulation |
| `Code/crossfit/pair_analysis.py` | Mutation-space Hamming distances and gated-pair audit logic |
| `Code/scripts/analyze_pair_distances.py` | Reproducible real-data pair-distance audit entry point |
| `Reports/step2/pair_distance_audit.csv` | Per-assay Hamming distributions and counts with/without the constraint |
| `Code/tests/test_losses.py` | Loss and pair-construction tests, including sigma-weighting safeguards |
| `Code/tests/test_sampling.py` | 5 allocation and epoch-sampling tests |
| `Code/tests/test_representations.py` | 11 representation tests |
| `Code/tests/test_evaluation.py` | Rank recovery and per-stratum conformal coverage tests |
| `Code/tests/test_optimization.py` | Cross-assay accumulation and rejection tests |
| `Code/tests/test_pair_analysis.py` | Mutation Hamming and audit-gate tests |

Final artifact hashes:

```text
step2.yaml                  78de9193e49658c4834856db7dcf86c85db5297e62a9412e45d454c61777ac01
pair_distance_audit.csv     8d2fc377708a228fc94f0d3759d5cd7bf5717520491bd998dd33a387dc469604
georgiev_2009_standardized  ef8ba3e53df6ea0ac770cb86978fb68d679c28a1eff8b1ec3463e0fa17b161b4
```

## 2. Composite objective

### 2.1 Pair interface

```python
from crossfit.losses import construct_pairs

pairs = construct_pairs(
    targets, assay_index, mutant_sequences, censored, sigma_v,
    margin_sd=cfg["pairs"]["margin_sd"],
    max_hamming_distance=cfg["pairs"]["max_hamming_distance"],
)
```

`PairBatch` fields are aligned one-dimensional tensors:

| Field | Meaning |
|---|---|
| `u`, `v` | Variant positions in the input batch |
| `assay_index` | Integer assay id for each pair; cross-assay pairs never enter |
| `target_difference` | `y_u - y_v` |
| `order_sign` | `sign(y_u - y_v)` |
| `censor_sign` | `+1` upper-bound, `-1` lower-bound, `0` unconstrained |
| `sigma_u`, `sigma_v` | Curated per-variant uncertainty; NaN means unavailable |
| `censor_u`, `censor_v` | Encoded `none`, `floor`, or `ceiling` states |

The gate is strict: `abs(y_u - y_v) > margin_sd`. Pairs must also be from the same assay
and have full-sequence Hamming distance no greater than the configured delta. The constructor
returns an empty batch when nothing passes; the loss raises rather than silently ranking all
pairs.

### 2.2 Censoring rule

For difference models:

```text
L_cens = -log Phi(s * (Delta_y - d_hat) / sigma_assay)
```

`s=+1` for `u` floor or `v` ceiling; `s=-1` for `u` ceiling or `v` floor. Same-bound
double-censored pairs are dropped. Opposite-bound pairs are kept with the reinforced sign.
All four single-censored and both double-censored cases have regression tests.

For score-head models, `CompositeFitnessLoss(prediction_mode="score")` applies standard
Gaussian/Tobit terms to the unique variants participating in the gated pairs: Gaussian density
for uncensored, left-tail probability for floor, and right-tail probability for ceiling.

`LearnedResidualScale` owns one positive parameter per assay, transformed by softplus plus the
configured floor. It is the model residual scale, not `sigma_v`. A gradient test verifies it is
learned through the censoring term.

### 2.3 One objective

`CompositeFitnessLoss` returns one differentiable `total`:

```text
lambda_rank * sum softplus(-sign(Delta_y) * d_hat / T)
+ lambda_mag * magnitude
+ lambda_cens * censoring NLL
+ lambda_reg * readout L1
```

Magnitude is either Huber or squared error. Its default weight is exactly one for every pair;
`sigma_v` is not used. The retained uncertainty-weighting path is guarded by
`loss.magnitude_uncertainty_weighting.enabled: false`. If explicitly enabled, it requires the
configured minimum assay-level coverage in every participating assay, uses inverse-pair-variance
weights where both pair members have finite positive sigma, leaves other raw pair weights at
one, and normalizes all pair weights to mean one within each assay. Optimizer weight decay is
separate and recorded in `step2.yaml`.

Coverage re-read over all curated rows including the layer-freeze reserve:

| Assay | Non-NaN / total | Rate |
|---|---:|---:|
| CcdB | 0 / 1,176 | 0% |
| EstA | 0 / 2,172 | 0% |
| FecA | 1,886 / 1,886 | 100% |
| Gcn4 | 0 / 2,638 | 0% |
| GFP | 0 / 51,714 | 0% |
| DAAO | 0 / 6,396 | 0% |
| KRAS | 0 / 26,012 | 0% |
| Ubiquitin | 0 / 1,380 | 0% |
| GB1 | 0 / 536,962 | 0% |
| TPMT | 0 / 3,648 | 0% |

Only FecA has values: 1,818 `mv_trend` and 68 `replicate_sd`. Corpus coverage is
1,886/633,984 (0.2975%); 632,098 values are NaN.

`CompositeLossOutput` also returns the four unweighted component tensors and pair/constraint
counts for logging. These diagnostics do not define separate training objectives.

For `prediction_mode="difference"`, pass one predicted difference per pair. For
`prediction_mode="score"`, pass one score per variant; the module forms `score_u-score_v`.
Step 3 can therefore use the fixed `Linear(d,1)` per-assay heads without another loss.

## 3. Assay-balance sampler

`CappedSqrtSampler` accepts a mapping from assay id to eligible training-row indices. It must be
fed loader-default data so the layer-freeze reserve and active test indices remain absent.

For a requested epoch budget it allocates counts proportional to `sqrt(n_assay)`, iteratively
removing assays that hit `min(n_assay, cap)`, then uses deterministic largest-remainder integer
allocation. Rows are redrawn without replacement from each assay for every epoch. Every emitted
`AssayBatch` contains one assay. `EpochSample.records()` has schema:

```text
epoch: int, assay_id: str, n_sampled: int, share: float, per_assay_cap: int
```

At the configured default, `epoch_size: capped_pool`, every available capped quota is consumed.
Numbers re-read from the default loader:

| Assay | Usable | Sampled | Share |
|---|---:|---:|---:|
| CcdB | 1,117 | 1,117 | 0.020623 |
| EstA | 2,063 | 2,063 | 0.038089 |
| FecA | 1,792 | 1,792 | 0.033085 |
| Gcn4 | 2,506 | 2,506 | 0.046268 |
| GFP | 49,128 | 11,944 | 0.220520 |
| DAAO | 6,076 | 6,076 | 0.112180 |
| KRAS | 24,711 | 11,944 | 0.220520 |
| Ubiquitin | 1,311 | 1,311 | 0.024205 |
| GB1 | 510,114 | 11,944 | 0.220520 |
| TPMT | 3,466 | 3,466 | 0.063992 |

Median usable size is 2,986; the `4x` cap is 11,944; the capped epoch contains 54,163 variants.
For an epoch budget below 54,163, the realized counts follow the square-root weights until a cap
binds. At the full capped pool, all capacities are necessarily exhausted, so square-root
weighting does not shape the realized default epoch: the counts are purely `min(n_assay, cap)`.
CcdB exposes every one of its 1,117 usable rows each epoch; GB1 exposes 11,944/510,114 =
2.3414%. A CcdB row therefore receives about 42.71 times the per-epoch exposure of a GB1 row.
The epoch was not reduced merely to make square-root weighting active.

## 4. Representations

### 4.1 Georgiev descriptor provenance

The raw 19 descriptors are Georgiev's published scales: Georgiev, *Journal of Computational
Biology* 16(5), 2009, DOI `10.1089/cmb.2008.0173`, Table 3. The numerical transcription was
cross-checked against `andreas-wagner-uzh/fitness_landscape_sampling` commit
`d227fd63c763c6673ed19192b595797b9e6cf26a`.

Each column was centered and divided by its population SD (`ddof=0`) across the 20 canonical
amino acids. Frozen CSV sha256:
`ef8ba3e53df6ea0ac770cb86978fb68d679c28a1eff8b1ec3463e0fa17b161b4`.
Maximum absolute column mean is `5.002e-14`; maximum unit-variance error is `2.809e-13` after
CSV rounding.

### 4.2 Shapes and semantics

| Function | Return |
|---|---|
| `mutation_indicator` | CSR `N x p`, where `p` is the observed assay-specific `(position, mutant AA)` vocabulary |
| `aaindex_flat` | CSR `N x (L*19)` |
| `aaindex_positional` | Dense `N x L x 19`, an exact reshape of `aaindex_flat` |
| `esm_mean` | `d` or `N x d`, all-residue mean of supplied activations |
| `esm_mutpool` | `d` or `N x d`, mutated-position mean of supplied activations |
| `esm_residue` | Validated supplied `L x d` or `N x L x d` tensor |
| `pair_representation(..., "early")` | `rep(u) - rep(v)`; sparse inputs stay sparse |
| `pair_representation(..., "late")` | `(rep(u), rep(v))` control tuple |

AAIndex entries are `G(mutant) - G(wild_type)` only at mutated positions. Unmutated positions
are zero. Actual dimensions include GB1 8,512, GFP 4,522, and FecA 1,368, matching `L*19`.
Non-canonical amino acids raise; no imputation path exists.

The published rounded table has one exactly equal channel for 8 of 380 ordered substitutions:
`A<->V` channel 4, `M<->W` channel 1, `P<->W` channel 10, and `S<->W` channel 16. Those
substitutions have 18 mathematical nonzeros while retaining a 19-channel positional block. The
published values were not perturbed to make the plan's “19 nonzero” shorthand literally true.

The ESM functions consume already-computed arrays only. They do not load a PLM or create an
activation cache.

## 5. Rank aggregation and conformal calibration

`aggregate_pairwise_scores` constructs the oriented comparison incidence matrix and solves

```text
min_s sum_uv (s_u - s_v - d_hat_uv)^2  subject to s_wild_type = 0.
```

It returns the anchored scores, residual norm, and sparse least-squares iteration count. The
comparison graph must be connected; otherwise it raises because each disconnected component
has an unidentified additive constant. A synthetic cyclic graph test recovers known scores to
numerical precision with the wild-type score exactly zero.

`MondrianConformal(alpha)` calibrates absolute residual quantiles independently in exact strata
`(mutation_count, mutates_unseen_position)`. It uses finite-sample rank
`ceil((n + 1) * (1 - alpha))`; when that rank exceeds the calibration-stratum size, its interval
is infinite rather than anti-conservative. A prediction stratum absent from calibration raises;
there is no pooled fallback. Tests obtain approximately 90% evaluation coverage separately in
all four synthetic strata, not only marginally.

## 6. Cross-assay optimizer contract

`step_across_assays` receives scalar losses from distinct single-assay batches. It zeroes once,
backpropagates each loss divided by the number of participating assays, clips the accumulated
gradient if configured, and calls `optimizer.step()` exactly once. It requires at least
`optimizer_contract.min_assays_per_step: 2` and rejects duplicate assay ids within a step. A
closed-form SGD test verifies the update is the mean of two assay gradients, rather than two
alternating steps.

The GPU training loop must use this helper (or preserve its exact contract). Per-assay
validation curves are mandatory and pooled validation loss must not control early stopping, as
recorded in `step2.yaml`. The predictor readout L1 accepted by `CompositeFitnessLoss` applies
only to predictor readouts; it must never be passed into the Step 7 BatchTopK crosscoder.

## 7. Hamming-constraint audit

Reproduce with:

```bash
.venv/bin/python Code/scripts/analyze_pair_distances.py
```

The audit uses five deterministic epochs from the configured default sampler. It counts the
actual within-batch training pairs after same-bound double-censor removal and the strict
`abs(Delta_y) > 0.25` gate, first without a distance filter and then with `delta <= 2`.

| Assay | Without constraint | With constraint | Removed |
|---|---:|---:|---:|
| CcdB | 145,428 | 145,428 | 0.0000% |
| EstA | 276,852 | 276,852 | 0.0000% |
| FecA | 243,582 | 120,657 | 50.4656% |
| Gcn4 | 330,844 | 82 | 99.9752% |
| GFP | 1,215,991 | 677 | 99.9443% |
| DAAO | 810,310 | 810,310 | 0.0000% |
| KRAS | 1,621,540 | 70,234 | 95.6687% |
| Ubiquitin | 126,582 | 126,582 | 0.0000% |
| GB1 | 1,522,402 | 5,646 | 99.6291% |
| TPMT | 457,756 | 457,756 | 0.0000% |

The complete per-assay distributions are in `Reports/step2/pair_distance_audit.csv`; Hamming
distances extend to 4 for FecA/GB1/KRAS, 24 for GFP, and 44 for Gcn4. This audit is over the
realized training stream, not the all-pairs Cartesian product across 602,284 rows. Since five
assays exceed the 1% removal threshold, `max_hamming_distance: 2` remains in config under the
user's ruling. Its effect is large enough that GPU work must not start until these numbers are
reviewed.

## 8. Tests and integrity

New tests cover:

- strict margin gating, assay and Hamming strata;
- all four single-censor sign cases and both double-censor cases;
- allowed versus forbidden censoring directions;
- learned residual-scale gradients;
- standard score-head Tobit likelihood;
- exact weighted composite summation and readout L1;
- square-root allocation, cap enforcement, no oversampling, no row duplication, epoch redraw,
  deterministic replay, and single-assay batches;
- frozen-table normalization, sparse positional encoding, exact conv reshape, mutation
  indicators, non-canonical rejection, precomputed ESM pooling, and early/late fusion.
- default-off sigma weighting, coverage enforcement, and within-assay weight normalization;
- synthetic anchored rank recovery and disconnected-graph rejection;
- per-stratum Mondrian conformal coverage and absent-stratum rejection;
- one-step cross-assay gradient averaging and invalid accumulation rejection;
- mutation-space Hamming calculation and pair-audit censor/margin gates.

Full suite: **191 passed in 83.82 s**, comprising the original 147 plus 44 Step 2 tests.

## 9. Incomplete, uncertain, or intentionally out of scope

1. **No GPU work was performed.** ESM-C embeddings and per-residue activation tensors do not
   exist yet.
2. **No predictor architectures or bake-off were built.** They require GPU representations.
3. **No hyperparameter search or result grid exists.** The config records default values and
   sweep candidates; none has been selected from test performance.
4. **No tuned baseline artifacts, architecture shortlist, uncertainty estimates, or answer to
   the all-residue versus mutated-position pooling question exists.** Those require the bake-off.
5. **The Hamming constraint changes the training distribution substantially.** It remains
   configured under the explicit >1% rule, but GPU work is blocked pending review of Section 7.
6. **Full-pool epochs exhaust all assay capacities.** Square-root weighting affects count
   allocation only when a smaller explicit epoch budget is requested. The full-pool default is
   what yields the planned approximately 22% GB1 share.

## 10. Remaining GPU portion of Step 2

The next GPU allocation must generate ESM-C residue activations/pooled embeddings, build the
authorized predictor architectures, run validation-only hyperparameter search for every required
representation/architecture pairing and objective arm, evaluate all metrics over the ten low-N
seeds, save tuned baseline artifacts, report per-assay mean and SD, and settle the pooling-domain
comparison. It must use the sampler records above and must not read the layer-freeze reserve.
It must also accumulate across distinct assays before every optimizer step, emit per-assay
validation curves, and never early-stop from pooled validation loss. **Do not start this GPU
portion until the retained Hamming constraint and Section 7 pair losses have been reviewed.**
