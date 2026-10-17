# CROSSFIT Step 2 — Local CPU Handoff

**Scope.** This handoff covers only the four authorized local items: the composite fitness
objective, assay-balanced sampler, representation module, and tests. No predictor model,
embedding extraction, hyperparameter search, or bake-off was run.

**Repo root.** `/Users/songcao/workdir/Interpretable_PLM`

**Environment.** Python 3.11; local CPU PyTorch 2.2.2; NumPy 1.26.4.

**Test status.** `python3 -m pytest Code/tests -q` -> **181 passed in 70.94 s** on 2026-10-17.

## 1. Files built

| Path | Contents |
|---|---|
| `Code/config/step2.yaml` | Pair thresholds, loss weights and sweeps, residual-scale settings, sampler settings, representation metadata, optimizer contract |
| `Code/config/georgiev_2009_standardized.csv` | Frozen standardized 20 amino acid x 19 descriptor table |
| `Code/crossfit/losses.py` | Pair construction, censoring rules, learned per-assay residual scales, one composite objective |
| `Code/crossfit/sampling.py` | Capped square-root allocation, without-replacement epoch draws, single-assay batches, realized-share records |
| `Code/crossfit/representations.py` | Mutation indicator, positional Georgiev/AAIndex, and adapters for precomputed ESM activations |
| `Code/tests/test_losses.py` | 18 loss and pair-construction tests |
| `Code/tests/test_sampling.py` | 5 allocation and epoch-sampling tests |
| `Code/tests/test_representations.py` | 11 representation tests |

`step2.yaml` sha256:
`5fb4ec3408b705764c9c8e810a008a38845db107e3a17537252ac4312169cac0`.

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

Magnitude is either Huber or squared error. Where both variants have finite positive `sigma_v`,
the per-pair magnitude term is weighted by `1 / (sigma_u^2 + sigma_v^2)`; otherwise its weight
is one. Optimizer weight decay is separate and recorded in `step2.yaml`.

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
binds. At the full capped pool, all capacities are necessarily exhausted.

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

## 5. Tests

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

Full suite: **181 passed**, comprising the original 147 plus 34 Step 2 tests.

## 6. Incomplete, uncertain, or intentionally out of scope

1. **No GPU work was performed.** ESM-C embeddings and per-residue activation tensors do not
   exist yet.
2. **No predictor architectures or bake-off were built.** The authorized scope explicitly
   stopped after objective, sampler, representations, and tests.
3. **No hyperparameter search or result grid exists.** The config records default values and
   sweep candidates; none has been selected from test performance.
4. **No tuned baseline artifacts, architecture shortlist, uncertainty estimates, or answer to
   the all-residue versus mutated-position pooling question exists.** Those require the bake-off.
5. **Rank aggregation and Mondrian conformal calibration were not built.** They are part of the
   full Step 2 plan but were outside the four authorized local items; they remain CPU work, not
   GPU work.
6. **Full-pool epochs exhaust all assay capacities.** Square-root weighting affects count
   allocation only when a smaller explicit epoch budget is requested. The full-pool default is
   what yields the planned approximately 22% GB1 share.

## 7. Remaining GPU portion of Step 2

The next GPU allocation must generate ESM-C residue activations/pooled embeddings, build the
authorized predictor architectures, run validation-only hyperparameter search for every required
representation/architecture pairing and objective arm, evaluate all metrics over the ten low-N
seeds, save tuned baseline artifacts, report per-assay mean and SD, and settle the pooling-domain
comparison. It must use the sampler records above and must not read the layer-freeze reserve.
