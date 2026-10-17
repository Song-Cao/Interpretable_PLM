# CROSSFIT Step 2 — Local CPU Handoff

**Scope.** This handoff covers the local composite objective, assay-balanced sampler,
representation module, rank aggregation, Mondrian conformal calibration, cross-assay gradient
accumulation contract, real-data diagnostics, and tests. It also covers the GB1-only classical
additive baseline pilot and the preregistered stop it triggered. No GPU or ESM work was run.

**Repo root.** `/Users/songcao/workdir/Interpretable_PLM`

**Environment.** Python 3.11; local CPU PyTorch 2.2.2; NumPy 1.26.4.

**Test status.** `python3 -m pytest Code/tests -q` -> **199 passed in 79.80 s** on 2026-10-17.

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
| `Code/crossfit/models.py` | Bias-free additive model and composite-objective CPU fitting |
| `Code/crossfit/metrics.py` | Per-variant, pair, campaign, and Hamming-stratified metrics |
| `Code/scripts/analyze_pair_distances.py` | Reproducible real-data pair-distance audit entry point |
| `Code/scripts/run_additive_baselines.py` | Validation tuning, low-N evaluation, and artifact serialization |
| `Reports/step2/pair_distance_audit.csv` | Per-assay Hamming distributions; no distance is filtered |
| `Reports/step2/margin_gate_audit.csv` | Gated-pair counts at all three registered margins |
| `Reports/step2/gcn4_mutation_count_distribution.csv` | Curated and reserve-excluded Gcn4 mutation loads |
| `Reports/step2/additive_*.csv` | GB1-only tuning, run, summary, sanity, and distance-stratified results |
| `Reports/step2/additive_models/*.npz` | GB1 coefficients and residual scales for all N/seed fits |
| `Code/tests/test_losses.py` | Loss and pair-construction tests, including sigma-weighting safeguards |
| `Code/tests/test_sampling.py` | 5 allocation and epoch-sampling tests |
| `Code/tests/test_representations.py` | 11 representation tests |
| `Code/tests/test_evaluation.py` | Rank recovery and per-stratum conformal coverage tests |
| `Code/tests/test_optimization.py` | Cross-assay accumulation and rejection tests |
| `Code/tests/test_pair_analysis.py` | Mutation Hamming and audit-gate tests |

Selected final hashes:

```text
step2.yaml                       65e7065a5b4ef87ed3b7db90302082d65f0e4ffe8bd63a85b8012fe5380410ca
pair_distance_audit.csv          bbf0faa7b703613dde72066f60d20ccf47edcdfdd1ecb294fe33d7426250bb7e
margin_gate_audit.csv            c2031f58dc66490537e743af70428b2e4751008029567b29d2e949a03dfc85bc
gcn4_mutation_count_distribution c18496ace3caada6a0741dc8d14f2372ac2928a73c619ef3b84926befb67d4c8
additive_gb1_sanity.csv           b5aca177f237b557fae6d97990d4f0e84fb45b2e518105e027f10aa7900f6f76
georgiev_2009_standardized        ef8ba3e53df6ea0ac770cb86978fb68d679c28a1eff8b1ec3463e0fa17b161b4
```

## 2. Composite objective

### 2.1 Pair interface

```python
from crossfit.losses import construct_pairs

pairs = construct_pairs(
    targets, assay_index, censored, sigma_v,
    margin_sd=cfg["pairs"]["margin_sd"],
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

The gate is strict: `abs(y_u - y_v) > margin_sd`, and pairs must share an assay. There is no
Hamming argument, filtering branch, or Hamming config entry. The constructor returns an empty
batch when nothing passes; the loss raises rather than silently ranking all pairs.

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

## 7. Pair diagnostics and Hamming ruling

Reproduce with:

```bash
.venv/bin/python Code/scripts/analyze_pair_distances.py
```

The audit uses five deterministic epochs from the configured default sampler. It counts actual
within-batch training pairs after same-bound double-censor removal and the strict margin gate.
`pair_distance_audit.csv` retains the exact Hamming distribution as a diagnostic, but no
distance is excluded. Maximum observed distance is 4 for FecA/GB1/KRAS, 24 for GFP, and 44 for
Gcn4. No delta is safe and informative: delta 4 is inert for eight assays while retaining only
about 6.1% of GFP and 0.4% of Gcn4. The constraint and its config entry were therefore removed.
GPU bake-off metrics must be reported by exact pair Hamming distance.

The margin sweep was already present at `0.10, 0.25, 0.50`; it was not altered after inspection.
Realized gated-pair counts are:

| Assay | 0.10 | 0.25 | 0.50 |
|---|---:|---:|---:|
| CcdB | 156,954 | 145,428 | 116,729 |
| EstA | 304,405 | 276,852 | 233,196 |
| FecA | 266,675 | 243,582 | 206,186 |
| Gcn4 | 368,185 | 330,844 | 271,742 |
| GFP | 1,533,779 | 1,215,991 | 973,272 |
| DAAO | 897,551 | 810,310 | 671,931 |
| KRAS | 1,775,112 | 1,621,540 | 1,373,967 |
| Ubiquitin | 164,635 | 126,582 | 100,225 |
| GB1 | 1,676,117 | 1,522,402 | 1,299,483 |
| TPMT | 509,436 | 457,756 | 376,427 |

These are realized counts over five sampled epochs, not all possible corpus pairs. Exact rows
are in `margin_gate_audit.csv`.

Gcn4 is two mutation-load populations: 1,631/2,638 curated variants (61.83%) have 1-6
mutations, only 43 (1.63%) have 7-30, and 964 (36.54%) have 31-44. Reserve exclusion leaves the
same structure: 61.57%, 1.60%, and 36.83%. No filtering or correction was applied.

## 8. Classical additive baselines: GB1 stop

Both implemented arms are bias-free linear scores on early-subtracted sparse representations:

- one-hot additive on `mutation_indicator`;
- AAIndex additive on `aaindex_flat`.

Each uses the complete composite objective. Tuning is three-fold CV inside the N=512, seed=0
training subset, over all 27 combinations of three margins, three readout-L1 values, and three
AdamW weight decays. Early stopping uses validation composite loss; candidate selection uses
mean validation variant Spearman. The selected hyperparameters and epoch count are then fixed
across all six low-N sizes and ten seeds. Fold-0 test rows never enter tuning.

The preregistered GB1 sanity stop fired:

| Representation | N=16 | N=32 | N=64 | N=128 | N=256 | N=512 |
|---|---:|---:|---:|---:|---:|---:|
| One-hot | 0.0996±0.0160 | 0.1343±0.0149 | 0.1843±0.0165 | 0.2658±0.0195 | 0.3832±0.0204 | 0.5248±0.0185 |
| AAIndex | 0.2920±0.0759 | 0.4041±0.0650 | 0.5153±0.0446 | 0.6360±0.0299 | 0.7247±0.0152 | 0.7782±0.0083 |

Across the preregistered N=16,32,64 selection range, AAIndex exceeds one-hot by 0.26434 mean
Spearman, above the 0.05 bug threshold. No other assay was run.

The encoding itself passes the immediate rank check: at every one of GB1's 55 mutable
positions, the 19x19 one-hot-to-Georgiev contrast matrix has rank 19 and condition number
4.51-4.76. The likely protocol-level cause is low-N identifiability plus coordinate-dependent
regularization: N=16 observes only 31.4/1,045 one-hot features on average, and N=512 still only
654.9. AAIndex shares across residues at an observed position while unseen one-hot coefficients
stay at zero. This explains how minimum-norm/penalized fits can differ despite isomorphic full
feature spaces, but per the preregistration it is recorded as a bug signal, not a model win. No
model or threshold was changed in response.

## 9. Tests and integrity

New tests cover:

- strict margin gating and assay strata, with no distance-filter branch;
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
- exact early-subtraction linear scoring, composite-objective fitting, zero-pair failure, and
  deterministic CV folds;
- variant, pair, campaign, and exact-Hamming-stratified predictor metrics.

Full suite: **199 passed in 79.80 s**, comprising the original 147 plus 52 Step 2 tests.

## 10. Incomplete, uncertain, or intentionally out of scope

1. **No GPU work was performed.** ESM-C embeddings and per-residue activation tensors do not
   exist yet.
2. **The two additive arms are implemented, but the result suite is GB1-only.** The
   preregistered bug stop prevented the other nine assays from being tuned or evaluated.
3. **The full additive mean/SD grid is incomplete.** GB1 artifacts are saved and loadable; no
   result is reported for an assay that was not run.
4. **No ESM predictor architectures, uncertainty estimates, or pooling comparison exist.**
   Those require the GPU bake-off.
5. **The GB1 gap is unresolved.** Full-space rank and conditioning are sound, but low-N
   coordinate-dependent identifiability invalidates the expected empirical equivalence under
   the current fitting protocol. No unapproved correction was attempted.
6. **Full-pool epochs exhaust all assay capacities.** Square-root weighting affects count
   allocation only when a smaller explicit epoch budget is requested. The full-pool default is
   what yields the planned approximately 22% GB1 share.

## 11. Remaining GPU portion of Step 2

The next GPU allocation must generate ESM-C residue activations/pooled embeddings, build the
authorized predictor architectures, run validation-only hyperparameter search for every required
representation/architecture pairing and objective arm, evaluate all metrics over the ten low-N
seeds, save tuned baseline artifacts, report per-assay mean and SD, and settle the pooling-domain
comparison. It must use the sampler records above and must not read the layer-freeze reserve.
It must also accumulate across distinct assays before every optimizer step, emit per-assay
validation curves, never early-stop from pooled validation loss, and report pair performance by
Hamming distance. The additive suite must not be resumed or altered until the GB1 stop in
Section 8 is ruled on.
