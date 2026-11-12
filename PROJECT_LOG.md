# CROSSFIT — project log

Running record of every task, decision and adjustment, newest section at the bottom.
One line per event: what changed, why, and what it invalidates if anything.

Conventions for this file:
- `[DECISION]` — a ruling that downstream code must respect. Also mirrored into
  `README.md` standing rules if it is a standing rule.
- `[ADJUST]` — a change to something already built. Always says what it supersedes.
- `[FINDING]` — an empirical fact discovered from the data that constrains design.
- `[COST]` — GPU/wall-clock consumed, for the fixed-budget accounting the brief requires.
- Dates are the working date. Source-of-truth for numbers is always the artifact
  named in the entry, not this file's prose.

---

## Step 0 — Brief and protocol negotiation (2026-08-24 → 2026-08-27)

Scientific question, notation and the four outcomes (emergent / re-weighted /
rotated / conserved) fixed from the brief. Ten underspecification flags raised and
resolved over three rounds. Rulings that bind all later steps:

- `[DECISION]` Crosscoder encoder reads the concatenation `[h_evo_i ; h_fit_i]` and
  emits one shared `z`; `D_evo` and `D_fit` reconstruct separately. Consequence:
  computing `phi` needs forward passes through **both** models at train and inference.
- `[DECISION]` Activation normalization: one fixed frozen scalar per (model, layer)
  so `E[||h||_2] = sqrt(d)` on a fixed reference set. Raw ratio
  `E||h_fit|| / E||h_evo||` reported separately as an explicit gain diagnostic.
- `[DECISION]` Sparsity: **BatchTopK only. No L1 anywhere in the crosscoder
  objective, including auxiliaries.** Rationale: under L1, decoder-norm `Delta_j` is
  confounded by Complete Shrinkage and Latent Decoupling artifacts. The
  `lambda_reg * ||w||_1` penalty on the downstream fitness readout is a different
  object and is retained.
- `[DECISION]` Latent scaling (error ratio / reconstruction ratio) is a mandatory
  pre-registered gate on emergent candidates, with its own null distribution from
  the same null crosscoders as `tau`.
- `[DECISION]` One crosscoder per backbone, trained on pooled activations across all
  donor assays. Results aggregated **per assay** for reporting (standing rule 6
  governs reporting, not fitting).
- `[DECISION]` A single collective LoRA adapter is mandatory, not optional — pooled
  `(h_evo, h_fit)` pairs are only coherent against one `E_fit`.
- `[DECISION]` Per-assay heads are `Linear(d, 1)` with a within-assay ranking loss,
  so no term ever puts two assays' labels on a common scale. Single-assay batches,
  per-assay loss averaged over assays, assay sampling proportional to `sqrt(n)`,
  gradient accumulation across assays before each step.
- `[DECISION]` Reference null = shuffled-label LoRA (matched optimizer, steps, rank).
  Null A (two pretraining seeds) is **impossible** for ESM-C — it ships as a single
  checkpoint — replaced by a self-pair null (`h_evo` vs `h_evo`, must return
  `Delta == 0.5` exactly) plus crosscoder-seed variability.
- `[DECISION]` `tau` from a Beta fit to the null `Delta` tail, not empirical
  quantiles; QQ plot for auditability. R=2 null seeds on the primary backbone, R=1 on
  replication.
- `[DECISION]` Measurement track: one fixed donor adapter, one dictionary, one `tau`,
  nulls once. Prediction track: K=3 grouped assay folds, no null calibration.
- `[DECISION]` Cache layer must be at or after the first LoRA-adapted layer, asserted
  numerically (`max_i ||h_fit_i - h_evo_i|| > 0` plus a magnitude floor), not as a
  config check.

Still open, carried forward: pair-gating mechanism and what `c` multiplies; pooling
domain for `phi_j` (all residues vs mutated positions only); where `sigma_v` enters
the head loss.

---

## Step 1 — Dataset curation (2026-08-27 → )

### 1a. Repo scaffold and source pinning — DONE

- Created `Code/{crossfit,scripts,tests,config}`, `Data/{raw,interim,processed}`,
  `Reports/step1`. Registered dev environment `crossfit-dev` on the repo venv.
- `[DECISION]` All thresholds and seeds live in `Code/config/step1.yaml`. Nothing
  downstream hardcodes a number that appears there.
- `[DECISION]` Sources pinned by revision, not "latest":
  ProteinGym 2.0 HF revision `2e5c7b10c3dda55ca7295c5f1334887801294131`;
  v1 metadata at GitHub commit `62eab5b3c83a3727b5b57bfbe377b24ef7545ccc`.
  Recorded in `Data/raw/reference/sources_lock.json` + `SOURCES.md`.
- `[FINDING]` `.pgdata` is a nested zip: outer holds `dataset.pgdata` + `slices.json`
  (published 5-fold split masks); inner holds the assay CSV, WT fasta, structure PDB,
  MSA a3m and a TOML `manifest.lock` carrying phenotype, replicate count and
  `has_uncertainty`. All 211 bundles join onto the v1 reference on `DMS_id`.

### 1b. Candidate table and selection — DONE

- 211 bundles → 137 after dropping OrganismalFitness → 122 passing n>=500 and
  len<=1024 → 10 selected. Artifacts: `candidates.csv`, `fig_selection.png`,
  `Reports/step1/selection_rationale.md`.
- `[DECISION]` Selection is a documented judgement call; `crossfit.selection.verify()`
  raises if the hand-chosen set violates any brief criterion, so it cannot silently
  drift. All 10 checks pass.
- `[FINDING]` ProteinGym's `UniProt_ID` column holds UniProt **mnemonics**
  (`BLAT_ECOLX`), not accessions. Querying it as `accession:` returned zero EC
  annotations for all 137 assays — a silent-empty failure, not a crash. Also
  `ANCSZ` is an ancestral reconstruction with no UniProt record, and one malformed
  term makes UniProt reject an entire OR-joined batch with HTTP 400; the fetch now
  falls back to per-id queries.
- `[FINDING]` 62 of the 64 eligible Stability assays are one Tsuboyama mini-domain
  corpus (37-72 aa). Stability breadth is structurally thin in this benchmark; both
  selected stability assays essentially exhaust the alternatives.
- `[DECISION]` GB1 via `SPG1_STRSG_Olson_2014` (55 mutated positions) not
  `Wu_2016` (4 positions) — Wu makes an `unseen_position` split impossible.
- `[DECISION]` Rejected `A4_HUMAN_Seuma_2022` despite 14.8k mostly-multi variants:
  mutations span 42 positions of a 770 aa precursor, so per-residue activation
  caching would be dominated by never-mutated context.

### 1c. Download and per-assay QC — DONE (2026-08-27)

- Ten `.pgdata` bundles fetched at the pinned revision, 204 MB, every file's byte
  size checked against the pinned manifest. Written to
  `Data/raw/proteingym2/download_manifest.json`.
- `[FINDING]` HuggingFace publishes an xet content-defined-chunking `oid`, not a
  plain sha256, so file hashes cannot be recomputed locally. Size mismatch against
  the pinned manifest is treated as a hard failure instead.
- Three bugs fixed in `fetch_assays.py`: config keys are `reference` / `raw_pgdata`
  not `*_dir`; a scripted edit bound `.read_text()` to a string literal instead of
  the resolved Path; the manifest key is `file_manifest` not `files`.
- `[FINDING]` **Multi-line FASTA.** A first parse read only line 1 and reported
  `wt_len=60` for all ten assays. Fixed by joining every non-header line. This also
  retroactively dissolves the earlier "FecA WT 60 aa vs sequence 72 aa" puzzle —
  FecA's WT is genuinely 72 aa.
- `[FINDING]` Mutation-string QC is **clean for all ten**: 0 unparsed tokens, 0
  WT-reference mismatches, 0 out-of-range positions, 0 non-standard residues, 0
  nonsense rows, and mutant sequences rebuild byte-exactly from WT plus
  substitutions (1-based, offset 0). The main indexing hazard does not materialize.
- `[FINDING]` **Dedup identity resolved.** Strip synonymous tokens → drop
  fully-synonymous/WT rows → dedupe on mutant amino-acid sequence → average labels,
  reproduces v1's `DMS_total_number_mutants` exactly for 10/10 assays, single and
  multi counts included. So the declared counts *are* the deduplicated counts.
  Only FecA has sequence-level duplication (1,956 raw → 1,886 curated, 68 duplicate
  groups). Supersedes the earlier open question.
- `[DECISION]` Dedup unit is the **mutant amino-acid sequence**, not the mutation
  string, because activations are cached per sequence. Duplicates averaged, never
  deleted; disagreement beyond 3 sigma flagged in `dup_disagrees` (one FecA variant).
- `[FINDING]` **FecA published mask cannot be aligned** (length 2,025 vs 1,956
  source rows). Published masks index pre-dedup source row order; for the other nine
  they align 1:1. FecA's fold re-derived deterministically, provenance recorded as
  `re_derived_deterministic`. Its `pg_kfold` numbers are therefore not comparable to
  published leaderboard entries; the other nine are.
- `[DECISION]` **Do not apply `raw_DMS_directionality`.** ProteinGym's `DMS Score` is
  already oriented higher = fitter; the field records the sign ProteinGym has
  *already* applied. Re-applying would have flipped CcdB (the one assay with −1).
  Verified against each bundle's own `DMS Score Bin`. Encoded as
  `apply_directionality: false` plus a build-time `assert_orientation`.
- `[FINDING]` **sigma_v is genuinely unavailable for 9/10 assays.** Every
  `manifest.lock` declares `has_uncertainty = true`, but no bundle ships a
  per-variant uncertainty column and `DMS_substitutions.csv` has no such field — the
  flag describes the experiment, not the repackaged table. Consistent with the
  supervised ProteinGym leaderboard consuming no per-variant sigma.
- `[DECISION]` `sigma_source ∈ {replicate_sd, mv_trend, unavailable}`; **NaN means no
  information, never zero.** Realized: 600,492 unavailable / 1,728 mv_trend / 64
  replicate_sd, all of the latter two in FecA. Consequence for Step 4: any
  1/sigma^2-weighted loss is a FecA-only sensitivity check, not a headline component.

### 1d. Unified table — DONE

- `[ADJUST]` First build produced a **655 MB** `variants.csv` because `wt_sequence`
  and `mutant_sequence` were repeated per row. Restructured to a narrow schema:
  per-assay constants live once in `assay_manifest.csv`; `mutant_sequence` is not
  stored but reconstructed by `apply_mutations`, asserted byte-exact at build time and
  re-validated on read. Now 77.6 MB CSV / 14.9 MB parquet, 633,984 rows.
- `[FINDING]` **`apply_mutations` did not check the token's FROM residue.** The build
  catches corruption by comparing to the shipped sequence, but the loader's
  `with_sequences=True` path has no reference, so a bad token would silently yield a
  wrong sequence and every activation cached from it would be wrong. Added
  `check_reference=True`; returns `None` on mismatch.

### 1e. Splits — DONE, with one outcome-changing correction

- `[ADJUST]` **The unseen-position split leaked.** v1 partitioned positions into
  folds then assigned each variant to the *lowest* fold among its mutated positions.
  For multi-mutants that leaks: **23,619 of GFP's 26,687 fold-0 test variants (88%)
  had every one of their positions seen in training.** The B2 test was largely not
  testing extrapolation. Superseded by a **position-fold bitmask** with a strict
  three-way decode: `test = mask == 1<<f` (every mutated position held out),
  `train = mask & 1<<f == 0` (none held out), otherwise **drop**. Straddlers must be
  dropped — to test leaks context, to train leaks the held-out position. Leakage
  across all assays and folds is now 0, asserted in the test suite.
- `[COST]` The strict rule discards 0 for the six single-mutant-dominated assays,
  0.31 (KRAS), 0.33 (GB1), 0.53 (GFP), 0.79 (Gcn4) of each fold.
- `[DECISION]` **B2 availability gate.** A k-mutant survives with probability
  ~n_folds^-k, so a K sweep over {5,3,2} was run. K stays uniformly **5** for
  cross-assay comparability; per-fold usability is checked against
  `min_test_size=200`, `min_train_size=500`. Result: 9/10 assays available with 5
  usable folds; **Gcn4 alone unavailable** (mean 17.1 mutations over only 44 mutated
  positions; test sets of 3/23/72 at K=5/3/2). `get_split` raises `UnusableSplit`
  rather than returning a 3-variant test set. **B2 conclusions come from nine
  assays; Gcn4 contributes to B4 instead.** This is a declared limitation.
- `[DECISION]` `low_n` subsets are prefixes of one permutation per (assay, seed), so
  they are exactly nested and the low-N curve is monotone in information, not luck.
  Test set is the fixed `pg_kfold` fold-0 holdout.
- Layer-freeze reserve: 31,700 of 633,984 rows (5%, seed 424242), excluded by
  default and asserted unreachable from all four schemes.
- Prediction-track assay groups (K=3): {CcdB, Gcn4, DAAO, KRAS} / {EstA, FecA, TPMT}
  / {GFP, Ubiquitin, GB1}.

### 1f. Loader and tests — DONE

- `crossfit.datasets`: `list_assays`, `manifest`, `load_assay`, `load_all`,
  `get_split`, `prediction_groups`. Splits are a pure function of
  `(assay_id, scheme, seed)` via `stable_hash`, never of iteration order.
- `Code/tests/test_splits.py` — **147 tests passing in 70 s**, including
  `selection.verify()`, zero unseen-position leakage (checked with
  `require_usable=False` so refused folds are covered too), low_n nesting, reserve
  unreachability, and `stable_hash` cross-process stability.

### 1g. Schmirler et al. audit — DONE, no new assays

Checked `RSchmirler/data-repo_plm-finetune-eval` per request.

- `[FINDING]` Their GFP sequences are ProteinGym's with the leading residue trimmed
  (`pg.sequence.str[1:]` maps exactly), so the TAPE Hamming<=3 split is transferable
  onto our GFP rows. Their copy also has duplicate sequences needing score averaging.
- `[DECISION]` GB1 — they use `Wu_2016` (4 mutated positions); we keep `Olson_2014`
  (55 positions) because 4 positions cannot support B2.
- `[DECISION]` Rejected AAV (~4% position coverage, viral-titer proxy phenotype),
  Rocklin Stability and Meltome (both cross-protein, no single WT to cache residue
  positions against — structurally incompatible with the crosscoder design).
- Optional follow-up left open: a fifth `schmirler_gfp` scheme from their split, for
  one directly-comparable number. Under standing rule 4 it can only be a sanity
  check, never a baseline, since their backbone differs.

### Step 1 report

`Reports/step1/step1_report.md` + `Reports/step1/fig_qc.png`.

---

## 1h. Step 1 closeout — directory cleanup and Step 2 handoff

### Directory cleanup

- `ORBIT_PROJECT_LOG.md` (9,439 B) was checked against
  `/Users/songcao/workdir/ORBIT_GNN/PROJECT_LOG.md` (44,808 B) and found **unique**, not a
  stale duplicate — its Step 0 proposal triage, ranking and open-risk sections appear nowhere
  in the ORBIT log. Copied to `ORBIT_GNN/PROJECT_LOG_step0_triage.md` *before* removal.
- Trashed (user-approved, recoverable): `ORBIT_PROJECT_LOG.md`,
  `_view_001..005_fig_qc.png` (338 KB of figure-inspection scratch), 4 × `.DS_Store`.
- `.gitignore` extended with `.DS_Store` and `_view_*.png`.
- `Reports/` had been wiped by an idle workspace sweep; all four files
  (`step1_report.md`, `fig_qc.png`, `selection_rationale.md`, `fig_selection.png`) were
  **restored from artifacts**.
- Kept deliberately: `.pytest_cache/`, `Code/crossfit.egg-info/`, `__pycache__/` (gitignored
  byproducts) and the empty `Cache/` + `Models/` (needed from Step 2 on).
- Post-cleanup integrity re-verified: `load_all()` → 602,284 rows × 10 assays; manifest 10×36;
  all 10 `.pgdata` bundles (195 MB) present; **147 tests pass** (86.9 s).

### Step 2 handoff — `Reports/step1/data_curation_summary.md`

Reproducibility-focused handoff, every number re-read from disk rather than carried from
memory. Six sections: curated datasets + schema; QC actually performed; splits; Step 2
interface; incomplete/uncertain/deviating items; reproduce chain.

New findings surfaced while writing it (all previously unrecorded):

- `[FINDING]` **No batch/plate metadata exists anywhere.** Every shipped assay CSV inside the
  `.pgdata` bundles carries exactly `sequence, DMS Score, mutant` (+ `DMS Score Bin`, absent
  for FecA). No batch, plate, replicate-id, timepoint or sort-gate field.
  `n_biological_replicates` is assay-level only and NaN for GFP, EstA, CcdB.
  **B3's batch-confounding sub-question is therefore not addressable with this corpus**;
  only the noise and censoring sub-questions are.
- `[FINDING]` **FecA orientation is unverified** — no `DMS Score Bin` column, so
  `assert_orientation` could not run; recorded as `no_bin_column_orientation_unverified`.
  The other nine are empirically verified `higher_is_fitter`.
- `[GAP]` **`y_center` / `y_scale` are not persisted** — `standardize()` writes them to
  `DataFrame.attrs`, which does not survive the CSV/parquet write. Must be recomputed from
  `y_raw` (`mean`, `std(ddof=0)`) over the full curated set **including the reserve**.
- `[GAP]` **No validation split exists.** No scheme emits one, `get_split` returns a 2-tuple.
  Step 2 must carve val from train.
- `[CORRECTION]` Censoring counts over the curated set *including* reserve, superseding the
  earlier post-reserve figures: CcdB ceiling 178 / GFP floor 11,092 / GB1 floor 124,723;
  corpus totals 497,991 none, 135,815 floor, 178 ceiling.
- `[CORRECTION]` σ_v provenance over all 633,984 curated rows: `unavailable` 632,098,
  `mv_trend` 1,818, `replicate_sd` 68 (all FecA). Exactly 68 rows have `n_measurements == 2`;
  exactly 1 row has `dup_disagrees == True`. `per_variant_sigma_present` is True for FecA only.
- `[CORRECTION]` The brief's size profile **is** met: 3 assays at 500–2k (CcdB 1176,
  Ubiquitin 1380, FecA 1886), 4 at 2k–10k, 3 above 10k — an earlier draft of the handoff
  miscounted this as 2/4/3 and has been corrected.
- `[VERIFIED]` All eight QC hard checks are zero across all ten assays over 1,384,881 tokens.
  `unseen_position_report.json` reports 0 fully-covered test variants for every assay × fold.
  `pg_kfold` folds: 126,802 / 126,799 / 126,796 / 126,794 / 126,793, no `-1`.
  Layer-freeze reserve 31,700 rows = exactly 0.0500.

### Unresolved, carried into Step 2+

ProteinGym's **modulo-5** and **contiguous-5** CV schemes were scoped (re-derivable from
position alone, no download) and deliberately **not built**; `schmirler_gfp` likewise
deferred. Protocol questions still open from Step 0: where σ_v enters the head loss — now
sharpened by its absence for 9/10 assays, so the live options are to drop σ-weighting
entirely or keep it as a FecA-only sensitivity check; the pooling domain for `phi_j`; the
pair-gating mechanism and what `c` multiplies; the donor/eval assay partition; and whether
the prediction track needs 0 or 3 crosscoders.

**Step 1 is complete.**

---

## Repository onboarding (2026-10-16)

- `[DECISION]` Git tracks source, configuration, provenance, reports, figures, and small
  result artifacts. Raw `.pgdata` bundles, unified variant tables, regenerable split CSVs
  larger than a few MB, activation caches, checkpoints, environments, and build/test caches
  are excluded.
- `[ADJUST]` Replaced the stale root README, which named nonexistent Step 1 scripts, paths,
  and loader arguments, with the verified reproduction chain and current directory map.
- `[FINDING]` Repository onboarding preserved the remote MIT license and initial commit; the
  pre-Step-2 baseline remains green at 147 tests passing in 73.15 seconds.

---

## Step 2 — local implementation (2026-10-16)

- `[DECISION]` Corrected project claim: Gate 1 drift is only a precondition. The scientific
  claim concerns the feature-level structure of adaptation in a shared dictionary: reuse,
  re-weighting, rotation, abandonment/evolution-only structure, emergence, and conservation.
  Conserved structure is a substantive outcome; Gate 1 passing is not a finding.
- `[DECISION]` The 31,700-row layer-freeze reserve is the only valid dataset for Gate 1
  representational-drift measurements as well as analysis-layer selection. It remains excluded
  from all adapter training; training-set drift is inflated and invalid.
- `[FINDING]` Implementation blocked before code changes by a sign inconsistency in the fixed
  difference-model censoring likelihood. With `Delta_y = y_u - y_v`, a floor-censored `u`
  implies the latent difference is at most the observed difference, requiring
  `-log Phi((Delta_y - d_hat) / sigma)`. The plan instead specifies
  `-log Phi((d_hat - Delta_y) / sigma)`, which assigns low loss in the impossible direction.
  A ruling on which convention is authoritative is required before the composite objective can
  be implemented without changing a fixed formulation.
- `[FINDING]` The 19-channel AAIndex representation is not reproducibly specified: the plan
  fixes its output shape but not the AAIndex1 release, incomplete-record filtering,
  per-property normalization, or PCA basis. These choices change every encoded value. Proposed
  resolution: pin AAIndex1, retain complete nonconstant indices, z-score each index across the
  20 amino acids, fit the full 19-component PCA once, freeze the 20 x 19 score table, and encode
  each mutation as `PC(mutant) - PC(wild_type)` at its sequence position.
- `[ADJUST]` Censoring ruling supersedes the erroneous single-sign formula above. For
  `Delta_y = y_u - y_v`, the implemented constraint is
  `-log Phi(s * (Delta_y - d_hat) / sigma)`: `s=+1` for `u` floor or `v` ceiling,
  and `s=-1` for `u` ceiling or `v` floor. Same-bound double-censored pairs are dropped;
  opposite-bound pairs are retained. `sigma` is one learned positive residual scale per assay.
- `[ADJUST]` AAIndex PCA proposal above was rejected. The representation now uses Georgiev's
  published 19 descriptors (Georgiev, J Comput Biol 2009, DOI `10.1089/cmb.2008.0173`,
  Table 3). The numerical transcription was cross-checked from
  `andreas-wagner-uzh/fitness_landscape_sampling` commit
  `d227fd63c763c6673ed19192b595797b9e6cf26a`; each column was centered and divided by its
  population SD across the 20 canonical amino acids. Frozen artifact:
  `Code/config/georgiev_2009_standardized.csv`, sha256
  `ef8ba3e53df6ea0ac770cb86978fb68d679c28a1eff8b1ec3463e0fa17b161b4`.
- `[DECISION]` AAIndex encodes `G(mutant) - G(wild_type)` at the mutated sequence position.
  `aaindex_flat` is CSR with shape `N x (L*19)`; the convolutional form is the exact same
  matrix reshaped to `N x L x 19`. Non-canonical residues raise; none are imputed.
- `[FINDING]` The plan's statement that every single mutant has 19 nonzero AAIndex entries is
  not literally true for the mandated published table: 8 of 380 ordered substitutions have
  one exactly equal descriptor value (`A<->V` channel 4, `M<->W` channel 1, `P<->W` channel
  10, `S<->W` channel 16). They therefore have 18 mathematical nonzeros. The values are not
  perturbed; every mutation still occupies a fixed 19-channel positional block.
- `[DECISION]` Epoch allocation is proportional to `sqrt(n_assay)` for a requested epoch
  budget, with water-filling at `min(n_assay, 4 * median(n))` and no replacement. The config's
  default `capped_pool` epoch consumes every assay's available capped quota. On the 602,284
  usable rows, median `n=2,986`, cap `11,944`, epoch size `54,163`; GB1, GFP, and KRAS each
  contribute 11,944 rows (22.0520%), while no small assay is repeated.
- `[ADJUST]` Added PyTorch as the differentiable objective dependency and constrained NumPy to
  `<2`; the available macOS x86 PyTorch 2.2.2 wheel was compiled against NumPy 1.x and emitted
  an initialization warning under the prior NumPy 2.4.6 environment. Local CPU validation uses
  NumPy 1.26.4.
- `[DECISION]` `CompositeFitnessLoss` is the sole training objective. It returns one weighted
  scalar plus unweighted diagnostics. Ranking, magnitude, censoring, and readout-L1 terms are
  summed over the gated batch; optimizer weight decay remains separate. Magnitude uses pair
  variance `sigma_u^2 + sigma_v^2` only when both curated sigmas are finite and positive,
  otherwise that pair is unweighted.
- `[FINDING]` Step 2 local implementation added 34 tests; the full suite is 181 passing in
  70.94 seconds. The exact interfaces, realized assay shares, representation dimensions,
  provenance, and deliberately unbuilt work are in `Reports/step2/step2_handoff.md`.

### Step 2 local corrections and prerequisites (2026-10-17)

- `[ADJUST]` Superseded the default magnitude weighting above. Curated `sigma_v` is now
  ignored by the magnitude term unless `loss.magnitude_uncertainty_weighting.enabled` is
  explicitly true; the default is false and every pair has weight one. If enabled, the
  implementation requires the configured minimum sigma coverage in every participating assay,
  uses inverse-pair-variance weights where both values are finite and positive, leaves other
  raw weights at one, and normalizes weights to mean one separately within each assay.
- `[FINDING]` Re-read over all 633,984 curated rows including the reserve: FecA is the only
  assay with non-NaN `sigma_v`, at 1,886/1,886 (100%; 1,818 `mv_trend`, 68
  `replicate_sd`). Each other assay has 0% coverage. Corpus coverage is 1,886/633,984
  (0.2975%); 632,098 values are NaN. Thus uncertainty weighting is unusable for the joint
  default objective.
- `[FINDING]` Square-root weights are inert at the default `epoch_size: capped_pool`. The
  allocation is exactly `min(n_assay, 11,944)` because the epoch exhausts every capped
  capacity. CcdB contributes all 1,117 usable rows every epoch; GB1 contributes 11,944 of
  510,114 (2.3414%), so a CcdB row receives about 42.71 times the per-epoch exposure of a
  GB1 row. Keep the full capped pool rather than discarding GB1 rows merely to make square-root
  weights bind.
- `[DECISION]` GPU predictor training must record validation curves separately for every
  assay and must not early-stop on a pooled validation loss. If small assays overfit before
  large assays, record the result for Steps 9 and 10; do not modify the run in response.
- `[FINDING]` The added Hamming constraint is strongly active, not vacuous. Across five
  deterministic default capped-pool epochs, after the censoring and `|Delta_y| > 0.25` gates
  and before the distance constraint, `delta <= 2` removes FecA 50.4656%, Gcn4 99.9752%,
  GFP 99.9443%, KRAS 95.6687%, and GB1 99.6291% of within-batch gated pairs. It removes 0%
  in CcdB, EstA, DAAO, Ubiquitin, and TPMT. Exact per-assay distance distributions and pair
  counts are in `Reports/step2/pair_distance_audit.csv`.
- `[DECISION]` Per the >1% ruling, retain `pairs.max_hamming_distance: 2` and stop before the
  GPU run so its large effect on the training distribution can be reviewed. The audit covers
  the actual deterministic sampled training stream (five epochs of single-assay batches),
  not an infeasible all-pairs Cartesian product of the 602,284 usable variants.
- `[DECISION]` One optimizer update accumulates equally averaged gradients from at least two
  distinct single-assay batches and calls `step()` only once afterward. The shared helper
  rejects a single assay or repeated assay id within an accumulation group; gradient clipping
  occurs after accumulation.
- `[DECISION]` Difference-head predictions are converted to per-variant scores by an anchored
  graph-Laplacian least-squares solve of `s_u - s_v = d_hat_uv`, with the wild-type score fixed
  at zero. Disconnected graphs are rejected because their component offsets are unidentified.
- `[DECISION]` Split conformal intervals use exact Mondrian strata `(mutation_count,
  mutates_unseen_position)` and the finite-sample corrected absolute-residual quantile. There
  is no pooled fallback for an absent calibration stratum.
- `[DECISION]` The Step 2 readout L1 term applies only to predictor readout parameters. It must
  never enter the Step 7 crosscoder objective, which uses BatchTopK and no L1 term anywhere.
- `[FINDING]` The corrected Step 2 CPU prerequisites bring the full suite to 191 passing tests
  in 83.82 seconds: the original 147 plus 44 Step 2 tests. No GPU work was performed.

### Step 2 additive-baseline preregistration (2026-10-17)

- `[FINDING]` No Hamming threshold is both informative and safe: `delta=4` is inert for eight
  assays but retains only about 6.1% of GFP and 0.4% of Gcn4 gated pairs, while smaller values
  also remove most pairs from complete single/double libraries.
- `[ADJUST]` Drop the pair Hamming constraint and its config entry entirely, superseding the
  temporary `delta=2` retention above. Pair distance becomes a reported performance stratum,
  not a training filter.
- `[DECISION]` Before fitting the classical baselines: per-position mutation indicators and
  the 19 Georgiev differences span the same linear amino-acid contrast space when all residues
  are observed. Their difference is the regularization geometry and AAIndex's ability to share
  strength when a one-hot coefficient is unidentified. Expect similar GB1 performance because
  its single/double library has complete coverage, with divergence confined mainly to sparse
  assays. A mean low-N GB1 AAIndex Spearman advantage greater than 0.05 is preregistered as a
  bug signal and stops the remaining baseline run.
- `[DECISION]` Tune each assay/representation once by three-fold validation internal to the
  `N=512, seed=0` training subset, selecting mean validation variant-level Spearman. Reuse the
  selected margin, L1, weight decay, and epoch count for every low-N size and all ten seeds;
  no test row participates in tuning. This avoids hyperparameter luck changing the
  sample-efficiency curve, including at N=16 where a validation carve is not defensible.
- `[FINDING]` Gcn4 is a sharply separated mixture by mutation load, confirming the pair-distance
  bimodality is compositional rather than a Hamming implementation artifact. Of 2,638 curated
  variants, 1,631 (61.83%) have 1-6 mutations, only 43 (1.63%) have 7-30, and 964 (36.54%)
  have 31-44. The reserve-excluded distribution is nearly identical: 1,543/2,506 (61.57%),
  40/2,506 (1.60%), and 923/2,506 (36.83%). Preserve both populations and take no corrective
  action before Step 9. Exact counts are in
  `Reports/step2/gcn4_mutation_count_distribution.csv`.
- `[FINDING]` The Step 2 margin sweep was already registered at 0.10, 0.25, and 0.50. Across
  the same five deterministic capped-pool epochs, total gated pairs across assays are
  7,652,849, 6,751,287, and 5,623,158 respectively. Per-assay counts are recorded in
  `Reports/step2/margin_gate_audit.csv`; no candidate was added or removed after seeing them.
- `[FINDING]` The preregistered GB1 stop condition fired before any other additive baseline was
  run. Across N=16,32,64 and ten seeds, mean held-out variant Spearman was 0.13944 for the
  mutation indicator and 0.40378 for AAIndex, an AAIndex advantage of 0.26434 versus the 0.05
  bug threshold. At N=512 the means were 0.52485 and 0.77821. The selected one-hot setting was
  margin 0.50, L1 1e-4, weight decay 1e-3, 30 epochs; AAIndex selected margin 0.25, L1 1e-5,
  no weight decay, 4 epochs. All 120 fits per arm completed without a zero-pair failure.
- `[FINDING]` The GB1 diagnostic does not indicate a rank-deficient Georgiev table: at each of
  55 mutable positions, the 19 by 19 matrix mapping non-wild-type residue indicators to
  Georgiev differences has rank 19 and condition number 4.51-4.76. It does expose a low-N
  identifiability mismatch in the preregistered expectation. Although the full assay contains
  all 1,045 position/residue features, an N=16 training subset observes only 31.4 on average;
  even N=512 observes 654.9. AAIndex therefore shares across unobserved residues at an observed
  position, while zero-initialized one-hot coefficients remain unidentified. Coordinate-
  dependent explicit and implicit regularization can differ even though the full feature spaces
  are isomorphic. Per instruction this remains classified as a bug signal, not a finding about
  model superiority; the other nine assays were not run and no corrective change was made.
- `[FINDING]` After removing Hamming filtering and adding the additive/metric interfaces, the
  full suite passes 199 tests in 79.80 seconds. No GPU work was performed.

### v10 migration and IRED-first pipeline (2026-10-21)

- `[ADJUST]` `CROSSFIT_Implementation_Plan_v10.docx` is now the sole implementation plan.
  A repository inventory found no tracked earlier plan document and no `CODEX_HANDOFF.md`,
  so no such files required deletion. Added root `AGENTS.md` to make the v10 standing rules
  persistent across sessions. The committed plan SHA-256 is
  `1552d61191b54aa4c7ee2ffc7d56bc803b41c31876c3f91c4ac24e1a7953b5ed`.
- `[ADJUST]` Archived each invalid pre-v10 GB1 additive artifact under
  `Reports/archive/step2_pre_v10_gb1_additive/`: `additive_baseline_runs.csv`,
  `additive_baseline_summary.csv`, `additive_gb1_sanity.csv`, the `additive_models/`
  directory and both model files, `additive_pair_hamming_metrics.csv`,
  `additive_selected_hyperparameters.csv`, and `additive_tuning_grid.csv`. These fits use
  pairwise early subtraction and low-N selection and must never be loaded by the v10 pipeline.
- `[ADJUST]` Deleted superseded balancing modules `Code/crossfit/sampling.py` and
  `Code/crossfit/optimization.py`, and tests `Code/tests/test_sampling.py` and
  `Code/tests/test_optimization.py`. Independent dataset pipelines have neither assay
  sampling nor cross-assay gradient accumulation.
- `[ADJUST]` Deleted superseded pairwise modules `Code/crossfit/pair_analysis.py` and
  `Code/crossfit/metrics.py`, scripts `Code/scripts/analyze_pair_distances.py` and
  `Code/scripts/run_additive_baselines.py`, and tests `Code/tests/test_pair_analysis.py` and
  `Code/tests/test_metrics.py`. v10 predicts variant fitness directly and forms ranking
  comparisons only transiently inside a batch.
- `[ADJUST]` Deleted `Data/processed/splits/prediction_groups.csv`; grouped assay folds are
  invalid under independent per-dataset pipelines.
- `[ADJUST]` Archived each superseded pair diagnostic under
  `Reports/archive/step2_pre_v10_pair_diagnostics/`: `pair_distance_audit.csv`,
  `margin_gate_audit.csv`, and `gcn4_mutation_count_distribution.csv`.
- `[ADJUST]` Replaced the Step 2 objective and model interfaces rather than retaining v4
  compatibility flags. Scores are per variant; ranking pairs are formed only inside a batch;
  censoring is a standard per-variant Tobit likelihood with a learned dataset residual scale;
  and predictor regularization is L2. The margin `c=0.25` SD and temperature `T=1.0` are fixed
  config values and absent from every sweep.
- `[ADJUST]` Replaced sparse mutation-difference encodings with the published dense forms:
  whole-sequence 20-by-L one-hot, whole-sequence absolute 19-by-L Georgiev, full ESM-C
  residues, residue deltas, all-residue mean pooling, and mutated-position pooling as a
  diagnostic only. The frozen Georgiev CSV remains byte-identical at SHA-256
  `ef8ba3e53df6ea0ac770cb86978fb68d679c28a1eff8b1ec3463e0fa17b161b4`.
- `[DECISION]` The Step 2 architecture factory contains ridge, MLP, random forest, gradient-
  boosted regression trees, a residual 1-D convolutional stack, and attention pooling. The
  optional boosted-tree condition is met by Sandhu et al. (Protein Science, 2025), which
  evaluates scikit-learn GBTs for supervised protein-fitness prediction. Added scikit-learn
  as a project dependency.
- `[DECISION]` Augmented one-hot and AAIndex linear models use explicit normal equations with
  an unpenalized intercept, the selected site penalty on every site coefficient, and `1e-8`
  on the single dPLL coefficient. dPLL uses variant context and sums only mutated-position
  masked-log-probability differences; wild-type terms are cached by position.
- `[ADJUST]` Removed top-k recall. Evaluation now reports direct-score Spearman, Pearson, MSE,
  NDCG, wild-type-cutoff top-k precision with test hit rate, five-member deep ensembles, and
  overall/per-stratum Mondrian coverage. The IRED plate loader deliberately provides no
  precision cutoff.
- `[FINDING]` Load-time ceiling-censor assertions pass for all five v10 datasets: IRED, GB1,
  GFP, KRAS, and DAAO each have zero ceiling-censored variants. The symmetric code branch is
  unit-tested but remains inactive on real selected data.
- `[FINDING]` IRED has 1,183 source rows: one wild type and 1,182 unique non-WT variants. All
  290-residue sequences are canonical and reconstruct exactly; no mutation token or label is
  invalid. The wild-type campaign fitness is -0.035637535033723. The curated variants contain
  173 floor-censored and zero ceiling-censored rows.
- `[DECISION]` The IRED layer-freeze reserve contains 59 variants, leaving 1,123 available by
  default. Primary pg_kfold fold 0 has 228 usable test variants and a training pool of 895.
  The evaluation-only rounds-1-4 versus round-5 split has 758 train and 365 test variants
  after reserve exclusion.
- `[FINDING]` The IRED plate source has 55 measurements and one duplicated sequence/mutation.
  Its two `1/s` values were averaged, leaving 54 unique variants; all reconstruct from the
  campaign wild type and none overlaps a campaign sequence. No wild-type kcat exists, so the
  plate contract is Spearman and NDCG only.
- `[COST]` This migration and IRED curation used 0 GPU-hours. Training and all ESM-C inference
  stopped at the requested post-curation review boundary.
- `[FINDING]` The post-migration full suite passes 186 tests in 93.74 seconds. A test-process
  thread cap prevents a reproducible macOS x86 Accelerate/OpenMP race when PyTorch is loaded
  before the first Arrow import; it changes no training or numerical configuration.

### v12 compute-bounded IRED run (2026-10-21)

- `[ADJUST]` Replaced v10 with `CROSSFIT_Implementation_Plan_v12.docx` (SHA-256
  `06e88344d02bd0adf8b132779f1fadfaffb74c8cc6fd301cd50802604cfab4ca`). v12 is the sole
  plan and root `AGENTS.md` now carries its 30-60-configuration random-search and one-dataset
  architecture-search budget rules.
- `[ADJUST]` The prior `selection_metric: pg_kfold_test_spearman` key was a leakage bug in
  naming, although no Step 2 training/search entry point existed yet and therefore no model
  had selected on test. Replaced it with validation Spearman from deterministic five-fold CV
  entirely inside the pg_kfold training pool; the pg_kfold test is reporting-only.
- `[ADJUST]` Replaced the 15% validation slice with five-fold internal CV. Registered random
  search at seed 20261212 with 60 configurations per arm initially and a budget floor of 30.
  Ridge site penalties now span `1e-4` through `1e4` and explicitly include `1e3` and `1e4`.
- `[ADJUST]` Registered exactly the nine v12 arm families. Removed gradient-boosted trees,
  attention pooling, their model-factory branches and grids, MLPs on classical encodings, and
  all but ridge on `esm_mutpool`. Renamed active Georgiev config/code identifiers to AAIndex;
  the frozen published table bytes and hash are unchanged.
- `[FINDING]` IRED pg_kfold fold 0 contains 228 test variants. Against the standardized
  wild-type cutoff 0.3217626895, 113 are hits, for a test hit rate of 0.4956140351.
- `[FINDING]` The 54-variant plate set has `1/s` mean 0.66115548, SD 0.84847707, median
  0.33893789, range -0.02174366 to 4.45589763, and 5th/95th percentiles 0.04373442/2.01776129.
  Mutation counts 1-8 have counts 5, 16, 15, 9, 6, 1, 1, 1. Seven variants mutate at least
  one position absent from the campaign; the unseen positions are 106, 116, 123, 137, 233.
- `[DECISION]` The exact diagonal-ridge baselines solve the explicitly penalized normal
  equations through their algebraically equivalent centered dual system. This preserves an
  unpenalized intercept, the selected site penalty, and the `1e-8` dPLL penalty while avoiding
  a 5,800-by-5,800 factorization for IRED one-hot. Ten representative CPU fits averaged
  0.10-0.11 seconds, projecting 31-33 seconds for each 60-configuration, five-fold search.
- `[FINDING]` The unaugmented one-hot and AAIndex ridge arms are effectively tied, as expected
  from their shared linear column space. Five-fold validation Spearman was 0.61514 for one-hot
  and 0.61504 for AAIndex; pg_kfold test Spearman was 0.53197 and 0.53224, respectively. The
  selected site penalties were 425.82 and 8326.54; neither is exactly on the registered
  `1e-4`/`1e4` boundary. Exact convex fits are seed-invariant, so their five-member ensemble
  means are identical and ensemble SD is zero.
- `[FINDING]` ESM-C 300M extraction on the A100 returned 960 channels at 30 hidden-state
  depths. IRED's realized cache is 1,674,691,432 bytes (1.67 GB decimal): final-layer
  residue tensors for 1,182 campaign variants and 54 plate variants, all-depth tensors only
  for the 59-variant reserve, plus wild type. The model work took 90.62 seconds (109.52
  seconds wall time including startup and writes).
- `[FINDING]` Mutated-position dPLL was computed for 1,237 unique sequences using 3,441
  variant-context masked tasks and 277 wild-type position tasks cached once. All values are
  finite and unique by sequence; mean is -5.7925, SD 5.3026, and range -32.4302 to 5.0523.
- `[ADJUST]` The pod deliberately lacks ignored `Data/processed/variants.parquet`. The loader
  now reads tracked `ired_variants.parquet` independently when the ProteinGym bundle is
  absent, with a regression test. Pinned `pyarrow>=14,<18`: PyArrow 26 required NumPy 2,
  while the ESM environment requires NumPy 1.26; PyArrow 17 is compatible with both.
- `[FINDING]` The augmented classical arms remain close to each other. IRED pg_kfold test
  Spearman is 0.54409 for one-hot+dPLL and 0.54419 for AAIndex+dPLL, versus 0.53197 and
  0.53224 unaugmented. On the independent plate set, augmentation is worse: Spearman is
  0.53002/0.52956 versus 0.56371/0.56341. This negative external result is retained.
- `[COST]` Ten representative `esm_residue -> conv residual` fits averaged 137.72 seconds,
  projecting 11.48 GPU-hours at 60 configurations and 5.74 hours even at the minimum 30 for
  that arm alone. Per the v12 budget rule, classical and pooled ESM arms take priority; the
  two required residue-conv arms cannot both fit under the nine-hour session ceiling, and
  the optional one-hot conv arm is not launched. No seed count or measurement is reduced.
- `[FINDING]` Five-fold validation selects all-residue mean pooling over mutated-position
  pooling under the matched ridge comparison (Spearman 0.60765 versus 0.55007), so Step 7's
  pooling domain is all residues. The held-out estimates reverse weakly (0.52243 versus
  0.54519) with strongly overlapping bootstrap intervals, and the plate estimates are
  0.33684 versus 0.44425. Those evaluation reversals are reported but do not override the
  pre-registered validation-only selection rule.
- `[FINDING]` The ranking among completed IRED arms is narrow at the top: one-hot ridge,
  AAIndex ridge, their dPLL-augmented versions, ESM-mean ridge, and ESM-mean MLP have inner-CV
  Spearman 0.6058-0.6151. External plate performance separates them: unaugmented classical
  ridge is best at about 0.564; augmented classical is about 0.530; mutated-position ESM
  ridge 0.444; ESM-mean ridge 0.337; MLP 0.239; random forest 0.232.
- `[FINDING]` Selected search boundaries: ESM-mean ridge chose minimum weight decay and batch
  size; ESM-mean MLP chose maximum dropout and learning rate plus minimum batch size; random
  forest chose maximum depth, maximum `min_samples_leaf`, and maximum feature fraction. The fixed `1e-8` dPLL
  penalty is not a searched hyperparameter and is therefore not mislabeled as a boundary.
- `[DECISION]` The exact ESM-C backbone is `biohub/ESMC-300M` revision
  `e4bac860f0c502cd80f3aeac3d0ce6524c6627cb`, loaded with `esm==3.4.1.post1`. Runtime loading
  now fails if the Hugging Face main ref differs. Step 3 uses batch size 16 with two-step
  accumulation, preserving the registered effective batch size 32 and 23 optimizer steps
  per CV training epoch as the earlier 4-by-8 setting.
- `[ADJUST]` Gate 1 task gain is measured on five cross-validation folds within IRED's
  pg_kfold training pool, not on the 59-row layer-freeze reserve. The reserve is used only
  for depth selection, drift, and shuffled-label separation. This supersedes v12's labelled
  reserve comparison and preserves the reserve as training-independent activation data.
- `[ADJUST]` The analysis depth is selected on IRED alone for this run. The common-depth
  agreement required across datasets is deferred until the second dataset; all 30 IRED
  depths are committed so that decision will not require recomputing this curve.
- `[ADJUST]` A preliminary fixed-80-epoch Gate 1 run completed, but PEFT
  `save_pretrained` emitted empty 40-byte states for ESM-C's compatibility wrapper. Explicit
  trainable-state export fixed that defect, after which a clean two-control refit exposed a
  second issue: initialization had occurred before the per-fit seed, so recovered checkpoints
  did not reproduce the cached activations. The preliminary reports, cache, empty exports,
  and complete-but-mismatched recovery fits are preserved in explicitly named archive paths
  and are not official Gate 1 results. Adapter construction is now seeded and checkpoint
  export is name-and-shape validated.
- `[DECISION]` Step 2 early stopping and best-checkpoint selection monitor validation
  Spearman. Final five-seed neural refits use the rounded median best epoch from the selected
  configuration's five CV folds. ReduceLROnPlateau also monitors validation Spearman by
  design, for IRED and every later dataset. Keeping stopping, restoration, and scheduling on
  the same pre-committed quantity is more coherent and preserves cross-dataset comparability.
  The completed IRED Step 2 results already follow this policy and are not rerun.
- `[DECISION]` Step 3 real adapters early-stop on validation Spearman and restore the best
  checkpoint. IRED fold 0 is the config-fixed final adapter fold. The shuffled-label control
  uses the same training rows, initialization seed, optimizer, rank, and exact learning-rate
  sequence for exactly the real adapter's best-checkpoint optimizer-step count; it never
  early-stops on shuffled validation labels. Both realized counts are written to Gate 1.
- `[ADJUST]` The fixed-epoch Gate rerun was interrupted after the stopping-rule clarification,
  before any active checkpoint or cache existed. Its timing file is archived and its partial
  fold value is not an official or reported Gate result. Every subsequent fit starts from its
  registered initialization; no partial training state is reused.
- `[FINDING]` The authoritative IRED Gate 1 task comparison completed all five folds. Frozen
  and adapted mean Spearman are 0.59878 and 0.61784, for mean task gain +0.01906. Fold gains
  are +0.01656, +0.04025, -0.02916, +0.08086, and -0.01321; two folds are negative. The
  final fold-0 real adapter restores epoch 9 at 207 optimizer steps after executing 483 steps
  to satisfy patience. The shuffled control completed exactly 207 steps with the real LR
  sequence.
- `[ADJUST]` Checkpoint load-back found that the first separation analysis of the authoritative
  checkpoints had extracted the shuffled model while it remained in training mode, injecting
  LoRA dropout noise. Training and checkpoints were complete and valid. All representation
  extraction now forces eval mode, and the active analysis was resumed from those checkpoints.
  The train-mode report and cache are preserved under explicit archive paths and are not used.
- `[FINDING]` Corrected eval-mode Gate 1 passes. Depth 29 maximizes real-minus-shuffled drift:
  real 1.29809, shuffled 0.84576, gap bootstrap 95% interval [0.38609, 0.52066]. Real and
  shuffled CKA are 0.06415 and 0.81942. Every one of the 30 depths has a positive gap lower
  bound. Depth 29 is the final layer, so +2/+4 sensitivity is impossible; -2/-4 gaps remain
  positive at 0.32644 and 0.24897. This boundary optimum and lack of downstream blocks must
  be revisited in the second-dataset common-layer agreement check.
- `[FINDING]` The active Step 3 pod cache contains eval-mode reserve tensors shaped
  `[59,30,290,960]` and base/adapted depth-29 tensors shaped `[1182,290,960]`, totaling
  3.1 GB. Full-corpus extraction retains only the selected layer batch-by-batch. No Step 4
  cache was written.
- `[COST]` Step 2 consumed approximately 1.9 A100 GPU-hours against its 5-hour estimate,
  including extraction, pooled-arm searches, the official conv timing audit, and discarded
  timing attempts. Step 3 consumed approximately 2.0 A100 GPU-hours against its 2-hour
  estimate, including the authoritative 0.43-hour run plus the fully preserved diagnostic
  reruns needed to expose checkpoint and eval-mode defects. Session total is approximately
  3.9 GPU-hours, below the 9-hour hard ceiling.
- `[FINDING]` Final repository verification passed all 203 tests in 99.92 seconds. The active
  Step 2 and Gate 1 reports contain only completed searches, folds, refits, and checkpoint-
  backed analyses; interrupted and superseded attempts remain isolated under
  `Reports/archive/` and are not used in any reported result.
- `[ADJUST]` The earlier Gate 1 verdict is withdrawn. The five paired fold gains have mean
  0.01906, SE 0.01955, `t(4)=0.97506`, and two-sided `p=0.38474`; a positive sample mean is
  not evidence that the adapted encoder beats the frozen encoder. Gate 1 is false because
  task gain is not established at alpha 0.05.
- `[ADJUST]` The former maximum-gap layer rule was miscalibrated because it selected collapsed
  final depth 29 (real CKA 0.06415) and left no downstream computation for causal patching.
  Layer selection now excludes the first and final three depths, requires real CKA at least
  0.5, and selects the earliest eligible depth within 90% of the eligible maximum drift gap.
  The depth-29 report is preserved under
  `Reports/archive/step3_gate1_depth29_pre_cka_floor/` rather than deleted.
- `[FINDING]` The corrected reserve-only rule nominates depth 26. Its real/shuffled drift is
  0.63382/0.34174 (ratio 1.85470), gap interval [0.28320, 0.30120], real CKA 0.91895, and
  mean activation-norm ratio `||h_fit||/||h_evo||=0.91650`. Sensitivity gaps at offsets -4,
  -2, and +2 are 0.17599, 0.21504, and 0.37331; +4 does not exist and is reported as such.
  The layer remains unfrozen pending review.
- `[ADJUST]` Layer selection and headline reporting use different held-out data by design.
  The layer is selected once from the 59-variant reserve curve; after it is frozen, headline
  drift is measured on the 228-row pg_kfold test split. Training/test stability profiles are
  deferred with that re-extraction and must not be used to re-select the layer.
- `[DECISION]` The proposed lower-learning-rate or stronger-regularization collapse remedy is
  withdrawn. The adapter already used learning rate 1e-4 and weight decay 0.01, and collapse
  was localized to the head-adjacent final depth. No adapter is retrained to repair that
  collapse. A bounded task-gain sweep changes only rank and learning rate upward:
  `(16,3e-4)`, `(32,1e-4)`, and `(32,3e-4)`, against the completed `(16,1e-4)` baseline.
- `[ADJUST]` The three convolutional Step 2 arms were deliberately omitted after the original
  ten-fit projection, not accidentally missed. That projection-based omission proved too
  conservative when the completed session used only about 1.9 GPU-hours, and the incomplete
  arm count was not surfaced prominently enough. The two required ESM-residue arms are now
  being completed at 30 random configurations; the optional one-hot arm will run if its timing
  is cheap.
- `[ADJUST]` A restarted-pod convolution timing attempt spent 52 minutes without completing
  its ten fits because identical fold normalization was recomputed for every configuration.
  It was interrupted before any search fold or result existed. Fold-specific training-only
  moments and transformed tensors are now cached once and reused exactly across configurations;
  the arm restarted from the beginning with no partial result carried forward.
- `[DECISION]` Neural random searches now atomically cache each completed configuration-fold
  fit under `Cache/step2/IRED/search_state/`. A resumed search validates arm identity, fold and
  configuration bounds, and uniqueness before skipping completed folds. A fit interrupted
  before its atomic row is written is rerun from that fold's beginning; partial predictions or
  weights are never promoted to reports. This changes restart behavior only, not training.
- `[ADJUST]` A second `esm_residue_conv` invocation was stopped during an unfinished repeat of
  the timing audit after 1.07 hours, before the official random search and before any result.
  The completed earlier ten-fit audit has the same arm, seed, model, and training logic, so it
  is reused to project the requested 30 configurations. The runner restarted the search from
  its beginning under atomic completed-fold caching; the interrupted timing fit is discarded.
- `[ADJUST]` The first official `esm_residue_conv` fold completed but exposed a host-transfer
  bottleneck: roughly 45 minutes for a fit whose best epoch was 15, while A100 utilization was
  mostly idle. CUDA training now keeps the current normalized fold tensors GPU-resident and
  shuffles lightweight row indices. This preserves values, batch membership, seeds, objective,
  stopping, and architecture. Completed fold 0 remains authoritative; the unfinished fold 1
  attempt was discarded and rerun from its beginning, completing in about 2.5 minutes. Peak
  conv-process memory was about 2.1 GB on the 80-GB card.
- `[FINDING]` The bounded IRED task-gain sweep completed every fold for all three new
  configurations. Rank 16/lr 3e-4 gives mean gain 0.02314, `t(4)=1.23124`, `p=0.28568`;
  rank 32/lr 1e-4 gives 0.03086, `t(4)=1.36380`, `p=0.24433`; rank 32/lr 3e-4 gives
  0.03104, `t(4)=1.58446`, `p=0.18826`. None establishes positive task gain at alpha
  0.05. The registered rank 16/lr 1e-4 Gate result remains failed, and the analysis layer
  is unchanged. This is a finding about what the 895-row IRED training pool can establish.
- `[COST]` The three-configuration IRED task-gain sweep consumed 3.02390 A100 GPU-hours.
  All 15 new cross-validation folds completed; the authoritative registered baseline was
  reused rather than retrained.
- `[DECISION]` After the adapter sweep completed, the two required ESM convolution arms were
  run concurrently on the A100. They have separate representations, random-search seeds,
  atomic fold-state files, reports, and model directories. GPU-resident fold tensors use only
  about 2.5 GB per process on the 80-GB card, and utilization remained below saturation. This
  changes wall-clock scheduling only; both arms retain 30 configurations, five CV folds, and
  five final seeds.
- `[ADJUST]` The GPU pod restarted after atomically completing 102/150 `esm_residue_conv`
  search folds and 93/150 `esm_residue_delta_conv` folds. The replacement container retained
  both cache files. Completed rows were reused after validation; each fold active at shutdown
  had no atomic row and was rerun from its beginning. No partial checkpoint, prediction, or
  metric was promoted. The completed task-gain reports on the pod matched the committed copies
  byte-for-byte before the pod fast-forwarded and both searches resumed.
- `[DECISION]` The optional `onehot_positional_conv` arm was defined as cheap enough to run if
  its completed ten-fit audit projected no more than 2 A100 GPU-hours for 30 configurations and
  five folds. The audit measured 28.9690 seconds per fit and projected 1.20704 GPU-hours, so the
  full optional search was launched. The threshold was applied before seeing any arm result.
- `[FINDING]` Both required ESM convolutional arms completed 30 random configurations across
  five folds and five final seeds. `esm_residue_conv` selected width 256, depth 3, kernel 5,
  dropout 0.1, learning rate 3e-4, weight decay 0.01, batch size 16, and 13 epochs; validation,
  pg-test, round-held-out, and plate Spearman are 0.63268, 0.54294, 0.49846, and 0.33364.
  `esm_residue_delta_conv` selected width 256, depth 5, kernel 3, dropout 0.1, learning rate
  1e-4, weight decay 0.001, batch size 16, and 2 epochs; the corresponding Spearman values are
  0.62910, 0.54552, 0.54081, and 0.38998. All reported results come from complete searches and
  complete five-member ensembles.
- `[FINDING]` The optional `onehot_positional_conv` arm completed all 150 search folds and five
  final seeds. It selected width 256, depth 5, kernel 7, dropout 0.1, learning rate 3e-4,
  weight decay 1e-4, batch size 32, and 22 epochs. Validation, pg-test, round-held-out, and
  plate Spearman are 0.63207, 0.51678, 0.50733, and 0.57423. Width, depth, and kernel size
  landed on upper search boundaries; dropout and weight decay landed on lower boundaries.
- `[COST]` Completing the three convolutional arms consumed approximately 5.9 A100 wall-hours:
  about 1.94 hours in two discarded pre-search timing/host-transfer attempts, about 2.9 hours
  for the two concurrent required searches across the pod restart, and 1.0198 hours for the
  optional one-hot search and refits. Concurrent required processes are counted once as GPU
  wall occupancy, not twice as process-hours. Together with the exact 3.02390-hour bounded
  adapter sweep, this continuation used approximately 8.9 A100 GPU-hours.
- `[FINDING]` The completed Step 2 summary contains all 11 expected result variants and no
  incomplete arms. The convolutional validation shortlist ranks ESM residue (0.63268), one-hot
  positional (0.63207), and ESM residue delta (0.62910). Final verification passes all 210
  tests in 100.56 seconds.
- `[DECISION]` IRED Gate 1 is failed permanently on the registered criterion. Mean paired gain
  is 0.01906 with per-fold SD 0.04371, Cohen's d 0.436, `t(4)=0.975`, and `p=0.385`.
  Approximately 42.5 independent folds are needed for 80% power at d=0.44; a hypothetical
  ten-fold test gives only `t(9)=1.379`, `p=0.201`. A variant bootstrap of fixed out-of-fold
  predictions is rejected because it conditions away model-fit variance. The small positive
  gain is monotone across the four tested capacity settings but is not establishable from 895
  training variants: underpowered is not the same as absent. Capacity, sweep size, and the Gate
  criterion will not be changed.
- `[FINDING]` Gate 1's independent representational precondition is met cleanly at depth 26:
  real relative drift is 0.63382 against shuffled drift 0.34174, with gap interval
  [0.28320, 0.30120] and real CKA 0.91894. Only the task-gain criterion fails.
- `[ADJUST]` Prediction persistence is now a required pre-metric contract. Every future fit
  writes held-out rows with dataset, step, arm, configuration, fold, seed, condition, variant,
  target, and prediction before reducing them to metrics. Full-search folds are partitioned as
  pod-only parquet; selected CV and individual ensemble-condition predictions are committed.
  Checkpoint classes are enabled by default and every retained checkpoint receives a SHA-256
  manifest entry. The historical audit records missing Step 2 CV/member rows and all missing
  Step 3 OOF rows as unrecoverable without retraining; no missing row is fabricated.
- `[DECISION]` Continuing past failed IRED Gate 1 is a post-hoc reinterpretation of the
  criteria's relative weight, not a revised Gate verdict. The justification is that drift and
  shuffled-label separation directly measure label-driven representation change over 17,110
  held-out residue positions and both passed cleanly, whereas task gain is an underpowered
  five-fold generalization proxy. No later result may retroactively mark IRED Gate 1 passed.
- `[DECISION]` Depth 26 is frozen as the analysis layer for exploratory IRED Steps 4-8. Later
  training/test stability profiles are reporting diagnostics and cannot re-select the layer.
- `[DECISION]` IRED Steps 4-8 are exploratory pipeline shakedown only. No IRED feature-level
  result is confirmatory or evidence for the central claim; every such result is
  hypothesis-generating for GB1. The run must exercise caching, self-pair and shuffled nulls,
  latent scaling, the per-decoder norm floor, classification precedence, and patching hygiene,
  and every implementation defect found must be reported. GB1 starts only after the IRED
  sequence is complete and never in parallel.
- `[DECISION]` IRED dictionary width will be sized to effective sample size rather than assuming
  width multiplier `w=8`. The training pool contains 259,550 residue-position samples before any
  cache exclusion, so `w=8` would create 7,680 features with only 33.8 position-samples per feature.
  Width selection must use participation ratio and dead-feature fraction as diagnostics and is
  expected to require a much smaller `w`. A high dead-feature fraction at this data volume is a
  sample-size consequence, not a biological or methodological finding.
- `[FINDING]` The approved pre-cache arithmetic for IRED uses all 895 pg_kfold fold-0 training
  variants at all 290 residues: 259,550 positions. At d=960, fp16, and two models, the realized
  activation payload is 996,672,000 bytes = 0.996672 decimal GB (0.928223 GiB), or 0.498336 GB
  per model. Test, layer-freeze reserve, round-held-out test, and plate variants are excluded.
  This is a budget calculation only; no activation cache was created.
- `[COST]` Gate reporting, prediction-persistence enforcement, the IRED conclusion, and the
  Step 4 budget calculation consumed 0 GPU-hours.
- `[FINDING]` Repository verification after the persistence changes passes all 217 tests in
  105.35 seconds. The new enforcement tests validate the common prediction schema, atomic CSV
  and parquet writes, checkpoint hashing, legacy exceptions, inference-only backfill, and the
  presence of condition predictions for every historical Step 2 result.

### v13 IRED activation cache and dictionary-width sweep (2026-10-25)

- `[ADJUST]` Replaced v12 with `CROSSFIT_Implementation_Plan_v13.docx` (SHA-256
  `1ac8f00a3cb6bfc90626aa9ecb796477132d0f48e481e60dc22e010535ce6b1d`). v13 is the sole
  implementation plan. Gate 2 moves from Step 5 to Step 7 because pooled sparse features do not
  exist until then; Step 5 now fits the dense references and pre-commits the comparison rule.
- `[DECISION]` Gate 2 uses pg_kfold test Spearman per dataset. Dense dominates only if its
  paired-bootstrap interval for the difference from the best sparse-feature arm excludes zero in
  the dense-favouring direction; sparse is competitive when the interval contains zero or favours
  sparse. Sample-efficiency curves are reporting-only and never decide the gate. This rule was
  written before any Step 7 sparse result was produced.
- `[ADJUST]` Dictionary expansion is named `width_multiplier` (`w`, with `m=w*d`) everywhere in
  Steps 4, 6, and 7. It is distinct from the Step 3 LoRA scaling factor, and no dictionary config
  uses the name alpha. The registered IRED sweep is `w in {1, 2, 4, 8}`.
- `[DECISION]` Fixed BatchTopK `k=32` before looking at any crosscoder result. This is the
  published BatchTopK operating point, is sparse relative to the 960-channel activation
  (`32/960=3.33%`), and remains constant across the width sweep so only dictionary width changes.
  Realized L0 is reported rather than assumed.
- `[DECISION]` The approved Step 4 cache contains all 290 positions for only the 895 variants in
  IRED's pg_kfold fold-0 training pool: 259,550 positions and 0.996672 decimal GB for two fp16
  models. Each model receives one fixed dataset/layer scalar making its training-pool mean
  activation norm `sqrt(960)`; test, reserve, round-held-out test, and plate variants are excluded.
- `[DECISION]` The IRED width curve is diagnostic, not an architecture-tuning opportunity. High
  dead fractions rising with `w` and low participation ratio are pre-registered as expected
  consequences of only 259,550 position samples. If every width is mostly dead, IRED is reported
  as unable to support a dictionary; the crosscoder architecture will not be changed to hide it.
- `[FINDING]` The first inference-only Gate 1 fold-0 backfill attempt exposed a device-placement
  defect before writing predictions: the reconstructed ESM-C backbone was on CUDA while its
  linear head remained on CPU. The checkpoint and training artifacts were unaffected. The
  backfill loader now moves the complete wrapper to the requested device and reruns inference
  from the beginning; no model is retrained and no partial prediction file is retained.
- `[ADJUST]` A formula audit after the first completed width sweep found that the implementation
  had used affine decoder layers with biases, while v13 specifies the matrix maps `D_evo z` and
  `D_fit z`. The symmetric self-pair test cannot expose this because the two biases remain equal.
  Those results and checkpoints are archived as invalid, `decoder_bias: false` is now asserted in
  both configs and tests, and the self-pair plus every width/seed run restarts from the beginning.
  The activation cache and Gate 1 inference backfill are unaffected.
- `[FINDING]` The approved IRED depth-26 cache completed with 259,550 positions. The fp16 tensor
  payload is 0.996672 decimal GB (0.996676 GB including NPY headers). Fixed normalization scalars
  are 0.0272291482 for E_evo and 0.0296749163 for E_fit, giving mean normalized norms 30.983911
  and 30.984206 against `sqrt(960)=30.983867`. The raw adapted/base mean-norm ratio is 0.917581.
  Token alignment and the real-adapter checkpoint hash both passed.
- `[FINDING]` The authoritative bias-free `(h_evo,h_evo)` self-pair null gives Delta minimum and
  maximum 0.5 with maximum absolute deviation 0.0 at tolerance `1e-7`. Realized L0 is exactly
  32. This clears the alignment/normalization check; it does not supply shuffled-null tau.
- `[FINDING]` The bias-free three-seed width sweep completed every registered run. Mean dead
  fractions at `w={1,2,4,8}` are `{0.3267,0.4993,0.6623,0.7815}`; participation-ratio effective
  ranks are `{122.8,112.8,99.1,95.0}`; samples/feature are `{270.4,135.2,67.6,33.8}`. Per-model
  FVE remains 0.9959-0.9987 and realized L0 remains 31.91-32.00. The monotone dead-fraction curve
  is the expected consequence of IRED's limited position count, not evidence against crosscoders.
- `[DECISION]` No IRED dictionary width is selected. `w=1` is not overcomplete and still has
  32.7% dead features; every overcomplete setting is at least 49.9% dead and worsens with width.
  Under v13's no-working-width rule, IRED cannot support an overcomplete dictionary at this data
  volume. Taxonomy, latent scaling, phi predictors, and Gate 2 are not run on a mostly-dead model.
- `[FINDING]` Gate 1 fold-0 checkpoint inference now persists 179 held-out rows each for the real
  and shuffled adapters in the common prediction schema. The remaining four registered folds and
  every sweep fold are still unrecoverable without retraining, which remains unauthorized.
- `[COST]` Authoritative runs consumed 0.02682 A100 wall-hours for caching, 0.03354 for the
  bias-free self-pair, and 0.10493 for the bias-free width sweep. The archived affine-decoder
  null and sweep consumed 0.14219 hours; the failed and corrected inference-only backfills used
  approximately 0.043 hours. Total GPU occupancy for this continuation was approximately 0.351
  A100 wall-hours. No seed, null, width, or epoch count was cut.
- `[FINDING]` Final repository verification passes all 225 tests in 94.85 seconds. The suite now
  enforces v13 as the sole plan, fixed `k=32`, registered width terminology, bias-free decoder
  matrices, exact self-pair symmetry, cache normalization, checkpoint-only backfill device
  placement, and the 179+179 persisted fold-0 prediction rows.

### v14 IRED width diagnostics and exploratory continuation (2026-10-31)

- `[ADJUST]` Replaced v13 with `CROSSFIT_Implementation_Plan_v14.docx` (SHA-256
  `3688be0d0bceadc955aa92fb18395c6487cf5ccc8822554d2b189030971e78d2`). v14 is the sole
  implementation plan. Step 8 now uses residue-level static aggregation, class-level tests,
  mutation-responsive codes, a six-block panel, and the revised semantic hypothesis.
- `[ADJUST]` The statement "IRED cannot support an overcomplete dictionary" was the consequence
  of a pre-committed stopping rule, not an empirical impossibility. For exploratory IRED work we
  proceed past it to distinguish data rank, BatchTopK competition, and optimization death. Gate 1
  remains failed, nothing here is confirmatory, and no later result may revise that verdict.
- `[FINDING]` The authoritative width checkpoints persist 20 scalar reconstruction losses but no
  per-epoch activation counts or intermediate states. Death timing cannot be reconstructed from
  final weights. The registered width/seed fits will therefore be rerun deterministically with a
  fixed-model full-pool activation-count pass at initialization and after every epoch; final state
  and loss traces must reproduce the existing checkpoints before those counts are accepted.
- `[DECISION]` The single optimization diagnostic uses random dead-feature resampling at `w=4`,
  seed 20261311, after epochs 5, 10, and 15. It changes no loss term and introduces no L1. A
  reduction of at least 0.10 in final dead fraction is pre-registered as substantial; a recovered
  `w=4` must also be no worse than the baseline `w=2` mean dead fraction to replace either default
  working width. Otherwise the exploratory working dictionaries are `w=1` and `w=2`.
- `[DECISION]` Before Part B, expect phi prediction to underperform the Step 2 one-hot arm on the
  external plate set because all Step 2 ESM-derived arms transferred at Spearman 0.23-0.44 while
  one-hot and AAIndex reached 0.53-0.57. Any reversal requires explanation and is not treated as
  an unqualified success.
- `[FINDING]` Dictionary-free PCA over all 259,550 training-pool positions gives raw activation
  participation ratios of 66.94 for E_evo and 61.24 for E_fit. The numbers of components needed
  for 90% variance are 148 and 162; for 99% they are 256 and 309. The 960-channel activations
  therefore have concentrated intrinsic variance before a dictionary is introduced. The prior
  sparse-code participation ratios of 122.83, 112.78, 99.12, and 95.02 are reported alongside
  this input-space reference but are not PCA dimensions in the same space.
- `[FINDING]` All 12 registered width/seed fits were replayed with full-pool activation counts at
  initialization and every epoch. Every loss trace and final parameter tensor exactly reproduced
  its authoritative checkpoint. Mean initially-dead fractions at `w={1,2,4,8}` are
  `{0.0316,0.1050,0.2238,0.3747}`, while mean final-dead fractions remain
  `{0.3267,0.4993,0.6623,0.7815}`. Of final-dead features, `{95.4%,84.9%,74.4%,66.1%}` had been
  active at least once. Most deaths at every width are therefore crowd-out during optimization,
  although never-active features rise strongly with width.
- `[FINDING]` The registered random dead-feature-resampling diagnostic at `w=4`, seed 20261311,
  reset 2,454, 2,524, and 2,549 full-pool-dead features after epochs 5, 10, and 15 and cleared
  their Adam moments without changing the objective. Final death was 0.66797 versus 0.66042 in
  the matched baseline, an absolute reduction of -0.00755. It fails the pre-registered 0.10
  recovery criterion and is worse than the `w=2` mean 0.49931, so random resampling did not
  rescue `w=4`.
- `[DECISION]` The two exploratory Part B dictionaries are `w=1` and `w=2`. Both will be run:
  `w=1` is the lowest-death non-overcomplete reference, while `w=2` tests the edge of supported
  overcompleteness. `w=4` and `w=8` are excluded after the failed registered recovery diagnostic.
  This choice does not revise Gate 1 or create confirmatory evidence from IRED.
- `[COST]` v14 Part A consumed 0.14059 A100 wall-hours, including raw PCA, all 12 instrumented
  exact-reproduction replays, and the `w=4` resampling run. The intervention fit itself used
  0.00971 hours. Cumulative documented IRED GPU occupancy through Part A is approximately 13.3
  A100 wall-hours, including retained superseded and diagnostic work recorded above.

### v15 effective-rank width correction and IRED Part B (2026-10-31)

- `[ADJUST]` Replaced v14 with `CROSSFIT_Implementation_Plan_v15.docx` (SHA-256
  `fcdbaf82f2774176717dd541d2b6f3279f5505726058ff9e9b28924214c81422`). v15 is the sole
  implementation plan. Overcompleteness is measured against raw activation effective rank, not
  ambient dimension `d`.
- `[ADJUST]` The earlier description of `w=1` as non-overcomplete was wrong. With 960 features
  and raw participation ratios 66.94 for E_evo and 61.24 for E_fit, `w=1` is already 14.34x and
  15.68x overcomplete; `w=2` is 28.68x and 31.35x. All future datasets, including GB1, report
  dictionary width divided by raw activation effective rank.
- `[ADJUST]` The GB1 base/real/shuffled effective-rank comparison remains a required control
  measurement but is no longer a directional hypothesis or pre-check. IRED already shows that
  rank contraction is generic to fine-tuning, not label-driven; GB1 repeats the control only to
  verify that the nuisance effect is measured there.
- `[FINDING]` On the matched 59-variant layer-freeze reserve at depth 26 (17,110 residue
  observations), participation ratios are 66.9329 for E_evo, 60.6730 for the real adapter, and
  60.2912 for the step-matched shuffled-label adapter. Relative to E_evo, the contractions are
  9.35% real and 9.92% shuffled. The shuffled adapter contracts at least as much despite its CKA
  of 0.9811 versus 0.9189 for the real adapter. IRED's effective-rank contraction is therefore a
  generic consequence of fine-tuning, not label-driven dimensionality reduction. The existing
  training-pool ranks 66.94 and 61.24 remain the width-denominator estimates; the reserve triplet
  is the matched causal comparison.
- `[DECISION]` Step 6 uses `q=0.05`. After excluding dead and own-decoder-norm-floor failures,
  fit one Beta distribution to each shuffled-null run's Delta values, never pool runs, compute
  one-sided p-values separately for the lower and upper tails, and apply Benjamini-Hochberg
  within each tail. The Delta value at the BH cutoff defines that tail's threshold. Take the
  more conservative threshold across null runs, and report QQ diagnostics plus leave-one-seed-out
  variability.
- `[DECISION]` Latent scaling is a conjunctive artifact gate. A prospective emergent feature is
  excluded if either its error ratio or reconstruction ratio is flagged against its own Step 6
  shuffled null at `q=0.05`. Evidence of either complete shrinkage or latent decoupling is enough
  to withhold the headline class.
- `[DECISION]` Decoder rotation and activation-frequency change each receive a dataset- and
  backbone-specific threshold from the shuffled-label runs, using null p-values and
  Benjamini-Hochberg at `q=0.05`. Neither threshold may be imported across datasets or backbones.
- `[DECISION]` Part B reports `emergent candidate`, defined as Delta-emergent and passing both
  latent-scaling tests but not yet causally validated. Every Part B class proportion is
  provisional. The final `emergent` label is unavailable until Part C supplies causal patching.
- `[DECISION]` Run Step 9 regardless of IRED's Gate 2 verdict and report that verdict unchanged.
  This is an IRED-only resource-allocation override: the protocol is already built, compute is
  negligible, and the requested scientific output is the sparse-feature prediction increment.
  It does not weaken a validity criterion. On GB1 and later datasets, obey Gate 2's conditional
  branch exactly; Gates 1 and 3 remain validity gates everywhere.
- `[FINDING]` The original step-matched-null provisional taxonomy, superseded by v16, is strongly
  width-dependent. Across
  three seeds, `w=1` assigns 32.67% dead, 28.16% emergent candidate, 1.63% evolution-only,
  36.04% rotated, 0% re-weighted, and 1.49% conserved. At `w=2` the corresponding values are
  49.93%, 0%, 0.94%, 47.81%, 0.05%, and 1.27%. At `w=2`, no feature reaches reconstruction-ratio
  significance after BH correction across 1,920 tests. This is a power limitation, not evidence
  that emergent features are absent, and class proportions are not comparable across widths
  because `w=1` corrects 960 tests while `w=2` corrects 1,920. All labels remain provisional.
- `[FINDING]` Real crosscoder reconstruction is high but differs by model and is reported
  separately: mean FVE is 0.9985 for E_evo and 0.9959 for E_fit at `w=1`, and 0.9986 and 0.9968
  at `w=2`. These aggregate values do not override dead-feature, norm-floor, or latent-scaling
  exclusions.
- `[FINDING]` Gate 2 returns `sparse_competitive`. The validation-selected dense reference
  (`dense_fit_mean`) and sparse arm (`phi_w1_max_all_live`) score 0.5007 and 0.5495 Spearman on
  pg_kfold test. The paired-bootstrap dense-minus-sparse difference is -0.04882 with 95% interval
  [-0.09564, -0.00153]. This supports competitiveness against the validation-selected reference,
  not superiority over every dense probe.
- `[FINDING]` The Step 9 validation-selected arm is `phi_w1_max_all_live` with MLP readout. Its
  Spearman is 0.5495 on pg_kfold test, 0.5652 round-held-out, and 0.5002 on the external plate
  set (plate 95% bootstrap interval [0.2442, 0.7094]). Sparse max pooling improves on the earlier
  dense ESM plate range but remains below the registered one-hot/AAIndex range. Several
  non-selected narrow `w=2` arms reach plate point estimates 0.56-0.57 with wide intervals;
  because they were not validation-selected and rely on a width-unstable taxonomy, they are
  post-hoc diagnostics rather than headline successes.
- `[FINDING]` The headline expressivity gap is `R2(L4)-R2(L0)=0.152725` on pg_kfold test. L0,
  L1.5, L1, and L4 test R2 values are -0.0481, 0.1646, -0.1604, and 0.1046. The interaction gate
  failed (`H^2=0.9357`, permutation `p=1.0`, 200 permutations), so L2 was not fitted. The
  reporting-only sample-efficiency curve rose from mean test Spearman 0.406 at N=16 to 0.539 at
  N=512 and selected nothing.
- `[FINDING]` One Step 5 delta-PCA fold and one Step 9 arm were interrupted. The runner now
  requires both a complete checkpoint and aligned held-out prediction file before resuming;
  otherwise it reruns the fold from the beginning. Every reported result completed its full
  registered fit, and no partial state or partial metric was accepted.
- `[COST]` Part B's instrumented IRED GPU wall time was 0.05253 hours for the shuffled activation
  cache, 0.03295 for six shuffled-null crosscoders, 0.35732 for dense probes and feature-set
  models, 0.09481 for the expressivity ladder, and 0.01290 for sample efficiency: 0.55052 hours.
  The lightweight phi-cache and taxonomy-statistic passes predated wall-time instrumentation and
  are not included; no search, seed, null, or reported arm was cut. The ten-fit ladder timing
  projected 0.105 hours, so the registered search count required no budget reduction.

### v15 IRED Step 8 preparation (2026-11-01)

- `[DECISION]` The supplied Hollfelder residue designations are encoded as separate tracks:
  catalytic proposed, substrate direct contact, NADPH direct contact, NADPH proximal/coupled,
  and evolutionary peak. The evolutionary-peak track is a positive control only and is never
  pooled into a claim-bearing functional block. Experimentally observed, crystallographic, and
  docking-derived contacts retain separate provenance; no docking track is inferred in the
  absence of docking input.
- `[FINDING]` The 290-residue IRED campaign wild type aligns at 100% identity and 100% coverage
  to canonical 5OCM positions 1-290; 5OCM has one extra terminal Ala at position 291. Every
  supplied residue maps without mismatch. W178 and H244 map exactly to 5OCM W178 and H244 but
  are distinct from the entry's contextual W208 and H238, so no residue was substituted.
- `[DECISION]` RCSB metadata defines the deposited six-chain 5OCM coordinate entry as three
  biological dimers: A/F, B/C, and D/E, not one biological hexamer. Structural contacts and
  catalytic distances use all three biological copies and allow cross-subunit contacts within
  each dimer, but never include crystal contacts between different dimers. Every contact records
  the biological assembly and both chains of origin.
- `[DECISION]` Functional shells are mutually exclusive in the order catalytic, first shell
  (<=5 A), second shell (>5 to <=8 A), distal functional network, and generic distal. The distal
  network contains the supplied substrate/cofactor mechanistic residues not already absorbed by
  a geometric shell; evolutionary-peak controls are excluded. The realized counts are 3, 37,
  36, 3, and 211 residues.
- `[DECISION]` Mutation-responsive codes use the exact arithmetic `Delta z_j(x)=z_j(x)-z_j(WT)`.
  Because BatchTopK is batch-contextual, each 290-position sequence and the wild type are encoded
  independently so competition is fixed within one sequence before subtraction. The cache is
  sparse, pod-only, and indexed by persisted variant rows; residue-level means and a manifest are
  the only report artifacts. No enrichment, patching, or oracle validation is authorized in this
  preparation stage.
- `[FINDING]` The <=5 A crystallographic proxy tracks contain 174 NAP contact rows over 29 unique
  sequence positions and 82 9RH ketone-analogue rows over 14 positions. Of these, 24 NAP and 42
  9RH rows cross the biological dimer interface, confirming that a single-chain calculation would
  discard relevant geometry. The 9RH track remains explicitly a proxy for the campaign imine.
- `[FINDING]` Residue-level mutation responses were cached for all 1,182 campaign variants at
  both working widths. `w=1` has shape 342,780 x 960 with 17,961,806 nonzeros (5.46%); `w=2` has
  shape 342,780 x 1,920 with 18,518,511 nonzeros (2.81%). The compressed pod-only cache is 177 MB.
  Every row is indexed to the persisted curated `variant_row`; both crosscoder and adapter hashes
  match the authoritative Step 7 and Gate 1 checkpoints.
- `[COST]` Wild-type extraction and complete `Delta z` caching at `w=1,2` consumed 0.04321 L40
  wall-hours. No enrichment, patching, oracle evaluation, or feature inspection was run.
- `[FINDING]` Final Part B and Step 8-preparation verification passes all 242 tests in 99.57
  seconds. The suite includes hard assertions for the 290-position 5OCM mapping, every supplied
  residue, official three-dimer assembly membership, shell precedence, and mutant-minus-WT code
  arithmetic. Two expected constant-input warnings verify that undefined rank correlations are
  recorded as undefined rather than silently replaced.

### IRED taxonomy and Gate 2 audits (2026-11-01)

- `[FINDING]` Gate 2's `sparse_competitive` verdict is accepted as a positive result. The
  validation-selected `dense_fit_mean` and `phi_w1_max_all_live` arms score 0.5007 and 0.5495
  on pg_kfold test, and the paired dense-minus-sparse interval [-0.09564, -0.00153] barely
  excludes zero relative to that selected reference. For audit, all dense references are reported:
  validation/test Spearman is
  0.6231/0.5007 for h_fit mean, 0.6077/0.5543 for h_evo mean, and 0.3948/0.4882 for delta PCs.
  h_evo is 0.0048 ahead of sparse on test. Selecting it after reading test would violate the
  pre-committed validation-only selection rule and does not change the verdict, but the claim is
  “sparse features are competitive with dense probes,” never “sparse beats dense.”
- `[FINDING]` The original latent-scaling nulls were matched by dictionary width. The real-pair
  reconstruction-ratio medians are 0.7849-0.7942 at `w=1` and 0.7797-0.7897 at `w=2`, versus
  step-matched-null medians 0.9295-0.9360 and 0.9301-0.9322. At `w=2`, no feature reaches BH
  significance against every null run, so the conservative lower cutoff becomes `-inf`. This is
  reduced power across 1,920 tests, not a mechanical distribution shift or evidence of absence.
- `[ADJUST]` The v15 intermediate recalibrated decoder-cosine rotation against a rotation-only drift-matched
  shuffled-label null because the original step-matched null had substantially lower global
  drift. Delta, latent-scaling, norm-floor, and activation-frequency thresholds remain on their
  registered nulls in that archived intermediate only; v16 supersedes it with matched-drift
  calibration for every threshold. The null uses registered rank 16, learning rate 0.0001, weight decay
  0.01, and shuffled labels; it reached drift 0.6484 after 575 optimizer steps versus real
  drift 0.6338, a 2.3% mismatch.
- `[FINDING]` In the archived cosine-only intermediate, drift matching reduces mean provisional rotated fractions from 36.04% to 7.12%
  at `w=1` and from 47.81% to 10.45% at `w=2`, confirming that most original rotation calls
  re-reported global drift. Original rotation-stage flags were near-saturated across quintiles.
  Residual drift-matched flags concentrate at high activation frequency and lower decoder norm,
  so the remaining signal is feature-structured but retains a decoder-norm caveat. Corrected
  provisional conserved fractions are 30.38% and 38.39% at `w=1` and `w=2`.
- `[FINDING]` The IRED expressivity gap is `R2(L4)-R2(L0)=0.152725`. The additive readout used
  for attribution in Steps 8 and 11 is materially weaker than the nonlinear ceiling, so those
  attributions describe the weaker additive model. Separately, IRED fitness requires nonlinear
  sparse-feature logic under this ladder.
- `[ADJUST]` The validation-selected Step 9 arm uses max pooling, while the plan's displayed
  notation defines `phi_j` as a residue sum. Computation and downstream interfaces remain valid;
  the notation must distinguish max-pooled and sum-pooled arms rather than describing the
  selected result with the sum-only formula.
- `[COST]` The completed drift-matched adapter, activation cache, six null crosscoders, and
  calibration used 0.10186 L40 wall-hours; the persistence-only replay from complete artifacts
  used 0.02423 hours. One preceding 30-epoch non-crossing attempt used approximately 0.13 hours
  and produced no accepted checkpoint or statistic. Total diagnostic cost was approximately
  0.26 L40 hours. No Part C enrichment, patching, or oracle validation was run.
- `[FINDING]` Post-audit verification passes all 243 tests in 103.83 seconds. The two warnings
  are the expected constant-input rank-correlation checks; no partial fit or Part C result was
  admitted.

### v16 matched-drift recalibration (2026-11-01)

- `[ADJUST]` `CROSSFIT_Implementation_Plan_v16.docx` replaces v15, which was deleted. The
  matched-drift shuffled-label null is now unconditional and calibrates Delta, error ratio,
  reconstruction ratio, decoder cosine, and activation-frequency change. The earlier
  cosine-only matched-drift result remains archived but is not the headline taxonomy.
- `[DECISION]` When step-matched and drift-matched nulls disagree, the headline taxonomy is the
  one with fewer features across emergent-candidate, evolution-only, rotated, and re-weighted
  classes. Drift-matched has fewer shifted features for every real seed at both widths and is
  the IRED headline.
- `[FINDING]` Mean step-matched versus drift-matched provisional proportions are, at `w=1`:
  emergent candidate 28.16% versus 0%, evolution-only 1.63% versus 0.03%, rotated 36.04% versus
  25.00%, re-weighted 0% versus 0.07%, and conserved 1.49% versus 42.22%. At `w=2`, the matched
  values are 0%, 0.14%, 10.76%, 0.05%, and 39.11%, with dead fractions unchanged at 32.67% and
  49.93%.
- `[FINDING]` The zero emergent-candidate counts are power limitations, not evidence of absence.
  At `w=1`, two seeds have no conservative symmetric Delta cutoff because one tail has no BH
  discoveries, and all three have no reconstruction-ratio discoveries against every null run.
  At `w=2`, the reconstruction gate likewise has none while correcting 1,920 simultaneous tests
  rather than 960. Class proportions are not directly comparable across widths because the
  multiple-testing burden scales with dictionary size.
- `[FINDING]` Reconstruction-ratio effect sizes show closely overlapping real/null medians:
  0.7849-0.7942 versus 0.7767-0.7909 at `w=1`, and 0.7797-0.7897 versus 0.7818-0.7896 at `w=2`.
  Real pairs have heavier lower tails, but no feature survives the conservative all-null BH
  requirement. The `-inf` cutoff records insufficient power after correction, not zero effect.
- `[FINDING]` Rotation survives exclusion of low-norm decoder columns. Among features at or above
  both decoder medians, 27.3-32.2% are rotated at `w=1` and 17.4-18.5% at `w=2`.
  Activation-frequency concentration remains (Spearman 0.40-0.54 and 0.23-0.36); the highest
  frequency quintile has 35-73% rotation versus 2-11% in the lowest. Surviving rotation is not
  explained by poorly determined cosine on low-norm columns.
- `[DECISION]` Gate 2 remains `sparse_competitive`. Sparse is 0.0488 ahead of the
  validation-selected dense reference with an interval barely excluding zero, but 0.0048 behind
  the strongest dense arm on test. Report only that sparse features are competitive with dense
  probes; never claim that sparse features beat dense probes.
- `[DECISION]` Part C does not begin because v16 leaves zero `w=1` emergent candidates, failing
  the user's explicit condition for enrichment. Prepared annotations and Delta-z caches remain
  intact. No enrichment, patching-budget calculation, ablation, patching, or oracle validation
  was run.
- `[COST]` Recomputing all five threshold families and the full taxonomy from completed caches
  and checkpoints used approximately 0.016 L40 wall-hours. No model or crosscoder was retrained.
- `[FINDING]` v16 recalibration verification passes all 243 tests in 106.23 seconds. The two
  warnings are the registered constant-input rank-correlation checks.

### v17 pooled-null recalibration (2026-11-08)

- `[ADJUST]` `CROSSFIT_Implementation_Plan_v17.docx` replaces v16, which is deleted. The
  per-null-run conjunction used in v16 is superseded before inspecting the v17 taxonomy. It
  made the effective error rate depend on the arbitrary number of null seeds. Every statistic
  now uses one distribution fitted to eligible samples pooled across drift-matched null runs,
  followed by tail-specific BH at q=0.05. Leave-one-null-seed-out thresholds and QQ plots are
  uncertainty diagnostics, not additional gates. The latent-scaling ratios remain conjunctive.
- `[DECISION]` Five independently initialized real-pair crosscoders are the replication set at
  each IRED working width. Unexpected results are answered by their mean and spread, not by a
  dataset-specific calibration-rule change. The full v16 proportions remain above as an
  explicitly superseded audit record rather than an alternative result to prefer.
- `[FINDING]` The IRED expressivity gap is `R2(L4)-R2(L0)=0.152725`. The additive readout used
  for attribution in Steps 8 and 11 is materially weaker than the nonlinear ceiling, so its
  attributions describe that weaker model. Separately, the IRED ladder indicates that fitness
  requires nonlinear sparse-feature logic.
- `[FINDING]` The validation-selected sparse arm uses max residue pooling, while the plan's
  displayed notation `phi_j = sum_i z_ij` describes sum pooling. Downstream computation is not
  affected, but reports and notation must name the pooling operator explicitly.
- `[DECISION]` Gate 2 is a routing gate. The interval against the validation-selected dense arm
  justified running Step 9; substantive reporting must show that comparison beside the strongest
  dense test arm and the classical plate-set arms. The permitted conclusion is competitive with
  dense probes and classical encodings, never improved over them.
- `[PREDICTION]` Before GB1 taxonomy is run: if rotation without emergence replicates on GB1,
  treat it as a general property of fitness adaptation under LoRA. If GB1 has emergent features,
  interpret IRED's null result in light of its data volume. This comparison is pre-committed and
  costs no additional run once GB1 is analyzed.
- `[ADJUST]` The complete v17 five-seed taxonomy supersedes the v16 three-seed, runwise-conjunction
  proportions. At `w=1`, mean (SD) proportions are dead 32.94% (1.41%), emergent candidate 0%,
  evolution-only 0.02% (0.05%), rotated 25.23% (1.55%), re-weighted 0.04% (0.06%), and conserved
  41.77% (1.18%). At `w=2`, they are 50.83% (1.33%), 0%, 0.13% (0.06%), 11.36% (0.75%), 0.05%
  (0%), and 37.63% (0.96%), respectively. These are the IRED results under the corrected rule;
  the absence of emergent candidates is retained rather than interpreted away.
- `[FINDING]` Rotation is the only shifted class populated in all five `w=1` dictionaries and is
  therefore the pre-registered Step 8 target. It remains well replicated under pooled-null
  calibration, whereas evolution-only and re-weighted calls are too sparse for a class-level
  semantic comparison. Conserved is the registered comparison class.
- `[COST]` Completing four missing real crosscoders (seeds 20261314-20261315 at `w=1,2`) and
  diagnosing all ten real checkpoints used 0.02940 A100 wall-hours. Streaming all statistics for
  the five-seed, two-width pooled-null taxonomy used 0.10148 A100 wall-hours. Complete checkpoints
  were loaded; no completed run was retrained and no partial result was reported.
- `[FINDING]` The nominal Step 2/Step 5 dense-reference discrepancy is resolved. Step 2
  `esm_mean_ridge` uses ESM-C's final layer, fits 761 rows after reserving 134 for conformal
  calibration, and selected weight decay 1e-4 for 16 epochs. Step 5 `dense_evo_mean` uses
  analysis layer 26, all 895 training-pool rows, weight decay 0.01, and 21 epochs. Their
  validation Spearman values are nearly equal (0.607648 and 0.607708), but they are not duplicate
  arms; the 0.522426 versus 0.554287 test values do not expose an implementation error.
- `[FINDING]` Gate 2's routing comparison remains sparse-competitive. Sparse minus the
  validation-selected dense reference is +0.04882 with paired 95% interval [0.00153, 0.09564];
  sparse minus the strongest dense test arm is -0.00478 with interval [-0.07623, 0.06314]. On
  the 54-variant plate set, paired intervals against all four classical ridge arms include zero.
  Report only competitiveness: second on pg_kfold test, first round-held-out, and statistically
  indistinguishable from the classical arms on plate.
- `[FINDING]` The v17 five-seed residue-level semantic test selects rotation as the target and
  conserved as comparison. After FDR across nine claim-bearing families, rotated features are
  enriched at catalytic (q=0.0364), first-shell, second-shell, and substrate-contact positions
  (each q<=0.000225) and depleted at generic-distal positions (q=0.000225). Cofactor and 5OCM
  ketone-proxy tracks are not significant. The campaign-derived evolutionary-peak positive
  control is positive in every seed (p=0.00010) and remains ineligible for a claim.
- `[FINDING]` Rotated and conserved top-activating positions have distinct shell distributions:
  mean across-seed total-variation distance 0.0808, blocked-permutation p=0.000500. The primary-
  seed mutation-responsive panel uses 289 distal-mutant variants in 257 exact mutated-position
  background groups and reproduces stronger rotated responses at catalytic/first/second shells.
  This mutation panel is single-seed and exploratory.
- `[DECISION]` Causal patching remains unstarted pending approval. The planned scope is the 116
  rotated primary-seed features above both decoder medians across 64 stratified pg_kfold training-
  pool variants: 7,424 candidate partial forward passes plus 192 no-op/dead/random controls,
  7,616 total. Error-preserving splicing inside E_fit and control verification remain mandatory.
- `[COST]` Five-seed residue-profile extraction and the Step 8 semantic statistics used 0.01748
  A100 wall-hours. Approximate cumulative documented IRED occupancy is 14.32 A100 wall-hours,
  including retained superseded and diagnostic runs. No patching or oracle computation ran.

### IRED emergent-resolution audit and approved causal scope (2026-11-08)

- `[DECISION]` The zero-emergent result is not interpreted as an absence until the empirical-null
  resolution is adequate. The registered dead/norm-floor filter leaves about 603 eligible
  features per `w=1` null run, so 20 total runs would provide only about 12,000 observations and
  could not reach the first 960-test BH cutoff (`5.2083e-5`). Thirty-two realized runs produced
  18,458 eligible observations, below the exact requirement of 19,199, so the replication set is
  extended to 34 total matched-drift null seeds to cross the actual empirical-resolution floor.
  This changes replication only: q=0.05, the latent-scaling conjunction, and every taxonomy rule
  remain fixed.
- `[DECISION]` Latent error and reconstruction ratios retain pooled empirical calibration.
  Reconstruction ratios can be negative or exceed one, so fitting a Beta to them would violate
  its support. A generalized-Pareto fit below the pooled empirical fifth percentile is added only
  as the requested tail-fit diagnostic and sensitivity check. Bounded Delta, cosine, and
  frequency statistics retain pooled-Beta calibration.
- `[ADJUST]` The approved causal panel supersedes the earlier planned-only 64-variant budget.
  It fixes the target at the 116 rotated primary-dictionary features above both decoder medians
  and uses 256 pg_kfold fold-0 training-pool variants stratified by clipped mutation count and
  fitness quartile: 29,696 candidate and 768 no-op/dead/random control partial forward passes,
  30,464 total. Patches must occur inside `E_fit` with the original fit reconstruction residual
  spliced back, and candidate effects may be computed only after all three hygiene controls pass.
- `[FINDING]` The robust IRED semantic profile is a spatial gradient: the rotated-minus-conserved
  effect peaks in the second shell (+0.399 SD), remains positive in the first shell (+0.152), and
  reverses to depletion at generic-distal positions (-0.267). Catalytic residues are only three
  positions and their interval barely excludes zero; the single substrate-contact position cannot
  carry a standalone claim. A hypothesis for later testing, not a conclusion, is that directed-
  evolution campaigns rarely retain catalytic-site substitutions because they abolish activity,
  leaving too little variation there for an adapted feature to encode.
- `[FINDING]` All NADPH-related semantic tracks are null, including direct contact, supplied
  proximal/functionally coupled positions, and the crystallographic cofactor context. This is an
  informative negative because the dETM analysis attributes the A72T and G37C dead ends to
  competing NADPH interactions; rotated-feature enrichment does not recapitulate that mechanism.
- `[FINDING]` The `w=1` emergent-candidate count remains zero after correcting empirical
  resolution. Thirty-four matched-drift null seeds supply 19,609 eligible observations, making
  the minimum empirical p-value `5.0994e-5`, finer than the first 960-test BH cutoff
  `5.2083e-5`. Zero is reproduced across all five real dictionary seeds under both the registered
  empirical latent-scaling ratios and a lower-tail generalized-Pareto sensitivity analysis.
  The result is no longer attributable to the three-null p-value floor.
- `[FINDING]` The reconstruction tail fit is not the binding constraint. Reconstruction ratios
  are unbounded, so Beta is invalid; the diagnostic lower-5% generalized-Pareto fit gives top-ten
  fitted p-values within 0.259 log10 units of their pooled empirical p-values. The primary seed's
  most extreme ratio has empirical p=0.001785 and fitted p=0.001132, far above the first BH
  threshold. The expanded null changes mean rotation only from 25.23% to 25.06%.
- `[COST]` Training the 31 additional matched-drift null crosscoders and re-reading completed
  checkpoints used 0.1412 A100 wall-hours. The successful 34-null/five-real statistic pass used
  0.2427 hours. A preceding fully streamed pass consumed approximately 0.22 hours before its
  exact-resolution guard correctly rejected the 18,458-observation 32-null pool. Total new GPU
  occupancy is approximately 0.604 hours; cumulative documented IRED occupancy is approximately
  14.92 A100 wall-hours. Causal patching was not started before the user terminated the pod.

### IRED causal-patching hygiene after pod reconnection (2026-11-08)

- `[ADJUST]` The ESM-C loader now requests the registered Hugging Face revision directly when
  constructing the model. The previous implementation downloaded the moving `main` revision and
  checked its cache ref afterward, which correctly stopped a mismatched run but could not load the
  historical pin. The causal run used `biohub/ESMC-300M` revision
  `e4bac860f0c502cd80f3aeac3d0ce6524c6627cb` and the hash-verified Gate 1 adapter.
- `[FINDING]` Causal patching stopped in the complete 256-variant hygiene phase before any of the
  116 rotated candidate features was ablated. The no-op maximum raw prediction effect was
  `7.95424e-05`, above the registered `1e-05` tolerance, so status is `failed_no_op`. The inactive
  dead-feature control had the same numerical maximum and mean standardized effect `-8.78e-06`;
  the random-live control was near zero (mean standardized effect `-0.000357`, mean absolute
  `0.000410`). The freshly evaluated layer-26 state differed from the cached state by as much as
  `2284` raw activation units despite exact weights, indicating that the replacement runtime is
  not numerically interchangeable with the cache-producing runtime. No tolerance, statistic, or
  patching result was changed to make the control pass.
- `[COST]` Failed pre-inference pin checks plus the complete control-only pass used approximately
  0.03 RTX 6000 Ada wall-hours. Historical occupancy is reported separately as approximately
  14.92 A100 wall-hours; heterogeneous device-hours are not added. No causal effect was computed
  or reported.

### IRED activation-cache diagnosis (2026-11-12)

- `[ADJUST]` Causal no-op and dead-feature hygiene are now evaluated in the same
  baseline-prediction-SD units as the random-feature control. The earlier raw `1e-05` tolerance
  tested float32 accumulation rather than splice correctness and is superseded for a reason
  stated independently of its result. The retained raw maximum `7.95424e-05` is `0.0010763` SD,
  below the common standardized tolerance `0.05`; the hygiene phase passes. Both units remain in
  the artifact. Candidate ablations remain unrun because the separate cache-depth audit blocks
  them. The earlier `[FINDING]` labeling this value `failed_no_op` is retained above only as the
  superseded audit record.
- `[FINDING]` The `2284` cache discrepancy is a depth-label/storage mismatch, not normalization
  and not a different numerical model. `Cache/step3/IRED/full_selected_layer_base.npy` matches
  current depth 29 at Pearson/Spearman `0.999685/0.999558`, scale `0.999504`; its depth-26
  correlations are `0.446689/0.311353`. The adapted file likewise matches depth 29 at
  `0.999788/0.999762`, scale `0.997500`, but depth 26 only at `0.175909/0.128945`. These are the
  original final-layer files already marked superseded in the Step 3 handoff, but later configs
  consumed them as depth 26.
- `[FINDING]` The authoritative Step 4 training-only cache is correctly depth 26. After reversing
  its fixed scalars, current single-sequence re-extraction agrees at Pearson/Spearman
  `0.999788/0.999726` for base and `0.999605/0.999529` for adapted, with unit through-origin
  scales. The small remaining scatter has the expected float16-storage signature. The all-layer
  reserve also matches every current same depth above Pearson `0.99968`, and cached depth 26
  matches current depth 26 best. Thus Step 6/7 nulls, crosscoders, thresholds, and the 25.23%
  five-seed rotated taxonomy remain valid.
- `[ADJUST]` Step 5 dense probes, Gate 2, Step 9 phi/expressivity/sample-efficiency results, and
  Step 8 semantic and mutation-responsive enrichment are superseded pending correct depth-26
  recomputation. Their campaign paths point to the depth-29 Step 3 files while their reports call
  the inputs depth 26. The earlier shell enrichment is withdrawn as a current biological result;
  underlying files are retained for audit. No full cache was re-extracted and no downstream
  result was recomputed in this diagnosis.
- `[FINDING]` Runtime provenance is incomplete for the A100 cache-producing container: Torch,
  CUDA, TF32 flags, realized attention kernel, and Transformer Engine availability were not
  persisted. The diagnostic used an RTX PRO 4500 Blackwell with Torch 2.11.0+cu130, CUDA 13.0,
  matmul TF32 off, cuDNN TF32 on, no autocast, and no Flash-Attention, xFormers, or Transformer
  Engine. The preceding hygiene run used an RTX 6000 Ada. Correctly identified layers agree
  across devices, so device precision does not explain the large mismatch.
- `[COST]` The single-sequence, all-depth cache diagnosis used approximately 0.02 RTX PRO 4500
  Blackwell wall-hours. No patching, full-corpus extraction, crosscoder training, or predictor
  refit ran.

### v18 cache provenance and depth-26 recomputation pre-commitment (2026-11-12)

- `[DECISION]` Representation depth is now a dataset-independent convention. Step 2 embedding
  baselines (`esm_mean`, `esm_mutpool`, `esm_residue`, and `esm_residue_delta`) use ESM-C's final
  output layer. Analysis-layer objects—Step 5 dense probes, Step 4/6/7 crosscoder inputs, Step 9
  phi, Step 8 semantics, and causal patching—use the frozen selected hidden-state depth, which is
  26 for IRED. Every reported number must name its depth. The withdrawn IRED Step 5/9 artifacts
  violated the analysis-layer half of this convention by loading superseded depth-29 tensors;
  they remain non-comparable to Step 2 until recomputed at depth 26.
- `[ADJUST]` The superseded depth-29 base, adapted, and coupled row-index files were moved from
  active `Cache/step3/IRED/` paths to
  `Cache/archive/IRED_depth29_superseded_2026-11-12/` before recomputation. Active Step 8/9
  configuration now resolves only the explicitly named depth-26 cache. An invariant test forbids
  the old generic paths and all archive paths in active configuration.
- `[DECISION]` v18's outcome-neutral recomputation rule is pre-committed before extraction. The
  withdrawn Gate 2, Step 9 expressivity, and Step 8 enrichment results are withdrawn, not
  refuted: the dictionary and depth-26 taxonomy remain valid, but those downstream inputs were
  wrong. Whatever the depth-26 recomputation returns is the IRED result. Reproduction is the
  expected consequence of fixing the input and is not described as confirmation; failure to
  reproduce is reported without changing the analysis. Gate 2's rule and all Step 8/9 selection
  procedures remain unchanged.
