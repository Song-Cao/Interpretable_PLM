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
