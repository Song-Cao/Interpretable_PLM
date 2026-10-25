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
