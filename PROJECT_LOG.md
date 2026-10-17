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
