# CROSSFIT Step 1 — Dataset curation report

**Deliverable:** ten ProteinGym DMS assays in one schema, with QC, sigma provenance,
four split schemes, two reservations, a uniform loader, and a 147-test suite.

**Date:** 2026-08-28 · **Config:** `Code/config/step1.yaml` · **Log:** `PROJECT_LOG.md`

---

## 1. Pinned sources

| source | identifier |
|---|---|
| ProteinGym 2.0 bundles | HF `ProteinGym/ProteinGym2.0` @ `2e5c7b10c3dda55ca7295c5f1334887801294131` (211 `.pgdata`, 4.85 GB total) |
| ProteinGym v1 reference | `OATML-Markslab/ProteinGym` @ `62eab5b3c83a3727b5b57bfbe377b24ef7545ccc`, `reference_files/DMS_substitutions.csv`, sha256 `a8f498011532a74a…` |
| pinned at | 2026-08-27T17:30:53Z |

Recorded in `Data/raw/reference/sources_lock.json`. Ten bundles fetched
(204 MB), each byte-size-verified against the pinned file manifest;
`Data/raw/proteingym2/download_manifest.json` records what was actually
written. HuggingFace publishes an xet content-defined-chunking oid rather
than a plain sha256, so the file hash cannot be recomputed locally — size
mismatch against the pinned manifest is treated as a hard failure instead.

## 2. The ten assays

Ordered by curated size. `pos cov` = fraction of WT positions that carry at
least one mutation; `B2 folds` = usable unseen-position folds (§6).

| assay | protein | phenotype | organism | len | curated n | single / multi | max k | pos cov | B2 folds |
|---|---|---|---|---:|---:|---|---:|---:|---:|
| `CCDB_ECOLI_Adkar_2012` | CcdB | Activity | *E. coli* | 101 | 1,176 | 1,176 / 0 | 1 | 0.990 | 5 |
| `RL40A_YEAST_Roscoe_2014` | Ubiquitin | Activity | *S. cerevisiae* | 128 | 1,380 | 1,380 / 0 | 1 | 0.586 | 5 |
| `FECA_ECOLI_Tsuboyama_2023_2D1U` | FecA domain | Stability | *E. coli* | 72 | 1,886 | 1,219 / 667 | 2 | 1.000 | 5 |
| `ESTA_BACSU_Nutschel_2020` | EstA | Stability | *B. subtilis* | 212 | 2,172 | 2,172 / 0 | 1 | 0.793 | 5 |
| `GCN4_YEAST_Staller_2018` | Gcn4 AD | Binding | *S. cerevisiae* | 281 | 2,638 | 33 / 2,605 | 44 | 0.157 | **0** |
| `TPMT_HUMAN_Matreyek_2018` | TPMT | Expression | *H. sapiens* | 245 | 3,648 | 3,648 / 0 | 1 | 0.984 | 5 |
| `OXDA_RHOTO_Vanella_2023_activity` | DAAO | Activity | *R. gracilis* | 364 | 6,396 | 6,396 / 0 | 1 | 1.000 | 5 |
| `RASK_HUMAN_Weng_2022_abundance` | KRAS | Expression | *H. sapiens* | 188 | 26,012 | 3,066 / 22,946 | 2 | 0.995 | 5 |
| `GFP_AEQVI_Sarkisyan_2016` | avGFP | Activity | *A. victoria* | 238 | 51,714 | 1,084 / 50,630 | 15 | 0.979 | 5 |
| `SPG1_STRSG_Olson_2014` | GB1 | Binding | *Streptococcus* sp. G | 448 | 536,962 | 1,045 / 535,917 | 2 | 0.123 | 5 |

Total 633,984 variants. Size profile: 3 assays at 500–2k, 4 at 2k–10k, 3
above 10k, as specified. Seven organisms, four phenotype classes, EC
classes 1/2/3 represented, five assays with multi-mutants.

**Accepted deviations.** FecA is 72 aa, below the preferred 100–500 band —
the entire Tsuboyama corpus is 37–72 aa, so there is no in-band
alternative for a mega-scale proteolysis-stability assay. GB1 (Olson) and
Gcn4 both have low position coverage; both were kept because they are the
only large multi-mutant epistasis assays in the eligible pool, and Olson
was chosen over `SPG1_STRSG_Wu_2016` (149k variants but only 4 mutated
positions) precisely so that a position-based split is possible at all.
`A4_HUMAN_Seuma_2022` was rejected: 42 mutated positions in a 770-aa
precursor means per-residue activation caching would be dominated by
never-mutated context.

![Step 1 dataset QC]({{artifact:art_0820d11e-0655-425e-8aa2-82580f2321b0}})

*Figure. (a) Size versus position coverage; the two large binding assays
buy variant count by saturating few positions. (b) Standardized label
distributions — the pathology is a spike of dead variants (GFP, GB1), not
heavy tails. (c) Composition of unseen-position fold 0: strict position
holdout discards every variant that straddles the boundary, which is
cheap for single-mutant assays and expensive for combinatorial ones. (d)
Fraction of variants at a censoring bound and sharing a tied label value.*

## 3. Mutation-string QC — clean for all ten

Per-assay checks in `Code/crossfit/qc.py`, results in
`Data/processed/qc_report.csv`. For **all ten assays**:

- unparsed mutation tokens: **0**
- tokens whose FROM residue disagrees with the WT sequence: **0**
- out-of-range positions: **0**
- non-standard / stop-codon residues: **0**
- variants whose reconstructed sequence ≠ the shipped sequence: **0**
- variants whose length ≠ WT length: **0**
- missing labels: **0**

Mutation positions are 1-based with zero offset, and mutant sequences
rebuild **byte-exactly** from WT plus substitutions. The main indexing
risk in this kind of curation therefore does not materialize here.

Two traps found and closed along the way:

1. **Multi-line FASTA.** A first parse read only the first sequence line
   and reported `wt_len=60` for every assay. This also retroactively
   dissolves an earlier apparent inconsistency — FecA's WT is genuinely
   72 aa, matching its variant sequences.
2. **`apply_mutations` did not verify the FROM residue.** The build catches
   a corrupted mutation string by comparing against the shipped sequence,
   but the loader's `with_sequences=True` path has no reference to compare
   to, so a bad token would silently produce a wrong sequence and every
   activation cached from it would be wrong. `apply_mutations` now takes
   `check_reference=True` and returns `None` on mismatch.

## 4. Deduplication — the declared counts *are* the deduplicated counts

The counts above are post-deduplication. Procedure: strip synonymous
tokens from each mutation string (`K34A:S56S` → `K34A`), drop
fully-synonymous / WT rows, then **deduplicate on the mutant amino-acid
sequence** and average labels within a duplicate group. The dedup unit is
the sequence, not the mutation string, because CROSSFIT caches activations
per sequence: two identical sequences with different labels are pure label
noise, and leakage if they land on opposite sides of a fold boundary.

Applying that recipe reproduces ProteinGym v1's
`DMS_total_number_mutants` **exactly for 10/10 assays**, and the
single/multi breakdown matches too. Only FecA has any sequence-level
duplication: 1,956 raw rows → 1,886 curated (68 duplicate-sequence
groups, 136 rows involved, 79 synonymous tokens, 2 fully-synonymous rows).
The other nine satisfy `rows == unique mutation strings == unique
sequences` with zero synonymous tokens. Nothing is silently deleted;
duplicate groups whose members disagree by more than
`duplicate_disagreement_sigma = 3.0` are flagged in `dup_disagrees`
(one variant, in FecA).

## 5. Labels, orientation, and σ_v

**Orientation — a silent sign-flip averted.** ProteinGym's `DMS Score` is
*already* oriented higher = fitter; the `raw_DMS_directionality` field
records the sign ProteinGym **has already applied** to the raw phenotype.
Re-applying it would have flipped CcdB, the one assay in our ten with
`raw_DMS_directionality = -1`. Verified empirically
against each bundle's own `DMS Score Bin`: mean score is higher in the
functional bin than the non-functional bin for every assay carrying the
column — including `CCDB_ECOLI_Adkar_2012`, whose directionality is −1.
Encoded as `apply_directionality: false` plus a build-time
`assert_orientation` that raises on contradiction.

**σ_v is genuinely unavailable for 9/10 assays.** Every bundle's
`manifest.lock` declares `has_uncertainty = true`, but no bundle ships a
per-variant uncertainty column, and `DMS_substitutions.csv` has no
uncertainty field — the flag describes the underlying experiment, not the
repackaged table. This is consistent with the supervised ProteinGym
leaderboard (ProteinNPT, Kermut) consuming no per-variant σ. Policy: a
`sigma_source` column with values `replicate_sd` (SEM = sd/√n from
duplicate measurements), `mv_trend` (imputed from a fitted
mean–variance trend, only where ≥30 replicate groups exist), or
`unavailable`. **NaN means "no information", never zero.** Realized
distribution over the 602,284 non-reserve variants: 600,492 `unavailable`,
1,728 `mv_trend`, 64 `replicate_sd` — all of the latter two in FecA.

Consequence for the protocol: any loss term weighting by 1/σ_v² is
available for FecA only. Either the head loss drops σ weighting entirely,
or it is used as a FecA-only sensitivity check. This must be decided
before Step 4.

**Standardization.** Within-assay z-score, with σ_v rescaled by the same
factor; `y_center` / `y_scale` retained so raw units are recoverable.
`y_raw` is kept alongside `y`.

**Censoring and ties.** GFP 10,519 variants at the floor (21.4%), GB1
118,680 at the floor (23.3%), CcdB 168 at the ceiling (15.0%); no other
assay is censored. Tied-label fractions are high for CcdB (0.958 — only
49 unique label values across 1,176 variants), GB1 (0.683) and EstA
(0.596). Both are flagged per row (`censored ∈ {none, floor, ceiling}`)
rather than filtered, and the loader exposes `drop_censored=True`.
Rank-based metrics on CcdB will be tie-dominated; report Spearman with an
explicit tie correction there.

## 6. Splits

Four schemes, all a pure function of `(assay_id, scheme, seed)` through a
`stable_hash`, never of iteration order.

**`pg_kfold`** — ProteinGym's published 5-fold masks, used verbatim for
nine assays (`pg_kfold_provenance = published`). FecA's published mask has
length 2,025 against 1,956 source rows and cannot be aligned, so its fold
is re-derived deterministically and marked
`re_derived_deterministic`. (The published masks index **pre-dedup** source
row order; for the other nine they align 1:1.)

**`unseen_position`** — the B2 test. This is the scheme that needed the
most care, and the first implementation was wrong in an
outcome-changing way.

*The flaw.* Positions were partitioned into 5 folds and each variant
assigned to the lowest fold among its mutated positions. For a
multi-mutant that leaks: a variant in test fold *f* still has its other
mutated positions represented in training. Quantified: **23,619 of GFP's
26,687 fold-0 test variants (88%) had every one of their positions seen in
training.** The B2 test was largely not testing extrapolation to unseen
positions.

*The fix.* The build now stores a **bitmask over position folds** per
variant (bit *f* set iff the variant mutates ≥1 position in group *f*),
and fold *f* decodes as a strict three-way partition:

```
test  = mask == (1 << f)        every mutated position is held out
train = mask & (1 << f) == 0    no mutated position is held out
drop  = otherwise               straddles the boundary — unusable
```

Straddlers must be dropped: assigning them to test leaks training context,
assigning them to train puts a held-out position into training. Leakage
across all assays and all folds is now **0**, verified in the test suite
(with `require_usable=False` so the guarantee is checked even on refused
folds). The cost is the drop fraction in Figure 1c: 0 for the six
single-mutant-dominated assays, 0.31 (KRAS), 0.33 (GB1), 0.53 (GFP), 0.79
(Gcn4).

*The availability gate.* A *k*-mutant survives the strict rule with
probability ≈ `n_folds^−k`, so combinatorially saturated assays cannot
form a position-disjoint split at any fold count. Gcn4 is the extreme
case: mean 17.1 mutations per variant, median 6, max 44, over only 44
mutated positions. Sweeping K ∈ {5,3,2} gives Gcn4 test sets of 3 / 23 /
72 variants — unusable throughout, and K=2 would also collapse the
training set. K therefore stays uniformly **5** for cross-assay
comparability, and the report emits per-fold usability against
`min_test_size = 200`, `min_train_size = 500`. Result: **9/10 assays
`unseen_position_available = True` with 5 usable folds each; Gcn4 alone
False, 0 usable folds.** `get_split(..., scheme="unseen_position")` raises
`UnusableSplit` for Gcn4 rather than returning a 3-variant test set.
This is a declared limitation, not a workaround: **B2 conclusions will be
drawn from nine assays, and Gcn4 contributes to B4 (epistasis) instead.**

**`mutation_count_extrapolation`** — train on n_mut ≤ 1, test on higher
order; defined for the five multi-mutant assays (FecA 1,219/667, Gcn4
33/2,605, GFP 1,084/50,630, KRAS 3,066/22,946, GB1 1,045/535,917),
`KeyError` by design for the five single-mutant-only assays.

**`low_n`** — N ∈ {16, 32, 64, 128, 256, 512} × 10 seeds = 60 subsets per
assay, each a **prefix of one permutation**, so subsets are exactly nested
and the low-N curve is monotone in information rather than in luck. Test
set is the `pg_kfold` fold-0 holdout, held fixed across N and seed.

## 7. Two reservations

**Layer-freeze reserve** — a within-assay 5% variant subset
(`seed = 424242`), 31,700 of 633,984 rows, used only to choose the
activation-caching layer. Excluded by default from the loader and from
every split scheme; the test suite asserts it is unreachable from all four
schemes.

**Prediction-track assay groups** — K=3 grouped **assay** folds in
`splits/prediction_groups.csv`, for the prediction track only. Group 0:
CcdB, Gcn4, DAAO, KRAS. Group 1: EstA, FecA, TPMT. Group 2: GFP,
Ubiquitin, GB1. The measurement track keeps one donor adapter, one
dictionary, one τ, nulls run once.

## 8. Loader API

`Code/crossfit/datasets.py`, no dependencies beyond pandas/numpy:

```python
from crossfit import datasets as D
D.list_assays()                       # the ten ids, sorted
D.manifest()                          # one row per assay, 36 columns
D.load_assay(aid, with_sequences=True)  # rebuilds + re-validates sequences
D.load_all()                          # 602,284 rows × 13 cols (reserve excluded)
D.get_split(aid, "pg_kfold", fold=0)  # -> (train_idx, test_idx) positional
D.get_split(aid, "low_n", N=64, seed=3)
D.prediction_groups()
```

The variant table is deliberately **narrow**: a first build that repeated
`wt_sequence` and `mutant_sequence` on every row produced a 655 MB CSV.
Per-assay constants live once in `assay_manifest.csv`; `mutant_sequence`
is not stored but reconstructed by `apply_mutations`, asserted byte-exact
at build time and re-validated on read. Result: `variants.csv` 77.6 MB,
`variants.parquet` 14.9 MB.

## 9. Tests

`Code/tests/test_splits.py` — **147 tests, all passing, 70 s**. Coverage:

- the frozen ten-assay selection still verifies against the candidate pool
- manifest is one row per assay; no NaN labels
- σ_v is never silently zero: NaN ⟺ `sigma_source == "unavailable"`
- within-assay standardization holds (|mean| < 0.05, |sd − 1| < 0.05)
- sequences regenerate, are unique, and differ from WT; `n_mutations` and
  `mutated_positions` agree with the mutation string
- layer-freeze reserve excluded by default and unreachable from every scheme
- prediction groups cover all ten assays and are balanced
- train/test disjoint in every scheme × assay
- **unseen-position leakage = 0**, checked including refused folds
- `pg_kfold` folds partition each assay
- `mutation_count_extrapolation` holds out strictly higher order
- `low_n` nesting, sizing, seed-distinctness, and grid-matches-config
- `stable_hash` stable across processes; masks reproducible from scratch
  against the stored CSV
- `apply_mutations` rejects a wrong FROM residue

## 10. Outputs

| file | size |
|---|---:|
| `Data/processed/variants.csv` | 77,591,928 |
| `Data/processed/variants.parquet` | 14,917,766 |
| `Data/processed/assay_manifest.csv` | 6,256 |
| `Data/processed/qc_report.csv` | 3,759 |
| `Data/processed/unseen_position_report.json` | 5,571 |
| `Data/processed/splits/pg_kfold.csv` | 25,604,051 |
| `Data/processed/splits/unseen_position.csv` | 20,234,677 |
| `Data/processed/splits/mutation_count_extrapolation.csv` | 37,341,929 |
| `Data/processed/splits/low_n.csv` | 5,042,366 |
| `Data/processed/splits/prediction_groups.csv` | 302 |

Variant schema: `assay_id, variant_row, mutations, n_mutations,
mutated_positions, y_raw, y, sigma_v, sigma_source, n_measurements,
censored, dup_disagrees, is_layer_freeze_reserve`.

## 11. Audit: Schmirler, Heinzinger & Rost (Nat Commun 2024)

Requested check on whether to incorporate the datasets from
*"Fine-tuning protein language models boosts predictions across diverse
tasks"* (`10.1038/s41467-024-51844-2`, PMC11358375). Their
`training data.zip` (repo `RSchmirler/data-repo_plm-finetune-eval`) ships
per-task `train/valid/test.pkl` for five tasks. Legacy pickle protocol —
`pd.read_pickle(io.BytesIO(...))` works, plain `pickle.loads` fails.

| their task | verdict | reason |
|---|---|---|
| GFP | **already in our set; their split reusable** | Their sequences are ProteinGym's with the leading residue trimmed (`pg.sequence.str[1:]` maps onto them exactly), so the TAPE Hamming ≤ 3 train/test assignment can be transferred onto our GFP rows. Their copy also contains duplicate sequences whose scores must be averaged — the same dedup step we already apply. |
| GB1 | **family covered, their split not used** | FLIP/Schmirler use `SPG1_STRSG_Wu_2016` on a 56-aa construct; we deliberately chose `Olson_2014` on the full 448-aa WT (55 mutated positions vs 4) so an unseen-position split exists. Their 4-position split cannot support B2. |
| AAV | **rejected** | Addable in principle, but mutations cover only ~4% of positions, and the phenotype is a viral-titer proxy several causal steps from a protein-level property. |
| Stability (Rocklin) | **rejected** | Cross-protein: mean length ~45 aa across many designed miniproteins. Incompatible with a per-protein activation-caching crosscoder — there is no single WT to cache residue positions against. |
| Meltome | **rejected** | Cross-protein Tm regression over 13 species, same structural incompatibility. |

**Net effect on Step 1: no new assays.** The one transferable asset is
their GFP split, which is the TAPE Hamming ≤ 3 assignment. Adding it as a
fifth scheme (`schmirler_gfp`) would let us report one number directly
comparable to their Table 2 — worth doing, but it is a single-assay
comparison on a different backbone, so under standing rule 4 it can only
be a sanity check, never a baseline. Deferred pending your call.

## 12. Limitations to carry into Step 2

1. **σ_v exists for FecA only.** Any σ-weighted objective is a FecA-only
   sensitivity analysis, not a headline method component.
2. **B2 is a nine-assay result.** Gcn4 cannot support a position-disjoint
   split at any fold count.
3. **Strict B2 discards data.** Test sets on the multi-mutant assays are
   small (GFP 747, KRAS 1,625) relative to their totals. Bootstrap CIs on
   B2 will be correspondingly wide, and should be reported per assay.
4. **CcdB is nearly ordinal.** 49 unique label values, 95.8% ties, 15%
   ceiling-censored. Treat as a coarse-grained assay; do not report
   uncorrected Spearman.
5. **FecA's published fold is re-derived**, so its `pg_kfold` numbers are
   not comparable to published ProteinGym leaderboard entries for that
   assay. The other nine are.
6. **Two backbones, one curation.** Nothing in this dataset is
   backbone-specific, but τ and every null must be recomputed per
   backbone (standing rule 1).
