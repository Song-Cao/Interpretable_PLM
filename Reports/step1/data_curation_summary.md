# CROSSFIT Step 1 — Data Curation Handoff

**Purpose.** Reproducibility handoff for the agent taking Step 2 (activation caching / LoRA
adaptation). Everything below was verified by reading the actual files, code, and test results
on 2026-10-14; nothing is inferred from intended behaviour. Where something is incomplete or
deviates from the Step 1 brief it is marked in §5 rather than smoothed over.

**Repo root.** `/Users/songcao/workdir/Interpretable_PLM`
**Environment.** `crossfit-dev` (path-venv at `<root>/.venv`) — Python 3.11.15, pandas 2.3.3,
numpy 2.4.6, pyarrow 25.0.0.
**Test status.** `python3 -m pytest Code/tests -q` → **147 passed in 86.90s** (re-run 2026-10-14).

---

## 1. Final curated datasets

### 1.1 Source and pinning

All ten assays come from one source: the HuggingFace dataset **`ProteinGym/ProteinGym2.0`**,
pinned at revision **`2e5c7b10c3dda55ca7295c5f1334887801294131`** (`last_modified`
2026-08-25T18:49:13Z; 211 `.pgdata` bundles, 4,846,635,552 B total in the repo).
Pin recorded in `Data/raw/reference/sources_lock.json` (`pinned_at_utc`
2026-08-27T17:30:53+00:00).

A second source is used **for cross-checking counts only, not for labels**: ProteinGym v1
`reference_files/DMS_substitutions.csv`, GitHub commit
**`62eab5b3c83a3727b5b57bfbe377b24ef7545ccc`**, local sha256
`a8f498011532a74aa9fe556a50555a75e928c5837d19c06a87592ae04049b308`.

The ten downloaded bundles are verified by size and sha256 against the pinned manifest in
`Data/raw/proteingym2/download_manifest.json` (keys: `repo`, `revision`, `n_assays`, `files`;
each file entry carries `size_bytes`, `pinned_size`, `pinned_oid`, `local_sha256`). 10 bundles,
195 MB on disk.

### 1.2 The ten assays

`n_curated` is the post-deduplication count (see §2.3) and is the row count in
`variants.csv`. `n_usable` excludes the 5% layer-freeze reserve (§3.4) and is what
`load_assay()` returns by default.

| assay_id | phenotype | protein | organism | wt_len | n_raw | n_curated | n_usable | single / multi | max k | pos. cov. |
|---|---|---|---|---|---|---|---|---|---|---|
| `GFP_AEQVI_Sarkisyan_2016` | Activity | GFP | *Aequorea victoria* | 238 | 51714 | 51714 | 49128 | 1084 / 50630 | 15 | 0.979 |
| `SPG1_STRSG_Olson_2014` | Binding | GB1 | *Streptococcus* sp. group G | 448 | 536962 | 536962 | 510114 | 1045 / 535917 | 2 | 0.123 |
| `RASK_HUMAN_Weng_2022_abundance` | Expression | KRAS | *Homo sapiens* | 188 | 26012 | 26012 | 24711 | 3066 / 22946 | 2 | 0.995 |
| `ESTA_BACSU_Nutschel_2020` | Stability | EstA | *Bacillus subtilis* | 212 | 2172 | 2172 | 2063 | 2172 / 0 | 1 | 0.793 |
| `OXDA_RHOTO_Vanella_2023_activity` | Activity | DAAO | *Rhodotorula gracilis* | 364 | 6396 | 6396 | 6076 | 6396 / 0 | 1 | 1.000 |
| `GCN4_YEAST_Staller_2018` | Binding | Gcn4 | *S. cerevisiae* | 281 | 2638 | 2638 | 2506 | 33 / 2605 | 44 | 0.157 |
| `TPMT_HUMAN_Matreyek_2018` | Expression | TPMT | *Homo sapiens* | 245 | 3648 | 3648 | 3466 | 3648 / 0 | 1 | 0.984 |
| `CCDB_ECOLI_Adkar_2012` | Activity | CcdB | *Escherichia coli* | 101 | 1176 | 1176 | 1117 | 1176 / 0 | 1 | 0.990 |
| `RL40A_YEAST_Roscoe_2014` | Activity | Ubiquitin | *S. cerevisiae* | 128 | 1380 | 1380 | 1311 | 1380 / 0 | 1 | 0.586 |
| `FECA_ECOLI_Tsuboyama_2023_2D1U` | Stability | FecA | *Escherichia coli* | 72 | 1956 | 1886 | 1792 | 1219 / 667 | 2 | 1.000 |

**Totals: 633,984 curated / 31,700 reserved / 602,284 usable.** 7 distinct organisms;
4 phenotype classes. FecA is the only assay where `n_raw != n_curated`.

Per-assay provenance (DOI, readout, library, biological-replicate count, MSA Neff category,
UniProt id, PDB presence) is in `Data/processed/assay_manifest.csv` — 36 columns, one row per
assay. Do not re-derive it.

### 1.3 Exact file paths

All paths relative to the repo root.

| Path | Bytes | Contents |
|---|---|---|
| `Data/processed/variants.csv` | 77,591,928 | 633,984 variant rows, 13 columns |
| `Data/processed/variants.parquet` | 14,917,766 | identical content; **the loader reads this one** |
| `Data/processed/assay_manifest.csv` | 6,256 | 10 rows × 36 cols, per-assay constants incl. `wt_sequence` |
| `Data/processed/qc_report.csv` | 3,759 | 10 rows × 37 cols, every QC counter in §2 |
| `Data/processed/unseen_position_report.json` | 5,571 | per-assay per-fold B2 sizes, usability, leakage |
| `Data/processed/splits/pg_kfold.csv` | 25,604,051 | 633,984 rows |
| `Data/processed/splits/unseen_position.csv` | 20,234,677 | 633,984 rows (bitmask, see §3.2) |
| `Data/processed/splits/mutation_count_extrapolation.csv` | 37,341,929 | 619,212 rows, 5 assays |
| `Data/processed/splits/low_n.csv` | 5,042,366 | 100,800 rows, 60 schemes |
| `Data/processed/splits/prediction_groups.csv` | 302 | 10 rows |
| `Data/raw/proteingym2/*.pgdata` | 195 MB | 10 pinned source bundles (gitignored) |
| `Data/raw/proteingym2/download_manifest.json` | 3,601 | download integrity record |
| `Data/raw/reference/sources_lock.json` | 31,834 | source pin |
| `Data/raw/reference/DMS_substitutions.csv` | 208,734 | ProteinGym v1 reference |
| `Data/interim/candidates.csv` | 76,903 | 137-row candidate table behind the selection |

### 1.4 Final schema — `variants.csv` / `variants.parquet`

Thirteen columns, in this order. The table is deliberately **narrow**: sequences are not stored
(a denormalized first build was 655 MB; this one is 77.6 MB).

| Column | dtype | Meaning |
|---|---|---|
| `assay_id` | object | join key into `assay_manifest.csv` |
| `variant_row` | int64 | within-assay row id; **the join key used by every split file** |
| `mutations` | object | colon-joined tokens, 1-based, e.g. `K3R:V55A:Q94R` |
| `n_mutations` | int64 | token count after synonymous stripping |
| `mutated_positions` | object | semicolon-joined 1-based positions, e.g. `3;55;94` |
| `y_raw` | float64 | ProteinGym `DMS Score`, post-dedup, post-averaging, **no sign flip** |
| `y` | float64 | within-assay z-score of `y_raw` — **the training target** |
| `sigma_v` | float64 | label SE on the `y` scale; **NaN = no information, never zero** |
| `sigma_source` | object | `replicate_sd` \| `mv_trend` \| `unavailable` |
| `n_measurements` | int64 | source rows collapsed onto this mutant sequence |
| `censored` | object | `none` \| `floor` \| `ceiling` |
| `dup_disagrees` | bool | duplicate group whose spread exceeded 3σ |
| `is_layer_freeze_reserve` | bool | **True ⇒ must not be used for training** (§3.4) |

Per-assay constants (`wt_sequence`, `protein_id`, `uniprot_id`, `selection_assay`, `organism`,
`phenotype_class`, `readout`, `library`, `growth_coupled`, `n_biological_replicates`,
`publication_doi`, `msa_neff_category`, …) live **once** in `assay_manifest.csv` and are joined
on `assay_id` by the loader.

---

## 2. QC and preprocessing actually performed

Every counter below is a column in `Data/processed/qc_report.csv`.

### 2.1 Mutation / reference consistency

Token grammar `^([A-Za-z])(\d+)([A-Za-z])$`, 1-based positions, offset 0. Four checks plus a
byte-exact rebuild, run per token and per row:

| Check (qc_report column) | Result, all 10 assays |
|---|---|
| `n_unparsed_tokens` | **0** |
| `n_wt_mismatch` (token FROM-residue vs WT at that position) | **0** |
| `n_out_of_range` | **0** |
| `n_invalid_residue_rows` | **0** |
| `n_nonsense_rows` (stop codons) | **0** |
| `n_seq_len_ne_wt` | **0** |
| `n_rebuild_mismatch` (WT + substitutions == shipped sequence, byte-for-byte) | **0** |
| `n_missing_label` | **0** |

Token volume checked: 1,384,881 tokens total (e.g. GB1 1,072,879; GFP 200,547). The indexing
risk did not materialize for any assay.

### 2.2 Sequence / alphabet filtering

`STANDARD_AA = ACDEFGHIKLMNPQRSTVWY` (`Code/crossfit/qc.py:24`). Any row whose shipped sequence
contains a character outside that set is counted in `n_invalid_residue_rows`. **Zero rows were
excluded on alphabet grounds in any assay** — no filtering was actually triggered, the check
simply passed. Sequence length is also asserted equal to `wt_len` per row (`n_seq_len_ne_wt`).

### 2.3 Deduplication

Three operations, in order:

1. **Strip synonymous tokens** — a token with `t[0] == t[-1]` (e.g. `K34A:S56S` → `K34A`).
2. **Drop fully-synonymous / WT-equivalent rows** — rows left with no tokens.
3. **Deduplicate on the mutant amino-acid sequence**, averaging `y_raw` across the group.

The dedup unit is the **mutant amino-acid sequence**, not the mutation string, because that is
the unit CROSSFIT caches activations for. Nothing is silently deleted: duplicate groups are
averaged and flagged.

Only **FecA** is affected. Its ladder: 1,956 raw rows → 1,956 unique `mutant` strings → 1,887
unique stripped strings → **1,886** unique sequences after excluding 2 fully-synonymous rows.
That 1,886 equals v1's `DMS_total_number_mutants` exactly. The same identity holds for all ten
assays, single/multi breakdown included — **so the counts in §1.2 are post-deduplication**.
FecA counters: `n_synonymous_tokens` 79, `n_rows_fully_synonymous` 2, `n_dup_seq_groups` 68,
`n_dup_seq_rows` 136, `n_dup_disagree` **1**, `dup_pooled_sd` 0.2266 against a global label SD of
0.8487. All nine other assays have 0 on every one of these counters.

**No variant was dropped for duplicate disagreement.** `duplicate_disagreement_sigma = 3.0`
flags (`dup_disagrees`) and never removes — dropping on disagreement would be selection on the
label. Exactly **1 row** in the whole corpus carries `dup_disagrees = True`.

### 2.4 Censoring handling

Flagged, never dropped. A bound is flagged only if ≥ `censor_mass_frac` = 2% of variants sit
within `censor_eps` = 1% of the label range of that extreme (`Code/crossfit/qc.py:242-253`).
Rows at a flagged bound get `censored = "floor"` / `"ceiling"`.

| assay | ceiling | floor | none |
|---|---|---|---|
| `CCDB_ECOLI_Adkar_2012` | 178 | 0 | 998 |
| `GFP_AEQVI_Sarkisyan_2016` | 0 | 11,092 | 40,622 |
| `SPG1_STRSG_Olson_2014` | 0 | 124,723 | 412,239 |
| other seven | 0 | 0 | all |

Corpus totals: 497,991 `none`, 135,815 `floor`, 178 `ceiling`. This flags **dynamic-range
saturation, not experimental error**. `load_assay(..., drop_censored=True)` is available but
**off by default**.

Related: `ties_frac` is high for CcdB (0.9583 — only 49 unique label values over 1,176 variants),
GB1 (0.6828) and EstA (0.5962). CcdB is effectively ordinal.

### 2.5 Label orientation and normalization

**Orientation: no sign flip was applied.** ProteinGym's `DMS Score` is already oriented
higher = fitter; `raw_DMS_directionality` records the sign ProteinGym *already applied*. Config
carries `apply_directionality: false`. Only one of the ten (`CCDB_ECOLI_Adkar_2012`) has
`raw_DMS_directionality = -1`; re-applying it would have inverted that assay.

This was **verified empirically, not assumed**: `assert_orientation`
(`Code/crossfit/labels.py:42`) groups each bundle by its own `DMS Score Bin` and raises unless
mean score in functional bin 1 exceeds bin 0. `qc_report.orientation`:
**`verified_higher_is_fitter` for nine assays**; FecA is
**`no_bin_column_orientation_unverified`** because its bundle ships no `DMS Score Bin` column —
its orientation rests on ProteinGym's upstream convention alone.

**Normalization:** `y` is a within-assay z-score of `y_raw`, population SD (`ddof=0`),
`y = (y_raw − mean) / sd`, raising on zero variance. `sigma_v` is divided by the same `sd` so it
stays on `y`'s scale. Verified: per-assay |mean(y)| < 0.05 and |sd(y) − 1| < 0.05 (test suite).

### 2.6 Source of `sigma_v`

**σ_v is genuinely unavailable for 9 of 10 assays.** Every bundle manifest declares
`has_uncertainty = true`, but that flag describes the underlying experiment, not the repackaged
table: **no bundle ships a per-variant uncertainty column**, and v1's `DMS_substitutions.csv`
has no uncertainty field. `assay_manifest.per_variant_sigma_present` is **False for nine
assays, True for FecA only**.

Realized provenance over all 633,984 curated rows:

| `sigma_source` | rows | derivation |
|---|---|---|
| `unavailable` | 632,098 | no information — `sigma_v` is NaN |
| `mv_trend` | 1,818 | FecA only; imputed |
| `replicate_sd` | 68 | FecA only; measured |

The 68 `replicate_sd` values are SEM = `sd/√n_measurements` over FecA's 68 duplicate groups
(`ddof=1`). Those 68 real SDs then fit `log(sd)` linear in `y_raw`
(`fit_mv_trend`, `MIN_GROUPS_FOR_TREND = 30`), and that trend imputes σ for FecA's remaining
1,818 variants — an informed guess, flagged `mv_trend` so it can never be mistaken for measured.

**NaN means "no uncertainty information", never zero.** Any σ-weighted loss must handle NaN
explicitly. Enforced by test: NaN ⇔ `sigma_source == "unavailable"`.

### 2.7 Batch / plate metadata

**None exists.** Verified by reading the columns of all ten shipped assay CSVs inside the
`.pgdata` bundles. Every bundle carries exactly `sequence`, `DMS Score`, `mutant` plus
`DMS Score Bin` — FecA lacks even the bin column. No batch, plate, replicate-id, timepoint, or
sort-gate column exists in any bundle, and therefore none exists in `variants.csv`.

`n_measurements` is **not** a replication count in the experimental sense: it counts source rows
collapsed onto one mutant sequence. 633,916 rows have `n_measurements = 1` and exactly 68 have
2 — all FecA.

The only batch-adjacent field is `assay_manifest.n_biological_replicates`, an **assay-level**
integer from the bundle manifest (GB1 3, KRAS 3, TPMT 8, DAAO 2, Ubiquitin 2, FecA 2, Gcn4 1;
**NaN for GFP, EstA, CcdB**). It cannot be mapped to individual variants.

⇒ **B3 (noisy/batch/censored labels) has censoring and FecA-only σ to work with, but no batch
structure at all.** This is a hard limitation of the source, not of the curation.

---

## 3. Splits

Five schemes were built. All are a pure function of `(assay_id, scheme, seed)` via
`stable_hash`, never of iteration order. Config: `Code/config/step1.yaml`,
`master_seed: 20260827`.

### 3.1 Published splits reused — `pg_kfold`

`slices.json`'s published `kfold` masks from each bundle, K = 5. `pg_kfold.csv` covers all
633,984 rows; fold value counts are 126,802 / 126,799 / 126,796 / 126,794 / 126,793 — a clean
partition with no `-1`.

Provenance per assay (`qc_report.pg_kfold_provenance` / `assay_manifest`): **`published`** for
nine, aligned 1:1 to source rows. **FecA is `re_derived_deterministic`** — its published mask
has length 2,025 against 1,956 source rows, cannot be aligned, so the fold was re-derived
deterministically. Recorded, not hidden.

Verified **random**, not position-based: single-mutant positions scatter across 4.5–4.9 of 5
folds and fold identity is not `position mod 5`.

### 3.2 `unseen_position` (B2) — bitmask encoding, strict three-way partition

Positions (not variants) are partitioned into `n_folds = 5` groups. Storage is a **bitmask**:
`unseen_position.csv` has columns `assay_id, variant_row, pos_fold_mask`, where bit *f* is set
iff the variant mutates ≥1 position in group *f*. Per fold *f* it decodes as:

```
test  = mask == (1 << f)        every mutated position is held out
train = mask & (1 << f) == 0    no mutated position is held out
drop  = otherwise               straddles the boundary — unusable this fold
```

**This replaced a leaking implementation.** The original assigned each variant to the lowest
fold among its positions, which for multi-mutants put variants in test whose positions were all
seen in training — 23,619 of GFP's 26,687 fold-0 test variants (88%), GB1 342,232, KRAS 16,419,
Gcn4 564. Under the strict rule, `unseen_position_report.json`'s
`n_test_variants_fully_covered_by_train_positions` is **0 for every assay and every fold**.

`train + test < n` by design. Fold 0, post-reserve:

| assay | train | test | dropped |
|---|---|---|---|
| `CCDB_ECOLI_Adkar_2012` | 936 | 240 | 0 |
| `ESTA_BACSU_Nutschel_2020` | 1736 | 436 | 0 |
| `FECA_ECOLI_Tsuboyama_2023_2D1U` | 1650 | 236 | 0 |
| `RL40A_YEAST_Roscoe_2014` | 1110 | 270 | 0 |
| `TPMT_HUMAN_Matreyek_2018` | 2892 | 756 | 0 |
| `OXDA_RHOTO_Vanella_2023_activity` | 5103 | 1293 | 0 |
| `RASK_HUMAN_Weng_2022_abundance` | 16435 | 1625 | 7952 |
| `SPG1_STRSG_Olson_2014` | 342232 | 20064 | 174666 |
| `GFP_AEQVI_Sarkisyan_2016` | 23619 | 747 | 27348 |
| `GCN4_YEAST_Staller_2018` | 564 | 3 | 2071 |

**Availability gate.** A *k*-mutant survives with probability ≈ `n_folds^−k`, so
combinatorially-saturated assays lose nearly everything. Thresholds `min_test_size: 200`,
`min_train_size: 500`. Result: **9 of 10 assays available with 5 usable folds each; Gcn4 is
`unseen_position_available = False` with 0 usable folds** (it mutates 44 positions with
mean k = 17.1, max 44 — test set is 3 variants at K=5, 23 at K=3, 72 at K=2, so no fold count
rescues it). `get_split` **raises `UnusableSplit`** for Gcn4 rather than returning an
underpowered fold. K stays uniformly 5 for cross-assay comparability.

⇒ **B2 is a nine-assay result.** Gcn4 contributes to B4 instead.

### 3.3 `mutation_count_extrapolation` (B4)

`k = 1`: fold 0 = train (`n_mutations ≤ 1`), fold 1 = test (`> 1`), `-1` = undefined. Defined for
the **5 multi-mutant assays only**; `get_split` raises `KeyError` for the other five, which is
correct by design.

| assay | fold 0 (train) | fold 1 (test) |
|---|---|---|
| `FECA_ECOLI_Tsuboyama_2023_2D1U` | 1219 | 667 |
| `GCN4_YEAST_Staller_2018` | 33 | 2605 |
| `GFP_AEQVI_Sarkisyan_2016` | 1084 | 50630 |
| `RASK_HUMAN_Weng_2022_abundance` | 3066 | 22946 |
| `SPG1_STRSG_Olson_2014` | 1045 | 535917 |

### 3.4 `low_n`

Grid `N ∈ {16, 32, 64, 128, 256, 512} × 10 seeds` = **60 schemes**, all ten assays, 100,800 rows.
Scheme strings are literally `low_n_N{N}_seed{seed}` and the loader depends on that format.

Subsets are **nested by construction** — each is a prefix of one permutation, so the N=16 set is
a subset of N=32 and so on. Verified at call time (CcdB seed 0: train sizes exactly
16/32/64/128/256/512, each nested in the next). The test set is the **`pg_kfold` fold-0
holdout**, so `low_n` absorbs the random-CV comparison for free. The training pool excludes both
fold 0 and the layer-freeze reserve.

### 3.5 Reserved variants — layer-freeze reserve

A within-assay **5%** random subset, `seed = 424242`, flagged `is_layer_freeze_reserve`.
Realized: **31,700 of 633,984 rows, exactly 0.0500**; per assay CcdB 59, Ubiquitin 69, FecA 94,
EstA 109, Gcn4 132, TPMT 182, DAAO 320, KRAS 1301, GFP 2586, GB1 26848.

Its only sanctioned use is **choosing the activation-caching layer**. It is excluded by default
from `load_assay`/`load_all` and unreachable through every split scheme (asserted by test).

### 3.6 Grouped assay folds — prediction track only

K = 3 grouped folds over **assays**, not variants (`prediction_groups.csv`):

| group | assays |
|---|---|
| 0 | CcdB, Gcn4, DAAO, KRAS |
| 1 | EstA, FecA, TPMT |
| 2 | GFP, Ubiquitin, GB1 |

Per the standing rule this partitions the **prediction track only**; the measurement track stays
one donor adapter, one dictionary, one τ, nulls once.

---

## 4. Step 2 interface

### 4.1 The loader

Import `crossfit.datasets`. Do **not** read the CSVs directly — the split storage format is not
uniform across schemes (bitmask vs fold id) and `get_split` is what hides that.

```python
import sys; sys.path.insert(0, "Code")
import crossfit.datasets as D
```

Verified signatures:

```python
D.list_assays() -> list[str]
D.manifest() -> pd.DataFrame                       # 10 x 36, per-assay constants
D.wt_sequence(assay_id) -> str
D.load_assay(assay_id, with_sequences=False, include_reserve=False,
             drop_censored=False) -> pd.DataFrame
D.load_all(**kwargs) -> pd.DataFrame               # 602,284 x 13 at defaults
D.get_split(assay_id, scheme, fold=None, N=None, seed=None,
            include_reserve=False, require_usable=True) -> tuple[np.ndarray, np.ndarray]
D.prediction_groups() -> dict[str, int]
D.unseen_position_report_json() -> dict
D.UnusableSplit                                     # subclasses RuntimeError
```

`_SCHEMES = ("pg_kfold", "unseen_position", "mutation_count_extrapolation", "low_n")`.

**Sequences are reconstructed, not stored.** `load_assay(..., with_sequences=True)` adds
`mutant_sequence` and `wt_sequence`, and re-validates on every read: parse must succeed, every
length must equal `len(wt)`, and each reconstruction must differ from WT at exactly
`n_mutations` positions — otherwise it raises. `apply_mutations` takes `check_reference=True`
by default (the token's FROM residue must match WT) and returns `None` on mismatch. GFP check:
49,128 rows, all length 238.

### 4.2 Required columns

For activation caching: `assay_id`, `variant_row`, `mutations` (+ `wt_sequence` from the
manifest, or `with_sequences=True`).
For supervision: `y` (target), `assay_id` (per-assay head), `variant_row` (split join key).
For the standing rules: `is_layer_freeze_reserve`, `sigma_v`, `sigma_source`, `censored`.

### 4.3 How train / val / test are identified

`get_split` returns **positional indices into the frame `load_assay()` just returned** — not
`variant_row` values, not labels. Pass the *same* `include_reserve` to both or the indices will
not align.

```python
df = D.load_assay("GFP_AEQVI_Sarkisyan_2016")
tr, te = D.get_split("GFP_AEQVI_Sarkisyan_2016", "pg_kfold", fold=0)
train, test = df.iloc[tr], df.iloc[te]
```

Per-scheme argument contract:

| scheme | required args | semantics |
|---|---|---|
| `pg_kfold` | `fold` ∈ 0–4 | `fold` held out, all others train |
| `unseen_position` | `fold` ∈ 0–4 | held-out **position group**; `train + test < n`; raises `UnusableSplit` for Gcn4 |
| `mutation_count_extrapolation` | *(none)* | fold 0 train, fold 1 test; `KeyError` on the 5 single-mutant assays |
| `low_n` | `N` **and** `seed` | nested size-N train; test = `pg_kfold` fold-0 holdout |

**There is no validation split.** No scheme emits one and `get_split` returns a 2-tuple only
(grep for `val`/`validation` in `datasets.py`/`splits.py` → no hits). Step 2 must carve
validation out of the returned **train** indices, and must do so without touching test or the
reserve.

### 4.4 How σ_v, censoring and batch metadata are accessed

- **σ_v** — columns `sigma_v` / `sigma_source` on the per-variant frame. Branch on
  `sigma_source`; treat NaN as "no information", never zero. Usable σ exists for FecA only
  (68 measured, 1,818 trend-imputed).
- **Censoring** — column `censored` ∈ {`none`, `floor`, `ceiling`}; assay-level flags
  `censor_floor` / `censor_ceil` and `ties_frac` in the manifest. `drop_censored=True` filters
  to `censored == "none"`; it is off by default.
- **Batch metadata** — **does not exist** (§2.7). The nearest field is assay-level
  `manifest().n_biological_replicates`, NaN for GFP/EstA/CcdB, not mappable to variants.

### 4.5 Data reserved — must not be used for training

1. **`is_layer_freeze_reserve == True`** — 31,700 rows. Only for choosing the activation-caching
   layer. Excluded by default; never pass `include_reserve=True` in a training path.
2. **The test indices of whichever split is in play** — including, for `unseen_position`, the
   dropped straddling variants, which belong to neither side for that fold.
3. **Gcn4 `unseen_position`** — not a reserve but unusable; report B2 as unavailable for it
   rather than passing `require_usable=False`, which exists for diagnostics only.

**Standing rule 2 reminder:** the activation-caching layer must be at or after the first
LoRA-adapted layer, or `Delta_j` is meaningless by construction. Assert this in Step 2 code —
nothing in Step 1 can check it.

---

## 5. Incomplete, uncertain, or deviating items

**Deviations from the Step 1 brief**

1. **FecA length is out of band.** 72 aa against the preferred 100–500. Accepted because the
   entire Tsuboyama corpus is 37–72 aa, so there is no in-band alternative for that slot.
2. **No validation split was built** (§4.3). Step 2 must carve it from train.

The brief's size profile *is* met and is not a deviation: on `n_variants_curated` the bins are
3 at 500–2k (CcdB 1176, Ubiquitin 1380, FecA 1886), 4 at 2k–10k (EstA 2172, Gcn4 2638,
TPMT 3648, DAAO 6396), 3 above 10k (KRAS 26,012, GFP 51,714, GB1 536,962). The same 3/4/3
holds on post-reserve usable counts (FecA 1792 is the only one near a bin edge).

**Uncertain / unverified**

3. **FecA orientation is unverified.** Its bundle ships no `DMS Score Bin`, so
   `assert_orientation` could not run; `qc_report.orientation` records
   `no_bin_column_orientation_unverified`. The other nine are empirically verified.
4. **FecA `pg_kfold` is re-derived, not published** — published mask length 2,025 vs 1,956
   source rows, unalignable. Any comparison of FecA random-CV numbers against published
   ProteinGym results is not like-for-like.
5. **FecA n differs from what ProteinNPT / Kermut would consume.** They read the shipped
   per-assay CSVs as-is without re-deriving dedup, so their FecA n is 1,956 against our 1,886.
   Belongs in the methods section.
6. **`unseen_position` is ours, not a ProteinGym scheme.** ProteinGym's supervised benchmark
   defines three CV schemes — random 5-fold, **modulo 5**, **contiguous 5**. We reuse only the
   random one. Modulo and contiguous were scoped, shown to be exactly re-derivable from position
   alone with no download, and then **not built** at the user's direction (the five original
   schemes were kept). This gap is unresolved.

**Known gaps in the artifacts**

7. **`y_center` / `y_scale` are not persisted.** `standardize()` writes them to
   `DataFrame.attrs`, which does not survive the CSV/parquet write — grep confirms they appear
   nowhere else in `Code/`. To map `y` back to raw units, recompute from `y_raw`
   (`mean`, `std(ddof=0)`) over the **full curated set including the reserve**, which is how
   they were derived.
8. **Split storage is non-uniform** — `unseen_position` stores a bitmask, every other scheme a
   fold id. `get_split()` hides this; anyone reading the CSVs directly will get it wrong.
9. **A `schmirler_gfp` scheme was proposed and deliberately not built.** The Schmirler/
   Heinzinger/Rost (Nat Commun 2024, `10.1038/s41467-024-51844-2`) audit concluded: GFP is
    already in our set and their TAPE Hamming ≤ 3 split is transferable (their sequences are
   ProteinGym's with the leading residue trimmed), GB1 family is covered by Olson, and AAV /
   Rocklin Stability / Meltome were rejected. **Net: no new assays.** Under standing rule 4 the
   transferred split could only ever be a sanity check, never a baseline.

**Protocol questions still open from Step 0** (not Step 1's to settle, but they gate Step 2+):
where σ_v enters the head loss — now sharpened by its absence for 9/10 assays, so the live
options are to drop σ-weighting entirely or keep it as a FecA-only sensitivity check; the
pooling domain for `phi_j` (all residues vs mutated positions only); the pair-gating mechanism
and what `c` multiplies; the donor/eval assay partition; and whether the prediction track needs
0 or 3 crosscoders.

---

## 6. How to reproduce

```bash
cd /Users/songcao/workdir/Interpretable_PLM
python3 Code/scripts/pin_sources.py       # refresh sources_lock.json (changes the pin)
python3 Code/scripts/fetch_assays.py      # download the 10 .pgdata bundles (~195 MB)
python3 Code/scripts/build_dataset.py     # rebuild Data/processed/ (~50 s)
python3 -m pytest Code/tests -q           # 147 tests, ~87 s
```

Code map: `Code/crossfit/{pgdata,qc,labels,splits,datasets,selection,utils}.py`;
config `Code/config/step1.yaml` (single source of every threshold and seed);
log `PROJECT_LOG.md` (sections 1a–1g); report `Reports/step1/step1_report.md`.
