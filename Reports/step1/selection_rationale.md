# Step 1 — assay selection rationale

Source: ProteinGym 2.0 (HF `ProteinGym/ProteinGym2.0`, revision `2e5c7b10c3dda55ca7295c5f1334887801294131`), assay-level metadata from the v1 reference file at commit `62eab5b3c83a3727b5b57bfbe377b24ef7545ccc`. See `Data/raw/reference/SOURCES.md`.

![selection landscape]({{artifact:art_eefd9cb3-b872-4164-9a9e-f3bfee0a06eb}})

## Funnel

| stage | assays |
|---|---|
| `.pgdata` bundles in ProteinGym 2.0 | 211 |
| join to v1 metadata on `DMS_id` | 211 (all) |
| phenotype in ['Activity', 'Binding', 'Stability', 'Expression'] — excludes OrganismalFitness | 137 |
| and n_variants ≥ 500, len ≤ 1024 | 122 |
| selected | 10 |

The phenotype filter removes the 77 OrganismalFitness assays (viral replication fitness, bacterial/yeast growth rate). Those are proxies several causal steps removed from a protein-level property, so a representational change measured against them could not be attributed to the protein's molecular function.

## Selected assays

| # | assay_id | phenotype | slot | n | multi | len | mut. positions (~) | organism | EC | MSA depth |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `GFP_AEQVI_Sarkisyan_2016` | Activity | >10k | 51,714 | 50,630 | 238 | 57 (24%) | *Aequorea victoria* | — | Low |
| 2 | `SPG1_STRSG_Olson_2014` | Binding | >10k | 536,962 | 535,917 | 448 | 55 (12%) | *Streptococcus sp. group G* | — | Low |
| 3 | `RASK_HUMAN_Weng_2022_abundance` | Expression | >10k | 26,012 | 22,946 | 188 | 161 (86%) | *Homo sapiens* | 3.6.5.2 | High |
| 4 | `ESTA_BACSU_Nutschel_2020` | Stability | 2k-10k | 2,172 | 0 | 212 | 114 (54%) | *Bacillus subtilis* | 3.1.1.3 | High |
| 5 | `OXDA_RHOTO_Vanella_2023_activity` | Activity | 2k-10k | 6,396 | 0 | 364 | 337 (93%) | *Rhodotorula gracilis* | 1.4.3.3 | High |
| 6 | `GCN4_YEAST_Staller_2018` | Binding | 2k-10k | 2,638 | 2,605 | 281 | 2 (1%) | *Saccharomyces cerevisiae* | — | Low |
| 7 | `TPMT_HUMAN_Matreyek_2018` | Expression | 2k-10k | 3,648 | 0 | 245 | 192 (78%) | *Homo sapiens* | 2.1.1.67 | Medium |
| 8 | `CCDB_ECOLI_Adkar_2012` | Activity | 500-2k | 1,176 | 0 | 101 | 62 (61%) | *Escherichia coli* | — | High |
| 9 | `RL40A_YEAST_Roscoe_2014` | Activity | 500-2k | 1,380 | 0 | 128 | 73 (57%) | *Saccharomyces cerevisiae* | — | Medium |
| 10 | `FECA_ECOLI_Tsuboyama_2023_2D1U` | Stability | 500-2k | 1,886 | 667 | 72 | 64 (89%) | *Escherichia coli* | — | High |

Totals: 633,984 scored variants, 7 organisms, 4 enzymes spanning EC classes 1, 2, 3 (oxidoreductase, transferase, hydrolase), 5 assays with multi-mutants, 204 MB to download.

`mut. positions (~)` is estimated as single mutants / 19 and is the ceiling on how many positions an `unseen_position` split can hold out. Two selected assays have low coverage by design: GB1 (55 of 448 positions — the library is a defined two-site-exhaustive design) and Gcn4 (only 33 single mutants, so its position sets must be derived from multi-mutant variants).

## Per-assay rationale

**1. `GFP_AEQVI_Sarkisyan_2016`** (>10k)  
Avian/jellyfish GFP fluorescence, 51.7k variants of which 50.6k are multi-mutant. The canonical epistasis landscape (B4): mutation effects here are strongly background-dependent, so it is the primary test of whether crosscoder features capture non-additive structure. Low MSA depth also probes whether adaptation helps where evolutionary signal is weak. 238 aa, inside the preferred length band.

**2. `SPG1_STRSG_Olson_2014`** (>10k)  
Protein G B1 domain / IgG binding. The complete single+double mutant landscape over 55 positions — the reference dataset for epistasis and low-N benchmarking, and a FLIP task, so results connect to published hard splits. Chosen over SPG1_STRSG_Wu_2016 (149k variants) because Wu mutates only 4 positions, which makes an unseen_position split impossible; Olson's 55 positions support B2 directly.

**3. `RASK_HUMAN_Weng_2022_abundance`** (>10k)  
KRAS protein abundance by yeast growth, 26k variants (22.9k multi-mutant), 188 aa, 86% position coverage. Supplies expression phenotype at large N with multi-mutants, on a GTPase (EC 3.6.5.2). Its sibling assay RASK_HUMAN_Weng_2022_binding-DARPin_K55 measures binding on the identical variant library — retained as a documented alternate for a same-protein phenotype contrast, not selected here to avoid double-weighting one family.

**4. `ESTA_BACSU_Nutschel_2020`** (2k-10k)  
Bacillus subtilis lipase A thermostability (EC 3.1.1.3), 2.2k variants, 212 aa. The only non-Tsuboyama stability assay in the eligible pool inside the preferred length band, and the only assay measuring thermostability proper rather than proteolysis-inferred folding stability. Directly represents the engineering objective the project targets.

**5. `OXDA_RHOTO_Vanella_2023_activity`** (2k-10k)  
Rhodotorula gracilis D-amino acid oxidase activity (EC 1.4.3.3), 6.4k variants, 364 aa, 93% position coverage. Adds an oxidoreductase (EC class 1) and a fungal organism, both otherwise absent. Chosen over AMIE_PSEAE_Wrenbeck_2017 (amidase, EC 3.5.1.4) which would have made three of four enzymes hydrolases; AMIE is the documented alternate.

**6. `GCN4_YEAST_Staller_2018`** (2k-10k)  
Yeast Gcn4 transcription activation domain binding, 2.6k variants of which 2.6k are multi-mutant. An intrinsically disordered activation domain — a family with no folded structure and no stability-mediated fitness, which is the sharpest available contrast to the folded enzymes and the strongest test that findings are not stability artifacts. Note only 33 single mutants, so its unseen_position split must be derived from multi-mutant position sets.

**7. `TPMT_HUMAN_Matreyek_2018`** (2k-10k)  
Human thiopurine methyltransferase abundance by VAMP-seq (EC 2.1.1.67), 3.6k variants, 245 aa, 78% position coverage. Supplies expression at mid-N and an EC class 2 transferase. Single-mutant only, which is what the mutation_count_extrapolation training side needs.

**8. `CCDB_ECOLI_Adkar_2012`** (500-2k)  
E. coli CcdB toxin activity, 1.2k variants, 101 aa, 61% position coverage. Small prokaryotic non-enzyme; toxicity readout is mechanically unlike the fluorescence and growth reporters elsewhere in the set.

**9. `RL40A_YEAST_Roscoe_2014`** (500-2k)  
Yeast ubiquitin E1 reactivity, 1.4k variants, 128 aa. Ubiquitin is one of the most constrained folds known, with very high MSA depth — the opposite regime to GFP, so together they bracket the range over which evolutionary likelihood and measured fitness can disagree.

**10. `FECA_ECOLI_Tsuboyama_2023_2D1U`** (500-2k)  
E. coli FecA N-terminal domain folding stability from cDNA-display proteolysis, 1.9k variants (667 multi-mutant), 89% position coverage, two biological replicates so sigma_v is measured rather than imputed. Represents the Tsuboyama mega-scale source, which is 62 of the 64 eligible stability assays and cannot be omitted without misrepresenting the benchmark. At 72 aa it is below the preferred 100-500 band; this is an accepted, reported cost — the whole Tsuboyama corpus is 37-72 aa, so there is no in-band alternative.

## Criterion check

Verified programmatically by `crossfit.selection.verify()`, which raises if any check fails; re-run as part of the test suite so the set cannot drift out of spec.

| criterion | required | actual |
|---|---|---|
| assay count | 10 | 10 |
| assays in size bin 500-2000 | 3 | 3 |
| assays in size bin 2000-10000 | 4 | 4 |
| assays in size bin 10000-inf | 3 | 3 |
| multi-mutant assays | ≥ 3 | 5 |
| phenotype classes | all 4 | 4 |
| distinct organisms | ≥ 7 | 7 |
| distinct EC classes | ≥ 3 | 3 |
| min variants per assay | ≥ 500 | 1,176 |
| max WT length | ≤ 1024 | 448 |
| structure available | all | 10/10 |

## Accepted deviations

- **Length band.** 8 of 10 assays fall in the preferred 100–500 aa band. `FECA_ECOLI_Tsuboyama_2023_2D1U` is 72 aa and `CCDB_ECOLI_Adkar_2012` is 101 aa (just inside). The entire Tsuboyama corpus — 62 of the 64 eligible stability assays — is 37–72 aa, so representing that source at all requires going below the band. Consequence to carry forward: this assay contributes proportionally fewer residue activations per variant to the crosscoder's pooled training set.

- **Stability breadth.** Only two eligible stability assays are not Tsuboyama mini-domains. Both selected stability assays therefore cover the available space: one thermostability assay (LipA, 212 aa) and one proteolysis-inferred folding stability assay (FecA domain, 72 aa). Stability conclusions cannot be claimed to generalise across many folds.

- **Family independence.** All 10 assays are distinct proteins from 7 organisms; no two share a Pfam family. The KRAS abundance and KRAS binding assays share a variant library, so only one is selected.

## Alternates held in reserve

Documented so the rejected set is explicit. Any substitution during QC must be recorded here with its reason.

| assay_id | status |
|---|---|
| `AMIE_PSEAE_Wrenbeck_2017` | Amidase activity (EC 3.5.1.4), 6.2k variants. Displaced by OXDA for EC-class diversity; first substitute if OXDA fails QC. |
| `SPG1_STRSG_Wu_2016` | GB1 four-site combinatorial library, 149k variants up to quadruple mutants. Supports mutation-count extrapolation to n=3,4 but only 4 mutated positions, so no unseen_position split. |
| `RASK_HUMAN_Weng_2022_binding-DARPin_K55` | Binding on the same KRAS library as the selected abundance assay; enables a same-protein phenotype contrast if that becomes a question. |
| `A4_HUMAN_Seuma_2022` | Amyloid-beta aggregation, 14.8k variants mostly multi-mutant. Rejected: mutations span only 42 positions of a 770-aa precursor, so per-residue activation caching is dominated by never-mutated context. |
| `MET_HUMAN_Estevam_2023` | MET kinase activity (EC 2.7.10.1), 5.4k variants, 99% position coverage. Strong alternate; displaced only to keep four phenotype classes balanced. |
| `NUD15_HUMAN_Suiter_2020` | NUDT15 abundance, 2.8k variants, 91% position coverage. Alternate for TPMT in the expression slot. |
| `CBPA2_HUMAN_Tsuboyama_2023_1O6X` | Tsuboyama 72-aa domain, 2.1k variants, 99% position coverage. Alternate for FECA if its QC fails. |
