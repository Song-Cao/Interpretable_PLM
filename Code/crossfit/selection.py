"""The Step 1 assay selection, with per-assay rationale, plus automated
verification that the chosen set satisfies every stated criterion.

The selection itself is a documented judgement call, not the output of an
optimiser: with ~10 slots the binding constraint is phenotype/family/size
coverage, which no scalar objective captures honestly. What *is* automated is
the check that the hand-chosen set meets the brief's criteria — `verify()`
raises if it does not, so the set cannot silently drift out of spec.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class Choice:
    assay_id: str
    slot: str  # size bin the assay was chosen to fill
    rationale: str


# ---------------------------------------------------------------------------
# The selection. Ordered by size bin, then by assay id.
# ---------------------------------------------------------------------------
SELECTION: tuple[Choice, ...] = (
    # ---- >10k -------------------------------------------------------------
    Choice(
        "GFP_AEQVI_Sarkisyan_2016",
        ">10k",
        "Avian/jellyfish GFP fluorescence, 51.7k variants of which 50.6k are "
        "multi-mutant. The canonical epistasis landscape (B4): mutation effects "
        "here are strongly background-dependent, so it is the primary test of "
        "whether crosscoder features capture non-additive structure. Low MSA "
        "depth also probes whether adaptation helps where evolutionary signal is "
        "weak. 238 aa, inside the preferred length band.",
    ),
    Choice(
        "SPG1_STRSG_Olson_2014",
        ">10k",
        "Protein G B1 domain / IgG binding. The complete single+double mutant "
        "landscape over 55 positions — the reference dataset for epistasis and "
        "low-N benchmarking, and a FLIP task, so results connect to published "
        "hard splits. Chosen over SPG1_STRSG_Wu_2016 (149k variants) because Wu "
        "mutates only 4 positions, which makes an unseen_position split "
        "impossible; Olson's 55 positions support B2 directly.",
    ),
    Choice(
        "RASK_HUMAN_Weng_2022_abundance",
        ">10k",
        "KRAS protein abundance by yeast growth, 26k variants (22.9k "
        "multi-mutant), 188 aa, 86% position coverage. Supplies expression "
        "phenotype at large N with multi-mutants, on a GTPase (EC 3.6.5.2). Its "
        "sibling assay RASK_HUMAN_Weng_2022_binding-DARPin_K55 measures binding "
        "on the identical variant library — retained as a documented alternate "
        "for a same-protein phenotype contrast, not selected here to avoid "
        "double-weighting one family.",
    ),
    # ---- 2k-10k -----------------------------------------------------------
    Choice(
        "ESTA_BACSU_Nutschel_2020",
        "2k-10k",
        "Bacillus subtilis lipase A thermostability (EC 3.1.1.3), 2.2k variants, "
        "212 aa. The only non-Tsuboyama stability assay in the eligible pool "
        "inside the preferred length band, and the only assay measuring "
        "thermostability proper rather than proteolysis-inferred folding "
        "stability. Directly represents the engineering objective the project "
        "targets.",
    ),
    Choice(
        "OXDA_RHOTO_Vanella_2023_activity",
        "2k-10k",
        "Rhodotorula gracilis D-amino acid oxidase activity (EC 1.4.3.3), 6.4k "
        "variants, 364 aa, 93% position coverage. Adds an oxidoreductase (EC "
        "class 1) and a fungal organism, both otherwise absent. Chosen over "
        "AMIE_PSEAE_Wrenbeck_2017 (amidase, EC 3.5.1.4) which would have made "
        "three of four enzymes hydrolases; AMIE is the documented alternate.",
    ),
    Choice(
        "GCN4_YEAST_Staller_2018",
        "2k-10k",
        "Yeast Gcn4 transcription activation domain binding, 2.6k variants of "
        "which 2.6k are multi-mutant. An intrinsically disordered activation "
        "domain — a family with no folded structure and no stability-mediated "
        "fitness, which is the sharpest available contrast to the folded "
        "enzymes and the strongest test that findings are not stability "
        "artifacts. Note only 33 single mutants, so its unseen_position split "
        "must be derived from multi-mutant position sets.",
    ),
    Choice(
        "TPMT_HUMAN_Matreyek_2018",
        "2k-10k",
        "Human thiopurine methyltransferase abundance by VAMP-seq (EC 2.1.1.67), "
        "3.6k variants, 245 aa, 78% position coverage. Supplies expression at "
        "mid-N and an EC class 2 transferase. Single-mutant only, which is what "
        "the mutation_count_extrapolation training side needs.",
    ),
    # ---- 500-2k -----------------------------------------------------------
    Choice(
        "CCDB_ECOLI_Adkar_2012",
        "500-2k",
        "E. coli CcdB toxin activity, 1.2k variants, 101 aa, 61% position "
        "coverage. Small prokaryotic non-enzyme; toxicity readout is mechanically "
        "unlike the fluorescence and growth reporters elsewhere in the set.",
    ),
    Choice(
        "RL40A_YEAST_Roscoe_2014",
        "500-2k",
        "Yeast ubiquitin E1 reactivity, 1.4k variants, 128 aa. Ubiquitin is one "
        "of the most constrained folds known, with very high MSA depth — the "
        "opposite regime to GFP, so together they bracket the range over which "
        "evolutionary likelihood and measured fitness can disagree.",
    ),
    Choice(
        "FECA_ECOLI_Tsuboyama_2023_2D1U",
        "500-2k",
        "E. coli FecA N-terminal domain folding stability from cDNA-display "
        "proteolysis, 1.9k variants (667 multi-mutant), 89% position coverage, "
        "two biological replicates so sigma_v is measured rather than imputed. "
        "Represents the Tsuboyama mega-scale source, which is 62 of the 64 "
        "eligible stability assays and cannot be omitted without "
        "misrepresenting the benchmark. At 72 aa it is below the preferred "
        "100-500 band; this is an accepted, reported cost — the whole Tsuboyama "
        "corpus is 37-72 aa, so there is no in-band alternative.",
    ),
)

SELECTED_IDS: tuple[str, ...] = tuple(c.assay_id for c in SELECTION)

# Assays deliberately kept in reserve, with the reason. Documented so the
# rejected set is explicit rather than implied by absence.
ALTERNATES: dict[str, str] = {
    "AMIE_PSEAE_Wrenbeck_2017": "Amidase activity (EC 3.5.1.4), 6.2k variants. Displaced by OXDA for EC-class diversity; first substitute if OXDA fails QC.",
    "SPG1_STRSG_Wu_2016": "GB1 four-site combinatorial library, 149k variants up to quadruple mutants. Supports mutation-count extrapolation to n=3,4 but only 4 mutated positions, so no unseen_position split.",
    "RASK_HUMAN_Weng_2022_binding-DARPin_K55": "Binding on the same KRAS library as the selected abundance assay; enables a same-protein phenotype contrast if that becomes a question.",
    "A4_HUMAN_Seuma_2022": "Amyloid-beta aggregation, 14.8k variants mostly multi-mutant. Rejected: mutations span only 42 positions of a 770-aa precursor, so per-residue activation caching is dominated by never-mutated context.",
    "MET_HUMAN_Estevam_2023": "MET kinase activity (EC 2.7.10.1), 5.4k variants, 99% position coverage. Strong alternate; displaced only to keep four phenotype classes balanced.",
    "NUD15_HUMAN_Suiter_2020": "NUDT15 abundance, 2.8k variants, 91% position coverage. Alternate for TPMT in the expression slot.",
    "CBPA2_HUMAN_Tsuboyama_2023_1O6X": "Tsuboyama 72-aa domain, 2.1k variants, 99% position coverage. Alternate for FECA if its QC fails.",
}


@dataclass
class VerificationResult:
    ok: bool
    checks: dict[str, tuple[bool, str]] = field(default_factory=dict)

    def raise_if_failed(self) -> None:
        bad = {k: v[1] for k, v in self.checks.items() if not v[0]}
        if bad:
            msg = "\n".join(f"  - {k}: {m}" for k, m in bad.items())
            raise ValueError(f"selection violates Step 1 criteria:\n{msg}")


def verify(candidates: pd.DataFrame, cfg: dict) -> VerificationResult:
    """Check the hand-chosen selection against every stated criterion."""
    sel = cfg["selection"]
    s = candidates[candidates.DMS_id.isin(SELECTED_IDS)].copy()
    checks: dict[str, tuple[bool, str]] = {}

    missing = set(SELECTED_IDS) - set(s.DMS_id)
    checks["all_present"] = (not missing, f"not in candidate pool: {sorted(missing)}")
    checks["count"] = (
        len(s) == sel["n_assays"],
        f"selected {len(s)}, config expects {sel['n_assays']}",
    )
    checks["hard_filters"] = (
        bool(s.eligible.all()),
        f"fail hard filters: {sorted(s.loc[~s.eligible, 'DMS_id'])}",
    )
    for bin_name, want in sel["size_bins"].items():
        lo_hi = bin_name.replace("-inf", "-999999999").split("-")
        lo, hi = int(lo_hi[0]), int(lo_hi[1])
        got = int(s.DMS_total_number_mutants.between(lo, hi - 1).sum())
        checks[f"size_bin[{bin_name}]"] = (got == want, f"got {got}, want {want}")
    n_multi = int(s.includes_multiple_mutants.astype(bool).sum())
    checks["multimutant"] = (
        n_multi >= sel["min_multimutant_assays"],
        f"{n_multi} multi-mutant assays, need >= {sel['min_multimutant_assays']}",
    )
    got_ph = set(s.coarse_selection_type)
    checks["phenotype_coverage"] = (
        got_ph == set(sel["keep_phenotypes"]),
        f"missing phenotypes: {sorted(set(sel['keep_phenotypes']) - got_ph)}",
    )
    n_org = s.source_organism.nunique()
    checks["organism_diversity"] = (n_org >= 7, f"only {n_org} distinct organisms")
    if "ec_class" in s:
        n_ec = s.ec_class.dropna().nunique()
        checks["ec_class_diversity"] = (n_ec >= 3, f"only {n_ec} distinct EC classes")
    return VerificationResult(ok=True, checks=checks)


def selection_frame(candidates: pd.DataFrame) -> pd.DataFrame:
    """Selected assays joined to their metadata, in selection order."""
    lut = {c.assay_id: (i, c.slot, c.rationale) for i, c in enumerate(SELECTION)}
    s = candidates[candidates.DMS_id.isin(SELECTED_IDS)].copy()
    s["sel_order"] = s.DMS_id.map(lambda a: lut[a][0])
    s["slot"] = s.DMS_id.map(lambda a: lut[a][1])
    s["rationale"] = s.DMS_id.map(lambda a: lut[a][2])
    return s.sort_values("sel_order").reset_index(drop=True)
