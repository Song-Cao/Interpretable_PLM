"""Per-assay quality control for DMS assays.

Design rules (see README standing rules):
  * Every check reports, and nothing is silently dropped. Rows removed from the
    regression set are recorded with a reason code, not deleted from the audit.
  * The mutation-string / reference-sequence consistency check is the critical
    one: a one-off indexing convention would corrupt every downstream
    per-residue activation without ever raising an error.
  * Deduplication is at the level of the *mutant amino-acid sequence*, because
    that is the unit CROSSFIT caches activations for. Two rows with identical
    sequences and different labels are label noise, and leakage if they land in
    different folds.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

TOKEN_RE = re.compile(r"^([A-Za-z])(\d+)([A-Za-z])$")
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


@dataclass
class AssayQC:
    assay_id: str
    n_rows_raw: int = 0
    n_tokens: int = 0
    n_unparsed_tokens: int = 0
    n_wt_mismatch: int = 0
    wt_mismatch_examples: list[str] = field(default_factory=list)
    n_out_of_range: int = 0
    wt_len: int = 0
    seq_len_min: int = 0
    seq_len_max: int = 0
    n_seq_len_ne_wt: int = 0
    n_invalid_residue_rows: int = 0
    n_rebuild_mismatch: int = 0
    n_synonymous_tokens: int = 0
    n_rows_fully_synonymous: int = 0
    n_dup_seq_groups: int = 0
    n_dup_seq_rows: int = 0
    n_dup_disagree: int = 0
    dup_pooled_sd: float | None = None
    n_rows_kept: int = 0
    n_nonsense_rows: int = 0
    n_missing_label: int = 0
    n_unique_y: int = 0
    ties_frac: float = 0.0
    frac_at_floor: float = 0.0
    frac_at_ceil: float = 0.0
    censor_floor_flagged: bool = False
    censor_ceil_flagged: bool = False
    published_mask_len: int | None = None
    published_mask_usable: bool = False
    published_mask_note: str = ""
    notes: list[str] = field(default_factory=list)

    def as_row(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["wt_mismatch_examples"] = ";".join(self.wt_mismatch_examples[:5])
        d["notes"] = " | ".join(self.notes)
        return d


def strip_synonymous(mutant: str) -> str:
    """Drop tokens whose WT and mutant residue are identical.

    ProteinGym mutation strings for the Tsuboyama assays carry synonymous
    components (e.g. ``K34A:S56S``). They do not change the protein sequence, so
    ``K34A`` and ``K34A:S56S`` denote the same molecule. Returns ``"WT"`` when
    every token is synonymous.
    """
    toks = [t for t in str(mutant).split(":") if not (len(t) > 2 and t[0] == t[-1])]
    return ":".join(toks) if toks else "WT"


def n_mutations(stripped: str) -> int:
    return 0 if stripped == "WT" else stripped.count(":") + 1


def apply_mutations(wt: str, stripped: str, check_reference: bool = True) -> str | None:
    """Apply a ':'-joined substitution string to ``wt``; None if unapplicable.

    Positions are 1-based with no offset (verified for all ten assays).

    With ``check_reference`` (the default) the token's FROM residue must match
    ``wt`` at that position. This matters because this function is the loader's
    sequence-reconstruction path: without the check, an off-by-one or
    wrong-isoform mutation string produces a plausible-looking sequence with the
    substitution at the wrong place, and every activation cached from it is
    silently wrong. Returning None instead makes that a loud failure.
    """
    if stripped == "WT":
        return wt
    s = list(wt)
    for t in stripped.split(":"):
        g = TOKEN_RE.match(t)
        if g is None:
            return None
        pos = int(g.group(2))
        if pos < 1 or pos > len(s):
            return None
        if check_reference and s[pos - 1] != g.group(1):
            return None
        s[pos - 1] = g.group(3)
    return "".join(s)


def check_mutation_strings(table: pd.DataFrame, wt: str, qc: AssayQC) -> None:
    """Validate every token against the reference sequence. Critical check."""
    for m in table["mutant"].astype(str):
        if m == "WT":
            continue
        for t in m.split(":"):
            g = TOKEN_RE.match(t)
            if g is None:
                qc.n_unparsed_tokens += 1
                continue
            qc.n_tokens += 1
            wt_aa, pos, mut_aa = g.group(1), int(g.group(2)), g.group(3)
            if wt_aa == mut_aa:
                qc.n_synonymous_tokens += 1
            if pos < 1 or pos > len(wt):
                qc.n_out_of_range += 1
                continue
            if wt[pos - 1] != wt_aa:
                qc.n_wt_mismatch += 1
                if len(qc.wt_mismatch_examples) < 5:
                    qc.wt_mismatch_examples.append(f"{t}(ref={wt[pos-1]})")


def run_assay_qc(
    assay_id: str,
    table: pd.DataFrame,
    wt: str,
    score_col: str = "DMS Score",
    published_masks: list[list[bool]] | None = None,
    censor_eps: float = 0.01,
    censor_mass_frac: float = 0.02,
    dup_disagreement_sigma: float = 3.0,
) -> tuple[pd.DataFrame, AssayQC]:
    """Return (clean_table, qc_record).

    The returned table is one row per unique mutant amino-acid sequence, with
    labels averaged over duplicates and a ``published_fold`` column where the
    bundle's own split masks could be aligned.
    """
    qc = AssayQC(assay_id=assay_id, wt_len=len(wt), n_rows_raw=len(table))
    df = table.reset_index(drop=True).copy()
    df["_row"] = np.arange(len(df))

    check_mutation_strings(df, wt, qc)

    qc.seq_len_min = int(df["sequence"].str.len().min())
    qc.seq_len_max = int(df["sequence"].str.len().max())
    qc.n_seq_len_ne_wt = int((df["sequence"].str.len() != len(wt)).sum())

    non_std = df["sequence"].map(lambda s: bool(set(str(s)) - STANDARD_AA))
    qc.n_invalid_residue_rows = int(non_std.sum())

    df["mut_stripped"] = df["mutant"].map(strip_synonymous)
    df["n_mutations"] = df["mut_stripped"].map(n_mutations)
    qc.n_rows_fully_synonymous = int((df["mut_stripped"] == "WT").sum())

    rebuilt = [apply_mutations(wt, s) for s in df["mut_stripped"]]
    qc.n_rebuild_mismatch = int(sum(r != s for r, s in zip(rebuilt, df["sequence"])))

    # Nonsense/stop variants: ProteinGym substitution files should not contain
    # them, but check rather than assume.
    qc.n_nonsense_rows = int(df["mutant"].astype(str).str.contains(r"\*|X", regex=True).sum())

    y = pd.to_numeric(df[score_col], errors="coerce")
    qc.n_missing_label = int(y.isna().sum())
    df["y_raw"] = y

    # ---- align published masks BEFORE dedup (masks index source row order) ----
    fold = None
    if published_masks is not None:
        L = len(published_masks[0])
        qc.published_mask_len = L
        if L == len(df):
            arr = np.array(published_masks, dtype=bool)
            if (arr.sum(axis=0) == 1).all():
                fold = arr.argmax(axis=0)
                qc.published_mask_usable = True
                qc.published_mask_note = "aligned 1:1 to source rows; disjoint partition"
            else:
                qc.published_mask_note = "masks are not a disjoint partition"
        else:
            qc.published_mask_note = (
                f"mask length {L} != {len(df)} source rows; cannot align, fold re-derived"
            )
    if fold is not None:
        df["published_fold"] = fold
    else:
        df["published_fold"] = -1

    # ---- drop rows unusable for regression, keeping the audit trail ----
    drop = y.isna() | (df["mut_stripped"] == "WT") | non_std
    kept = df.loc[~drop].copy()

    # ---- dedup on mutant sequence ----
    grp = kept.groupby("sequence", sort=False)
    sizes = grp.size()
    dupes = sizes[sizes > 1]
    qc.n_dup_seq_groups = int(len(dupes))
    qc.n_dup_seq_rows = int(dupes.sum())

    if len(dupes):
        sub = kept[kept["sequence"].isin(dupes.index)]
        gs = sub.groupby("sequence")["y_raw"]
        within_sd = gs.std(ddof=1)
        qc.dup_pooled_sd = float(np.sqrt(np.nanmean(within_sd.values ** 2)))
        spread = (gs.max() - gs.min()).abs()
        if qc.dup_pooled_sd and qc.dup_pooled_sd > 0:
            qc.n_dup_disagree = int((spread > dup_disagreement_sigma * qc.dup_pooled_sd).sum())

    agg = grp.agg(
        y_raw=("y_raw", "mean"),
        n_measurements=("y_raw", "size"),
        y_replicate_sd=("y_raw", lambda s: float(s.std(ddof=1)) if len(s) > 1 else np.nan),
        mutant=("mut_stripped", "first"),
        n_mutations=("n_mutations", "first"),
        published_fold=("published_fold", "first"),
        _row=("_row", "min"),
    ).reset_index()
    agg = agg.sort_values("_row").drop(columns="_row").reset_index(drop=True)
    agg["dup_disagrees"] = False
    if len(dupes) and qc.dup_pooled_sd:
        bad = spread[spread > dup_disagreement_sigma * qc.dup_pooled_sd].index
        agg.loc[agg["sequence"].isin(bad), "dup_disagrees"] = True

    qc.n_rows_kept = len(agg)

    yy = agg["y_raw"].to_numpy(float)
    qc.n_unique_y = int(len(np.unique(yy)))
    qc.ties_frac = float(1.0 - qc.n_unique_y / len(yy))
    lo, hi = float(yy.min()), float(yy.max())
    rng = hi - lo
    at_lo = float(np.mean(yy <= lo + censor_eps * rng))
    at_hi = float(np.mean(yy >= hi - censor_eps * rng))
    qc.frac_at_floor, qc.frac_at_ceil = at_lo, at_hi
    qc.censor_floor_flagged = at_lo >= censor_mass_frac
    qc.censor_ceil_flagged = at_hi >= censor_mass_frac

    censored = np.full(len(agg), "none", dtype=object)
    if qc.censor_floor_flagged:
        censored[yy <= lo + censor_eps * rng] = "floor"
    if qc.censor_ceil_flagged:
        censored[yy >= hi - censor_eps * rng] = "ceiling"
    agg["censored"] = censored

    return agg, qc
