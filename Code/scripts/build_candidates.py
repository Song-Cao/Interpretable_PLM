"""Build the Step 1 candidate table: join ProteinGym 2.0 bundles to v1 metadata,
apply the selection filters, and annotate the enzyme subset via UniProt EC codes.

Emits Data/interim/candidates.csv — every assay that survives the phenotype
filter, with per-criterion boolean flags so the selection is auditable and the
rejected set is explicit.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from crossfit.utils import REPO_ROOT, load_config  # noqa: E402

UNIPROT = "https://rest.uniprot.org/uniprotkb/search"


ACC_RE = re.compile(r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}$")
FIELDS = "accession,id,protein_name,ec,xref_pfam,organism_name,length,keyword"


def _uniprot_term(ident: str) -> str | None:
    """ProteinGym's `UniProt_ID` is a UniProt *mnemonic* (entry name) such as
    `BLAT_ECOLX` for most assays and a bare accession (`A0A1I9GEU1`) for some
    unreviewed entries. A few are neither — e.g. `ANCSZ`, an ancestral
    reconstruction with no UniProt record. Return None for those; a single
    malformed term makes UniProt reject the entire OR-joined query with 400.
    """
    if "_" in ident:
        return f"id:{ident}"
    if ACC_RE.match(ident):
        return f"accession:{ident}"
    return None


def _get_tsv(query: str, contact: str | None) -> list[dict]:
    params = {"query": query, "fields": FIELDS, "format": "tsv", "size": 500}
    if contact:
        params["email"] = contact
    for attempt in range(3):
        try:
            r = requests.get(UNIPROT, params=params, timeout=90)
            r.raise_for_status()
            lines = r.text.strip().split("\n")
            if len(lines) < 2:
                return []
            hdr = lines[0].split("\t")
            return [dict(zip(hdr, ln.split("\t"))) for ln in lines[1:]]
        except Exception as exc:
            if attempt == 2:
                raise exc
            time.sleep(2 * (attempt + 1))
    return []


def fetch_uniprot_annotations(ids: list[str], contact: str | None = None) -> pd.DataFrame:
    """Batch-query UniProt for EC numbers, protein names, Pfam and keywords.

    Chunked OR-joined queries, falling back to per-id queries when a chunk is
    rejected, so one unresolvable identifier cannot blank out 20 others.
    """
    rows: list[dict] = []
    terms = {i: _uniprot_term(i) for i in ids}
    unresolvable = [i for i, t in terms.items() if t is None]
    if unresolvable:
        print(f"[info] no UniProt query form for: {unresolvable}", file=sys.stderr)
    good = [i for i, t in terms.items() if t is not None]

    for i in range(0, len(good), 20):
        chunk = good[i : i + 20]
        try:
            rows += _get_tsv(" OR ".join(terms[a] for a in chunk), contact)
        except Exception as exc:
            print(f"[warn] chunk at {i} failed ({exc}); retrying per-id", file=sys.stderr)
            for a in chunk:
                try:
                    rows += _get_tsv(terms[a], contact)
                except Exception as exc2:
                    print(f"[warn]   {a}: {exc2}", file=sys.stderr)
                time.sleep(0.2)
        time.sleep(0.3)
    return pd.DataFrame(rows)


def size_bin(n: int) -> str:
    if n < 500:
        return "<500"
    if n < 2000:
        return "500-2k"
    if n < 10000:
        return "2k-10k"
    return ">10k"


def main() -> None:
    cfg = load_config()
    sel = cfg["selection"]
    ref_dir = REPO_ROOT / cfg["paths"]["reference"]
    interim = REPO_ROOT / cfg["paths"]["interim"]
    interim.mkdir(parents=True, exist_ok=True)

    lock = json.loads((ref_dir / "sources_lock.json").read_text())
    manifest = lock["proteingym2"]["file_manifest"]
    avail = {Path(p).stem: v for p, v in manifest.items()}

    ref = pd.read_csv(ref_dir / "DMS_substitutions.csv")
    ref["has_pgdata"] = ref.DMS_id.isin(avail)
    df = ref[ref.has_pgdata].copy()
    df["pgdata_bytes"] = df.DMS_id.map(lambda i: avail[i]["size"])
    df["pgdata_oid"] = df.DMS_id.map(lambda i: avail[i]["oid"])

    # ---- selection criteria, one boolean column each -------------------------
    df["f_phenotype"] = df.coarse_selection_type.isin(sel["keep_phenotypes"])
    df["f_size"] = df.DMS_total_number_mutants >= sel["min_variants"]
    df["f_length"] = df.seq_len <= sel["max_seq_len"]
    lo, hi = sel["preferred_seq_len"]
    df["f_length_preferred"] = df.seq_len.between(lo, hi)
    df["f_multimutant"] = df.includes_multiple_mutants.astype(bool)
    df["size_bin"] = df.DMS_total_number_mutants.map(size_bin)
    df["frac_multi"] = df.DMS_number_multiple_mutants / df.DMS_total_number_mutants
    df["eligible"] = df.f_phenotype & df.f_size & df.f_length

    cand = df[df.f_phenotype].copy()

    # ---- enzyme subset via UniProt EC ---------------------------------------
    ids = sorted({str(u) for u in cand.UniProt_ID.dropna().unique()})
    ann = fetch_uniprot_annotations(ids, contact=None)
    if not ann.empty:
        ann = ann.rename(
            columns={
                "Entry": "acc",
                "Entry Name": "entry_name",
                "EC number": "ec_number",
                "Protein names": "protein_name",
            }
        )
        keep_cols = [c for c in ("acc", "entry_name", "ec_number", "protein_name", "Pfam", "Keywords") if c in ann.columns]
        keep = ann[keep_cols].copy()
        # UniProt can return several records per entry name (e.g. demerged
        # entries): prefer the informative one, then drop remaining duplicates.
        keep["_rank"] = (
            keep.get("protein_name", pd.Series("", index=keep.index)).fillna("").str.contains("demerged").astype(int)
        )
        keep = keep.sort_values("_rank").drop(columns="_rank")
        # join on whichever identifier form ProteinGym used
        lut: dict[str, dict] = {}
        if "entry_name" in keep:
            lut.update(keep.drop_duplicates("entry_name").set_index("entry_name").to_dict("index"))
        lut.update(keep.drop_duplicates("acc").set_index("acc").to_dict("index"))
        cand = cand.join(
            pd.DataFrame.from_dict(
                {i: lut.get(i, {}) for i in cand.UniProt_ID.astype(str)}, orient="index"
            ).rename_axis("UniProt_ID").drop(columns=["acc", "entry_name"], errors="ignore"),
            on="UniProt_ID",
        )
        cand["is_enzyme"] = cand.ec_number.notna() & (cand.ec_number.astype(str).str.strip() != "")
        cand["ec_class"] = cand.ec_number.astype(str).str.split(".").str[0].where(cand.is_enzyme)
    else:
        print("[warn] UniProt annotation empty — is_enzyme unresolved", file=sys.stderr)
        cand["ec_number"] = pd.NA
        cand["is_enzyme"] = pd.NA
        cand["ec_class"] = pd.NA

    out_cols = [
        "DMS_id", "UniProt_ID", "coarse_selection_type", "selection_assay", "selection_type",
        "source_organism", "taxon", "molecule_name", "protein_name", "ec_number", "is_enzyme",
        "ec_class", "seq_len", "region_mutated", "DMS_total_number_mutants",
        "DMS_number_single_mutants", "DMS_number_multiple_mutants", "frac_multi", "size_bin",
        "includes_multiple_mutants", "MSA_Neff_L", "MSA_Neff_L_category", "pdb_file", "pdb_range",
        "raw_DMS_directionality", "first_author", "year", "title", "pgdata_bytes", "pgdata_oid",
        "f_phenotype", "f_size", "f_length", "f_length_preferred", "f_multimutant", "eligible",
    ]
    out_cols = [c for c in out_cols if c in cand.columns]
    cand = cand[out_cols].sort_values(["coarse_selection_type", "DMS_total_number_mutants"])
    cand.to_csv(interim / "candidates.csv", index=False)

    print(f"candidates (phenotype-filtered): {len(cand)}")
    print(f"eligible (all hard filters):     {int(cand.eligible.sum())}")
    print(cand.groupby("coarse_selection_type").eligible.agg(["size", "sum"]).to_string())
    if cand.is_enzyme.notna().any():
        print(f"enzymes among eligible: {int(cand.loc[cand.eligible, 'is_enzyme'].fillna(False).sum())}")


if __name__ == "__main__":
    main()
