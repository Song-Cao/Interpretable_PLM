"""Uniform loader for the curated CROSSFIT assay set.

    from crossfit.datasets import list_assays, load_assay, load_all, get_split

    load_assay("CCDB_ECOLI_Adkar_2012")            -> DataFrame, one row/variant
    load_assay(aid, with_sequences=True)           -> adds mutant_sequence
    get_split(aid, "unseen_position", fold=0)      -> (train_idx, test_idx)
    get_split(aid, "low_n", N=64, seed=3)          -> (train_idx, test_idx)

Design notes
------------
* ``mutant_sequence`` is RECONSTRUCTED from ``wt_sequence`` + ``mutations`` on
  demand, never stored. ``load_assay(..., with_sequences=True)`` verifies the
  reconstruction against the stored ``n_mutations`` and position list, so a
  corrupted mutation string cannot silently produce a wrong sequence.
* The layer-freeze reserve is EXCLUDED by default from every split and from
  ``load_assay``. Pass ``include_reserve=True`` only in the layer-selection
  script. This is the mechanism that keeps the reserve out of reported results
  rather than a convention someone has to remember.
* Splits are returned as positional indices into the DataFrame returned by
  ``load_assay`` with the same arguments, so index alignment is not the caller's
  problem.
"""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pandas as pd

from .qc import apply_mutations
from .utils import load_config, resolve

_SCHEMES = ("pg_kfold", "unseen_position", "mutation_count_extrapolation", "low_n")


def _processed_dir() -> Path:
    return resolve(load_config(), "processed")


@functools.lru_cache(maxsize=1)
def _variants() -> pd.DataFrame:
    return pd.read_parquet(_processed_dir() / "variants.parquet")


@functools.lru_cache(maxsize=1)
def manifest() -> pd.DataFrame:
    """One row per assay: provenance, WT sequence, QC verdict, counts."""
    return pd.read_csv(_processed_dir() / "assay_manifest.csv")


class UnusableSplit(RuntimeError):
    """Raised when a requested fold exists but is too small to report on."""


@functools.lru_cache(maxsize=1)
def unseen_position_report_json() -> dict:
    import json

    p = _processed_dir() / "unseen_position_report.json"
    return json.loads(p.read_text()) if p.exists() else {}


@functools.lru_cache(maxsize=1)
def _splits() -> dict[str, pd.DataFrame]:
    d = _processed_dir() / "splits"
    return {s: pd.read_csv(d / f"{s}.csv") for s in _SCHEMES if (d / f"{s}.csv").exists()}


def list_assays() -> list[str]:
    return sorted(manifest()["assay_id"].tolist())


def wt_sequence(assay_id: str) -> str:
    m = manifest()
    row = m.loc[m["assay_id"] == assay_id]
    if row.empty:
        raise KeyError(f"unknown assay: {assay_id}")
    return str(row["wt_sequence"].iloc[0])


def load_assay(
    assay_id: str,
    with_sequences: bool = False,
    include_reserve: bool = False,
    drop_censored: bool = False,
) -> pd.DataFrame:
    v = _variants()
    df = v.loc[v["assay_id"] == assay_id].copy()
    if df.empty:
        raise KeyError(f"unknown assay: {assay_id}")
    if not include_reserve:
        df = df.loc[~df["is_layer_freeze_reserve"]]
    if drop_censored:
        df = df.loc[df["censored"] == "none"]
    df = df.reset_index(drop=True)

    if with_sequences:
        wt = wt_sequence(assay_id)
        seqs = [apply_mutations(wt, m) for m in df["mutations"]]
        if any(s is None for s in seqs):
            raise ValueError(f"{assay_id}: unparseable mutation string")
        lens = {len(s) for s in seqs}
        if lens != {len(wt)}:
            raise ValueError(f"{assay_id}: reconstructed length {lens} != wt {len(wt)}")
        # the reconstruction must differ from WT at exactly n_mutations positions
        nd = np.array([sum(a != b for a, b in zip(s, wt)) for s in seqs])
        bad = int((nd != df["n_mutations"].to_numpy()).sum())
        if bad:
            raise ValueError(
                f"{assay_id}: {bad} variants differ from WT at a number of positions "
                "inconsistent with n_mutations"
            )
        df["mutant_sequence"] = seqs
        df["wt_sequence"] = wt
    return df


def load_all(**kwargs) -> pd.DataFrame:
    return pd.concat([load_assay(a, **kwargs) for a in list_assays()], ignore_index=True)


def get_split(
    assay_id: str,
    scheme: str,
    fold: int | None = None,
    N: int | None = None,
    seed: int | None = None,
    include_reserve: bool = False,
    require_usable: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (train_idx, test_idx) as positional indices into load_assay(...).

    ``pg_kfold``: ``fold`` is the held-out fold; train is every other fold.
    ``unseen_position``: ``fold`` is the held-out POSITION group. Test contains
    only variants ALL of whose mutated positions are held out; train only
    variants NONE of whose positions are held out. Multi-mutants that straddle
    the boundary are dropped for that fold — train + test < n by design, and the
    per-fold loss is in ``unseen_position_report.json``. This is what makes the
    B2 fold a real extrapolation test rather than a partially-seen one.
    ``mutation_count_extrapolation``: no fold argument; fold 0 is train, 1 test.
    ``low_n``: requires ``N`` and ``seed``; the returned train set is the nested
    size-N subset and the test set is the pg_kfold fold-0 holdout.
    """
    df = load_assay(assay_id, include_reserve=include_reserve)
    row_to_pos = {int(r): i for i, r in enumerate(df["variant_row"].to_numpy())}
    sp = _splits()

    def pos(rows: np.ndarray) -> np.ndarray:
        return np.array([row_to_pos[int(r)] for r in rows if int(r) in row_to_pos], dtype=int)

    if scheme == "low_n":
        if N is None or seed is None:
            raise ValueError("low_n requires N and seed")
        t = sp["low_n"]
        sel = t[(t["assay_id"] == assay_id) & (t["scheme"] == f"low_n_N{N}_seed{seed}")]
        if sel.empty:
            raise KeyError(f"no low_n subset for {assay_id} N={N} seed={seed}")
        k = sp["pg_kfold"]
        ktab = k[k["assay_id"] == assay_id]
        test_rows = ktab.loc[ktab["fold"] == 0, "variant_row"].to_numpy()
        return pos(sel["variant_row"].to_numpy()), pos(test_rows)

    if scheme == "unseen_position":
        if fold is None:
            raise ValueError("unseen_position requires fold")
        t = sp["unseen_position"]
        tab = t[t["assay_id"] == assay_id]
        if tab.empty:
            raise KeyError(f"unseen_position not defined for {assay_id}")
        # rows in the mask table are pre-reserve-exclusion variant_rows, so
        # realign onto the loaded frame before decoding
        keep = tab["variant_row"].map(lambda r: int(r) in row_to_pos)
        tab = tab[keep]
        mask = np.zeros(len(df), dtype=int)
        mask[pos(tab["variant_row"].to_numpy())] = tab["pos_fold_mask"].to_numpy(int)
        from .splits import unseen_position_split

        tr, te = unseen_position_split(mask, fold)
        if not require_usable:
            return tr, te
        rep = unseen_position_report_json().get(assay_id, {})
        usable = rep.get("fold_usable")
        if usable is not None and not usable[int(fold)]:
            raise UnusableSplit(
                f"{assay_id}: unseen_position fold {fold} has "
                f"train={len(tr)} test={len(te)}, below the configured minimum. "
                "This assay is combinatorially saturated over few positions, so a "
                "position-disjoint split cannot be formed without leakage. Report "
                "B2 as unavailable for it (manifest.unseen_position_available) "
                "rather than reporting an underpowered fold; pass "
                "require_usable=False only for diagnostics."
            )
        return tr, te

    if scheme not in sp:
        raise KeyError(f"unknown scheme {scheme}; have {sorted(sp)}")
    t = sp[scheme]
    tab = t[t["assay_id"] == assay_id]
    if tab.empty:
        raise KeyError(f"scheme {scheme} not defined for {assay_id}")

    if scheme == "mutation_count_extrapolation":
        tr = tab.loc[tab["fold"] == 0, "variant_row"].to_numpy()
        te = tab.loc[tab["fold"] == 1, "variant_row"].to_numpy()
        return pos(tr), pos(te)

    if fold is None:
        raise ValueError(f"{scheme} requires fold")
    te = tab.loc[tab["fold"] == fold, "variant_row"].to_numpy()
    tr = tab.loc[(tab["fold"] != fold) & (tab["fold"] >= 0), "variant_row"].to_numpy()
    return pos(tr), pos(te)


def prediction_groups() -> dict[str, int]:
    """Assay -> grouped fold id for the PREDICTION track only (K=3).

    Never use this for the measurement track: measurement uses one donor
    adapter, one dictionary, one tau (README standing rule 10).
    """
    t = pd.read_csv(_processed_dir() / "splits" / "prediction_groups.csv")
    return dict(zip(t["assay_id"], t["pred_group"].astype(int)))
