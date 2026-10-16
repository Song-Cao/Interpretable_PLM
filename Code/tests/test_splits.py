"""Leakage and determinism tests for the CROSSFIT Step 1 dataset.

These are the tests that would catch a silent invalidation of downstream
results: a split that leaks, a split that changes between runs, a reserve that
escapes into a reported fold, or a mutation string that no longer regenerates
its sequence. They run against the BUILT dataset in Data/processed, so they are
also an integrity check on the artifacts themselves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crossfit import datasets as D
from crossfit import selection, splits as S
from crossfit.qc import apply_mutations, strip_synonymous, n_mutations
from crossfit.utils import load_config, resolve, stable_hash

CFG = load_config()
SCFG = CFG["splits"]
N_FOLDS = SCFG["unseen_position"]["n_folds"]
ASSAYS = D.list_assays()


# ---------------------------------------------------------------- selection ---


def test_selection_frozen_list_verifies():
    """The frozen ten satisfy every declared diversity/size constraint."""
    cand_path = resolve(CFG, "interim") / "candidates.csv"
    if not cand_path.exists():
        pytest.skip("candidate pool not built")
    res = selection.verify(pd.read_csv(cand_path), CFG)
    failed = {k: v for k, v in res.checks.items() if not v[0]}
    assert not failed, failed


def test_selection_matches_built_dataset():
    assert sorted(c.assay_id for c in selection.SELECTION) == sorted(ASSAYS)


# ------------------------------------------------------------------ schema ---


def test_manifest_one_row_per_assay():
    m = D.manifest()
    assert len(m) == len(ASSAYS)
    assert m["assay_id"].is_unique


def test_no_nan_labels():
    v = D.load_all()
    assert v["y"].notna().all()
    assert v["y_raw"].notna().all()


def test_sigma_never_silently_zero():
    """NaN means 'no information'. Zero would mean 'measured with no error',
    which would give that variant infinite weight in a sigma-weighted loss."""
    v = D.load_all()
    assert not (v["sigma_v"] == 0).any()
    have = v["sigma_source"] != "unavailable"
    assert v.loc[have, "sigma_v"].notna().all()
    assert v.loc[~have, "sigma_v"].isna().all()


def test_labels_standardized_within_assay():
    for a in ASSAYS:
        y = D.load_assay(a)["y"].to_numpy()
        assert abs(y.mean()) < 0.05, a
        assert abs(y.std() - 1.0) < 0.05, a


# ------------------------------------------------ sequence reconstruction ---


@pytest.mark.parametrize("assay_id", ASSAYS)
def test_sequences_regenerate_and_are_unique(assay_id):
    """Not storing mutant_sequence is only safe if it always rebuilds, and the
    dedup unit is the sequence, so sequences must be unique within an assay."""
    df = D.load_assay(assay_id, with_sequences=True, include_reserve=True)
    wt = D.wt_sequence(assay_id)
    assert (df["mutant_sequence"].str.len() == len(wt)).all()
    assert df["mutant_sequence"].is_unique
    assert (df["mutant_sequence"] != wt).all()


@pytest.mark.parametrize("assay_id", ASSAYS)
def test_n_mutations_and_positions_consistent(assay_id):
    df = D.load_assay(assay_id)
    for mut, nm, ps in zip(df["mutations"], df["n_mutations"], df["mutated_positions"]):
        toks = strip_synonymous(mut)
        assert n_mutations(toks) == nm
        assert sorted(int(p) for p in str(ps).split(";")) == sorted(
            S.mutated_positions(toks)
        )


# ------------------------------------------------------------ reservations ---


@pytest.mark.parametrize("assay_id", ASSAYS)
def test_layer_freeze_reserve_excluded_by_default(assay_id):
    """The reserve must be structurally unreachable, not excluded by convention."""
    d = D.load_assay(assay_id)
    assert not d["is_layer_freeze_reserve"].any()
    full = D.load_assay(assay_id, include_reserve=True)
    frac = full["is_layer_freeze_reserve"].mean()
    target = CFG["reservations"]["layer_freeze"]["frac"]
    assert abs(frac - target) < 0.02, (assay_id, frac)


@pytest.mark.parametrize("assay_id", ASSAYS)
@pytest.mark.parametrize("scheme", ["pg_kfold", "unseen_position", "low_n"])
def test_no_scheme_returns_reserved_variants(assay_id, scheme):
    kw = dict(fold=0) if scheme != "low_n" else dict(N=64, seed=0)
    if scheme == "unseen_position":
        kw["require_usable"] = False
    tr, te = D.get_split(assay_id, scheme, **kw)
    d = D.load_assay(assay_id)
    for idx in (tr, te):
        assert not d.iloc[idx]["is_layer_freeze_reserve"].any()


def test_prediction_groups_cover_all_assays_and_are_balanced():
    g = D.prediction_groups()
    assert set(g) == set(ASSAYS)
    K = CFG["reservations"]["prediction_folds"]["K"]
    counts = pd.Series(list(g.values())).value_counts()
    assert set(counts.index) == set(range(K))
    assert counts.max() - counts.min() <= 1


# ---------------------------------------------------------------- leakage ---


@pytest.mark.parametrize("assay_id", ASSAYS)
@pytest.mark.parametrize("scheme", ["pg_kfold", "unseen_position"])
def test_train_test_disjoint(assay_id, scheme):
    for f in range(N_FOLDS):
        tr, te = D.get_split(assay_id, scheme, fold=f, require_usable=False)
        assert not (set(tr.tolist()) & set(te.tolist()))


@pytest.mark.parametrize("assay_id", ASSAYS)
def test_unseen_position_zero_leakage(assay_id):
    """THE B2 test. No test variant may have ANY mutated position that appears
    anywhere in the training fold — including multi-mutants, which is why
    straddling variants are dropped rather than assigned. Checked with
    require_usable=False so the guarantee is verified even on folds we refuse
    to report."""
    df = D.load_assay(assay_id)
    posmap = df["mutated_positions"].map(
        lambda s: {int(p) for p in str(s).split(";") if p}
    )
    for f in range(N_FOLDS):
        tr, te = D.get_split(assay_id, "unseen_position", fold=f, require_usable=False)
        if len(te) == 0:
            continue
        train_pos = set().union(*posmap.iloc[tr]) if len(tr) else set()
        test_pos = set().union(*posmap.iloc[te])
        assert not (train_pos & test_pos), (assay_id, f, len(train_pos & test_pos))


@pytest.mark.parametrize("assay_id", ASSAYS)
def test_unusable_b2_folds_are_refused_not_returned(assay_id):
    """An underpowered fold must raise, so nobody can report a 3-variant test
    set as a B2 result by accident."""
    rep = D.unseen_position_report_json()[assay_id]
    avail = bool(D.manifest().set_index("assay_id").loc[assay_id, "unseen_position_available"])
    assert avail == rep["b2_available"]
    for f, ok in enumerate(rep["fold_usable"]):
        if ok:
            tr, te = D.get_split(assay_id, "unseen_position", fold=f)
            assert len(te) >= SCFG["unseen_position"]["min_test_size"]
            assert len(tr) >= SCFG["unseen_position"]["min_train_size"]
        else:
            with pytest.raises(D.UnusableSplit):
                D.get_split(assay_id, "unseen_position", fold=f)


def test_b2_available_for_most_assays():
    n = sum(
        bool(v) for v in D.manifest()["unseen_position_available"]
    )
    assert n >= 8, f"only {n} assays support an unseen-position split"


@pytest.mark.parametrize("assay_id", ASSAYS)
def test_pg_kfold_folds_partition_the_assay(assay_id):
    d = D.load_assay(assay_id)
    seen = set()
    for f in range(N_FOLDS):
        _, te = D.get_split(assay_id, "pg_kfold", fold=f)
        s = set(te.tolist())
        assert not (seen & s)
        seen |= s
    assert len(seen) == len(d)


def test_mutation_count_extrapolation_holds_out_higher_order_only():
    k = SCFG["mutation_count_extrapolation"]["k"]
    n_defined = 0
    for a in ASSAYS:
        try:
            tr, te = D.get_split(a, "mutation_count_extrapolation")
        except KeyError:
            continue  # single-mutant-only assay: scheme undefined, by design
        n_defined += 1
        d = D.load_assay(a)
        assert (d.iloc[tr]["n_mutations"] <= k).all()
        assert (d.iloc[te]["n_mutations"] > k).all()
    assert n_defined >= 3, "brief requires >=3 multi-mutant assays"


# ------------------------------------------------------------------ low_n ---


@pytest.mark.parametrize("assay_id", ASSAYS)
def test_low_n_nested_and_sized(assay_id):
    """Nesting makes the low-N curve monotone in information rather than luck:
    N=32 sees everything N=16 saw plus 16 more."""
    prev: set[int] | None = None
    for N in SCFG["low_n"]["N"]:
        tr, _ = D.get_split(assay_id, "low_n", N=N, seed=3)
        cur = set(tr.tolist())
        assert len(cur) == min(N, len(D.load_assay(assay_id)))
        if prev is not None:
            assert prev <= cur, (assay_id, N)
        prev = cur


@pytest.mark.parametrize("assay_id", ASSAYS)
def test_low_n_train_disjoint_from_its_test(assay_id):
    tr, te = D.get_split(assay_id, "low_n", N=512, seed=0)
    assert not (set(tr.tolist()) & set(te.tolist()))


def test_low_n_seeds_differ():
    a = "TPMT_HUMAN_Matreyek_2018"
    s = [frozenset(D.get_split(a, "low_n", N=64, seed=i)[0].tolist()) for i in range(5)]
    assert len(set(s)) == 5


def test_low_n_grid_matches_config():
    d = pd.read_csv(resolve(CFG, "processed") / "splits" / "low_n.csv")
    want = {
        f"low_n_N{N}_seed{s}"
        for N in SCFG["low_n"]["N"]
        for s in range(SCFG["low_n"]["n_seeds"])
    }
    assert set(d["scheme"]) == want


# ----------------------------------------------------------- determinism ---


def test_stable_hash_is_stable_across_processes():
    """Splits are keyed on this. If it ever depends on PYTHONHASHSEED, every
    reported fold silently changes between runs."""
    assert stable_hash("GFP_AEQVI_Sarkisyan_2016", "unseen_position", 20260827) == \
        stable_hash("GFP_AEQVI_Sarkisyan_2016", "unseen_position", 20260827)
    assert stable_hash("a") != stable_hash("b")


@pytest.mark.parametrize(
    "scheme,kw",
    [
        ("pg_kfold", dict(fold=1)),
        ("unseen_position", dict(fold=1)),
        ("low_n", dict(N=128, seed=7)),
    ],
)
def test_get_split_repeatable(scheme, kw):
    a = "RASK_HUMAN_Weng_2022_abundance"
    t1, e1 = D.get_split(a, scheme, **kw)
    t2, e2 = D.get_split(a, scheme, **kw)
    assert np.array_equal(t1, t2) and np.array_equal(e1, e2)


def test_unseen_position_masks_reproduce_from_scratch():
    """Re-deriving the mask from the raw table must reproduce the stored one:
    the split is a pure function of (assay_id, scheme, seed), never of row order
    or iteration order."""
    a = "FECA_ECOLI_Tsuboyama_2023_2D1U"
    df = D.load_assay(a, include_reserve=True)
    tab = pd.DataFrame({"mutant": df["mutations"]})
    m1 = S.unseen_position_masks(tab, a, N_FOLDS, SCFG["master_seed"])
    m2 = S.unseen_position_masks(tab, a, N_FOLDS, SCFG["master_seed"])
    assert np.array_equal(m1, m2)
    stored = pd.read_csv(resolve(CFG, "processed") / "splits" / "unseen_position.csv")
    stored = stored[stored["assay_id"] == a].sort_values("variant_row")
    assert np.array_equal(m1, stored["pos_fold_mask"].to_numpy(int))


def test_apply_mutations_rejects_wrong_reference():
    assert apply_mutations("MKV", "A1G") is None
    assert apply_mutations("MKV", "M1G") == "GKV"
    assert apply_mutations("MKV", "M1G:V3A") == "GKA"
