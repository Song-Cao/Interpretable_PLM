"""Split construction.

Every split is a deterministic function of (assay_id, scheme, seed) via
``utils.stable_hash``, so nothing depends on dict or file iteration order.

Schemes
-------
``pg_kfold``
    The bundle's own published 5-fold masks, carried through verbatim where they
    align 1:1 to the source rows. Where alignment fails (FecA, whose masks index
    the pre-dedup row order) the fold is re-derived deterministically and the
    manifest records the substitution.

``unseen_position``
    Held-out MUTATED POSITIONS, not held-out variants. A variant goes to test
    iff any of its mutated positions is in the held-out position set. This is
    the split that tests B2 (OOD failure at unseen positions), and it is the
    reason position coverage was a selection criterion.

``mutation_count_extrapolation``
    Train on variants with n_mutations <= k, test on n_mutations > k. Only
    defined for assays that actually contain multi-mutants.

``low_n``
    Nested training subsets of size N in {16..512} x 10 seeds, drawn from the
    pg_kfold training portion. Nested means the N=16 subset is a subset of
    N=32, so the low-N curve is monotone in information rather than in luck.

Reservations (carved out BEFORE any of the above)
-------------------------------------------------
``layer_freeze``  - within-assay variant subset used only to pick the crosscoder
                    layer; excluded from every reported analysis.
``pred_group``    - K grouped assay folds for the prediction track only. This is
                    an assay-level label, not a variant-level one.
"""
from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd

from .utils import stable_hash

TOKEN_RE = re.compile(r"^([A-Za-z])(\d+)([A-Za-z])$")


def mutated_positions(mutant: str) -> tuple[int, ...]:
    out = []
    for t in str(mutant).split(":"):
        g = TOKEN_RE.match(t)
        if g:
            out.append(int(g.group(2)))
    return tuple(sorted(set(out)))


def _rng(assay_id: str, scheme: str, seed: int) -> np.random.Generator:
    return np.random.default_rng(stable_hash(assay_id, scheme, seed) % (2**32))


def carve_layer_freeze(
    df: pd.DataFrame, assay_id: str, frac: float, seed: int
) -> np.ndarray:
    """Boolean mask of the layer-selection reserve. Never enters any analysis."""
    rng = _rng(assay_id, "layer_freeze", seed)
    n = len(df)
    k = max(1, int(round(frac * n)))
    idx = rng.choice(n, size=k, replace=False)
    m = np.zeros(n, dtype=bool)
    m[idx] = True
    return m


def kfold_fallback(df: pd.DataFrame, assay_id: str, n_folds: int, seed: int) -> np.ndarray:
    rng = _rng(assay_id, "pg_kfold_fallback", seed)
    return rng.permutation(np.arange(len(df)) % n_folds)


def unseen_position_masks(
    df: pd.DataFrame, assay_id: str, n_folds: int, seed: int
) -> np.ndarray:
    """Bitmask over position-folds touched by each variant.

    Positions (not variants) are partitioned into ``n_folds`` groups. Each
    variant gets an integer whose bit f is set iff the variant mutates at least
    one position assigned to group f. Bit 0 is group 0, so a single mutant always
    has exactly one bit set and a k-mutant has 1..k bits set.

    Decoding for held-out fold f (see ``unseen_position_split``):

        test  = mask == (1 << f)     -> EVERY mutated position is unseen
        train = mask & (1 << f) == 0 -> NO mutated position is held out
        drop  = otherwise            -> straddles the boundary, unusable

    The three-way rule is the point. Assigning a straddling multi-mutant to test
    (e.g. by "any held-out position") leaks: its other positions are in train, so
    the model has seen most of the variant's context and the fold no longer
    measures extrapolation to unseen positions. Assigning it to train leaks in the
    other direction — it puts a held-out position into training. Such variants
    have to be discarded per fold, which is why ``unseen_position_report``
    tabulates how many are lost.
    """
    pos_sets = df["mutant"].map(mutated_positions)
    all_pos = sorted({p for ps in pos_sets for p in ps})
    rng = _rng(assay_id, "unseen_position", seed)
    perm = rng.permutation(len(all_pos))
    pos_fold = {p: int(perm[i] % n_folds) for i, p in enumerate(all_pos)}
    out = np.zeros(len(df), dtype=int)
    for i, ps in enumerate(pos_sets):
        m = 0
        for p in ps:
            m |= 1 << pos_fold[p]
        out[i] = m
    return out


def unseen_position_split(mask: np.ndarray, fold: int) -> tuple[np.ndarray, np.ndarray]:
    """(train_idx, test_idx) positional indices from the bitmask, for one fold."""
    bit = 1 << int(fold)
    train = np.where((mask > 0) & ((mask & bit) == 0))[0]
    test = np.where(mask == bit)[0]
    return train, test


def unseen_position_report(
    df: pd.DataFrame,
    mask: np.ndarray,
    n_folds: int,
    min_test: int = 0,
    min_train: int = 0,
) -> dict[str, object]:
    pos_sets = df["mutant"].map(mutated_positions)
    n_pos = len({p for ps in pos_sets for p in ps})
    train_sizes, test_sizes, dropped, leaks = [], [], [], 0
    for f in range(n_folds):
        tr, te = unseen_position_split(mask, f)
        train_sizes.append(int(len(tr)))
        test_sizes.append(int(len(te)))
        dropped.append(int(len(mask) - len(tr) - len(te)))
        train_pos = {p for i in tr for p in pos_sets.iloc[i]}
        for i in te:
            if all(p in train_pos for p in pos_sets.iloc[i]):
                leaks += 1
    usable = [
        bool(te >= min_test and tr >= min_train)
        for tr, te in zip(train_sizes, test_sizes)
    ]
    return {
        "n_positions": n_pos,
        "encoding": "position_fold_bitmask",
        "train_sizes": train_sizes,
        "test_sizes": test_sizes,
        "dropped_straddling": dropped,
        "n_test_variants_fully_covered_by_train_positions": leaks,
        "fold_usable": usable,
        "n_usable_folds": int(sum(usable)),
        "b2_available": bool(any(usable)),
    }


def mutation_count_extrapolation(df: pd.DataFrame, k: int) -> np.ndarray:
    """0 = train (n_mut <= k), 1 = test (n_mut > k), -1 = undefined."""
    nm = df["n_mutations"].to_numpy()
    out = np.where(nm <= k, 0, 1)
    if (out == 1).sum() == 0:
        return np.full(len(df), -1, dtype=int)
    return out


def nested_low_n(
    train_idx: np.ndarray, assay_id: str, sizes: Iterable[int], seed: int
) -> dict[int, np.ndarray]:
    """Nested subsets: subsets[N] ⊂ subsets[N'] for N < N'.

    A single permutation of the training pool is taken and the first N entries
    form the size-N subset, which makes nesting exact by construction.
    """
    rng = _rng(assay_id, "low_n", seed)
    order = rng.permutation(train_idx)
    return {int(N): order[: int(N)] for N in sizes if int(N) <= len(order)}


def grouped_assay_folds(assay_ids: list[str], K: int, seed: int) -> dict[str, int]:
    """K grouped assay folds for the prediction track.

    Assays are sorted by identity (not by any data property) and dealt into K
    folds after a seeded permutation, so the assignment is reproducible and
    independent of the order assays were curated in.
    """
    ids = sorted(assay_ids)
    rng = np.random.default_rng(stable_hash("pred_group", seed) % (2**32))
    perm = rng.permutation(len(ids))
    return {ids[i]: int(perm[i] % K) for i in range(len(ids))}
