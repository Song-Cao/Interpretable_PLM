from __future__ import annotations

import numpy as np
import pytest

from crossfit.representations import (
    AMINO_ACIDS,
    AAINDEX_CHANNELS,
    aaindex_flat,
    encode_representation,
    esm_mean,
    esm_mutpool,
    esm_residue_delta,
    load_aaindex_table,
    onehot,
    onehot_positional,
)
from crossfit.utils import REPO_ROOT


TABLE_PATH = REPO_ROOT / "Code" / "config" / "aaindex_georgiev_2009_standardized.csv"


@pytest.fixture(scope="module")
def aaindex_table():
    return load_aaindex_table(TABLE_PATH)


def test_aaindex_table_is_frozen_standardized_20_by_19(aaindex_table):
    values = aaindex_table.to_numpy()
    assert values.shape == (20, AAINDEX_CHANNELS)
    assert np.allclose(values.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(values.var(axis=0), 1.0, atol=1e-12)


def test_onehot_encodes_every_position_and_keeps_all_twenty_columns():
    positional = onehot_positional(["AC", "WA"])
    assert positional.shape == (2, 2, len(AMINO_ACIDS))
    assert np.all(positional.sum(axis=-1) == 1)
    assert onehot(["AC", "WA"]).shape == (2, 40)
    assert np.array_equal(onehot(["AC", "WA"]), positional.reshape(2, -1))


def test_aaindex_is_dense_absolute_whole_sequence_encoding(aaindex_table):
    encoded = aaindex_flat(["AC", "WA"], aaindex_table)
    assert isinstance(encoded, np.ndarray)
    assert encoded.shape == (2, 2 * AAINDEX_CHANNELS)
    np.testing.assert_allclose(encoded[0].reshape(2, -1)[0], aaindex_table.loc["A"])
    np.testing.assert_allclose(encoded[0].reshape(2, -1)[1], aaindex_table.loc["C"])


@pytest.mark.parametrize("sequences", [["AX"], ["A", "AA"], []])
def test_classical_encodings_reject_noncanonical_or_unaligned_sequences(sequences, aaindex_table):
    with pytest.raises(ValueError):
        aaindex_flat(sequences, aaindex_table)


def test_esm_representations_keep_alignment_and_pooling_domain():
    variant = np.arange(2 * 4 * 3, dtype=float).reshape(2, 4, 3)
    wild_type = np.ones((4, 3))
    np.testing.assert_array_equal(esm_residue_delta(variant, wild_type), variant - wild_type)
    np.testing.assert_array_equal(esm_mean(variant), variant.mean(axis=1))
    pooled = esm_mutpool(variant, ["A1G:D3Y", "C2W"])
    np.testing.assert_array_equal(pooled[0], variant[0, [0, 2]].mean(axis=0))
    np.testing.assert_array_equal(pooled[1], variant[1, 1])


def test_representation_factory_selects_published_names(aaindex_table):
    direct = aaindex_flat(["AC"], aaindex_table)
    selected = encode_representation(
        "aaindex_flat", sequences=["AC"], aaindex_table=aaindex_table
    )
    np.testing.assert_array_equal(selected, direct)
    with pytest.raises(KeyError):
        encode_representation("obsolete_encoding", sequences=["AC"])
