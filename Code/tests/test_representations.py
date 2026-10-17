from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from crossfit.representations import (
    GEORGIEV_CHANNELS,
    MutationVocabulary,
    aaindex_flat,
    aaindex_positional,
    esm_mean,
    esm_mutpool,
    esm_residue,
    load_georgiev_table,
    mutation_indicator,
    pair_representation,
)
from crossfit.utils import REPO_ROOT


TABLE_PATH = REPO_ROOT / "Code" / "config" / "georgiev_2009_standardized.csv"


@pytest.fixture(scope="module")
def georgiev():
    return load_georgiev_table(TABLE_PATH)


def test_georgiev_table_is_frozen_standardized_20_by_19(georgiev):
    values = georgiev.to_numpy()
    assert values.shape == (20, GEORGIEV_CHANNELS)
    assert np.allclose(values.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(values.var(axis=0, ddof=0), 1.0, atol=1e-12)


def test_aaindex_flat_is_sparse_positional_mutant_minus_wild_type(georgiev):
    wild_type = "ACDE"
    encoded = aaindex_flat(wild_type, ["C2W", "A1G:D3Y"], georgiev)
    assert sparse.isspmatrix_csr(encoded)
    assert encoded.shape == (2, len(wild_type) * GEORGIEV_CHANNELS)
    assert encoded[0].nnz == GEORGIEV_CHANNELS
    first = encoded[0].toarray().reshape(len(wild_type), GEORGIEV_CHANNELS)
    assert np.count_nonzero(first[0]) == 0
    assert np.allclose(first[1], georgiev.loc["W"] - georgiev.loc["C"])
    assert np.count_nonzero(first[2:]) == 0
    assert encoded[1].nnz == 2 * GEORGIEV_CHANNELS


def test_conv_representation_is_exact_unflattened_sparse_matrix(georgiev):
    mutations = ["C2W", "A1G:D3Y"]
    flat = aaindex_flat("ACDE", mutations, georgiev)
    positional = aaindex_positional("ACDE", mutations, georgiev)
    assert positional.shape == (2, 4, GEORGIEV_CHANNELS)
    assert np.array_equal(positional.reshape(2, -1), flat.toarray())


def test_published_exact_zero_difference_is_not_changed(georgiev):
    encoded = aaindex_flat("A", ["A1V"], georgiev)
    assert encoded.shape == (1, GEORGIEV_CHANNELS)
    assert encoded.nnz == GEORGIEV_CHANNELS - 1
    assert encoded[0, 3] == 0.0


@pytest.mark.parametrize("mutation", ["A1X", "B1A", "A9C", "C1A", "A1A"])
def test_aaindex_rejects_noncanonical_or_inconsistent_mutations(georgiev, mutation):
    with pytest.raises(ValueError):
        aaindex_flat("ACDE", [mutation], georgiev)


def test_mutation_indicator_uses_observed_position_amino_acid_pairs_only():
    mutations = ["A1G", "A1V:D3Y", "D3Y"]
    vocabulary = MutationVocabulary.from_mutations("ACDE", mutations)
    assert vocabulary.features == ((1, "G"), (1, "V"), (3, "Y"))
    encoded = mutation_indicator("ACDE", mutations, vocabulary)
    assert sparse.isspmatrix_csr(encoded)
    assert encoded.shape == (3, 3)
    assert encoded.toarray().tolist() == [[1, 0, 0], [0, 1, 1], [0, 0, 1]]


def test_precomputed_esm_representation_shapes_and_pooling():
    activations = np.arange(2 * 4 * 3, dtype=float).reshape(2, 4, 3)
    assert esm_residue(activations).shape == (2, 4, 3)
    assert np.array_equal(esm_mean(activations), activations.mean(axis=1))
    pooled = esm_mutpool(activations, ["A1G:D3Y", "C2W"])
    assert np.array_equal(pooled[0], activations[0, [0, 2]].mean(axis=0))
    assert np.array_equal(pooled[1], activations[1, 1])
    assert esm_mutpool(activations[0], "A1G").shape == (3,)


def test_mutated_pooling_rejects_wild_type_and_bad_positions():
    activations = np.ones((4, 3))
    with pytest.raises(ValueError):
        esm_mutpool(activations, "WT")
    with pytest.raises(ValueError):
        esm_mutpool(activations, "A9G")


def test_pair_fusion_rule_subtracts_early_and_preserves_late_control():
    left = sparse.csr_matrix([[1.0, 2.0]])
    right = sparse.csr_matrix([[0.5, 1.5]])
    early = pair_representation(left, right, fusion="early")
    assert sparse.issparse(early)
    assert np.array_equal(early.toarray(), [[0.5, 0.5]])
    late = pair_representation(left, right, fusion="late")
    assert late[0] is left
    assert late[1] is right
