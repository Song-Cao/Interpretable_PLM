from __future__ import annotations

import numpy as np
import pytest

from crossfit.sampling import CappedSqrtSampler, allocate_capped_sqrt_counts


def test_sqrt_allocation_before_caps_bind():
    counts = allocate_capped_sqrt_counts(
        {"small": 100, "large": 400}, per_assay_cap=1000, epoch_size=150
    )
    assert counts == {"large": 100, "small": 50}


def test_allocation_never_oversamples_or_exceeds_cap():
    counts = allocate_capped_sqrt_counts(
        {"small": 10, "large": 10000}, per_assay_cap=100, epoch_size=110
    )
    assert counts == {"large": 100, "small": 10}
    with pytest.raises(ValueError):
        allocate_capped_sqrt_counts(
            {"small": 10, "large": 10000}, per_assay_cap=100, epoch_size=111
        )


def test_epoch_sampling_is_capped_unique_deterministic_and_single_assay():
    indices = {
        "a": np.arange(12),
        "b": np.arange(100, 140),
        "c": np.arange(1000, 1200),
    }
    sampler = CappedSqrtSampler(
        indices,
        cap_median_multiplier=2.0,
        batch_size=7,
        seed=17,
    )
    first = sampler.sample_epoch(epoch=3, epoch_size=60)
    repeated = sampler.sample_epoch(epoch=3, epoch_size=60)
    assert first.counts == repeated.counts
    assert [batch.assay_id for batch in first.batches] == [
        batch.assay_id for batch in repeated.batches
    ]
    assert all(
        np.array_equal(left.indices, right.indices)
        for left, right in zip(first.batches, repeated.batches)
    )
    for assay_id in indices:
        drawn = np.concatenate(
            [batch.indices for batch in first.batches if batch.assay_id == assay_id]
        )
        assert len(drawn) == first.counts[assay_id]
        assert len(np.unique(drawn)) == len(drawn)
        assert len(drawn) <= min(len(indices[assay_id]), first.per_assay_cap)
    assert all(batch.assay_id in indices for batch in first.batches)
    assert sum(first.shares.values()) == pytest.approx(1.0)


def test_new_epoch_redraws_rows():
    sampler = CappedSqrtSampler(
        {"a": np.arange(100), "b": np.arange(100, 300)},
        cap_median_multiplier=4.0,
        batch_size=10,
        seed=23,
    )
    first = sampler.sample_epoch(epoch=0, epoch_size=50)
    second = sampler.sample_epoch(epoch=1, epoch_size=50)
    first_rows = {
        assay: set(
            np.concatenate([b.indices for b in first.batches if b.assay_id == assay]).tolist()
        )
        for assay in first.counts
    }
    second_rows = {
        assay: set(
            np.concatenate([b.indices for b in second.batches if b.assay_id == assay]).tolist()
        )
        for assay in second.counts
    }
    assert any(first_rows[assay] != second_rows[assay] for assay in first_rows)


def test_default_epoch_is_complete_capped_pool_and_records_shares():
    sampler = CappedSqrtSampler(
        {"small": np.arange(5), "large": np.arange(100, 200)},
        cap_median_multiplier=1.0,
        batch_size=16,
        seed=5,
    )
    epoch = sampler.sample_epoch(epoch=0)
    assert epoch.n_variants == sampler.capped_pool_size
    assert epoch.counts["small"] == 5
    assert epoch.counts["large"] == epoch.per_assay_cap
    assert {row["assay_id"] for row in epoch.records()} == {"small", "large"}
