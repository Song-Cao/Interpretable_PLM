from __future__ import annotations

import numpy as np
import pytest

from crossfit.activation_cache import normalization_scalar, normalize_shards_in_place


def test_fixed_scalar_normalization_uses_one_dataset_level_value(tmp_path):
    values = np.array([[[3.0, 4.0], [0.0, 2.0]], [[1.0, 0.0], [2.0, 2.0]]], dtype=np.float16)
    path = tmp_path / "shard.npy"
    np.save(path, values)
    raw = values.astype(np.float32)
    scalar = normalization_scalar(float(np.linalg.norm(raw, axis=-1).sum()), 4, 2)
    realized = normalize_shards_in_place([path], scalar)
    observed = np.load(path).astype(np.float32)
    assert np.allclose(observed, raw * scalar, atol=2e-3)
    assert np.isclose(realized, np.sqrt(2), atol=2e-3)


@pytest.mark.parametrize(("total", "count"), [(0.0, 4), (1.0, 0)])
def test_normalization_rejects_empty_or_zero_norm_input(total, count):
    with pytest.raises(ValueError):
        normalization_scalar(total, count, 2)
