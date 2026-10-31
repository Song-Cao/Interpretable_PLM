from __future__ import annotations

import numpy as np
import pytest

from crossfit.activation_cache import (
    load_cached_model,
    normalization_scalar,
    normalize_shards_in_place,
)


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


def test_load_cached_model_materializes_named_aligned_shards(tmp_path):
    model_dir = tmp_path / "shuffled"
    model_dir.mkdir()
    first = np.arange(12, dtype=np.float16).reshape(2, 2, 3)
    second = np.arange(6, dtype=np.float16).reshape(1, 2, 3)
    np.save(model_dir / "shard_0000.npy", first)
    np.save(model_dir / "shard_0001.npy", second)
    (tmp_path / "manifest.json").write_text(
        '{"status":"complete","dimension":3,"models":{"shuffled":{"shards":'
        '[{"name":"shard_0000.npy"},{"name":"shard_0001.npy"}]}}}'
    )
    observed = load_cached_model(tmp_path, "shuffled")
    assert observed.shape == (6, 3)
    assert np.array_equal(observed[:4], first.reshape(-1, 3))
    with pytest.raises(KeyError):
        load_cached_model(tmp_path, "missing")
