from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from crossfit.cache_provenance import (
    manifest_provenance,
    verify_reference_blocks,
    write_reference_block,
)


THRESHOLDS = {
    "minimum_pearson": 0.999,
    "maximum_relative_rmse": 0.05,
    "minimum_through_origin_scale": 0.98,
    "maximum_through_origin_scale": 1.02,
}


def _manifest(tmp_path: Path, values: np.ndarray) -> Path:
    reference = write_reference_block(
        tmp_path, "base", values, activation_units="raw", normalization_scalar=1.0
    )
    manifest = manifest_provenance(
        contract_version=1,
        dataset="IRED",
        split="test_fixture",
        layer_index=26,
        storage_dtype="float16",
        compute_dtype="float32",
        backbone_revision="fixture",
        activation_units="raw",
        reference_blocks={"base": reference},
        requested_attention="fixture",
    )
    manifest["dimension"] = values.shape[-1]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path


def test_reference_verification_accepts_small_storage_error(tmp_path):
    rng = np.random.default_rng(4)
    values = rng.normal(size=(12, 8)).astype(np.float32)
    manifest = _manifest(tmp_path, values)
    observed = values + rng.normal(scale=1e-4, size=values.shape)
    result = verify_reference_blocks(manifest, {"base": observed}, THRESHOLDS)
    assert result["base"]["pearson"] > 0.999


def test_reference_verification_rejects_wrong_layer(tmp_path):
    rng = np.random.default_rng(5)
    values = rng.normal(size=(12, 8)).astype(np.float32)
    manifest = _manifest(tmp_path, values)
    with pytest.raises(AssertionError, match="Pearson"):
        verify_reference_blocks(
            manifest, {"base": rng.normal(size=values.shape)}, THRESHOLDS
        )


def test_reference_verification_rejects_changed_reference(tmp_path):
    values = np.arange(96, dtype=np.float32).reshape(12, 8)
    manifest = _manifest(tmp_path, values)
    path = tmp_path / "reference_wild_type_base.npy"
    changed = values.copy()
    changed[0, 0] += 1
    np.save(path, changed)
    with pytest.raises(AssertionError, match="checksum"):
        verify_reference_blocks(manifest, {"base": values}, THRESHOLDS)
