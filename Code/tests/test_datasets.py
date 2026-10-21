from __future__ import annotations

import pandas as pd

from crossfit import datasets


def test_ired_loader_does_not_require_ignored_primary_variant_bundle(tmp_path, monkeypatch):
    expected = pd.DataFrame({"assay_id": ["IRED"], "variant_row": [0]})
    expected.to_parquet(tmp_path / "ired_variants.parquet", index=False)
    monkeypatch.setattr(datasets, "_processed_dir", lambda: tmp_path)
    datasets._variants.cache_clear()
    try:
        observed = datasets._variants()
        pd.testing.assert_frame_equal(observed, expected)
    finally:
        datasets._variants.cache_clear()
