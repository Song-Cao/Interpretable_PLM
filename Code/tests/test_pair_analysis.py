from __future__ import annotations

import numpy as np
import pandas as pd

from crossfit.pair_analysis import (
    audit_epoch_pair_distances,
    mutation_map,
    variant_hamming_distance,
)
from crossfit.sampling import AssayBatch, EpochSample


def test_variant_hamming_counts_shared_positions_once():
    assert variant_hamming_distance(mutation_map("A1C:B2D"), mutation_map("A1E:B2D:C3F")) == 2
    assert variant_hamming_distance(mutation_map("WT"), mutation_map("A1C:B2D")) == 2


def test_pair_distance_audit_applies_margin_and_same_bound_censor_gate():
    frame = pd.DataFrame(
        {
            "mutations": ["WT", "A1C", "A1D:B2C", "A1E:B2D:C3F"],
            "y": [0.0, 1.0, 2.0, 2.1],
            "censored": ["none", "none", "floor", "floor"],
        }
    )
    epoch = EpochSample(
        epoch=0,
        batches=(AssayBatch("a", np.arange(4)),),
        counts={"a": 4},
        shares={"a": 1.0},
        per_assay_cap=4,
    )
    audit = audit_epoch_pair_distances({"a": frame}, [epoch], margin_sd=0.25)
    assert audit.counts["a"] == {1: 1, 2: 2, 3: 2}
    result = audit.to_frame().drop_duplicates("assay_id").iloc[0]
    assert result["total_gated_pairs"] == 5
