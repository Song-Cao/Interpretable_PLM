"""Diagnostics for the Hamming-distance composition of gated training pairs."""
from __future__ import annotations

import itertools
from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .sampling import EpochSample


def mutation_map(mutations: str) -> dict[int, str]:
    """Map one curated substitution string to final residues by position."""
    if mutations == "WT":
        return {}
    output: dict[int, str] = {}
    for token in mutations.split(":"):
        if len(token) < 3 or not token[1:-1].isdigit():
            raise ValueError(f"invalid curated mutation token: {token}")
        position = int(token[1:-1])
        if position in output:
            raise ValueError(f"position occurs twice in mutation string: {mutations}")
        output[position] = token[-1]
    return output


def variant_hamming_distance(left: Mapping[int, str], right: Mapping[int, str]) -> int:
    """Count sequence positions with different final residues for two variants."""
    return sum(left.get(position) != right.get(position) for position in left.keys() | right.keys())


@dataclass
class PairDistanceAudit:
    """Accumulated gated-pair Hamming counts by assay."""

    counts: dict[str, Counter[int]]

    def to_frame(self) -> pd.DataFrame:
        """Return the long-form Hamming distribution of all gated pairs."""
        rows: list[dict[str, int | float | str]] = []
        for assay_id in sorted(self.counts):
            distribution = self.counts[assay_id]
            total = sum(distribution.values())
            for distance in sorted(distribution):
                rows.append(
                    {
                        "assay_id": assay_id,
                        "hamming_distance": distance,
                        "gated_pairs": distribution[distance],
                        "gated_pair_fraction": distribution[distance] / total,
                        "total_gated_pairs": total,
                    }
                )
        return pd.DataFrame(rows)


def audit_epoch_pair_distances(
    frames_by_assay: Mapping[str, pd.DataFrame],
    epochs: Sequence[EpochSample],
    *,
    margin_sd: float,
) -> PairDistanceAudit:
    """Count distances for pairs passing the censor and margin gates in sampled batches."""
    if margin_sd < 0:
        raise ValueError("margin must be non-negative")
    required = {"mutations", "y", "censored"}
    for assay_id, frame in frames_by_assay.items():
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{assay_id} is missing pair-audit columns {sorted(missing)}")
    parsed = {
        assay_id: [mutation_map(value) for value in frame["mutations"].astype(str)]
        for assay_id, frame in frames_by_assay.items()
    }
    counts = {assay_id: Counter() for assay_id in frames_by_assay}
    for epoch in epochs:
        for batch in epoch.batches:
            frame = frames_by_assay[batch.assay_id]
            indices = np.asarray(batch.indices, dtype=np.int64)
            y = frame["y"].to_numpy(dtype=np.float64)[indices]
            censoring = frame["censored"].astype(str).to_numpy()[indices]
            mutations = [parsed[batch.assay_id][index] for index in indices]
            for u, v in itertools.combinations(range(len(indices)), 2):
                if censoring[u] == censoring[v] and censoring[u] != "none":
                    continue
                if abs(y[u] - y[v]) <= margin_sd:
                    continue
                distance = variant_hamming_distance(mutations[u], mutations[v])
                counts[batch.assay_id][distance] += 1
    return PairDistanceAudit(counts)
