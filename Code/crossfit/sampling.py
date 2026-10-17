"""Assay-balanced, capped epoch sampling for Step 2 and adaptation."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from .utils import stable_hash


@dataclass(frozen=True)
class AssayBatch:
    """One single-assay training batch."""

    assay_id: str
    indices: np.ndarray


@dataclass(frozen=True)
class EpochSample:
    """Sampled batches and realized per-assay epoch shares."""

    epoch: int
    batches: tuple[AssayBatch, ...]
    counts: dict[str, int]
    shares: dict[str, float]
    per_assay_cap: int

    @property
    def n_variants(self) -> int:
        return sum(self.counts.values())

    def records(self) -> list[dict[str, int | float | str]]:
        """Return rows suitable for an epoch sampling log."""
        return [
            {
                "epoch": self.epoch,
                "assay_id": assay_id,
                "n_sampled": self.counts[assay_id],
                "share": self.shares[assay_id],
                "per_assay_cap": self.per_assay_cap,
            }
            for assay_id in sorted(self.counts)
        ]


def allocate_capped_sqrt_counts(
    assay_sizes: Mapping[str, int], per_assay_cap: int, epoch_size: int
) -> dict[str, int]:
    """Allocate an integer epoch budget proportional to sqrt(n), subject to caps."""
    if not assay_sizes:
        raise ValueError("at least one assay is required")
    if per_assay_cap < 1 or epoch_size < 1:
        raise ValueError("cap and epoch size must be positive")
    names = sorted(assay_sizes)
    sizes = np.array([assay_sizes[name] for name in names], dtype=np.int64)
    if np.any(sizes < 1):
        raise ValueError("assay sizes must be positive")
    capacity = np.minimum(sizes, per_assay_cap)
    total_capacity = int(capacity.sum())
    if epoch_size > total_capacity:
        raise ValueError(
            f"epoch_size={epoch_size} exceeds the without-replacement capacity {total_capacity}"
        )

    weights = np.sqrt(sizes.astype(np.float64))
    continuous = np.zeros(len(names), dtype=np.float64)
    active = np.ones(len(names), dtype=bool)
    remaining = float(epoch_size)
    while remaining > 0 and np.any(active):
        proposed = remaining * weights[active] / weights[active].sum()
        active_indices = np.flatnonzero(active)
        saturates = proposed >= capacity[active_indices]
        if not np.any(saturates):
            continuous[active_indices] = proposed
            remaining = 0.0
            break
        saturated_indices = active_indices[saturates]
        continuous[saturated_indices] = capacity[saturated_indices]
        remaining -= float(capacity[saturated_indices].sum())
        active[saturated_indices] = False

    counts = np.floor(continuous).astype(np.int64)
    remainder = epoch_size - int(counts.sum())
    fractions = continuous - counts
    order = sorted(
        range(len(names)),
        key=lambda idx: (-fractions[idx], names[idx]),
    )
    for idx in order:
        if remainder == 0:
            break
        if counts[idx] < capacity[idx]:
            counts[idx] += 1
            remainder -= 1
    if remainder != 0:
        raise RuntimeError("failed to allocate the complete epoch budget")
    return {name: int(count) for name, count in zip(names, counts)}


class CappedSqrtSampler:
    """Draw deterministic epochs without replacement inside each assay.

    The assay allocation follows square-root dataset-size weights until a
    dataset reaches either its number of available rows or the configured
    multiple of the median assay size. Every emitted batch contains one assay.
    """

    def __init__(
        self,
        indices_by_assay: Mapping[str, Sequence[int] | np.ndarray],
        *,
        cap_median_multiplier: float,
        batch_size: int,
        seed: int,
    ) -> None:
        if cap_median_multiplier <= 0 or batch_size < 1:
            raise ValueError("cap multiplier and batch size must be positive")
        if not indices_by_assay:
            raise ValueError("at least one assay is required")
        self.indices_by_assay = {
            assay_id: np.asarray(indices, dtype=np.int64)
            for assay_id, indices in sorted(indices_by_assay.items())
        }
        if any(indices.ndim != 1 or len(indices) == 0 for indices in self.indices_by_assay.values()):
            raise ValueError("each assay must provide a non-empty one-dimensional index array")
        if any(len(np.unique(indices)) != len(indices) for indices in self.indices_by_assay.values()):
            raise ValueError("input indices must be unique within each assay")
        self.assay_sizes = {
            assay_id: len(indices) for assay_id, indices in self.indices_by_assay.items()
        }
        median_size = float(np.median(list(self.assay_sizes.values())))
        self.per_assay_cap = int(round(cap_median_multiplier * median_size))
        if self.per_assay_cap < 1:
            raise ValueError("the realized per-assay cap must be positive")
        self.batch_size = int(batch_size)
        self.seed = int(seed)

    @property
    def capped_pool_size(self) -> int:
        return sum(min(size, self.per_assay_cap) for size in self.assay_sizes.values())

    def sample_epoch(self, epoch: int, epoch_size: int | None = None) -> EpochSample:
        """Sample one epoch and report its realized assay composition."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        requested = self.capped_pool_size if epoch_size is None else int(epoch_size)
        counts = allocate_capped_sqrt_counts(
            self.assay_sizes, self.per_assay_cap, requested
        )
        rng = np.random.default_rng(stable_hash("step2_sampler", self.seed, epoch))
        batches: list[AssayBatch] = []
        for assay_id, count in counts.items():
            selected = rng.permutation(self.indices_by_assay[assay_id])[:count]
            for start in range(0, count, self.batch_size):
                batches.append(
                    AssayBatch(assay_id, selected[start : start + self.batch_size].copy())
                )
        rng.shuffle(batches)
        shares = {assay_id: count / requested for assay_id, count in counts.items()}
        if not math.isclose(sum(shares.values()), 1.0):
            raise RuntimeError("realized assay shares do not sum to one")
        return EpochSample(
            epoch=epoch,
            batches=tuple(batches),
            counts=counts,
            shares=shares,
            per_assay_cap=self.per_assay_cap,
        )
