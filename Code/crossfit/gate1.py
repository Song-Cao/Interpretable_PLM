"""Representational-drift statistics for Gate 1."""
from __future__ import annotations

import numpy as np
import torch
from scipy.stats import ttest_1samp


def tokenwise_relative_drift(base: np.ndarray, adapted: np.ndarray) -> np.ndarray:
    """Return one relative activation-change magnitude per variant and layer."""
    left = np.asarray(base, dtype=np.float32)
    right = np.asarray(adapted, dtype=np.float32)
    if left.shape != right.shape or left.ndim != 4:
        raise ValueError("layer activations must align as N x depth x L x d")
    numerator = np.linalg.norm(right - left, axis=-1).mean(axis=-1)
    denominator = np.linalg.norm(left, axis=-1).mean(axis=-1)
    if np.any(denominator == 0):
        raise ValueError("base activation norm is zero")
    return numerator / denominator


def activation_norms_by_layer(
    base: np.ndarray, adapted: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return base and adapted mean token norms and their ratio for every layer."""
    left = np.asarray(base, dtype=np.float32)
    right = np.asarray(adapted, dtype=np.float32)
    if left.shape != right.shape or left.ndim != 4:
        raise ValueError("layer activations must align as N x depth x L x d")
    base_norm = np.linalg.norm(left, axis=-1).mean(axis=(0, 2))
    adapted_norm = np.linalg.norm(right, axis=-1).mean(axis=(0, 2))
    if np.any(base_norm == 0):
        raise ValueError("base activation norm is zero")
    return base_norm, adapted_norm, adapted_norm / base_norm


def paired_task_gain_test(gains: np.ndarray) -> dict[str, float | int]:
    """Test whether paired cross-validation gains differ from zero."""
    values = np.asarray(gains, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("task gains must be a finite vector with at least two folds")
    result = ttest_1samp(values, popmean=0.0)
    return {
        "n_folds": int(len(values)),
        "mean": float(values.mean()),
        "standard_error": float(values.std(ddof=1) / np.sqrt(len(values))),
        "t_statistic": float(result.statistic),
        "degrees_of_freedom": int(result.df),
        "p_value_two_sided": float(result.pvalue),
    }


def select_analysis_layer(
    drift_gap: np.ndarray,
    real_cka: np.ndarray,
    *,
    cka_floor: float,
    exclude_first: int,
    exclude_last: int,
    near_max_fraction: float,
) -> tuple[int, np.ndarray, np.ndarray, float]:
    """Select the earliest eligible layer within a fraction of the maximum drift gap."""
    gap = np.asarray(drift_gap, dtype=np.float64)
    cka = np.asarray(real_cka, dtype=np.float64)
    if gap.ndim != 1 or cka.shape != gap.shape or not np.isfinite(gap).all():
        raise ValueError("drift gap and CKA must be aligned finite vectors")
    if not np.isfinite(cka).all():
        raise ValueError("drift gap and CKA must be aligned finite vectors")
    if exclude_first < 0 or exclude_last < 0 or exclude_first + exclude_last >= len(gap):
        raise ValueError("layer exclusions leave no selectable depth")
    if not 0.0 < near_max_fraction <= 1.0:
        raise ValueError("near-max fraction must be in (0, 1]")
    layer = np.arange(len(gap))
    last = len(gap) - exclude_last
    eligible = (
        (layer >= exclude_first)
        & (layer < last)
        & (cka >= float(cka_floor))
    )
    if not eligible.any():
        raise ValueError("no layer satisfies the CKA and mid-network constraints")
    maximum = float(gap[eligible].max())
    threshold = float(near_max_fraction) * maximum
    near_max = eligible & (gap >= threshold)
    return int(np.flatnonzero(near_max)[0]), eligible, near_max, threshold


def linear_cka_by_layer(
    base: np.ndarray, adapted: np.ndarray, *, device: str | torch.device = "cpu"
) -> np.ndarray:
    """Compute exact linear CKA over flattened reserve residue observations."""
    left = np.asarray(base)
    right = np.asarray(adapted)
    if left.shape != right.shape or left.ndim != 4:
        raise ValueError("layer activations must align as N x depth x L x d")
    output = []
    for layer in range(left.shape[1]):
        x = torch.as_tensor(left[:, layer], dtype=torch.float32, device=device).flatten(0, 1)
        y = torch.as_tensor(right[:, layer], dtype=torch.float32, device=device).flatten(0, 1)
        x = x - x.mean(dim=0, keepdim=True)
        y = y - y.mean(dim=0, keepdim=True)
        cross = torch.linalg.matrix_norm(x.T @ y).square()
        left_norm = torch.linalg.matrix_norm(x.T @ x)
        right_norm = torch.linalg.matrix_norm(y.T @ y)
        output.append(float((cross / (left_norm * right_norm)).cpu()))
    return np.asarray(output, dtype=np.float64)


def bootstrap_drift_gap(
    real_drift: np.ndarray,
    shuffled_drift: np.ndarray,
    *,
    samples: int,
    seed: int,
    confidence_level: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap variants to bound the mean real-minus-shuffled drift per layer."""
    real = np.asarray(real_drift, dtype=np.float64)
    shuffled = np.asarray(shuffled_drift, dtype=np.float64)
    if real.shape != shuffled.shape or real.ndim != 2:
        raise ValueError("drift arrays must align as variants by layer")
    rng = np.random.default_rng(seed)
    values = np.empty((samples, real.shape[1]), dtype=np.float64)
    for draw in range(samples):
        index = rng.integers(0, len(real), size=len(real))
        values[draw] = (real[index] - shuffled[index]).mean(axis=0)
    tail = (1.0 - confidence_level) / 2.0
    return np.quantile(values, tail, axis=0), np.quantile(values, 1.0 - tail, axis=0)
