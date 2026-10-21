"""Representational-drift statistics for Gate 1."""
from __future__ import annotations

import numpy as np
import torch


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
