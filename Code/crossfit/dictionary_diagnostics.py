"""Diagnostics for activation rank and crosscoder feature death."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .crosscoder import CrossCoder


@torch.inference_mode()
def activation_counts(
    model: CrossCoder,
    evo: np.ndarray | torch.Tensor,
    fit: np.ndarray | torch.Tensor,
    *,
    batch_size: int,
    device: str | torch.device,
) -> np.ndarray:
    """Count positive code activations over an aligned activation pool."""
    if evo.shape != fit.shape or evo.ndim != 2:
        raise ValueError("activation arrays must be aligned matrices")
    was_training = model.training
    model.eval()
    counts = torch.zeros(model.features, dtype=torch.int64, device=device)
    for start in range(0, len(evo), batch_size):
        stop = min(start + batch_size, len(evo))
        left = torch.as_tensor(evo[start:stop], dtype=torch.float32, device=device)
        right = torch.as_tensor(fit[start:stop], dtype=torch.float32, device=device)
        counts += (model.encode(left, right) > 0).sum(dim=0)
    model.train(was_training)
    return counts.cpu().numpy()


@torch.inference_mode()
def covariance_spectrum(
    values: np.ndarray | torch.Tensor,
    *,
    batch_size: int,
    device: str | torch.device,
) -> np.ndarray:
    """Return descending eigenvalues of the population covariance matrix."""
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("values must contain at least two matrix rows")
    dimension = int(values.shape[1])
    total = torch.zeros(dimension, dtype=torch.float64, device=device)
    second = torch.zeros((dimension, dimension), dtype=torch.float64, device=device)
    for start in range(0, len(values), batch_size):
        stop = min(start + batch_size, len(values))
        batch = torch.as_tensor(values[start:stop], dtype=torch.float64, device=device)
        total += batch.sum(dim=0)
        second += batch.T @ batch
    mean = total / len(values)
    covariance = second / len(values) - torch.outer(mean, mean)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min_(0).flip(0)
    return eigenvalues.cpu().numpy()


def participation_ratio(eigenvalues: np.ndarray) -> float:
    """Return trace(C)^2 / trace(C^2) from covariance eigenvalues."""
    values = np.asarray(eigenvalues, dtype=np.float64)
    denominator = float(np.square(values).sum())
    return float(values.sum() ** 2 / denominator) if denominator > 0 else 0.0


@dataclass(frozen=True)
class DeathTiming:
    """Feature-death categories measured at initialization and epoch boundaries."""

    features: int
    never_active: int
    crowded_out: int
    revived: int
    live_at_initialization: int
    live_at_final: int


def death_timing(counts_by_epoch: np.ndarray) -> DeathTiming:
    """Classify final-dead features from full-pool activation counts."""
    counts = np.asarray(counts_by_epoch)
    if counts.ndim != 2 or counts.shape[0] < 2 or np.any(counts < 0):
        raise ValueError("counts must be nonnegative checkpoint-by-feature values")
    active = counts > 0
    initial = active[0]
    final = active[-1]
    ever = active.any(axis=0)
    return DeathTiming(
        features=int(counts.shape[1]),
        never_active=int((~ever).sum()),
        crowded_out=int((~final & active[:-1].any(axis=0)).sum()),
        revived=int((~initial & final).sum()),
        live_at_initialization=int(initial.sum()),
        live_at_final=int(final.sum()),
    )


def _zero_adam_slices(
    optimizer: torch.optim.Optimizer,
    parameter: torch.nn.Parameter,
    indices: torch.Tensor,
    *,
    axis: int,
) -> None:
    state = optimizer.state.get(parameter, {})
    for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        value = state.get(name)
        if not isinstance(value, torch.Tensor) or value.ndim == 0:
            continue
        selector = [slice(None)] * value.ndim
        selector[axis] = indices
        value[tuple(selector)] = 0


@torch.no_grad()
def resample_dead_features(
    model: CrossCoder,
    optimizer: torch.optim.Optimizer,
    dead_mask: np.ndarray | torch.Tensor,
    *,
    generator: torch.Generator,
) -> int:
    """Reinitialize dead feature parameters and clear their Adam moments."""
    if model.evo_decoder.bias is not None or model.fit_decoder.bias is not None:
        raise ValueError("dead-feature resampling is registered for bias-free decoders")
    mask = torch.as_tensor(dead_mask, dtype=torch.bool, device=model.encoder.weight.device)
    if mask.ndim != 1 or len(mask) != model.features:
        raise ValueError("dead mask does not match dictionary width")
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    if len(indices) == 0:
        return 0

    encoder_bound = 1.0 / np.sqrt(2 * model.dimension)
    decoder_bound = 1.0 / np.sqrt(model.features)
    encoder_rows = torch.empty(
        (len(indices), 2 * model.dimension),
        dtype=model.encoder.weight.dtype,
        device=model.encoder.weight.device,
    ).uniform_(-encoder_bound, encoder_bound, generator=generator)
    encoder_bias = torch.empty(
        len(indices), dtype=model.encoder.bias.dtype, device=model.encoder.bias.device
    ).uniform_(-encoder_bound, encoder_bound, generator=generator)
    decoder_columns = torch.empty(
        (model.dimension, len(indices)),
        dtype=model.evo_decoder.weight.dtype,
        device=model.evo_decoder.weight.device,
    ).uniform_(-decoder_bound, decoder_bound, generator=generator)
    model.encoder.weight[indices] = encoder_rows
    model.encoder.bias[indices] = encoder_bias
    model.evo_decoder.weight[:, indices] = decoder_columns
    model.fit_decoder.weight[:, indices] = decoder_columns

    _zero_adam_slices(optimizer, model.encoder.weight, indices, axis=0)
    _zero_adam_slices(optimizer, model.encoder.bias, indices, axis=0)
    _zero_adam_slices(optimizer, model.evo_decoder.weight, indices, axis=1)
    _zero_adam_slices(optimizer, model.fit_decoder.weight, indices, axis=1)
    return int(len(indices))
