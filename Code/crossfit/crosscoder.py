"""BatchTopK crosscoder and reconstruction diagnostics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class BatchTopK(nn.Module):
    """Keep the largest positive batch_size times k pre-activations."""

    def __init__(self, k: int) -> None:
        super().__init__()
        if k <= 0:
            raise ValueError("k must be positive")
        self.k = int(k)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if values.ndim != 2:
            raise ValueError("BatchTopK expects batch by feature values")
        positive = torch.relu(values)
        flat = positive.flatten()
        keep = min(self.k * values.shape[0], flat.numel())
        selected, indices = torch.topk(flat, keep, sorted=False)
        output = torch.zeros_like(flat)
        output.scatter_(0, indices, selected)
        return output.view_as(values)


class CrossCoder(nn.Module):
    """Shared sparse encoder with separate evo and fit linear decoders."""

    def __init__(self, dimension: int, width_multiplier: int, k: int) -> None:
        super().__init__()
        if dimension <= 0 or width_multiplier <= 0:
            raise ValueError("dimension and width_multiplier must be positive")
        self.dimension = int(dimension)
        self.width_multiplier = int(width_multiplier)
        self.features = self.dimension * self.width_multiplier
        self.encoder = nn.Linear(2 * self.dimension, self.features)
        self.sparsity = BatchTopK(k)
        self.evo_decoder = nn.Linear(self.features, self.dimension)
        self.fit_decoder = nn.Linear(self.features, self.dimension)
        self.reset_parameters()

    @property
    def k(self) -> int:
        return self.sparsity.k

    def reset_parameters(self) -> None:
        self.encoder.reset_parameters()
        self.evo_decoder.reset_parameters()
        # Symmetric initialization makes the self-pair null an exact alignment test.
        with torch.no_grad():
            self.fit_decoder.weight.copy_(self.evo_decoder.weight)
            self.fit_decoder.bias.copy_(self.evo_decoder.bias)

    def encode(self, evo: torch.Tensor, fit: torch.Tensor) -> torch.Tensor:
        if evo.shape != fit.shape or evo.ndim != 2 or evo.shape[1] != self.dimension:
            raise ValueError("crosscoder inputs must be aligned batch-by-dimension matrices")
        return self.sparsity(self.encoder(torch.cat((evo, fit), dim=-1)))

    def forward(
        self, evo: torch.Tensor, fit: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        code = self.encode(evo, fit)
        return code, self.evo_decoder(code), self.fit_decoder(code)


def reconstruction_loss(
    evo: torch.Tensor,
    fit: torch.Tensor,
    evo_reconstruction: torch.Tensor,
    fit_reconstruction: torch.Tensor,
) -> torch.Tensor:
    """Return the sum of the two mean-squared reconstruction errors."""
    return torch.mean((evo_reconstruction - evo) ** 2) + torch.mean(
        (fit_reconstruction - fit) ** 2
    )


def decoder_delta(model: CrossCoder) -> torch.Tensor:
    """Compute the exact per-feature adapted-decoder norm ratio."""
    evo = torch.linalg.vector_norm(model.evo_decoder.weight, dim=0)
    fit = torch.linalg.vector_norm(model.fit_decoder.weight, dim=0)
    denominator = evo + fit
    return torch.where(denominator > 0, fit / denominator, torch.full_like(fit, 0.5))


@dataclass(frozen=True)
class CrossCoderDiagnostics:
    """Full-pool diagnostics for one fitted crosscoder."""

    dead_feature_fraction: float
    participation_ratio: float
    evo_fraction_variance_explained: float
    fit_fraction_variance_explained: float
    realized_l0: float
    active_features: int
    features: int


@torch.inference_mode()
def crosscoder_diagnostics(
    model: CrossCoder,
    evo: np.ndarray | torch.Tensor,
    fit: np.ndarray | torch.Tensor,
    *,
    batch_size: int,
    device: str | torch.device,
) -> CrossCoderDiagnostics:
    """Measure dead features, activation-energy rank, FVE and realized L0."""
    if evo.shape != fit.shape or evo.ndim != 2:
        raise ValueError("diagnostic arrays must be aligned matrices")
    model.eval()
    counts = torch.zeros(model.features, dtype=torch.float64, device=device)
    latent_sum = torch.zeros(model.features, dtype=torch.float32, device=device)
    second_moment = torch.zeros(
        (model.features, model.features), dtype=torch.float32, device=device
    )
    evo_sum = torch.zeros(model.dimension, dtype=torch.float64, device=device)
    fit_sum = torch.zeros(model.dimension, dtype=torch.float64, device=device)
    evo_sq = fit_sq = evo_error = fit_error = 0.0
    total_active = 0
    n = len(evo)
    for start in range(0, n, batch_size):
        stop = min(start + batch_size, n)
        left = torch.as_tensor(evo[start:stop], dtype=torch.float32, device=device)
        right = torch.as_tensor(fit[start:stop], dtype=torch.float32, device=device)
        code, left_hat, right_hat = model(left, right)
        active = code > 0
        counts += active.sum(dim=0).double()
        latent_sum += code.sum(dim=0)
        second_moment += code.T @ code
        total_active += int(active.sum())
        evo_sum += left.double().sum(dim=0)
        fit_sum += right.double().sum(dim=0)
        evo_sq += float((left.double() ** 2).sum())
        fit_sq += float((right.double() ** 2).sum())
        evo_error += float(((left_hat.double() - left.double()) ** 2).sum())
        fit_error += float(((right_hat.double() - right.double()) ** 2).sum())
    evo_tss = evo_sq - float((evo_sum**2).sum()) / n
    fit_tss = fit_sq - float((fit_sum**2).sum()) / n
    latent_mean = latent_sum / n
    covariance = second_moment / n - torch.outer(latent_mean, latent_mean)
    covariance_trace = float(torch.trace(covariance))
    covariance_frobenius_squared = float((covariance**2).sum())
    participation = (
        covariance_trace**2 / covariance_frobenius_squared
        if covariance_frobenius_squared > 0 else 0.0
    )
    live = int((counts > 0).sum())
    return CrossCoderDiagnostics(
        dead_feature_fraction=1.0 - live / model.features,
        participation_ratio=participation,
        evo_fraction_variance_explained=1.0 - evo_error / evo_tss,
        fit_fraction_variance_explained=1.0 - fit_error / fit_tss,
        realized_l0=total_active / n,
        active_features=live,
        features=model.features,
    )
