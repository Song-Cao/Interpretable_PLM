"""Step 2 predictor architectures and exact penalized linear baselines."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn


@dataclass
class DiagonalRidgeRegressor:
    """Linear regression with an explicit feature-wise diagonal penalty."""

    site_penalty: float
    density_penalty: float = 1e-8
    coefficient_: np.ndarray | None = None
    intercept_: float | None = None

    def fit(
        self,
        features: np.ndarray,
        target: np.ndarray,
        *,
        density: np.ndarray | None = None,
    ) -> "DiagonalRidgeRegressor":
        """Solve the penalized normal equations with an unpenalized intercept."""
        matrix = np.asarray(features, dtype=np.float64)
        response = np.asarray(target, dtype=np.float64)
        if matrix.ndim != 2 or response.shape != (len(matrix),) or len(matrix) == 0:
            raise ValueError("features and target must be aligned and non-empty")
        if self.site_penalty < 0 or self.density_penalty < 0:
            raise ValueError("diagonal penalties must be non-negative")
        penalties = np.full(matrix.shape[1], self.site_penalty, dtype=np.float64)
        if density is not None:
            density_column = np.asarray(density, dtype=np.float64)
            if density_column.shape != response.shape or not np.isfinite(density_column).all():
                raise ValueError("density must be a finite scalar aligned to each row")
            matrix = np.column_stack((matrix, density_column))
            penalties = np.concatenate((penalties, [self.density_penalty]))
        if not np.isfinite(matrix).all() or not np.isfinite(response).all():
            raise ValueError("linear baseline inputs must be finite")

        design = np.column_stack((np.ones(len(matrix)), matrix))
        penalty = np.diag(np.concatenate(([0.0], penalties)))
        normal_matrix = design.T @ design + penalty
        normal_rhs = design.T @ response
        solution = np.linalg.solve(normal_matrix, normal_rhs)
        self.intercept_ = float(solution[0])
        self.coefficient_ = solution[1:]
        return self

    def predict(self, features: np.ndarray, *, density: np.ndarray | None = None) -> np.ndarray:
        """Predict direct per-variant scores from a fitted linear baseline."""
        if self.coefficient_ is None or self.intercept_ is None:
            raise RuntimeError("fit must be called before predict")
        matrix = np.asarray(features, dtype=np.float64)
        if density is not None:
            matrix = np.column_stack((matrix, np.asarray(density, dtype=np.float64)))
        if matrix.ndim != 2 or matrix.shape[1] != len(self.coefficient_):
            raise ValueError("prediction features do not match fitted coefficients")
        return self.intercept_ + matrix @ self.coefficient_


class RidgeHead(nn.Module):
    """Direct linear score head trained with the composite objective and L2 decay."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features).squeeze(-1)


class MLPRegressor(nn.Module):
    """Regularized multilayer score predictor for fixed-length representations."""

    def __init__(self, input_dim: int, hidden_dims: Sequence[int], dropout: float) -> None:
        super().__init__()
        if not hidden_dims or not 0 <= dropout < 1:
            raise ValueError("MLP requires hidden layers and dropout in [0, 1)")
        layers: list[nn.Module] = []
        width = input_dim
        for hidden in hidden_dims:
            layers.extend((nn.Linear(width, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout)))
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


class ResidualConvBlock(nn.Module):
    """Two normalized convolutions with a residual connection."""

    def __init__(self, channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.GroupNorm(1, channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.GroupNorm(1, channels),
        )
        self.activation = nn.GELU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.activation(features + self.block(features))


class ConvResidualRegressor(nn.Module):
    """One-dimensional residual convolutional stack over N by L by C inputs."""

    def __init__(
        self,
        input_channels: int,
        width: int,
        depth: int,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if depth < 2 or kernel_size < 3 or kernel_size % 2 == 0:
            raise ValueError("conv stack requires depth >= 2 and an odd kernel >= 3")
        self.stem = nn.Conv1d(input_channels, width, kernel_size, padding=kernel_size // 2)
        self.blocks = nn.Sequential(
            *(ResidualConvBlock(width, kernel_size, dropout) for _ in range(depth))
        )
        self.readout = nn.Linear(width, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.blocks(self.stem(features.transpose(1, 2)))
        return self.readout(hidden.mean(dim=-1)).squeeze(-1)


class AttentionPoolingRegressor(nn.Module):
    """Learned attention pooling over N by L by d residue activations."""

    def __init__(self, input_dim: int, attention_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim), nn.Tanh(), nn.Linear(attention_dim, 1)
        )
        self.readout = nn.Sequential(
            nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attention(features).squeeze(-1), dim=-1)
        pooled = torch.sum(features * weights.unsqueeze(-1), dim=1)
        return self.readout(pooled).squeeze(-1)


def build_model(name: str, input_shape: Sequence[int], parameters: Mapping[str, Any]) -> Any:
    """Build one configured architecture without changing representation shape."""
    if name == "ridge":
        return RidgeHead(int(input_shape[-1]))
    if name == "mlp":
        return MLPRegressor(
            int(input_shape[-1]), parameters["hidden_dims"], float(parameters["dropout"])
        )
    if name == "random_forest":
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(**dict(parameters))
    if name == "gradient_boosted_trees":
        from sklearn.ensemble import GradientBoostingRegressor

        return GradientBoostingRegressor(**dict(parameters))
    if name == "conv_residual":
        return ConvResidualRegressor(
            int(input_shape[-1]), int(parameters["width"]), int(parameters["depth"]),
            int(parameters["kernel_size"]), float(parameters["dropout"]),
        )
    if name == "attention_pooling":
        return AttentionPoolingRegressor(
            int(input_shape[-1]), int(parameters["attention_dim"]),
            int(parameters["hidden_dim"]), float(parameters["dropout"]),
        )
    raise KeyError(f"unknown model architecture: {name}")
