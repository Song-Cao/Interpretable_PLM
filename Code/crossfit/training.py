"""Training utilities for one-dataset Step 2 predictor searches."""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from scipy.stats import spearmanr
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .losses import CompositeFitnessLoss


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and Torch for one ensemble member."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class FeatureNormalizer:
    """Training-only feature moments applied channel-wise or feature-wise."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "FeatureNormalizer":
        array = np.asarray(values, dtype=np.float32)
        if array.ndim not in {2, 3} or not len(array):
            raise ValueError("features must have shape N x D or N x L x C")
        axes = (0,) if array.ndim == 2 else (0, 1)
        mean = array.mean(axis=axes, keepdims=True)
        scale = array.std(axis=axes, keepdims=True)
        scale[scale < 1e-8] = 1.0
        return cls(mean=mean, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        """Apply frozen training moments without refitting on evaluation data."""
        return ((np.asarray(values, dtype=np.float32) - self.mean) / self.scale).astype(
            np.float32, copy=False
        )


@dataclass
class TorchFit:
    """A selected Torch model, residual scale, normalizer, and validation trace."""

    model: nn.Module
    loss: CompositeFitnessLoss
    normalizer: FeatureNormalizer
    best_epoch: int
    validation_spearman: float
    history: list[dict[str, float]]


def _loss_from_config(config: dict) -> CompositeFitnessLoss:
    residual = config["residual_scale"]
    weights = config["weights"]
    return CompositeFitnessLoss(
        margin_sd=float(config["margin_sd"]),
        temperature=float(config["temperature"]),
        magnitude_mode=str(config["magnitude_mode"]),
        huber_delta=float(config["huber_delta"]),
        residual_scale_initial=float(residual["initial"]),
        residual_scale_floor=float(residual["floor"]),
        rank_weight=float(weights["rank"]),
        magnitude_weight=float(weights["magnitude"]),
        censoring_weight=float(weights["censoring"]),
        regularization_weight=float(weights["regularization"]),
    )


@torch.no_grad()
def predict_torch(
    model: nn.Module,
    normalizer: FeatureNormalizer,
    features: np.ndarray,
    *,
    device: str | torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Predict direct fitness scores with a frozen training normalizer."""
    model.eval()
    array = torch.from_numpy(normalizer.transform(features))
    loader = DataLoader(TensorDataset(array), batch_size=batch_size, shuffle=False)
    output = [model(batch[0].to(device)).detach().cpu().numpy() for batch in loader]
    return np.concatenate(output).astype(np.float64)


def fit_torch_predictor(
    model: nn.Module,
    train_features: np.ndarray,
    train_target: np.ndarray,
    train_censoring: Sequence[str],
    validation_features: np.ndarray,
    validation_target: np.ndarray,
    *,
    objective_config: dict,
    optimization_config: dict,
    parameters: dict,
    seed: int,
    device: str | torch.device,
) -> TorchFit:
    """Fit one neural predictor with validation-Spearman early stopping."""
    seed_everything(seed)
    device = torch.device(device)
    normalizer = FeatureNormalizer.fit(train_features)
    train_x = torch.from_numpy(normalizer.transform(train_features))
    train_y = torch.as_tensor(train_target, dtype=torch.float32)
    censor_codes = {"none": 0, "floor": 1, "ceiling": 2}
    train_c = torch.tensor([censor_codes[str(value)] for value in train_censoring])
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_x, train_y, train_c),
        batch_size=int(parameters["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    model = model.to(device)
    objective = _loss_from_config(objective_config).to(device)
    trainable = list(model.parameters()) + list(objective.residual_scale.parameters())
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(parameters["learning_rate"]),
        weight_decay=float(parameters["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=float(optimization_config["scheduler_factor"]),
        patience=int(optimization_config["scheduler_patience"]),
    )
    best_score = -float("inf")
    best_epoch = 0
    best_model = copy.deepcopy(model.state_dict())
    best_scale = copy.deepcopy(objective.residual_scale.state_dict())
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(int(optimization_config["max_epochs"])):
        model.train()
        totals: list[float] = []
        for batch_x, batch_y, batch_c in loader:
            batch_x, batch_y, batch_c = batch_x.to(device), batch_y.to(device), batch_c.to(device)
            optimizer.zero_grad(set_to_none=True)
            scores = model(batch_x)
            output = objective(
                scores,
                batch_y,
                batch_c,
                regularized_parameters=model.parameters(),
            )
            output.total.backward()
            torch.nn.utils.clip_grad_norm_(trainable, float(optimization_config["gradient_clip_norm"]))
            optimizer.step()
            totals.append(float(output.total.detach()))
        prediction = predict_torch(model, normalizer, validation_features, device=device)
        score = float(spearmanr(prediction, validation_target).statistic)
        score = score if np.isfinite(score) else -1.0
        scheduler.step(score)
        history.append({"epoch": float(epoch), "loss": float(np.mean(totals)), "validation_spearman": score})
        if score > best_score + float(optimization_config["early_stopping_min_delta"]):
            best_score, best_epoch, stale = score, epoch, 0
            best_model = copy.deepcopy(model.state_dict())
            best_scale = copy.deepcopy(objective.residual_scale.state_dict())
        else:
            stale += 1
        if stale >= int(optimization_config["early_stopping_patience"]):
            break
    model.load_state_dict(best_model)
    objective.residual_scale.load_state_dict(best_scale)
    return TorchFit(model, objective, normalizer, best_epoch, best_score, history)
