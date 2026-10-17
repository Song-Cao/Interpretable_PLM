"""Classical additive predictors and CPU training utilities for Step 2."""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from scipy import sparse, stats
import torch
from torch import nn

from .losses import CompositeFitnessLoss, PairBatch, construct_pairs


class AdditiveLinear(nn.Module):
    """A bias-free linear score whose pair prediction is an early subtraction."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        if n_features < 1:
            raise ValueError("an additive model requires at least one feature")
        self.weight = nn.Parameter(torch.zeros(n_features))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.weight


@dataclass(frozen=True)
class AdditiveFit:
    """A fitted additive predictor and its optimization diagnostics."""

    weight: np.ndarray
    residual_scale: float
    epochs: int
    n_pairs: int
    status: str
    best_validation_spearman: float


@dataclass(frozen=True)
class SavedAdditiveModel:
    """One selected N/seed model loaded from a consolidated artifact."""

    weight: np.ndarray
    residual_scale: float
    status: str
    feature_metadata: str
    selected_hyperparameters: str


def _loss_module(loss_config: Mapping, regularization_weight: float) -> CompositeFitnessLoss:
    uncertainty = loss_config["magnitude_uncertainty_weighting"]
    if uncertainty["enabled"]:
        raise ValueError("joint additive baselines cannot use corpus-wide sigma weighting")
    weights = loss_config["weights"]
    return CompositeFitnessLoss(
        n_assays=1,
        prediction_mode="difference",
        temperature=loss_config["temperature"],
        magnitude_mode=loss_config["magnitude_mode"],
        huber_delta=loss_config["huber_delta"],
        residual_scale_initial=loss_config["residual_scale"]["initial"],
        residual_scale_floor=loss_config["residual_scale"]["floor"],
        rank_weight=weights["rank"],
        magnitude_weight=weights["magnitude"],
        censoring_weight=weights["censoring"],
        regularization_weight=regularization_weight,
    )


def _pairs_for_rows(
    target: np.ndarray,
    censored: Sequence[str],
    sigma_v: np.ndarray,
    margin_sd: float,
) -> PairBatch:
    return construct_pairs(
        torch.as_tensor(target, dtype=torch.float32),
        torch.zeros(len(target), dtype=torch.long),
        list(censored),
        torch.as_tensor(sigma_v, dtype=torch.float32),
        margin_sd=margin_sd,
    )


def _spearman(prediction: np.ndarray, target: np.ndarray) -> float:
    if len(prediction) < 2 or np.ptp(prediction) == 0 or np.ptp(target) == 0:
        return float("nan")
    return float(stats.spearmanr(prediction, target).statistic)


def fit_additive_model(
    features: sparse.spmatrix,
    target: np.ndarray,
    censored: Sequence[str],
    sigma_v: np.ndarray,
    *,
    margin_sd: float,
    regularization_weight: float,
    weight_decay: float,
    learning_rate: float,
    max_epochs: int,
    patience: int,
    early_stopping_min_delta: float,
    scheduler_factor: float,
    scheduler_patience: int,
    gradient_clip_norm: float,
    training_seed: int,
    loss_config: Mapping,
    validation: tuple[sparse.spmatrix, np.ndarray, Sequence[str], np.ndarray] | None = None,
) -> AdditiveFit:
    """Fit one additive model with the composite pair objective on CPU."""
    matrix = sparse.csr_matrix(features, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    sigma_v = np.asarray(sigma_v, dtype=np.float32)
    if matrix.shape[0] != len(target) or len(censored) != len(target) or len(sigma_v) != len(target):
        raise ValueError("features and training metadata must align")
    if (
        max_epochs < 1
        or patience < 1
        or scheduler_patience < 1
        or learning_rate <= 0
        or weight_decay < 0
        or early_stopping_min_delta < 0
        or not 0 < scheduler_factor < 1
    ):
        raise ValueError("optimization settings are invalid")
    pairs = _pairs_for_rows(target, censored, sigma_v, margin_sd)
    if len(pairs) == 0:
        return AdditiveFit(
            np.zeros(matrix.shape[1], dtype=np.float32),
            float(loss_config["residual_scale"]["initial"]),
            0,
            0,
            "no_gated_pairs",
            float("nan"),
        )

    torch.manual_seed(training_seed)
    model = AdditiveLinear(matrix.shape[1])
    objective = _loss_module(loss_config, regularization_weight)
    parameters = tuple(model.parameters()) + tuple(objective.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=scheduler_factor, patience=scheduler_patience
    )
    dense = torch.from_numpy(matrix.toarray())
    validation_features = None
    validation_target = None
    validation_pairs = None
    if validation is not None:
        validation_features = torch.from_numpy(
            sparse.csr_matrix(validation[0], dtype=np.float32).toarray()
        )
        validation_target = np.asarray(validation[1], dtype=np.float64)
        validation_pairs = _pairs_for_rows(
            validation_target,
            validation[2],
            np.asarray(validation[3]),
            margin_sd,
        )

    best_score = float("nan")
    best_loss = math.inf
    best_epoch = 0
    best_state = None
    stale = 0
    epochs_run = 0
    for epoch in range(1, max_epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        scores = model(dense)
        difference = scores[pairs.u] - scores[pairs.v]
        output = objective(difference, pairs, readout_parameters=(model.weight,))
        output.total.backward()
        torch.nn.utils.clip_grad_norm_(parameters, gradient_clip_norm)
        optimizer.step()
        epochs_run = epoch

        if validation_features is None:
            continue
        with torch.no_grad():
            validation_scores = model(validation_features)
            prediction = validation_scores.numpy()
        score = _spearman(prediction, validation_target)
        if validation_pairs is not None and len(validation_pairs) > 0:
            with torch.no_grad():
                validation_difference = (
                    validation_scores[validation_pairs.u] - validation_scores[validation_pairs.v]
                )
                validation_loss = float(
                    objective(validation_difference, validation_pairs).total / len(validation_pairs)
                )
        else:
            validation_loss = math.inf
        scheduler.step(validation_loss)
        if validation_loss < best_loss - early_stopping_min_delta:
            best_loss = validation_loss
            best_score = score
            best_epoch = epoch
            best_state = (copy.deepcopy(model.state_dict()), copy.deepcopy(objective.state_dict()))
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state[0])
        objective.load_state_dict(best_state[1])
        epochs_run = best_epoch
    scale = float(objective.residual_scale(torch.tensor([0]))[0].detach())
    return AdditiveFit(
        model.weight.detach().numpy().astype(np.float32),
        scale,
        epochs_run,
        len(pairs),
        "ok",
        best_score,
    )


def predict_additive(features: sparse.spmatrix, weight: np.ndarray) -> np.ndarray:
    """Predict variant scores from a fitted additive coefficient vector."""
    matrix = sparse.csr_matrix(features)
    coefficient = np.asarray(weight)
    if matrix.shape[1] != coefficient.shape[0]:
        raise ValueError("feature matrix and coefficient vector do not align")
    return np.asarray(matrix @ coefficient, dtype=np.float64).ravel()


def load_additive_model(
    path: str | Path, *, n_train: int, seed: int
) -> SavedAdditiveModel:
    """Load one fitted model from a saved assay/representation bundle."""
    with np.load(path, allow_pickle=False) as bundle:
        selected = (bundle["n_train"] == n_train) & (bundle["seed"] == seed)
        indices = np.flatnonzero(selected)
        if len(indices) != 1:
            raise KeyError(f"expected one model for N={n_train}, seed={seed}; found {len(indices)}")
        index = int(indices[0])
        return SavedAdditiveModel(
            weight=bundle["weight"][index].copy(),
            residual_scale=float(bundle["residual_scale"][index]),
            status=str(bundle["status"][index]),
            feature_metadata=str(bundle["feature_metadata"]),
            selected_hyperparameters=str(bundle["selected_hyperparameters"]),
        )


def deterministic_folds(n_rows: int, n_folds: int, seed: int) -> tuple[np.ndarray, ...]:
    """Create deterministic validation folds without touching test indices."""
    if n_folds < 2 or n_rows < n_folds:
        raise ValueError("cross-validation requires at least one row per fold")
    permutation = np.random.default_rng(seed).permutation(n_rows)
    return tuple(np.sort(fold) for fold in np.array_split(permutation, n_folds))
