from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse
import torch

from crossfit.models import (
    AdditiveLinear,
    deterministic_folds,
    fit_additive_model,
    load_additive_model,
    predict_additive,
)


LOSS_CONFIG = {
    "temperature": 1.0,
    "magnitude_mode": "squared",
    "huber_delta": 1.0,
    "residual_scale": {"initial": 1.0, "floor": 0.0001},
    "magnitude_uncertainty_weighting": {"enabled": False},
    "weights": {"rank": 1.0, "magnitude": 1.0, "censoring": 1.0},
}


def test_additive_linear_uses_early_subtraction_exactly():
    model = AdditiveLinear(2)
    with torch.no_grad():
        model.weight[:] = torch.tensor([2.0, -1.0])
    features = torch.tensor([[1.0, 0.0], [0.0, 3.0]])
    score = model(features)
    pair_prediction = model(features[0] - features[1])
    assert float(pair_prediction.detach()) == pytest.approx(float(score[0] - score[1]))


def test_additive_fit_recovers_synthetic_ordering():
    features = sparse.csr_matrix(
        np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    )
    target = np.array([0.0, 2.0, -1.0, 1.0], dtype=np.float32)
    fit = fit_additive_model(
        features,
        target,
        ["none"] * 4,
        np.full(4, np.nan, dtype=np.float32),
        margin_sd=0.0,
        regularization_weight=0.0,
        weight_decay=0.0,
        learning_rate=0.05,
        max_epochs=100,
        patience=15,
        early_stopping_min_delta=1e-6,
        scheduler_factor=0.5,
        scheduler_patience=5,
        gradient_clip_norm=10.0,
        training_seed=3,
        loss_config=LOSS_CONFIG,
        validation=(features, target, ["none"] * 4, np.full(4, np.nan, dtype=np.float32)),
    )
    prediction = predict_additive(features, fit.weight)
    assert fit.status == "ok"
    assert fit.n_pairs == 6
    assert np.corrcoef(prediction, target)[0, 1] > 0.99


def test_additive_fit_reports_zero_pair_gate_without_relaxing_it():
    fit = fit_additive_model(
        sparse.eye(3, format="csr"),
        np.zeros(3, dtype=np.float32),
        ["floor"] * 3,
        np.full(3, np.nan, dtype=np.float32),
        margin_sd=0.1,
        regularization_weight=0.0,
        weight_decay=0.0,
        learning_rate=0.05,
        max_epochs=10,
        patience=2,
        early_stopping_min_delta=1e-6,
        scheduler_factor=0.5,
        scheduler_patience=1,
        gradient_clip_norm=1.0,
        training_seed=3,
        loss_config=LOSS_CONFIG,
    )
    assert fit.status == "no_gated_pairs"
    assert fit.n_pairs == 0
    assert np.all(fit.weight == 0)


def test_deterministic_folds_partition_rows_once():
    first = deterministic_folds(20, 3, 17)
    second = deterministic_folds(20, 3, 17)
    assert all(np.array_equal(left, right) for left, right in zip(first, second))
    np.testing.assert_array_equal(np.sort(np.concatenate(first)), np.arange(20))


def test_saved_additive_bundle_loads_exact_n_and_seed(tmp_path):
    path = tmp_path / "models.npz"
    np.savez_compressed(
        path,
        weight=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        residual_scale=np.array([0.5, 0.7]),
        n_train=np.array([16, 32]),
        seed=np.array([0, 1]),
        status=np.array(["ok", "ok"]),
        feature_metadata=np.asarray("features"),
        selected_hyperparameters=np.asarray("parameters"),
    )
    loaded = load_additive_model(path, n_train=32, seed=1)
    np.testing.assert_array_equal(loaded.weight, [3.0, 4.0])
    assert loaded.residual_scale == pytest.approx(0.7)
    with pytest.raises(KeyError):
        load_additive_model(path, n_train=64, seed=0)
