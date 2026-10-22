from __future__ import annotations

import numpy as np
import torch

from crossfit.models import RidgeHead
from crossfit.training import (
    FeatureNormalizer,
    fit_torch_fixed_epochs,
    fit_torch_predictor,
    predict_torch,
)


def test_feature_normalizer_uses_training_moments_for_vector_and_sequence_inputs():
    vector = np.arange(24, dtype=np.float32).reshape(6, 4)
    vector_normalizer = FeatureNormalizer.fit(vector)
    np.testing.assert_allclose(vector_normalizer.transform(vector).mean(axis=0), 0.0, atol=1e-6)

    sequence = np.arange(48, dtype=np.float32).reshape(4, 3, 4)
    sequence_normalizer = FeatureNormalizer.fit(sequence)
    normalized = sequence_normalizer.transform(sequence)
    np.testing.assert_allclose(normalized.mean(axis=(0, 1)), 0.0, atol=1e-6)


def test_torch_fit_selects_on_validation_and_returns_direct_scores():
    generator = np.random.default_rng(7)
    features = generator.normal(size=(48, 3)).astype(np.float32)
    target = (features[:, 0] - 0.5 * features[:, 1]).astype(np.float32)
    fit = fit_torch_predictor(
        RidgeHead(3),
        features[:36],
        target[:36],
        ["none"] * 36,
        features[36:],
        target[36:],
        ["none"] * 12,
        objective_config={
            "margin_sd": 0.1,
            "temperature": 1.0,
            "magnitude_mode": "huber",
            "huber_delta": 1.0,
            "residual_scale": {"initial": 1.0, "floor": 1e-4},
            "weights": {"rank": 1.0, "magnitude": 1.0, "censoring": 1.0, "regularization": 0.0},
        },
        optimization_config={
            "max_epochs": 20,
            "early_stopping_patience": 8,
            "early_stopping_min_delta": 1e-4,
            "scheduler_factor": 0.5,
            "scheduler_patience": 3,
            "gradient_clip_norm": 1.0,
        },
        parameters={"batch_size": 12, "learning_rate": 0.03, "weight_decay": 1e-4},
        seed=4,
        device="cpu",
    )
    prediction = predict_torch(fit.model, fit.normalizer, features[36:], device="cpu")
    assert prediction.shape == (12,)
    assert fit.validation_spearman > 0.8
    assert all("validation_loss" in row for row in fit.history)
    assert torch.isfinite(fit.loss.residual_scale())


def test_fixed_epoch_refit_never_requires_validation_data():
    features = np.arange(60, dtype=np.float32).reshape(20, 3)
    target = features[:, 0] / 10.0
    settings = {
        "margin_sd": 0.1,
        "temperature": 1.0,
        "magnitude_mode": "huber",
        "huber_delta": 1.0,
        "residual_scale": {"initial": 1.0, "floor": 1e-4},
        "weights": {"rank": 1.0, "magnitude": 1.0, "censoring": 1.0, "regularization": 0.0},
    }
    fit = fit_torch_fixed_epochs(
        RidgeHead(3), features, target, ["none"] * 20,
        objective_config=settings,
        optimization_config={"gradient_clip_norm": 1.0},
        parameters={"batch_size": 10, "learning_rate": 0.01, "weight_decay": 1e-4},
        epochs=3, seed=2, device="cpu",
    )
    assert fit.epochs == 3
    assert predict_torch(fit.model, fit.normalizer, features, device="cpu").shape == (20,)
