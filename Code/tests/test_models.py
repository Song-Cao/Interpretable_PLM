from __future__ import annotations

import numpy as np
import pytest
import torch

from crossfit.models import DiagonalRidgeRegressor, build_model


def test_diagonal_ridge_recovers_linear_scores_with_unpenalized_intercept():
    features = np.array([[0.0], [1.0], [2.0], [3.0]])
    target = 4.0 + 2.0 * features[:, 0]
    model = DiagonalRidgeRegressor(site_penalty=1e-12).fit(features, target)
    np.testing.assert_allclose(model.predict(features), target, atol=1e-9)
    assert model.intercept_ == pytest.approx(4.0, abs=1e-9)


def test_augmented_ridge_uses_explicit_near_unpenalized_density_column():
    features = np.eye(4)
    density = np.array([-2.0, -1.0, 1.0, 2.0])
    target = 3.0 * density
    model = DiagonalRidgeRegressor(site_penalty=1e6, density_penalty=1e-8)
    model.fit(features, target, density=density)
    assert model.coefficient_[-1] == pytest.approx(3.0, rel=1e-5)
    np.testing.assert_allclose(model.predict(features, density=density), target, atol=1e-5)


@pytest.mark.parametrize(
    "name,input_shape,parameters,shape",
    [
        ("ridge", (12,), {}, (3,)),
        ("mlp", (12,), {"hidden_dims": [16, 8], "dropout": 0.2}, (3,)),
        ("conv_residual", (20, 5), {"width": 16, "depth": 2, "kernel_size": 3, "dropout": 0.1}, (3,)),
    ],
)
def test_neural_model_factory_returns_direct_variant_scores(name, input_shape, parameters, shape):
    model = build_model(name, input_shape, parameters)
    features = torch.randn((3,) + input_shape)
    assert model(features).shape == shape


@pytest.mark.parametrize("name", ["random_forest"])
def test_tree_model_factory_uses_published_sklearn_regressors(name):
    model = build_model(name, (4,), {"n_estimators": 10, "random_state": 3})
    features = np.arange(40, dtype=float).reshape(10, 4)
    target = np.arange(10, dtype=float)
    assert model.fit(features, target).predict(features).shape == (10,)


def test_factory_has_no_unregistered_architecture():
    with pytest.raises(KeyError):
        build_model("unregistered_model", (4,), {})
