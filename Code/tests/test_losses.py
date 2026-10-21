from __future__ import annotations

import math

import pytest
import torch

from crossfit.losses import CompositeFitnessLoss, in_batch_rank_indices


def objective(**overrides):
    settings = dict(
        margin_sd=0.25,
        temperature=1.0,
        magnitude_mode="squared",
        huber_delta=1.0,
        residual_scale_initial=1.0,
        residual_scale_floor=0.0001,
        rank_weight=1.0,
        magnitude_weight=1.0,
        censoring_weight=1.0,
        regularization_weight=0.0,
    )
    settings.update(overrides)
    return CompositeFitnessLoss(**settings)


def test_margin_gate_forms_pairs_strictly_inside_batch():
    left, right = in_batch_rank_indices(torch.tensor([0.0, 0.25, 1.0]), 0.25)
    assert list(zip(left.tolist(), right.tolist())) == [(0, 2), (1, 2)]


def test_ranking_uses_direct_score_differences_and_temperature():
    module = objective(magnitude_weight=0.0, censoring_weight=0.0)
    result = module(torch.tensor([2.0, 0.0]), torch.tensor([1.0, -1.0]), ["none"] * 2)
    assert result.ranking == pytest.approx(float(torch.nn.functional.softplus(torch.tensor(-2.0))))
    assert result.n_rank_pairs == 1


@pytest.mark.parametrize(
    "label,score,target,expected",
    [
        ("none", 0.5, 0.0, 0.5 * 0.5**2 + 0.5 * math.log(2 * math.pi)),
        ("floor", -3.0, -2.0, -float(torch.special.log_ndtr(torch.tensor(1.0)))),
        ("ceiling", 3.0, 2.0, -float(torch.special.log_ndtr(torch.tensor(1.0)))),
    ],
)
def test_standard_per_variant_tobit_cases(label, score, target, expected):
    module = objective(rank_weight=0.0, magnitude_weight=0.0)
    output = module(torch.tensor([score]), torch.tensor([target]), [label])
    assert float(output.censoring.detach()) == pytest.approx(expected, abs=1e-6)


def test_floor_tobit_rewards_predictions_consistent_with_upper_bound():
    module = objective(rank_weight=0.0, magnitude_weight=0.0)
    below = module(torch.tensor([-4.0]), torch.tensor([-2.0]), ["floor"]).total
    above = module(torch.tensor([0.0]), torch.tensor([-2.0]), ["floor"]).total
    assert below < above


def test_magnitude_excludes_censored_rows():
    module = objective(rank_weight=0.0, censoring_weight=0.0)
    output = module(
        torch.tensor([2.0, 100.0]), torch.tensor([0.0, -2.0]), ["none", "floor"]
    )
    assert output.magnitude == pytest.approx(4.0)
    assert output.n_uncensored == 1


def test_l2_regularization_is_part_of_single_objective():
    parameter = torch.tensor([3.0, 4.0], requires_grad=True)
    module = objective(
        rank_weight=0.0, magnitude_weight=0.0, censoring_weight=0.0,
        regularization_weight=2.0,
    )
    output = module(
        torch.tensor([0.0], requires_grad=True), torch.tensor([0.0]), ["none"],
        regularized_parameters=[parameter],
    )
    assert float(output.regularization.detach()) == pytest.approx(25.0)
    assert float(output.total.detach()) == pytest.approx(50.0)
    output.total.backward()
    assert parameter.grad.tolist() == pytest.approx([12.0, 16.0])


def test_residual_scale_is_learned():
    module = objective(rank_weight=0.0, magnitude_weight=0.0)
    output = module(torch.tensor([0.5]), torch.tensor([0.0]), ["none"])
    output.total.backward()
    assert module.residual_scale.raw_scale.grad is not None
    assert torch.isfinite(module.residual_scale.raw_scale.grad)
