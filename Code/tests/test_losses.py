from __future__ import annotations

import math

import pytest
import torch

from crossfit.losses import (
    CompositeFitnessLoss,
    censor_constraint_sign,
    construct_pairs,
)


def make_pairs(censor_u: str, censor_v: str, y_u: float, y_v: float):
    return construct_pairs(
        torch.tensor([y_u, y_v]),
        torch.tensor([0, 0]),
        ["AAAA", "AAAT"],
        [censor_u, censor_v],
        torch.tensor([float("nan"), float("nan")]),
        margin_sd=0.0,
        max_hamming_distance=1,
    )


@pytest.mark.parametrize(
    "censor_u,censor_v,y_u,y_v,expected",
    [
        ("floor", "none", -2.0, 0.0, 1),
        ("ceiling", "none", 2.0, 0.0, -1),
        ("none", "floor", 0.0, -2.0, -1),
        ("none", "ceiling", 0.0, 2.0, 1),
    ],
)
def test_single_censored_pair_sign_rule(censor_u, censor_v, y_u, y_v, expected):
    pairs = make_pairs(censor_u, censor_v, y_u, y_v)
    assert len(pairs) == 1
    assert int(pairs.censor_sign[0]) == expected


def test_same_bound_double_censored_pair_is_dropped():
    pairs = make_pairs("floor", "floor", -2.0, -2.0)
    assert len(pairs) == 0


@pytest.mark.parametrize(
    "censor_u,censor_v,y_u,y_v,expected",
    [
        ("floor", "ceiling", -2.0, 2.0, 1),
        ("ceiling", "floor", 2.0, -2.0, -1),
    ],
)
def test_opposite_bound_double_censored_pair_is_kept(
    censor_u, censor_v, y_u, y_v, expected
):
    pairs = make_pairs(censor_u, censor_v, y_u, y_v)
    assert len(pairs) == 1
    assert int(pairs.censor_sign[0]) == expected


@pytest.mark.parametrize(
    "censor_u,censor_v,y_u,y_v",
    [
        ("floor", "none", -2.0, 0.0),
        ("ceiling", "none", 2.0, 0.0),
        ("none", "floor", 0.0, -2.0),
        ("none", "ceiling", 0.0, 2.0),
    ],
)
def test_difference_censoring_rewards_only_the_allowed_direction(
    censor_u, censor_v, y_u, y_v
):
    pairs = make_pairs(censor_u, censor_v, y_u, y_v)
    module = CompositeFitnessLoss(
        n_assays=1,
        prediction_mode="difference",
        temperature=1.0,
        magnitude_mode="squared",
        huber_delta=1.0,
        residual_scale_initial=1.0,
        residual_scale_floor=0.0001,
        rank_weight=0.0,
        magnitude_weight=0.0,
        censoring_weight=1.0,
        regularization_weight=0.0,
    )
    observed = pairs.target_difference
    sign = pairs.censor_sign.to(dtype=observed.dtype)
    allowed = observed - sign
    forbidden = observed + sign
    allowed_loss = module(allowed, pairs).total
    forbidden_loss = module(forbidden, pairs).total
    assert allowed_loss < forbidden_loss


def test_pair_gate_is_strict_and_enforces_assay_and_hamming_strata():
    pairs = construct_pairs(
        torch.tensor([0.0, 0.25, 1.0, 2.0]),
        torch.tensor([0, 0, 1, 0]),
        ["AAAA", "AAAT", "AAAG", "AATT"],
        ["none"] * 4,
        torch.full((4,), float("nan")),
        margin_sd=0.25,
        max_hamming_distance=1,
    )
    assert list(zip(pairs.u.tolist(), pairs.v.tolist())) == [(1, 3)]


def test_composite_objective_sums_weighted_terms_and_l1():
    pairs = make_pairs("none", "none", 1.0, -1.0)
    readout = torch.tensor([-1.0, 2.0], requires_grad=True)
    prediction = torch.tensor([0.5], requires_grad=True)
    module = CompositeFitnessLoss(
        n_assays=1,
        prediction_mode="difference",
        temperature=1.0,
        magnitude_mode="squared",
        huber_delta=1.0,
        residual_scale_initial=1.0,
        residual_scale_floor=0.0001,
        rank_weight=2.0,
        magnitude_weight=3.0,
        censoring_weight=4.0,
        regularization_weight=5.0,
    )
    output = module(prediction, pairs, readout_parameters=[readout])
    expected_rank = torch.nn.functional.softplus(torch.tensor(-0.5))
    expected = 2.0 * expected_rank + 3.0 * (0.5 - 2.0) ** 2 + 5.0 * 3.0
    assert torch.allclose(output.total, expected)
    assert output.n_pairs == 1
    assert output.n_censor_constraints == 0
    output.total.backward()
    assert prediction.grad is not None
    assert readout.grad is not None


def test_difference_censoring_learns_residual_scale():
    pairs = make_pairs("floor", "none", -2.0, 0.0)
    prediction = torch.tensor([0.0], requires_grad=True)
    module = CompositeFitnessLoss(
        n_assays=1,
        prediction_mode="difference",
        temperature=1.0,
        magnitude_mode="huber",
        huber_delta=1.0,
        residual_scale_initial=1.0,
        residual_scale_floor=0.0001,
        rank_weight=0.0,
        magnitude_weight=0.0,
        censoring_weight=1.0,
        regularization_weight=0.0,
    )
    output = module(prediction, pairs)
    output.total.backward()
    assert module.residual_scale.raw_scale.grad is not None
    assert torch.isfinite(module.residual_scale.raw_scale.grad).all()
    assert not torch.all(module.residual_scale.raw_scale.grad == 0)


def test_score_head_uses_standard_tobit_for_all_used_variants():
    targets = torch.tensor([-2.0, 0.0, 2.0])
    pairs = construct_pairs(
        targets,
        torch.zeros(3, dtype=torch.long),
        ["AAAA", "AAAT", "AAAG"],
        ["floor", "none", "ceiling"],
        torch.full((3,), float("nan")),
        margin_sd=0.1,
        max_hamming_distance=1,
    )
    scores = torch.tensor([-3.0, 0.5, 3.0], requires_grad=True)
    module = CompositeFitnessLoss(
        n_assays=1,
        prediction_mode="score",
        temperature=1.0,
        magnitude_mode="squared",
        huber_delta=1.0,
        residual_scale_initial=1.0,
        residual_scale_floor=0.0001,
        rank_weight=0.0,
        magnitude_weight=0.0,
        censoring_weight=1.0,
        regularization_weight=0.0,
    )
    output = module(
        scores,
        pairs,
        variant_targets=targets,
        variant_assay_index=torch.zeros(3, dtype=torch.long),
        variant_censoring=["floor", "none", "ceiling"],
    )
    expected = (
        -torch.special.log_ndtr(torch.tensor(1.0))
        + 0.5 * math.log(2.0 * math.pi)
        + 0.5 * 0.5**2
        - torch.special.log_ndtr(torch.tensor(1.0))
    )
    assert torch.allclose(output.censoring, expected, atol=1e-6)
    assert output.n_censor_constraints == 2


def test_constraint_sign_rejects_conflicting_codes():
    with pytest.raises(ValueError):
        censor_constraint_sign(1, 1 + 2)


def test_magnitude_sigma_weighting_is_off_by_default():
    pairs = construct_pairs(
        torch.tensor([2.0, 0.0]),
        torch.zeros(2, dtype=torch.long),
        ["AAAA", "AAAT"],
        ["none", "none"],
        torch.tensor([0.1, 0.1]),
        margin_sd=0.0,
        max_hamming_distance=1,
    )
    module = CompositeFitnessLoss(
        n_assays=1,
        prediction_mode="difference",
        temperature=1.0,
        magnitude_mode="squared",
        huber_delta=1.0,
        residual_scale_initial=1.0,
        residual_scale_floor=0.0001,
        rank_weight=0.0,
        magnitude_weight=1.0,
        censoring_weight=0.0,
        regularization_weight=0.0,
    )
    assert module(torch.tensor([0.0]), pairs).magnitude == pytest.approx(4.0)


def test_enabled_sigma_weighting_requires_coverage_and_normalizes_within_assay():
    pairs = construct_pairs(
        torch.tensor([3.0, 1.0, 0.0]),
        torch.zeros(3, dtype=torch.long),
        ["AAAA", "AAAT", "AAAG"],
        ["none"] * 3,
        torch.tensor([0.1, 0.2, 0.4]),
        margin_sd=0.0,
        max_hamming_distance=1,
    )
    kwargs = dict(
        n_assays=1,
        prediction_mode="difference",
        temperature=1.0,
        magnitude_mode="squared",
        huber_delta=1.0,
        residual_scale_initial=1.0,
        residual_scale_floor=0.0001,
        rank_weight=0.0,
        magnitude_weight=1.0,
        censoring_weight=0.0,
        regularization_weight=0.0,
        use_sigma_magnitude_weights=True,
        min_sigma_coverage=0.8,
    )
    without_threshold = dict(kwargs)
    without_threshold.pop("min_sigma_coverage")
    with pytest.raises(ValueError, match="threshold"):
        CompositeFitnessLoss(**without_threshold, sigma_coverage_by_assay=[1.0])
    low_coverage = CompositeFitnessLoss(**kwargs, sigma_coverage_by_assay=[0.5])
    with pytest.raises(ValueError, match="below threshold"):
        low_coverage(torch.zeros(3), pairs)

    weighted = CompositeFitnessLoss(**kwargs, sigma_coverage_by_assay=[1.0])
    raw = torch.tensor([1 / 0.05, 1 / 0.17, 1 / 0.20])
    expected = ((raw / raw.mean()) * pairs.target_difference.square()).sum()
    assert torch.allclose(weighted(torch.zeros(3), pairs).magnitude, expected)

    pairs_with_missing = construct_pairs(
        torch.tensor([3.0, 1.0, 0.0]),
        torch.zeros(3, dtype=torch.long),
        ["AAAA", "AAAT", "AAAG"],
        ["none"] * 3,
        torch.tensor([0.1, 0.2, float("nan")]),
        margin_sd=0.0,
        max_hamming_distance=1,
    )
    mixed = CompositeFitnessLoss(**kwargs, sigma_coverage_by_assay=[0.9])
    mixed_raw = torch.tensor([1 / 0.05, 1.0, 1.0])
    mixed_expected = (
        (mixed_raw / mixed_raw.mean()) * pairs_with_missing.target_difference.square()
    ).sum()
    assert torch.allclose(mixed(torch.zeros(3), pairs_with_missing).magnitude, mixed_expected)
