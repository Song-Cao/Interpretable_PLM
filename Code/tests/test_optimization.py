from __future__ import annotations

import pytest
import torch

from crossfit.optimization import step_across_assays


def test_optimizer_steps_once_after_averaging_distinct_assay_gradients():
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    losses = (("a", (parameter - 2.0).square()), ("b", (parameter - 4.0).square()))
    result = step_across_assays(
        optimizer,
        losses,
        [parameter],
        min_assays=2,
        gradient_clip_norm=None,
    )
    assert parameter.item() == pytest.approx(0.6)
    assert result.assay_ids == ("a", "b")
    assert result.mean_loss == pytest.approx(10.0)


def test_optimizer_rejects_single_or_repeated_assay_batches():
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    with pytest.raises(ValueError, match="several"):
        step_across_assays(
            optimizer,
            [("a", parameter.square())],
            [parameter],
            min_assays=2,
            gradient_clip_norm=None,
        )
    with pytest.raises(ValueError, match="distinct"):
        step_across_assays(
            optimizer,
            [("a", parameter.square()), ("a", parameter.square())],
            [parameter],
            min_assays=2,
            gradient_clip_norm=1.0,
        )
