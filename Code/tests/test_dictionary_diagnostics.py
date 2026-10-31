from __future__ import annotations

import numpy as np
import torch

from crossfit.crosscoder import CrossCoder
from crossfit.dictionary_diagnostics import (
    activation_counts,
    covariance_spectrum,
    death_timing,
    participation_ratio,
    resample_dead_features,
)


def test_covariance_spectrum_and_participation_ratio_match_numpy():
    values = np.random.default_rng(3).normal(size=(31, 5)).astype(np.float32)
    observed = covariance_spectrum(values, batch_size=7, device="cpu")
    expected = np.linalg.eigvalsh(np.cov(values, rowvar=False, bias=True))[::-1]
    np.testing.assert_allclose(observed, expected, rtol=1e-10, atol=1e-10)
    assert np.isclose(
        participation_ratio(observed), expected.sum() ** 2 / np.square(expected).sum()
    )


def test_activation_counts_restore_training_state():
    torch.manual_seed(4)
    model = CrossCoder(3, 2, 1, decoder_bias=False).train()
    values = np.random.default_rng(4).normal(size=(17, 3)).astype(np.float32)
    counts = activation_counts(model, values, values, batch_size=5, device="cpu")
    assert counts.shape == (6,)
    assert counts.sum() == 17
    assert model.training


def test_death_timing_distinguishes_never_active_and_crowded_out():
    counts = np.array(
        [
            [0, 3, 0, 2, 0],
            [0, 4, 5, 0, 0],
            [0, 0, 7, 0, 6],
        ]
    )
    result = death_timing(counts)
    assert result.never_active == 1
    assert result.crowded_out == 2
    assert result.revived == 2
    assert result.live_at_initialization == 2
    assert result.live_at_final == 2


def test_resampling_changes_only_dead_slices_and_clears_moments():
    torch.manual_seed(5)
    model = CrossCoder(3, 2, 1, decoder_bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    values = torch.randn(8, 3)
    code, left, right = model(values, values)
    (left.square().mean() + right.square().mean() + code.square().mean()).backward()
    optimizer.step()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    dead = np.array([False, True, False, False, True, False])
    changed = resample_dead_features(
        model, optimizer, dead, generator=torch.Generator().manual_seed(19)
    )
    assert changed == 2
    live = torch.tensor([0, 2, 3, 5])
    torch.testing.assert_close(model.encoder.weight[live], before["encoder.weight"][live])
    torch.testing.assert_close(
        model.evo_decoder.weight[:, live], before["evo_decoder.weight"][:, live]
    )
    assert torch.equal(
        model.evo_decoder.weight[:, dead], model.fit_decoder.weight[:, dead]
    )
    for parameter, axis in (
        (model.encoder.weight, 0),
        (model.encoder.bias, 0),
        (model.evo_decoder.weight, 1),
        (model.fit_decoder.weight, 1),
    ):
        moment = optimizer.state[parameter]["exp_avg"]
        selected = moment[dead] if axis == 0 else moment[:, dead]
        assert torch.count_nonzero(selected) == 0
