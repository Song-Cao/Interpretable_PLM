from __future__ import annotations

import numpy as np
import torch

from crossfit.crosscoder import (
    BatchTopK,
    CrossCoder,
    crosscoder_diagnostics,
    decoder_delta,
    reconstruction_loss,
)


def test_batch_topk_has_registered_batch_level_l0():
    layer = BatchTopK(k=2)
    values = torch.tensor([[1.0, 4.0, -2.0, 3.0], [8.0, 2.0, 7.0, 6.0]])
    output = layer(values)
    assert int((output > 0).sum()) == 4
    assert set(output[output > 0].tolist()) == {4.0, 6.0, 7.0, 8.0}


def test_crosscoder_self_pair_delta_is_exactly_half_after_update():
    torch.manual_seed(5)
    model = CrossCoder(dimension=4, width_multiplier=2, k=2)
    values = torch.randn(9, 4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    _, left, right = model(values, values)
    loss = reconstruction_loss(values, values, left, right)
    loss.backward()
    optimizer.step()
    assert torch.equal(model.evo_decoder.weight, model.fit_decoder.weight)
    assert torch.equal(decoder_delta(model), torch.full((8,), 0.5))


def test_crosscoder_diagnostics_reports_realized_l0_and_dead_features():
    torch.manual_seed(7)
    model = CrossCoder(dimension=3, width_multiplier=2, k=1)
    values = np.random.default_rng(7).normal(size=(11, 3)).astype(np.float32)
    result = crosscoder_diagnostics(model, values, values, batch_size=4, device="cpu")
    assert np.isclose(result.realized_l0, 1.0)
    assert 0 <= result.dead_feature_fraction <= 1
    assert result.features == 6
