from __future__ import annotations

import torch

from crossfit.crosscoder import CrossCoder
from crossfit.crosscoder_training import load_or_fit_crosscoder


def test_checkpointed_crosscoder_resumes_only_complete_matching_run(tmp_path):
    generator = torch.Generator().manual_seed(3)
    evo = torch.randn(20, 4, generator=generator)
    fit = evo + 0.1 * torch.randn(20, 4, generator=generator)
    training = {
        "epochs": 2, "batch_size": 5, "learning_rate": 1e-3,
        "weight_decay": 0.0, "gradient_clip_norm": 1.0,
    }
    path = tmp_path / "model.pt"
    first, trace, resumed = load_or_fit_crosscoder(
        path, dimension=4, width_multiplier=2, k=2, decoder_bias=False,
        evo=evo, fit=fit, training=training, seed=7, device="cpu",
    )
    assert isinstance(first, CrossCoder)
    assert len(trace) == 2
    assert not resumed
    second, second_trace, resumed = load_or_fit_crosscoder(
        path, dimension=4, width_multiplier=2, k=2, decoder_bias=False,
        evo=evo, fit=fit, training=training, seed=7, device="cpu",
    )
    assert resumed
    assert second_trace == trace
    for name, value in first.state_dict().items():
        assert torch.equal(value, second.state_dict()[name])
