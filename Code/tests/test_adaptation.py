from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from crossfit.adaptation import SequenceFitnessAdapter, fit_adapter, predict_adapter


class _Backbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(8, 4)
        self.linear = torch.nn.Linear(4, 4)

    def forward(self, sequence_tokens: torch.Tensor):
        return SimpleNamespace(embeddings=self.linear(self.embed(sequence_tokens)))


def _tokenize(sequences: list[str]) -> torch.Tensor:
    return torch.tensor([[6, *[1 if value == "A" else 2 for value in sequence], 7] for sequence in sequences])


def test_adapter_trains_fixed_accumulated_steps_and_scores_variants():
    model = SequenceFitnessAdapter(_Backbone(), 4)
    sequences = ["AA", "AC", "CA", "CC", "AA"]
    fit = fit_adapter(
        model, sequences, np.arange(5, dtype=np.float32), ["none"] * 5,
        backbone_tokenize=_tokenize,
        objective_config={
            "margin_sd": 0.1, "temperature": 1.0, "magnitude_mode": "huber", "huber_delta": 1.0,
            "residual_scale": {"initial": 1.0, "floor": 1e-4},
            "weights": {"rank": 1.0, "magnitude": 1.0, "censoring": 1.0, "regularization": 0.0},
        },
        adapter_config={
            "batch_size": 2, "gradient_accumulation_steps": 2, "learning_rate": 0.01,
            "weight_decay": 1e-4, "gradient_clip_norm": 1.0,
        },
        epochs=3, seed=4, device="cpu",
    )
    assert fit.optimizer_steps == 6
    assert len(fit.trace) == 3
    assert predict_adapter(
        fit.model, sequences, backbone_tokenize=_tokenize, batch_size=2, device="cpu"
    ).shape == (5,)
