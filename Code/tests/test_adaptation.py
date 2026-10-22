from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from crossfit.adaptation import (
    SequenceFitnessAdapter,
    fit_adapter,
    load_trainable_state,
    predict_adapter,
    save_trainable_state,
)


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


class _LoRALikeBackbone(_Backbone):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = torch.nn.Linear(4, 2, bias=False)


def test_trainable_adapter_state_round_trip(tmp_path):
    source = SequenceFitnessAdapter(_LoRALikeBackbone(), 4)
    for parameter in source.parameters():
        parameter.requires_grad_(False)
    source.backbone.lora_A.weight.requires_grad_(True)
    source.head.weight.requires_grad_(True)
    source.head.bias.requires_grad_(True)
    with torch.no_grad():
        source.backbone.lora_A.weight.fill_(1.25)
        source.head.weight.fill_(-0.75)
        source.head.bias.fill_(0.5)

    path = tmp_path / "adapter.pt"
    save_trainable_state(source, path)
    target = SequenceFitnessAdapter(_LoRALikeBackbone(), 4)
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    target.backbone.lora_A.weight.requires_grad_(True)
    target.head.weight.requires_grad_(True)
    target.head.bias.requires_grad_(True)
    load_trainable_state(target, path)

    assert torch.equal(target.backbone.lora_A.weight, source.backbone.lora_A.weight)
    assert torch.equal(target.head.weight, source.head.weight)
    assert torch.equal(target.head.bias, source.head.bias)


def test_initialization_seed_precedes_training_seed():
    def construct(seed: int) -> SequenceFitnessAdapter:
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = SequenceFitnessAdapter(_LoRALikeBackbone(), 4)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.backbone.lora_A.weight.requires_grad_(True)
        model.head.weight.requires_grad_(True)
        model.head.bias.requires_grad_(True)
        return model

    first = construct(17)
    torch.rand(100)
    restarted = construct(17)
    assert all(
        torch.equal(left, right)
        for left, right in zip(first.parameters(), restarted.parameters(), strict=True)
    )
