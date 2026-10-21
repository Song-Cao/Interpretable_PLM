from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from crossfit.esmc import extract_residue_embeddings, masked_log_probabilities


class _FakeESMC(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.tokenizer = SimpleNamespace(mask_token_id=5)

    def _tokenize(self, sequences: list[str]) -> torch.Tensor:
        lookup = {"A": 1, "C": 2}
        return torch.tensor([[3, *[lookup[value] for value in sequence], 4] for sequence in sequences])

    def forward(self, sequence_tokens: torch.Tensor):
        batch, length = sequence_tokens.shape
        base = torch.arange(batch * length * 3, dtype=torch.float32).reshape(batch, length, 3)
        hidden = torch.stack((base, base + 100.0), dim=0)
        logits = torch.zeros(batch, length, 6)
        logits[..., 1] = 1.0
        logits[..., 2] = 2.0
        return SimpleNamespace(embeddings=base, hidden_states=hidden, sequence_logits=logits)


def test_extract_residues_drops_special_tokens_and_orders_layers_batch_first():
    output = extract_residue_embeddings(_FakeESMC(), ["AC", "CA"], batch_size=1, all_layers=True)
    assert output.final.shape == (2, 2, 3)
    assert output.layers.shape == (2, 2, 2, 3)
    np.testing.assert_allclose(output.layers[:, 0], output.final)
    np.testing.assert_allclose(output.layers[:, 1], output.final + 100.0)


def test_masked_log_probability_reads_observed_amino_acid_token():
    values = masked_log_probabilities(
        _FakeESMC(), ["AC", "CA"], [0, 0], batch_size=2
    )
    expected_logits = torch.tensor([0.0, 1.0, 2.0, 0.0, 0.0, 0.0]).log_softmax(0)
    np.testing.assert_allclose(values, [expected_logits[1], expected_logits[2]])
