"""Local ESM-C extraction used by Step 2 and Gate 1."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class ESMCBatch:
    """Residue embeddings and optional layerwise states, excluding BOS/EOS."""

    final: np.ndarray
    layers: np.ndarray | None


def load_esmc(
    model_name: str,
    device: str | torch.device = "cuda",
    *,
    repository: str | None = None,
    expected_revision: str | None = None,
) -> torch.nn.Module:
    """Load the pinned local open ESM-C model in inference mode."""
    from esm.models.esmc import ESMC

    model = ESMC.from_pretrained(model_name, device=torch.device(device))
    if repository is not None and expected_revision is not None:
        cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        reference = cache_root / "hub" / f"models--{repository.replace('/', '--')}" / "refs" / "main"
        if not reference.exists() or reference.read_text().strip() != expected_revision:
            observed = reference.read_text().strip() if reference.exists() else "missing"
            raise RuntimeError(
                f"ESM-C revision mismatch for {repository}: expected {expected_revision}, found {observed}"
            )
    return model.eval()


def _batched(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


@torch.inference_mode()
def extract_residue_embeddings(
    model: torch.nn.Module,
    sequences: Sequence[str],
    *,
    batch_size: int,
    all_layers: bool = False,
) -> ESMCBatch:
    """Extract aligned residue states, dropping exactly the two special tokens."""
    values = [str(value) for value in sequences]
    if not values or len({len(value) for value in values}) != 1:
        raise ValueError("ESM-C extraction needs non-empty equal-length sequences")
    final_rows: list[np.ndarray] = []
    layer_rows: list[np.ndarray] = []
    for batch in _batched(values, batch_size):
        tokens = model._tokenize(list(batch))
        output = model(sequence_tokens=tokens)
        if output.embeddings.shape[1] != len(values[0]) + 2:
            raise AssertionError("ESM-C token count is not sequence length plus BOS/EOS")
        final_rows.append(output.embeddings[:, 1:-1].float().cpu().numpy())
        if all_layers:
            # ESM-C returns layer x batch x token x channel.
            layer_rows.append(output.hidden_states[:, :, 1:-1].float().cpu().numpy())
    final = np.concatenate(final_rows, axis=0)
    layers = np.concatenate(layer_rows, axis=1).transpose(1, 0, 2, 3) if all_layers else None
    return ESMCBatch(final=final, layers=layers)


@torch.inference_mode()
def masked_log_probabilities(
    model: torch.nn.Module,
    sequences: Sequence[str],
    positions: Sequence[int],
    *,
    batch_size: int,
) -> np.ndarray:
    """Read log P(observed residue | masked sequence context) for aligned tasks."""
    values = [str(value) for value in sequences]
    position_values = np.asarray(positions, dtype=np.int64)
    if len(values) != len(position_values) or not values:
        raise ValueError("masked tasks must align and be non-empty")
    if any(position < 0 or position >= len(sequence) for sequence, position in zip(values, position_values)):
        raise ValueError("masked position is outside its sequence")
    output_values: list[np.ndarray] = []
    for start in range(0, len(values), batch_size):
        batch = values[start : start + batch_size]
        batch_positions = position_values[start : start + batch_size]
        tokens = model._tokenize(batch)
        row = torch.arange(len(batch), device=tokens.device)
        token_positions = torch.as_tensor(batch_positions + 1, device=tokens.device)
        observed = tokens[row, token_positions].clone()
        tokens[row, token_positions] = int(model.tokenizer.mask_token_id)
        logits = model(sequence_tokens=tokens).sequence_logits
        values_at_position = torch.log_softmax(logits[row, token_positions].float(), dim=-1)
        output_values.append(values_at_position[row, observed].cpu().numpy())
    return np.concatenate(output_values).astype(np.float64)
