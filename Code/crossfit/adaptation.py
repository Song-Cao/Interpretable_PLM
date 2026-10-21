"""LoRA fitness adaptation for one ESM-C dataset."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .losses import CompositeFitnessLoss


class SequenceFitnessAdapter(nn.Module):
    """ESM-C with a constrained mean-pooled Linear(d,1) fitness head."""

    def __init__(self, backbone: nn.Module, dimension: int) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(dimension, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        output = self.backbone(sequence_tokens=tokens)
        pooled = output.embeddings[:, 1:-1].float().mean(dim=1)
        return self.head(pooled).squeeze(-1)


def build_lora_adapter(backbone: nn.Module, *, rank: int, alpha: int, dropout: float) -> nn.Module:
    """Attach LoRA to every backbone linear layer and freeze token embeddings."""
    from peft import LoraConfig, get_peft_model

    config = LoraConfig(
        r=int(rank),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        target_modules="all-linear",
        bias="none",
    )
    adapted = get_peft_model(backbone, config)
    for parameter in adapted.base_model.model.embed.parameters():
        parameter.requires_grad_(False)
    return adapted


class _SequenceDataset(Dataset):
    def __init__(self, sequences: Sequence[str], target: np.ndarray, censoring: Sequence[str]) -> None:
        self.sequences = list(map(str, sequences))
        self.target = np.asarray(target, dtype=np.float32)
        codes = {"none": 0, "floor": 1, "ceiling": 2}
        self.censoring = np.asarray([codes[str(value)] for value in censoring], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[str, float, int]:
        return self.sequences[index], float(self.target[index]), int(self.censoring[index])


@dataclass
class AdapterFit:
    """One trained adapter and its fixed-step loss trace."""

    model: SequenceFitnessAdapter
    objective: CompositeFitnessLoss
    trace: list[dict[str, float]]
    optimizer_steps: int


def _objective(config: dict) -> CompositeFitnessLoss:
    residual, weights = config["residual_scale"], config["weights"]
    return CompositeFitnessLoss(
        margin_sd=float(config["margin_sd"]), temperature=float(config["temperature"]),
        magnitude_mode=str(config["magnitude_mode"]), huber_delta=float(config["huber_delta"]),
        residual_scale_initial=float(residual["initial"]), residual_scale_floor=float(residual["floor"]),
        rank_weight=float(weights["rank"]), magnitude_weight=float(weights["magnitude"]),
        censoring_weight=float(weights["censoring"]), regularization_weight=float(weights["regularization"]),
    )


def fit_adapter(
    model: SequenceFitnessAdapter,
    sequences: Sequence[str],
    target: np.ndarray,
    censoring: Sequence[str],
    *,
    backbone_tokenize,
    objective_config: dict,
    adapter_config: dict,
    epochs: int,
    seed: int,
    device: str | torch.device,
) -> AdapterFit:
    """Train LoRA and the linear head for an exact fixed number of optimizer steps."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dataset = _SequenceDataset(sequences, target, censoring)
    loader = DataLoader(
        dataset, batch_size=int(adapter_config["batch_size"]), shuffle=True,
        generator=torch.Generator().manual_seed(seed), collate_fn=lambda rows: rows,
    )
    model = model.to(device)
    objective = _objective(objective_config).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable += list(objective.residual_scale.parameters())
    optimizer = torch.optim.AdamW(
        trainable, lr=float(adapter_config["learning_rate"]),
        weight_decay=float(adapter_config["weight_decay"]),
    )
    accumulation = int(adapter_config["gradient_accumulation_steps"])
    total_updates = epochs * ((len(loader) + accumulation - 1) // accumulation)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_updates, 1))
    trace: list[dict[str, float]] = []
    optimizer_steps = 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        values = []
        for batch_index, rows in enumerate(loader):
            batch_sequences = [row[0] for row in rows]
            batch_target = torch.tensor([row[1] for row in rows], dtype=torch.float32, device=device)
            batch_censor = torch.tensor([row[2] for row in rows], dtype=torch.long, device=device)
            tokens = backbone_tokenize(batch_sequences)
            output = objective(
                model(tokens), batch_target, batch_censor,
                regularized_parameters=(model.head.weight, model.head.bias),
            )
            (output.total / accumulation).backward()
            values.append(float(output.total.detach()))
            last = batch_index + 1 == len(loader)
            if (batch_index + 1) % accumulation == 0 or last:
                torch.nn.utils.clip_grad_norm_(trainable, float(adapter_config["gradient_clip_norm"]))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                optimizer_steps += 1
        trace.append({"epoch": float(epoch), "loss": float(np.mean(values))})
    return AdapterFit(model, objective, trace, optimizer_steps)


@torch.inference_mode()
def predict_adapter(
    model: SequenceFitnessAdapter,
    sequences: Sequence[str],
    *,
    backbone_tokenize,
    batch_size: int,
    device: str | torch.device,
) -> np.ndarray:
    """Predict per-variant fitness from one adapted ESM-C model."""
    model.eval()
    output = []
    values = list(map(str, sequences))
    for start in range(0, len(values), batch_size):
        tokens = backbone_tokenize(values[start : start + batch_size])
        output.append(model(tokens).float().cpu().numpy())
    return np.concatenate(output).astype(np.float64)
