"""LoRA fitness adaptation for one ESM-C dataset."""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from scipy.stats import spearmanr
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


def save_trainable_state(model: SequenceFitnessAdapter, path: str | Path) -> None:
    """Save the adapter and head parameters without relying on a backbone state dict."""
    state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    if not state or not any("lora_" in name for name in state):
        raise AssertionError("adapter state contains no trainable LoRA parameters")
    torch.save({"format": "crossfit_trainable_state_v1", "parameters": state}, path)


def load_trainable_state(
    model: SequenceFitnessAdapter, path: str | Path, *, map_location: str = "cpu"
) -> None:
    """Load an exact adapter and head parameter state and validate every name and shape."""
    artifact = torch.load(path, map_location=map_location, weights_only=True)
    if artifact.get("format") != "crossfit_trainable_state_v1":
        raise ValueError("unrecognized adapter artifact format")
    saved = artifact["parameters"]
    current = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(saved) != set(current):
        missing = sorted(set(current) - set(saved))
        unexpected = sorted(set(saved) - set(current))
        raise ValueError(f"adapter parameter mismatch: missing={missing}, unexpected={unexpected}")
    with torch.no_grad():
        for name, parameter in current.items():
            value = saved[name]
            if tuple(value.shape) != tuple(parameter.shape):
                raise ValueError(
                    f"adapter parameter shape mismatch for {name}: "
                    f"saved={tuple(value.shape)}, current={tuple(parameter.shape)}"
                )
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


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


@dataclass
class EarlyStoppedAdapterFit:
    """An adapter restored to its best validation-Spearman checkpoint."""

    model: SequenceFitnessAdapter
    objective: CompositeFitnessLoss
    trace: list[dict[str, float]]
    best_epoch: int
    best_optimizer_steps: int
    executed_optimizer_steps: int
    validation_spearman: float
    learning_rates_to_best: list[float]


def _trainable_state(model: SequenceFitnessAdapter) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _restore_trainable_state(
    model: SequenceFitnessAdapter, state: dict[str, torch.Tensor]
) -> None:
    current = {name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad}
    if set(state) != set(current):
        raise ValueError("trainable adapter parameter names changed")
    with torch.no_grad():
        for name, parameter in current.items():
            value = state[name]
            if tuple(value.shape) != tuple(parameter.shape):
                raise ValueError(f"trainable adapter parameter shape changed for {name}")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


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


def fit_adapter_early_stopping(
    model: SequenceFitnessAdapter,
    train_sequences: Sequence[str],
    train_target: np.ndarray,
    train_censoring: Sequence[str],
    validation_sequences: Sequence[str],
    validation_target: np.ndarray,
    validation_censoring: Sequence[str],
    *,
    backbone_tokenize,
    objective_config: dict,
    adapter_config: dict,
    max_epochs: int,
    patience: int,
    min_delta: float,
    scheduler_factor: float,
    scheduler_patience: int,
    seed: int,
    device: str | torch.device,
) -> EarlyStoppedAdapterFit:
    """Early-stop on validation Spearman and restore the exact best adapter state."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dataset = _SequenceDataset(train_sequences, train_target, train_censoring)
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=float(scheduler_factor), patience=int(scheduler_patience),
    )
    accumulation = int(adapter_config["gradient_accumulation_steps"])
    best_score = -float("inf")
    best_epoch = -1
    best_steps = 0
    best_model = _trainable_state(model)
    best_scale = copy.deepcopy(objective.residual_scale.state_dict())
    stale = 0
    optimizer_steps = 0
    learning_rates: list[float] = []
    trace: list[dict[str, float]] = []
    validation_target_array = np.asarray(validation_target, dtype=np.float32)
    validation_codes = torch.tensor(
        [{"none": 0, "floor": 1, "ceiling": 2}[str(value)] for value in validation_censoring],
        dtype=torch.long, device=device,
    )
    for epoch in range(int(max_epochs)):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = []
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
            totals.append(float(output.total.detach()))
            last = batch_index + 1 == len(loader)
            if (batch_index + 1) % accumulation == 0 or last:
                torch.nn.utils.clip_grad_norm_(trainable, float(adapter_config["gradient_clip_norm"]))
                learning_rates.append(float(optimizer.param_groups[0]["lr"]))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
        prediction = predict_adapter(
            model, validation_sequences, backbone_tokenize=backbone_tokenize,
            batch_size=int(adapter_config["batch_size"]), device=device,
        )
        score = float(spearmanr(prediction, validation_target_array).statistic)
        score = score if np.isfinite(score) else -1.0
        with torch.no_grad():
            validation_output = objective(
                torch.as_tensor(prediction, dtype=torch.float32, device=device),
                torch.as_tensor(validation_target_array, device=device), validation_codes,
                regularized_parameters=(model.head.weight, model.head.bias),
            )
            validation_loss = float(validation_output.total)
        scheduler.step(score)
        trace.append({
            "epoch": float(epoch), "loss": float(np.mean(totals)),
            "validation_loss": validation_loss, "validation_spearman": score,
            "optimizer_steps": float(optimizer_steps),
        })
        if score > best_score + float(min_delta):
            best_score, best_epoch, best_steps, stale = score, epoch, optimizer_steps, 0
            best_model = _trainable_state(model)
            best_scale = copy.deepcopy(objective.residual_scale.state_dict())
        else:
            stale += 1
        if stale >= int(patience):
            break
    _restore_trainable_state(model, best_model)
    objective.residual_scale.load_state_dict(best_scale)
    return EarlyStoppedAdapterFit(
        model=model, objective=objective, trace=trace, best_epoch=best_epoch,
        best_optimizer_steps=best_steps, executed_optimizer_steps=optimizer_steps,
        validation_spearman=best_score, learning_rates_to_best=learning_rates[:best_steps],
    )


def fit_adapter_for_steps(
    model: SequenceFitnessAdapter,
    sequences: Sequence[str],
    target: np.ndarray,
    censoring: Sequence[str],
    *,
    backbone_tokenize,
    objective_config: dict,
    adapter_config: dict,
    optimizer_steps: int,
    learning_rates: Sequence[float],
    seed: int,
    device: str | torch.device,
) -> AdapterFit:
    """Train a control for an exact optimizer-step count and a supplied LR schedule."""
    if optimizer_steps < 1 or len(learning_rates) != optimizer_steps:
        raise ValueError("the control requires one learning rate per optimizer step")
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
        trainable, lr=float(learning_rates[0]), weight_decay=float(adapter_config["weight_decay"]),
    )
    accumulation = int(adapter_config["gradient_accumulation_steps"])
    trace: list[dict[str, float]] = []
    completed = 0
    epoch = 0
    while completed < optimizer_steps:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = []
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
            totals.append(float(output.total.detach()))
            last = batch_index + 1 == len(loader)
            if (batch_index + 1) % accumulation == 0 or last:
                for group in optimizer.param_groups:
                    group["lr"] = float(learning_rates[completed])
                torch.nn.utils.clip_grad_norm_(trainable, float(adapter_config["gradient_clip_norm"]))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                completed += 1
                if completed == optimizer_steps:
                    break
        trace.append({"epoch": float(epoch), "loss": float(np.mean(totals))})
        epoch += 1
    return AdapterFit(model, objective, trace, completed)


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
