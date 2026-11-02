"""Streaming ESM-C and dPLL scoring for the isolated IRED design search."""
from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from crossfit.adaptation import (
    SequenceFitnessAdapter, build_lora_adapter, load_trainable_state, predict_adapter,
)
from crossfit.crosscoder import CrossCoder
from crossfit.esmc import load_esmc, masked_log_probabilities
from crossfit.feature_pooling import pool_codes
from crossfit.representations import esm_mean, esm_mutpool, esm_residue_delta
from crossfit.utils import REPO_ROOT, load_config

from .mutations import apply, parse_set, parse_mutation
from .scoring import GenerationScorer


BASE_ARMS = (
    "esm_mean_ridge", "esm_mean_mlp", "esm_mean_random_forest",
    "esm_mutpool_ridge", "esm_residue_conv", "esm_residue_delta_conv",
)


def _new_adapter(config: dict, checkpoint: Path, *, seed: int, device: str) -> SequenceFitnessAdapter:
    settings = config["training"]["selected"]["adapter_linear"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    backbone = load_esmc(
        str(config["backbone"]["name"]), device,
        repository=str(config["backbone"]["repository"]),
        expected_revision=str(config["backbone"]["revision"]),
    )
    adapted = build_lora_adapter(
        backbone, rank=int(settings["rank"]), alpha=int(settings["alpha"]),
        dropout=float(settings["dropout"]),
    )
    model = SequenceFitnessAdapter(adapted, int(backbone.embed.embedding_dim)).to(device)
    load_trainable_state(model, checkpoint, map_location="cpu")
    return model.eval()


class StreamingEngine:
    """Score candidate unions without retaining full-corpus residue tensors."""

    def __init__(self, config: dict, wild_type: str, *, device: str = "cuda") -> None:
        self.config = config
        self.wild_type = wild_type
        self.device = device
        self.base = load_esmc(
            str(config["backbone"]["name"]), device,
            repository=str(config["backbone"]["repository"]),
            expected_revision=str(config["backbone"]["revision"]),
        )
        self.scorers = {arm: GenerationScorer(arm, config, device=device) for arm in BASE_ARMS}
        with torch.inference_mode():
            tokens = self.base._tokenize([wild_type])
            output = self.base(sequence_tokens=tokens)
            self.wild_residue = output.embeddings[:, 1:-1].float().cpu().numpy()[0]
        self.wild_logp = self._wild_log_probabilities()
        self._phi_components = None
        self._adapter = None

    def _wild_log_probabilities(self) -> np.ndarray:
        sequences = [self.wild_type] * len(self.wild_type)
        positions = list(range(len(self.wild_type)))
        return masked_log_probabilities(self.base, sequences, positions, batch_size=32)

    def benchmark(self, mutation_sets: list[str], *, batch_size: int = 8) -> dict[str, float]:
        """Time representative embedding and masked-density work."""
        sequences = [apply(self.wild_type, value) for value in mutation_sets]
        torch.cuda.synchronize()
        started = time.perf_counter()
        self._base_outputs(sequences, batch_size=batch_size)
        torch.cuda.synchronize()
        embedding_seconds = time.perf_counter() - started
        tasks_seq, tasks_pos = [], []
        for sequence, value in zip(sequences, mutation_sets):
            for token in parse_set(value):
                _, position, _ = parse_mutation(token)
                tasks_seq.append(sequence); tasks_pos.append(position - 1)
        torch.cuda.synchronize()
        started = time.perf_counter()
        masked_log_probabilities(self.base, tasks_seq, tasks_pos, batch_size=32)
        torch.cuda.synchronize()
        masked_seconds = time.perf_counter() - started
        return {
            "sequences": len(sequences), "embedding_seconds": embedding_seconds,
            "seconds_per_sequence": embedding_seconds / len(sequences),
            "masked_tasks": len(tasks_seq), "masked_seconds": masked_seconds,
            "seconds_per_masked_task": masked_seconds / max(len(tasks_seq), 1),
        }

    @torch.inference_mode()
    def _base_outputs(self, sequences: list[str], *, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        residues, layers = [], []
        layer = int(self.config["backbone"]["selected_layer"])
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start : start + batch_size]
            output = self.base(sequence_tokens=self.base._tokenize(batch))
            residues.append(output.embeddings[:, 1:-1].float().cpu().numpy())
            layers.append(output.hidden_states[layer, :, 1:-1].float().cpu().numpy())
        return np.concatenate(residues), np.concatenate(layers)

    def score_base_arms(
        self, candidates: dict[str, list[str]], *, batch_size: int = 8
    ) -> dict[str, dict[str, np.ndarray]]:
        """Share base ESM-C forwards across all fixed-representation ESM arms."""
        requested = {arm: set(values) for arm, values in candidates.items() if arm in BASE_ARMS}
        union = sorted(set().union(*requested.values())) if requested else []
        result: dict[str, dict[str, np.ndarray]] = {arm: {} for arm in requested}
        for start in range(0, len(union), batch_size):
            names = union[start : start + batch_size]
            sequences = [apply(self.wild_type, name) for name in names]
            residues, _ = self._base_outputs(sequences, batch_size=batch_size)
            common = {
                "esm_mean": esm_mean(residues),
                "esm_residue": residues,
                "esm_residue_delta": esm_residue_delta(residues, self.wild_residue),
            }
            for arm, wanted in requested.items():
                indices = [index for index, name in enumerate(names) if name in wanted]
                if not indices:
                    continue
                if arm == "esm_mutpool_ridge":
                    common["esm_mutpool"] = esm_mutpool(
                        residues, [name.replace("-", ":") for name in names]
                    )
                feature = self.scorers[arm].feature_name
                predictions = self.scorers[arm].predict_prepared({feature: common[feature]})[:, indices]
                for column, index in enumerate(indices):
                    result[arm][names[index]] = predictions[:, column]
        return result

    def dpll(self, mutation_sets: list[str], *, batch_size: int = 32) -> dict[str, float]:
        """Compute exact mutated-position dPLL values, reusing WT terms by position."""
        unique = sorted(set(mutation_sets))
        task_sequences, task_positions, owners = [], [], []
        for owner, value in enumerate(unique):
            sequence = apply(self.wild_type, value)
            for token in parse_set(value):
                _, position, _ = parse_mutation(token)
                task_sequences.append(sequence)
                task_positions.append(position - 1)
                owners.append(owner)
        observed = masked_log_probabilities(
            self.base, task_sequences, task_positions, batch_size=batch_size
        )
        totals = np.zeros(len(unique), dtype=np.float64)
        for value, position, owner in zip(observed, task_positions, owners):
            totals[owner] += float(value - self.wild_logp[position])
        return dict(zip(unique, totals))

    def _load_phi(self) -> tuple[SequenceFitnessAdapter, CrossCoder, float, float]:
        if self._phi_components is None:
            checkpoint = REPO_ROOT / self.config["paths"]["step3_adapter"]
            adapted = _new_adapter(self.config, checkpoint, seed=20261216, device=self.device)
            artifact = torch.load(
                REPO_ROOT / self.config["paths"]["step7_crosscoder"],
                map_location="cpu", weights_only=True,
            )
            crosscoder = CrossCoder(960, 1, int(artifact["nominal_k"]), decoder_bias=bool(artifact["decoder_bias"]))
            crosscoder.load_state_dict(artifact["state_dict"])
            crosscoder.to(self.device).eval()
            manifest = load_config(REPO_ROOT / self.config["paths"]["step4_manifest"])
            self._phi_components = (
                adapted, crosscoder,
                float(manifest["models"]["evo"]["normalization_scalar"]),
                float(manifest["models"]["fit"]["normalization_scalar"]),
            )
        return self._phi_components

    @torch.inference_mode()
    def score_phi(self, mutation_sets: list[str], *, batch_size: int = 8) -> dict[str, np.ndarray]:
        """Score the selected max-pooled w=1 crosscoder arm."""
        adapted, crosscoder, evo_scalar, fit_scalar = self._load_phi()
        scorer = GenerationScorer("phi_w1_max_all_live", self.config, device=self.device)
        result = {}
        layer = int(self.config["backbone"]["selected_layer"])
        values = sorted(set(mutation_sets))
        for start in range(0, len(values), batch_size):
            names = values[start : start + batch_size]
            sequences = [apply(self.wild_type, name) for name in names]
            _, base_layer = self._base_outputs(sequences, batch_size=batch_size)
            fit_rows = []
            for offset in range(0, len(sequences), batch_size):
                batch = sequences[offset : offset + batch_size]
                output = adapted.backbone(sequence_tokens=adapted.backbone._tokenize(batch))
                fit_rows.append(output.hidden_states[layer, :, 1:-1].float().cpu().numpy())
            fit_layer = np.concatenate(fit_rows)
            left = torch.as_tensor(base_layer * evo_scalar, device=self.device)
            right = torch.as_tensor(fit_layer * fit_scalar, device=self.device)
            code = crosscoder.encode(left.flatten(0, 1), right.flatten(0, 1))
            pooled = pool_codes(code.reshape(len(names), len(self.wild_type), -1).cpu().numpy(), "max")
            predictions = scorer.predict_prepared({"phi": pooled})
            for column, name in enumerate(names):
                result[name] = predictions[:, column]
        return result

    def _load_generation_adapter(self) -> SequenceFitnessAdapter:
        if self._adapter is None:
            checkpoint = REPO_ROOT / self.config["paths"]["models"] / "adapter_linear/seed_0.pt"
            self._adapter = _new_adapter(self.config, checkpoint, seed=0, device=self.device)
        return self._adapter

    def score_adapter(self, mutation_sets: list[str], *, batch_size: int = 8) -> dict[str, np.ndarray]:
        """Score search candidates with the seed-0 all-campaign adapter."""
        model = self._load_generation_adapter()
        values = sorted(set(mutation_sets))
        predictions = predict_adapter(
            model, [apply(self.wild_type, name) for name in values],
            backbone_tokenize=model.backbone._tokenize, batch_size=batch_size, device=self.device,
        )
        return {name: np.repeat(float(value), 5) for name, value in zip(values, predictions)}

