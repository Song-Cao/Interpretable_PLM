"""Load isolated generation checkpoints and score prepared representations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch

from crossfit.models import build_model
from crossfit.representations import (
    aaindex_flat, load_aaindex_table, onehot, onehot_positional,
)
from crossfit.training import FeatureNormalizer, predict_torch
from crossfit.utils import REPO_ROOT


FEATURE_BY_ARM = {
    "onehot_ridge_unaugmented": "onehot",
    "onehot_ridge_augmented": "onehot",
    "aaindex_flat_ridge_unaugmented": "aaindex",
    "aaindex_flat_ridge_augmented": "aaindex",
    "esm_mean_ridge": "esm_mean",
    "esm_mean_mlp": "esm_mean",
    "esm_mean_random_forest": "esm_mean",
    "esm_mutpool_ridge": "esm_mutpool",
    "esm_residue_conv": "esm_residue",
    "esm_residue_delta_conv": "esm_residue_delta",
    "onehot_positional_conv": "onehot_positional",
    "phi_w1_max_all_live": "phi",
}


@dataclass
class _TorchMember:
    model: torch.nn.Module
    normalizer: FeatureNormalizer


class GenerationScorer:
    """Five-member predictor using only application checkpoints."""

    def __init__(self, arm: str, config: dict, *, device: str) -> None:
        if arm not in FEATURE_BY_ARM:
            raise KeyError(f"unknown generation arm: {arm}")
        self.arm = arm
        self.config = config
        self.device = device
        self.feature_name = FEATURE_BY_ARM[arm]
        self.members = self._load_members()

    def _load_members(self) -> list[object]:
        root = REPO_ROOT / self.config["paths"]["models"] / self.arm
        output: list[object] = []
        for seed in map(int, self.config["training"]["seeds"]):
            npz = root / f"seed_{seed}.npz"
            joblib_path = root / f"seed_{seed}.joblib"
            torch_path = root / f"seed_{seed}.pt"
            if npz.exists():
                artifact = np.load(npz, allow_pickle=False)
                output.append((artifact["coefficient"], float(artifact["intercept"]), bool(artifact["augmented"])))
            elif joblib_path.exists():
                output.append(joblib.load(joblib_path))
            elif torch_path.exists():
                artifact = torch.load(torch_path, map_location="cpu", weights_only=False)
                parameters = dict(artifact["configuration"])
                model_parameters = {
                    key: value for key, value in parameters.items()
                    if key not in {"learning_rate", "weight_decay", "batch_size"}
                }
                model = build_model(
                    str(artifact["architecture"]), tuple(artifact["input_shape"]), model_parameters
                )
                model.load_state_dict(artifact["state_dict"])
                model.to(self.device).eval()
                output.append(_TorchMember(
                    model=model,
                    normalizer=FeatureNormalizer(
                        mean=np.asarray(artifact["normalizer_mean"]),
                        scale=np.asarray(artifact["normalizer_scale"]),
                    ),
                ))
            else:
                raise FileNotFoundError(f"missing application checkpoint for {self.arm} seed {seed}")
        if len(output) != 5:
            raise AssertionError(f"{self.arm} requires five generation members")
        return output

    def predict_prepared(self, features: dict[str, np.ndarray]) -> np.ndarray:
        """Score a prepared feature mapping, returning member by candidate values."""
        matrix = np.asarray(features[self.feature_name])
        values = []
        for member in self.members:
            if isinstance(member, tuple):
                coefficient, intercept, augmented = member
                x = matrix
                if augmented:
                    x = np.column_stack((x, np.asarray(features["density"])))
                values.append(intercept + x @ coefficient)
            elif isinstance(member, _TorchMember):
                values.append(predict_torch(member.model, member.normalizer, matrix, device=self.device))
            else:
                values.append(member.predict(matrix))
        return np.stack(values)


class ClassicalSequenceScorer:
    """Sequence-to-score wrapper for one classical ridge arm."""

    def __init__(self, arm: str, config: dict, density_lookup: dict[str, float], *, device: str = "cpu") -> None:
        self.predictor = GenerationScorer(arm, config, device=device)
        self.config = config
        self.density_lookup = density_lookup
        self.table = load_aaindex_table(REPO_ROOT / config["paths"]["aaindex_table"])

    def predict_members(self, sequences: list[str], mutation_sets: list[str]) -> np.ndarray:
        """Encode and score a classical candidate batch."""
        if self.predictor.feature_name == "onehot":
            representation = onehot(sequences)
        else:
            representation = aaindex_flat(sequences, self.table)
        needs_density = self.predictor.arm.endswith("augmented") and not self.predictor.arm.endswith("unaugmented")
        if needs_density:
            missing = [sequence for sequence in sequences if sequence not in self.density_lookup]
            if missing:
                raise KeyError(f"dPLL missing for {len(missing)} candidate sequences")
        density = np.asarray([self.density_lookup.get(sequence, 0.0) for sequence in sequences])
        return self.predictor.predict_prepared({self.predictor.feature_name: representation, "density": density})

