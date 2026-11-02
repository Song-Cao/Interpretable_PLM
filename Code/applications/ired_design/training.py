"""Retrain frozen model choices on the complete IRED campaign."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

from crossfit.models import DiagonalRidgeRegressor, build_model
from crossfit.representations import (
    aaindex_flat, esm_mean, esm_mutpool, esm_residue_delta, load_aaindex_table,
    onehot, onehot_positional,
)
from crossfit.training import fit_torch_fixed_epochs, seed_everything
from crossfit.utils import REPO_ROOT, load_config


TORCH_ARMS = {
    "esm_mean_ridge": ("ridge", "esm_mean"),
    "esm_mean_mlp": ("mlp", "esm_mean"),
    "esm_mutpool_ridge": ("ridge", "esm_mutpool"),
    "esm_residue_conv": ("conv_residual", "esm_residue"),
    "esm_residue_delta_conv": ("conv_residual", "esm_residue_delta"),
    "onehot_positional_conv": ("conv_residual", "onehot_positional"),
    "phi_w1_max_all_live": ("mlp", "phi"),
}
CLASSICAL_ARMS = (
    "onehot_ridge_unaugmented", "onehot_ridge_augmented",
    "aaindex_flat_ridge_unaugmented", "aaindex_flat_ridge_augmented",
)


def sha256(path: Path) -> str:
    """Hash one persisted artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def campaign(config: dict) -> tuple[pd.DataFrame, str]:
    """Load the 1,182 non-WT curated variants and their exact sequences."""
    source = pd.read_csv(REPO_ROOT / config["paths"]["campaign_source"])
    wild_rows = source.loc[source["mut"].astype(str) == "WT"]
    if len(wild_rows) != 1:
        raise AssertionError("campaign must contain exactly one wild type")
    wild_type = str(wild_rows.iloc[0]["aa_seq"])
    source = source.loc[source["mut"].astype(str) != "WT"].reset_index(drop=True)
    curated = pd.read_parquet(REPO_ROOT / config["paths"]["curated_variants"])
    if len(source) != 1182 or len(curated) != 1182:
        raise AssertionError("generation retraining requires all 1,182 curated variants")
    by_mutation = source.set_index(source["mut"].astype(str))
    frame = curated.copy()
    frame["mutant_sequence"] = [
        str(by_mutation.loc[str(value).replace(":", "-"), "aa_seq"])
        for value in frame["mutations"]
    ]
    return frame, wild_type


def _residues(config: dict, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    root = REPO_ROOT / config["paths"]["step2_representations"]
    rows = np.load(root / "variant_rows.npy")
    lookup = {int(row): index for index, row in enumerate(rows)}
    indices = np.asarray([lookup[int(row)] for row in frame["variant_row"]], dtype=int)
    values = np.asarray(np.load(root / "variant_residue.npy", mmap_mode="r")[indices], dtype=np.float32)
    wild_type = np.asarray(np.load(root / "wild_type_residue.npy"), dtype=np.float32)
    return values, wild_type


def _density(config: dict, sequences: list[str]) -> np.ndarray:
    table = pd.read_csv(REPO_ROOT / config["paths"]["step2_density"])
    lookup = dict(zip(table["aa_seq"].astype(str), table["dpll"].astype(float)))
    values = np.asarray([lookup[sequence] for sequence in sequences], dtype=np.float64)
    if not np.isfinite(values).all():
        raise AssertionError("campaign dPLL cache is incomplete")
    return values


def _features(config: dict, frame: pd.DataFrame, wild_type: str) -> dict[str, np.ndarray]:
    sequences = frame["mutant_sequence"].astype(str).tolist()
    residues, wild_residue = _residues(config, frame)
    root = REPO_ROOT / "Cache/step9/IRED/phi/w1"
    phi_rows = np.load(REPO_ROOT / "Cache/step9/IRED/phi/campaign_rows.npy")
    phi_lookup = {int(row): index for index, row in enumerate(phi_rows)}
    phi_indices = np.asarray([phi_lookup[int(row)] for row in frame["variant_row"]], dtype=int)
    taxonomy = pd.read_csv(REPO_ROOT / "Cache/step9/IRED/phi/w1/taxonomy.csv")
    live = taxonomy["provisional_class"].astype(str).to_numpy() != "dead"
    if live.shape != (960,):
        raise AssertionError("w=1 taxonomy must contain exactly 960 dictionary features")
    return {
        "onehot": onehot(sequences),
        "aaindex": aaindex_flat(
            sequences, load_aaindex_table(REPO_ROOT / config["paths"]["aaindex_table"])
        ),
        "onehot_positional": onehot_positional(sequences),
        "esm_mean": esm_mean(residues),
        "esm_mutpool": esm_mutpool(
            residues, frame["mutations"].astype(str).str.replace("-", ":", regex=False).tolist()
        ),
        "esm_residue": residues,
        "esm_residue_delta": esm_residue_delta(residues, wild_residue),
        "phi": np.asarray(
            np.load(root / "campaign_max.npy", mmap_mode="r")[phi_indices][:, live],
            dtype=np.float32,
        ),
        "density": _density(config, sequences),
    }


def _torch_parameters(values: dict) -> tuple[dict, int]:
    parameters = dict(values)
    epochs = int(parameters.pop("epochs"))
    model_parameters = {
        key: value for key, value in parameters.items()
        if key not in {"learning_rate", "weight_decay", "batch_size"}
    }
    return parameters | {"epochs": epochs}, epochs


def _save_torch(fit: Any, path: Path, *, architecture: str, parameters: dict, input_shape: tuple[int, ...], epochs: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": "ired_design_generation_torch_v1",
        "architecture": architecture,
        "state_dict": fit.model.state_dict(),
        "normalizer_mean": fit.normalizer.mean,
        "normalizer_scale": fit.normalizer.scale,
        "configuration": parameters,
        "input_shape": input_shape,
        "epochs": int(epochs),
    }, path)


def train_generation_models(
    config_path: Path, *, device: str, only_arms: set[str] | None = None
) -> dict:
    """Fit every non-adapter generation model and return a hashed manifest."""
    config = load_config(config_path)
    step2 = load_config(REPO_ROOT / config["training"]["objective_config"])
    frame, wild_type = campaign(config)
    features = _features(config, frame, wild_type)
    target = frame["y"].to_numpy(np.float32)
    censoring = frame["censored"].astype(str).to_numpy()
    selected = config["training"]["selected"]
    destination = REPO_ROOT / config["paths"]["models"]
    destination.mkdir(parents=True, exist_ok=True)
    report = REPO_ROOT / config["paths"]["reports"] / "generation_model_manifest.json"
    artifacts: list[dict] = []
    if only_arms and report.exists():
        previous = json.loads(report.read_text())
        artifacts = [row for row in previous.get("artifacts", []) if row["arm"] not in only_arms]

    for arm in CLASSICAL_ARMS:
        if only_arms is not None and arm not in only_arms:
            continue
        representation = "onehot" if arm.startswith("onehot") else "aaindex"
        augmented = arm.endswith("augmented") and not arm.endswith("unaugmented")
        settings = selected[arm]
        model = DiagonalRidgeRegressor(
            site_penalty=float(settings["site_penalty"]),
            density_penalty=float(settings["density_penalty"]),
        ).fit(features[representation], target, density=features["density"] if augmented else None)
        for seed in map(int, config["training"]["seeds"]):
            path = destination / arm / f"seed_{seed}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, coefficient=model.coefficient_, intercept=model.intercept_, configuration=json.dumps(settings), augmented=augmented)
            artifacts.append({"arm": arm, "seed": seed, "path": str(path.relative_to(REPO_ROOT)), "sha256": sha256(path)})

    rf_arm = "esm_mean_random_forest"
    rf_settings = dict(selected[rf_arm])
    for seed in map(int, config["training"]["seeds"] if only_arms is None or rf_arm in only_arms else []):
        path = destination / rf_arm / f"seed_{seed}.joblib"
        model = build_model("random_forest", features["esm_mean"].shape[1:], rf_settings | {"random_state": seed, "n_jobs": -1})
        model.fit(features["esm_mean"], target)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, path)
        artifacts.append({"arm": rf_arm, "seed": seed, "path": str(path.relative_to(REPO_ROOT)), "sha256": sha256(path)})

    for arm, (architecture, feature_name) in TORCH_ARMS.items():
        if only_arms is not None and arm not in only_arms:
            continue
        settings = dict(selected[arm])
        epochs = int(settings.pop("epochs"))
        parameters = settings
        model_parameters = {key: value for key, value in parameters.items() if key not in {"learning_rate", "weight_decay", "batch_size"}}
        x = features[feature_name]
        for seed in map(int, config["training"]["seeds"]):
            seed_everything(seed)
            fit = fit_torch_fixed_epochs(
                build_model(architecture, x.shape[1:], model_parameters),
                x, target, censoring,
                objective_config=step2["objective"], optimization_config=step2["optimization"],
                parameters=parameters, epochs=epochs, seed=seed, device=device,
            )
            path = destination / arm / f"seed_{seed}.pt"
            _save_torch(fit, path, architecture=architecture, parameters=parameters, input_shape=tuple(x.shape[1:]), epochs=epochs)
            artifacts.append({"arm": arm, "seed": seed, "path": str(path.relative_to(REPO_ROOT)), "sha256": sha256(path)})

    manifest = {
        "status": "complete", "dataset": "IRED", "training_rows": len(frame),
        "includes_wild_type": False, "model_members": len(artifacts), "artifacts": artifacts,
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
