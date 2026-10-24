"""Prediction and checkpoint persistence shared by fitted models."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


PREDICTION_COLUMNS = (
    "dataset",
    "step",
    "arm",
    "config_id",
    "fold",
    "seed",
    "condition",
    "variant_row",
    "target",
    "prediction",
)


def prediction_frame(
    *,
    dataset: str,
    step: int,
    arm: str,
    config_id: str | int,
    fold: str | int,
    seed: str | int,
    condition: str,
    variant_rows: Sequence[int],
    targets: Sequence[float],
    predictions: Sequence[float],
    uncertainty: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Build validated prediction rows before any metric reduction."""
    rows = np.asarray(variant_rows)
    target = np.asarray(targets, dtype=np.float64)
    prediction = np.asarray(predictions, dtype=np.float64)
    if rows.ndim != 1 or target.ndim != 1 or prediction.ndim != 1:
        raise ValueError("prediction inputs must be one-dimensional")
    if not (len(rows) == len(target) == len(prediction)):
        raise ValueError("variant rows, targets, and predictions must have equal length")
    if len(np.unique(rows)) != len(rows):
        raise ValueError("one fitted model must have one prediction per held-out variant")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("targets and predictions must be finite")
    frame = pd.DataFrame(
        {
            "dataset": dataset,
            "step": int(step),
            "arm": arm,
            "config_id": str(config_id),
            "fold": fold,
            "seed": seed,
            "condition": condition,
            "variant_row": rows.astype(np.int64),
            "target": target,
            "prediction": prediction,
        }
    )
    if uncertainty is not None:
        values = np.asarray(uncertainty, dtype=np.float64)
        if values.ndim != 1 or len(values) != len(frame):
            raise ValueError("uncertainty must have one value per prediction")
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("uncertainty must be finite and non-negative")
        frame["uncertainty"] = values
    return frame


def validate_prediction_frame(frame: pd.DataFrame) -> None:
    """Validate the common schema and one-row-per-model-held-out-variant rule."""
    missing = [column for column in PREDICTION_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"prediction table is missing columns: {missing}")
    keys = ["dataset", "step", "arm", "config_id", "fold", "seed", "condition", "variant_row"]
    if frame[keys].duplicated().any():
        raise ValueError("prediction table contains duplicate fitted-model variant rows")
    if frame.empty:
        raise ValueError("prediction table is empty")
    if not np.isfinite(frame["target"].to_numpy(float)).all():
        raise ValueError("prediction targets must be finite")
    if not np.isfinite(frame["prediction"].to_numpy(float)).all():
        raise ValueError("predictions must be finite")
    if "uncertainty" in frame and frame["uncertainty"].notna().any():
        values = frame.loc[frame["uncertainty"].notna(), "uncertainty"].to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("reported uncertainties must be finite and non-negative")


def _atomic_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.with_name(path.name + ".tmp")


def write_prediction_csv(frame: pd.DataFrame, path: str | Path) -> None:
    """Atomically write a validated committed prediction table."""
    validate_prediction_frame(frame)
    destination = Path(path)
    temporary = _atomic_path(destination)
    frame.to_csv(temporary, index=False)
    os.replace(temporary, destination)


def write_prediction_parquet(frame: pd.DataFrame, path: str | Path) -> None:
    """Atomically write a validated pod-only prediction partition."""
    validate_prediction_frame(frame)
    destination = Path(path)
    temporary = _atomic_path(destination)
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, destination)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of one persisted artifact."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def upsert_checkpoint_manifest(
    path: str | Path,
    *,
    dataset: str,
    step: int,
    arm: str,
    config_id: str | int,
    fold: str | int,
    seed: str | int,
    condition: str,
    checkpoint: str | Path,
) -> None:
    """Atomically record a checkpoint path, size, and content hash."""
    destination = Path(path)
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    record = {
        "dataset": dataset,
        "step": int(step),
        "arm": arm,
        "config_id": str(config_id),
        "fold": fold,
        "seed": seed,
        "condition": condition,
        "checkpoint": str(checkpoint_path),
        "bytes": checkpoint_path.stat().st_size,
        "sha256": sha256_file(checkpoint_path),
    }
    keys = ["dataset", "step", "arm", "config_id", "fold", "seed", "condition"]
    if destination.exists():
        records = json.loads(destination.read_text())
        if not isinstance(records, list):
            raise ValueError(f"checkpoint manifest is not a list: {destination}")
    else:
        records = []
    records = [item for item in records if any(item[key] != record[key] for key in keys)]
    records.append(record)
    records.sort(key=lambda item: tuple(str(item[key]) for key in keys))
    temporary = _atomic_path(destination)
    temporary.write_text(json.dumps(records, indent=2) + "\n")
    os.replace(temporary, destination)
