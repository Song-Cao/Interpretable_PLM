"""Residue-level sparse-code pooling for downstream predictors."""
from __future__ import annotations

import numpy as np


def pool_codes(codes: np.ndarray, operator: str) -> np.ndarray:
    """Pool an N by L by M sparse-code tensor over residues."""
    values = np.asarray(codes)
    if values.ndim != 3:
        raise ValueError("codes must have shape variants by residues by features")
    if operator == "sum":
        return values.sum(axis=1)
    if operator == "max":
        return values.max(axis=1)
    if operator == "binarized":
        return (values > 0).any(axis=1).astype(np.float32)
    raise KeyError(f"unknown pooling operator: {operator}")


def select_feature_set(labels: np.ndarray, feature_set: str, config: dict) -> np.ndarray:
    """Return the registered feature mask for one provisional taxonomy."""
    labels = np.asarray(labels).astype(str)
    if feature_set == "all_live":
        return labels != "dead"
    if feature_set == "objective_shifted":
        return np.isin(labels, config["objective_shifted_classes"])
    if feature_set == "shifted_plus_conserved":
        return np.isin(labels, config["shifted_plus_conserved_classes"])
    raise KeyError(f"unknown feature set: {feature_set}")
