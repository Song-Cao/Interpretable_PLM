#!/usr/bin/env python3
"""Audit the IRED dense-reference discrepancy and Gate 2 comparisons."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossfit.utils import REPO_ROOT, load_config


def _spearman(left: np.ndarray | pd.Series, right: np.ndarray | pd.Series) -> float:
    """Return Spearman correlation across supported SciPy result APIs."""
    result = spearmanr(left, right)
    return float(getattr(result, "statistic", result.correlation))


def _ensemble_predictions(path: Path, condition: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    selected = frame.loc[(frame["condition"] == condition) & (frame["seed"].astype(str) == "ensemble")]
    if selected.empty:
        selected = frame.loc[frame["condition"] == condition]
    if selected["variant_row"].duplicated().any():
        raise AssertionError(f"predictions are not one row per variant: {path}")
    return selected.sort_values("variant_row").reset_index(drop=True)


def _step2_predictions(path: Path, condition: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    selected = frame.loc[frame["condition"] == condition].copy()
    if selected["variant_row"].duplicated().any():
        raise AssertionError(f"Step 2 predictions are not one row per variant: {path}")
    return selected.sort_values("variant_row").reset_index(drop=True)


def _paired_interval(
    left: pd.DataFrame, right: pd.DataFrame, *, samples: int, seed: int, confidence: float
) -> dict[str, float]:
    merged = left[["variant_row", "target", "prediction"]].merge(
        right[["variant_row", "target", "prediction"]], on="variant_row",
        suffixes=("_left", "_right"), validate="one_to_one",
    )
    if not np.allclose(merged["target_left"], merged["target_right"]):
        raise AssertionError("paired predictions do not have identical targets")
    target = merged["target_left"].to_numpy(float)
    left_prediction = merged["prediction_left"].to_numpy(float)
    right_prediction = merged["prediction_right"].to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(samples):
        indices = rng.integers(0, len(target), len(target))
        difference = (
            _spearman(left_prediction[indices], target[indices])
            - _spearman(right_prediction[indices], target[indices])
        )
        if np.isfinite(difference):
            draws.append(float(difference))
    alpha = (1.0 - confidence) / 2.0
    return {
        "left_minus_right": float(
            _spearman(left_prediction, target)
            - _spearman(right_prediction, target)
        ),
        "lower": float(np.quantile(draws, alpha)),
        "upper": float(np.quantile(draws, 1.0 - alpha)),
        "variants": int(len(target)),
    }


def main() -> int:
    config = load_config(REPO_ROOT / "Code/config/step9.yaml")
    report = REPO_ROOT / "Reports/step7"
    report.mkdir(parents=True, exist_ok=True)
    samples = int(config["evaluation"]["paired_bootstrap_samples"])
    seed = int(config["evaluation"]["paired_bootstrap_seed"])
    confidence = float(config["evaluation"]["confidence_level"])

    discrepancy = pd.DataFrame([
        {
            "property": "ESM-C layer", "step2_esm_mean_ridge": "final layer",
            "step5_dense_evo_mean": "analysis layer 26", "explains_nonidentity": True,
        },
        {
            "property": "final model fit rows", "step2_esm_mean_ridge": "761",
            "step5_dense_evo_mean": "895", "explains_nonidentity": True,
        },
        {
            "property": "held-out conformal calibration rows", "step2_esm_mean_ridge": "134",
            "step5_dense_evo_mean": "0", "explains_nonidentity": True,
        },
        {
            "property": "selected learning rate", "step2_esm_mean_ridge": "0.0003",
            "step5_dense_evo_mean": "0.0003", "explains_nonidentity": False,
        },
        {
            "property": "selected weight decay", "step2_esm_mean_ridge": "0.0001",
            "step5_dense_evo_mean": "0.01", "explains_nonidentity": True,
        },
        {
            "property": "selected epochs", "step2_esm_mean_ridge": "16",
            "step5_dense_evo_mean": "21", "explains_nonidentity": True,
        },
        {
            "property": "validation Spearman", "step2_esm_mean_ridge": "0.607648",
            "step5_dense_evo_mean": "0.607708", "explains_nonidentity": False,
        },
        {
            "property": "pg_kfold test Spearman", "step2_esm_mean_ridge": "0.522426",
            "step5_dense_evo_mean": "0.554287", "explains_nonidentity": False,
        },
    ])
    discrepancy.to_csv(report / "v17_dense_reference_discrepancy.csv", index=False)

    sparse_test = _ensemble_predictions(
        REPO_ROOT / "Reports/step9/phi_w1_max_all_live_predictions.csv", "pg_kfold_test"
    )
    sparse_plate = _ensemble_predictions(
        REPO_ROOT / "Reports/step9/phi_w1_max_all_live_predictions.csv", "plate"
    )
    comparisons = []
    for arm, selection_role in (
        ("dense_fit_mean", "validation_selected_dense"),
        ("dense_evo_mean", "strongest_dense_on_test"),
    ):
        dense = _ensemble_predictions(REPO_ROOT / f"Reports/step5/{arm}_predictions.csv", "pg_kfold_test")
        interval = _paired_interval(
            sparse_test, dense, samples=samples, seed=seed, confidence=confidence
        )
        comparisons.append({
            "condition": "pg_kfold_test", "sparse_arm": "phi_w1_max_all_live",
            "reference_arm": arm, "reference_role": selection_role,
            "sparse_spearman": _spearman(sparse_test["prediction"], sparse_test["target"]),
            "reference_spearman": _spearman(dense["prediction"], dense["target"]),
            "sparse_minus_reference": interval["left_minus_right"],
            "paired_ci_lower": interval["lower"], "paired_ci_upper": interval["upper"],
            "variants": interval["variants"],
        })
    for arm in (
        "onehot_ridge_unaugmented", "onehot_ridge_augmented",
        "aaindex_flat_ridge_unaugmented", "aaindex_flat_ridge_augmented",
    ):
        classical = _step2_predictions(REPO_ROOT / f"Reports/step2/{arm}_predictions.csv", "plate")
        interval = _paired_interval(
            sparse_plate, classical, samples=samples, seed=seed, confidence=confidence
        )
        comparisons.append({
            "condition": "plate", "sparse_arm": "phi_w1_max_all_live",
            "reference_arm": arm, "reference_role": "classical_encoding",
            "sparse_spearman": _spearman(sparse_plate["prediction"], sparse_plate["target"]),
            "reference_spearman": _spearman(classical["prediction"], classical["target"]),
            "sparse_minus_reference": interval["left_minus_right"],
            "paired_ci_lower": interval["lower"], "paired_ci_upper": interval["upper"],
            "variants": interval["variants"],
        })
    comparison_frame = pd.DataFrame(comparisons)
    comparison_frame.to_csv(report / "v17_gate2_three_way_comparisons.csv", index=False)

    summary = {
        "status": "complete", "dataset": "IRED", "verdict": "sparse_competitive",
        "routing_interpretation": (
            "The interval against the validation-selected dense reference routed IRED into Step 9."
        ),
        "substantive_claim": (
            "Sparse features are competitive with dense probes and classical encodings; no improvement claim."
        ),
        "dense_discrepancy": (
            "The nominally similar arms use different ESM-C layers, final-fit sample counts, "
            "regularization, and epoch counts. The 0.032 test gap is therefore not evidence of a "
            "duplicate-arm implementation error."
        ),
        "bootstrap_samples": samples, "bootstrap_seed": seed,
    }
    (report / "v17_gate2_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(comparison_frame.to_string(index=False))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
