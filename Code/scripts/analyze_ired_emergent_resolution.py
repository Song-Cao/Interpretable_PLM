#!/usr/bin/env python3
"""Audit IRED emergent-feature resolution with the expanded matched-drift null."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import beta as beta_distribution
from scipy.stats import genpareto

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossfit.activation_cache import load_cached_model
from crossfit.crosscoder import CrossCoder
from crossfit.taxonomy import (
    benjamini_hochberg,
    beta_pvalues,
    calibrate_pooled_beta_tail,
    calibrate_pooled_empirical_tail,
    empirical_pvalues,
    feature_statistics,
    fit_beta_null,
    null_eligible,
    provisional_classes,
)
from crossfit.utils import REPO_ROOT, load_config


def _model(path: Path, dimension: int) -> CrossCoder:
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    model = CrossCoder(
        dimension, int(artifact["width_multiplier"]), int(artifact["nominal_k"]),
        decoder_bias=bool(artifact["decoder_bias"]),
    )
    model.load_state_dict(artifact["state_dict"])
    return model.cuda().eval()


def _subset(
    values: np.ndarray, candidate: np.ndarray, nulls: list[np.ndarray],
    *, kind: str, tail: str, q: float,
) -> tuple[np.ndarray, float, dict[str, object]]:
    output = np.zeros(len(values), dtype=bool)
    calibrator = calibrate_pooled_beta_tail if kind == "beta" else calibrate_pooled_empirical_tail
    passed, cutoff, report = calibrator(values[candidate], nulls, tail=tail, q=q)
    output[candidate] = passed
    return output, cutoff, report


def _lower_tail_fit(null: np.ndarray, tail_fraction: float) -> tuple[float, float, float]:
    """Fit a generalized Pareto to lower-tail exceedances of an unbounded statistic."""
    values = np.asarray(null, dtype=float)
    values = values[np.isfinite(values)]
    threshold = float(np.quantile(values, tail_fraction))
    exceedance = threshold - values[values <= threshold]
    shape, _, scale = genpareto.fit(exceedance, floc=0.0)
    return threshold, float(shape), float(scale)


def _lower_tail_pvalues(
    values: np.ndarray, parameters: tuple[float, float, float], tail_fraction: float,
) -> np.ndarray:
    threshold, shape, scale = parameters
    observed = np.asarray(values, dtype=float)
    output = np.ones(len(observed), dtype=float)
    inside = observed <= threshold
    output[inside] = tail_fraction * genpareto.sf(
        threshold - observed[inside], shape, loc=0.0, scale=scale
    )
    return output


def _tail_qq_beta(null: np.ndarray, path: Path, title: str, fraction: float) -> None:
    values = np.sort(np.asarray(null, dtype=float))
    values = values[np.isfinite(values)]
    parameters = fit_beta_null(values)
    count = max(2, int(np.floor(fraction * len(values))))
    probabilities = (np.arange(count) + 0.5) / len(values)
    expected = beta_distribution.ppf(probabilities, *parameters)
    observed = values[:count]
    figure, axis = plt.subplots(figsize=(5.5, 5.0))
    axis.plot(expected, observed, ".", ms=3)
    limits = [min(expected.min(), observed.min()), max(expected.max(), observed.max())]
    axis.plot(limits, limits, color="black", lw=1)
    axis.set(xlabel="Fitted Beta lower-tail quantile", ylabel="Empirical null quantile", title=title)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _tail_qq_gpd(
    null: np.ndarray, parameters: tuple[float, float, float], path: Path,
    title: str, fraction: float,
) -> None:
    values = np.sort(np.asarray(null, dtype=float))
    values = values[np.isfinite(values)]
    count = max(2, int(np.floor(fraction * len(values))))
    probabilities = (np.arange(count) + 0.5) / len(values)
    threshold, shape, scale = parameters
    expected = threshold - genpareto.ppf(1.0 - probabilities / fraction, shape, scale=scale)
    observed = values[:count]
    figure, axis = plt.subplots(figsize=(5.5, 5.0))
    axis.plot(expected, observed, ".", ms=3)
    limits = [min(expected.min(), observed.min()), max(expected.max(), observed.max())]
    axis.plot(limits, limits, color="black", lw=1)
    axis.set(xlabel="Fitted GPD lower-tail quantile", ylabel="Empirical null quantile", title=title)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _cutoff(values: np.ndarray, rejected: np.ndarray, tail: str) -> float:
    if not rejected.any():
        return -np.inf if tail == "lower" else np.inf
    return float(np.max(values[rejected]) if tail == "lower" else np.min(values[rejected]))


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("emergent-resolution statistics require the GPU pod")
    started = time.monotonic()
    step6 = load_config(REPO_ROOT / "Code/config/step6.yaml")
    step7 = load_config(REPO_ROOT / "Code/config/step7.yaml")
    audit = step6["calibration"]["resolution_audit"]
    width = int(audit["width_multiplier"])
    null_seeds = list(map(int, audit["seeds"]))
    real_seeds = list(map(int, step7["training"]["seeds"]))
    q = float(step6["calibration"]["fdr_q"])
    tail_fraction = float(audit["tail_fraction"])
    floor_fraction = float(step7["taxonomy"]["decoder_norm_floor_fraction"])
    activation_floor = int(step7["taxonomy"]["activation_floor_count"])
    dimension = int(step6["dimension"])
    batch_size = int(step7["training"]["evaluation_batch_size"])
    report = REPO_ROOT / step6["paths"]["resolution_audit_report_dir"]
    report.mkdir(parents=True, exist_ok=True)

    evo = load_cached_model(REPO_ROOT / step6["paths"]["step4_cache"], "evo")
    fit = load_cached_model(REPO_ROOT / step6["paths"]["step4_cache"], "fit")
    matched = load_cached_model(REPO_ROOT / step6["paths"]["drift_matched_cache"], "drift_matched")
    if not (evo.shape == fit.shape == matched.shape):
        raise AssertionError("real and matched-drift activation pools are not aligned")

    null_statistics = []
    checkpoint_hashes = {}
    for seed in null_seeds:
        path = (
            REPO_ROOT / step6["paths"]["drift_matched_crosscoder_checkpoints"]
            / f"w{width}_seed{seed}_nobias.pt"
        )
        checkpoint_hashes[str(path.relative_to(REPO_ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        model = _model(path, dimension)
        null_statistics.append(feature_statistics(model, evo, matched, batch_size=batch_size, device="cuda"))
        del model
        torch.cuda.empty_cache()
    eligible = [null_eligible(values, floor_fraction) for values in null_statistics]
    nulls = {
        "delta": [values.delta[mask] for values, mask in zip(null_statistics, eligible)],
        "error": [values.error_ratio[mask] for values, mask in zip(null_statistics, eligible)],
        "reconstruction": [values.reconstruction_ratio[mask] for values, mask in zip(null_statistics, eligible)],
        "cosine": [(values.decoder_cosine[mask] + 1.0) / 2.0 for values, mask in zip(null_statistics, eligible)],
        "frequency": [values.activation_frequency_change[mask] for values, mask in zip(null_statistics, eligible)],
    }
    pooled = {name: np.concatenate(runs) for name, runs in nulls.items()}
    pooled = {name: values[np.isfinite(values)] for name, values in pooled.items()}
    if len(pooled["reconstruction"]) < 19000:
        raise RuntimeError("eligible pooled null still cannot resolve the first 960-test BH cutoff")

    _tail_qq_beta(
        pooled["delta"], report / "delta_beta_lower_tail_qq.png",
        "IRED matched-drift Delta null: lower 5%", tail_fraction,
    )
    reconstruction_fit = _lower_tail_fit(pooled["reconstruction"], tail_fraction)
    error_fit = _lower_tail_fit(pooled["error"], tail_fraction)
    _tail_qq_gpd(
        pooled["reconstruction"], reconstruction_fit,
        report / "reconstruction_ratio_gpd_lower_tail_qq.png",
        "IRED matched-drift reconstruction ratio: lower 5%", tail_fraction,
    )

    class_rows = []
    threshold_rows = []
    extreme_rows = []
    taxonomy_paths = []
    for seed in real_seeds:
        checkpoint = REPO_ROOT / step7["paths"]["checkpoints"] / f"w{width}_seed{seed}_nobias.pt"
        model = _model(checkpoint, dimension)
        stats = feature_statistics(model, evo, fit, batch_size=batch_size, device="cuda")
        del model
        torch.cuda.empty_cache()
        live = stats.activation_count >= activation_floor
        evo_clear = stats.evo_decoder_norm >= floor_fraction * np.median(stats.evo_decoder_norm)
        fit_clear = stats.fit_decoder_norm >= floor_fraction * np.median(stats.fit_decoder_norm)
        lower_candidate = live & evo_clear
        upper_candidate = live & fit_clear
        shared_candidate = live & evo_clear & fit_clear

        lower, lower_cutoff, lower_report = _subset(
            stats.delta, lower_candidate, nulls["delta"], kind="beta", tail="lower", q=q,
        )
        upper, upper_cutoff, upper_report = _subset(
            stats.delta, upper_candidate, nulls["delta"], kind="beta", tail="upper", q=q,
        )
        tau = min(lower_cutoff, 1.0 - upper_cutoff)
        lower = lower_candidate & (stats.delta <= tau)
        upper = upper_candidate & (stats.delta >= 1.0 - tau)
        error, error_cutoff, error_report = _subset(
            stats.error_ratio, upper_candidate, nulls["error"], kind="empirical", tail="lower", q=q,
        )
        reconstruction, reconstruction_cutoff, reconstruction_report = _subset(
            stats.reconstruction_ratio, upper_candidate, nulls["reconstruction"],
            kind="empirical", tail="lower", q=q,
        )
        rotated, cosine_cutoff, cosine_report = _subset(
            (stats.decoder_cosine + 1.0) / 2.0, shared_candidate, nulls["cosine"],
            kind="beta", tail="lower", q=q,
        )
        frequency, frequency_cutoff, frequency_report = _subset(
            stats.activation_frequency_change, shared_candidate, nulls["frequency"],
            kind="beta", tail="upper", q=q,
        )
        labels = provisional_classes(
            stats, lower_delta=lower, upper_delta=upper,
            latent_error_pass=error, latent_reconstruction_pass=reconstruction,
            rotated=rotated, frequency_shifted=frequency,
            floor_fraction=floor_fraction, activation_floor_count=activation_floor,
        )

        error_fitted_p = _lower_tail_pvalues(stats.error_ratio[upper_candidate], error_fit, tail_fraction)
        reconstruction_fitted_p = _lower_tail_pvalues(
            stats.reconstruction_ratio[upper_candidate], reconstruction_fit, tail_fraction
        )
        error_fitted = np.zeros(len(stats.delta), dtype=bool)
        reconstruction_fitted = np.zeros(len(stats.delta), dtype=bool)
        error_fitted[upper_candidate] = benjamini_hochberg(error_fitted_p, q)
        reconstruction_fitted[upper_candidate] = benjamini_hochberg(reconstruction_fitted_p, q)
        fitted_labels = provisional_classes(
            stats, lower_delta=lower, upper_delta=upper,
            latent_error_pass=error_fitted, latent_reconstruction_pass=reconstruction_fitted,
            rotated=rotated, frequency_shifted=frequency,
            floor_fraction=floor_fraction, activation_floor_count=activation_floor,
        )

        frame = pd.DataFrame({
            "dataset": "IRED", "width_multiplier": width, "seed": seed,
            "feature": np.arange(len(stats.delta)), "delta": stats.delta,
            "evo_decoder_norm": stats.evo_decoder_norm, "fit_decoder_norm": stats.fit_decoder_norm,
            "decoder_cosine": stats.decoder_cosine, "activation_count": stats.activation_count,
            "activation_frequency": stats.activation_frequency,
            "activation_frequency_change": stats.activation_frequency_change,
            "error_ratio": stats.error_ratio, "reconstruction_ratio": stats.reconstruction_ratio,
            "lower_delta_tail": lower, "upper_delta_tail": upper,
            "latent_error_empirical_pass": error,
            "latent_reconstruction_empirical_pass": reconstruction,
            "rotation_shift": rotated, "activation_frequency_shift": frequency,
            "provisional_class": labels,
            "gpd_tail_sensitivity_class": fitted_labels,
        })
        path = report / f"taxonomy_resolution_w{width}_seed{seed}.csv"
        frame.to_csv(path, index=False)
        taxonomy_paths.append(str(path.relative_to(REPO_ROOT)))
        for calibration, values in (("empirical_registered", labels), ("gpd_tail_sensitivity", fitted_labels)):
            for label in step7["taxonomy"]["classification_precedence"]:
                count = int(np.sum(values == label))
                class_rows.append({
                    "width_multiplier": width, "seed": seed, "calibration": calibration,
                    "provisional_class": label, "count": count, "fraction": count / len(values),
                })

        candidate_indices = np.flatnonzero(upper_candidate)
        order = candidate_indices[np.argsort(stats.reconstruction_ratio[candidate_indices])[:10]]
        empirical = empirical_pvalues(stats.reconstruction_ratio[order], pooled["reconstruction"], "lower")
        fitted = _lower_tail_pvalues(stats.reconstruction_ratio[order], reconstruction_fit, tail_fraction)
        for rank, (feature, empirical_p, fitted_p) in enumerate(zip(order, empirical, fitted), start=1):
            raw = float(stats.reconstruction_ratio[feature])
            extreme_rows.append({
                "seed": seed, "extremeness_rank": rank, "feature": int(feature),
                "reconstruction_ratio": raw,
                "pooled_null_values_le_statistic": int(np.sum(pooled["reconstruction"] <= raw)),
                "pooled_null_observations": int(len(pooled["reconstruction"])),
                "empirical_p_value": float(empirical_p),
                "fitted_gpd_lower_tail_p_value": float(fitted_p),
                "absolute_log10_p_difference": float(abs(np.log10(empirical_p) - np.log10(fitted_p))),
            })
        threshold_rows.append({
            "seed": seed, "tau": float(tau), "lower_delta_cutoff": float(lower_cutoff),
            "upper_delta_cutoff": float(upper_cutoff), "latent_error_cutoff": float(error_cutoff),
            "latent_reconstruction_cutoff": float(reconstruction_cutoff),
            "decoder_cosine_cutoff": float(2.0 * cosine_cutoff - 1.0),
            "activation_frequency_cutoff": float(frequency_cutoff),
            "lower_delta_report": lower_report, "upper_delta_report": upper_report,
            "error_report": error_report, "reconstruction_report": reconstruction_report,
            "cosine_report": cosine_report, "frequency_report": frequency_report,
        })

    classes = pd.DataFrame(class_rows)
    classes.to_csv(report / "class_proportions_by_seed.csv", index=False)
    summary = classes.groupby(["calibration", "provisional_class"], as_index=False).agg(
        count_mean=("count", "mean"), count_sd=("count", "std"),
        fraction_mean=("fraction", "mean"), fraction_sd=("fraction", "std"),
    )
    summary.to_csv(report / "class_proportions_summary.csv", index=False)
    pd.DataFrame(extreme_rows).to_csv(report / "most_extreme_reconstruction_ratios.csv", index=False)
    (report / "thresholds.json").write_text(json.dumps(threshold_rows, indent=2) + "\n")
    manifest = {
        "status": "complete", "dataset": "IRED", "width_multiplier": width,
        "registered_headline_calibration": "pooled empirical latent-scaling ratios",
        "fitted_tail_diagnostic": "generalized Pareto below the empirical fifth percentile; Beta is invalid because reconstruction ratios are not bounded to [0,1]",
        "null_seeds": null_seeds, "real_seeds": real_seeds,
        "eligible_null_features_by_seed": [int(mask.sum()) for mask in eligible],
        "eligible_null_features_total": int(len(pooled["reconstruction"])),
        "minimum_empirical_p_value": float(1.0 / (len(pooled["reconstruction"]) + 1.0)),
        "first_bh_cutoff_for_960_tests": float(q / 960.0),
        "reconstruction_gpd": {
            "tail_fraction": tail_fraction, "threshold": reconstruction_fit[0],
            "shape": reconstruction_fit[1], "scale": reconstruction_fit[2],
        },
        "taxonomy_paths": taxonomy_paths, "null_checkpoint_sha256": checkpoint_hashes,
        "analysis_gpu_wall_hours": (time.monotonic() - started) / 3600.0,
        "partial_results_reported": False,
    }
    (report / "resolution_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
