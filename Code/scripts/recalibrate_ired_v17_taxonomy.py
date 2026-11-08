#!/usr/bin/env python3
"""Recompute the five-seed IRED taxonomy from one pooled drift-matched null."""
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
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossfit.activation_cache import load_cached_model
from crossfit.crosscoder import CrossCoder
from crossfit.taxonomy import (
    FeatureStatistics,
    calibrate_pooled_beta_tail,
    calibrate_pooled_empirical_tail,
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


def _frame(statistics: FeatureStatistics, width: int, seed: int) -> pd.DataFrame:
    return pd.DataFrame({
        "dataset": "IRED", "width_multiplier": width, "seed": seed,
        "feature": np.arange(len(statistics.delta)), "delta": statistics.delta,
        "evo_decoder_norm": statistics.evo_decoder_norm,
        "fit_decoder_norm": statistics.fit_decoder_norm,
        "decoder_cosine": statistics.decoder_cosine,
        "activation_count": statistics.activation_count,
        "activation_frequency": statistics.activation_frequency,
        "self_pair_activation_frequency": statistics.self_pair_activation_frequency,
        "activation_frequency_change": statistics.activation_frequency_change,
        "error_ratio": statistics.error_ratio,
        "reconstruction_ratio": statistics.reconstruction_ratio,
    })


def _subset(
    values: np.ndarray,
    candidate: np.ndarray,
    nulls: list[np.ndarray],
    *,
    kind: str,
    tail: str,
    q: float,
) -> tuple[np.ndarray, float, dict[str, object]]:
    output = np.zeros(len(values), dtype=bool)
    calibrator = (
        calibrate_pooled_beta_tail if kind == "beta" else calibrate_pooled_empirical_tail
    )
    passed, cutoff, report = calibrator(values[candidate], nulls, tail=tail, q=q)
    output[candidate] = passed
    return output, cutoff, report


def _tau_uncertainty(lower: dict[str, object], upper: dict[str, object]) -> list[dict[str, float]]:
    lower_rows = lower["leave_one_null_run_out"]
    upper_rows = upper["leave_one_null_run_out"]
    rows = []
    for left, right in zip(lower_rows, upper_rows):
        if left["omitted_null_run"] != right["omitted_null_run"]:
            raise AssertionError("lower and upper leave-one-out runs are misaligned")
        rows.append({
            "omitted_null_run": int(left["omitted_null_run"]),
            "lower_cutoff": float(left["cutoff"]),
            "upper_cutoff": float(right["cutoff"]),
            "symmetric_tau": float(min(left["cutoff"], 1.0 - right["cutoff"])),
        })
    return rows


def _qq_plot(nulls: list[np.ndarray], path: Path, title: str) -> tuple[float, float]:
    pooled = np.concatenate([np.asarray(values)[np.isfinite(values)] for values in nulls])
    parameters = fit_beta_null(pooled)
    figure, axis = plt.subplots(figsize=(5.5, 5.0))
    for index, values in enumerate(nulls):
        observed = np.sort(np.asarray(values)[np.isfinite(values)])
        probabilities = (np.arange(len(observed)) + 0.5) / len(observed)
        expected = beta_distribution.ppf(probabilities, *parameters)
        axis.plot(expected, observed, ".", ms=2, alpha=0.55, label=f"null seed {index + 1}")
    limits = [
        min(axis.get_xlim()[0], axis.get_ylim()[0]),
        max(axis.get_xlim()[1], axis.get_ylim()[1]),
    ]
    axis.plot(limits, limits, color="black", lw=1)
    axis.set(xlabel="Pooled-Beta quantile", ylabel="Observed statistic", title=title)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return parameters


def _distribution(values: np.ndarray, width: int, source: str, seed: int) -> dict[str, object]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    quantiles = np.quantile(finite, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "width_multiplier": width, "source": source, "seed": seed, "n": len(finite),
        "mean": finite.mean(), "standard_deviation": finite.std(ddof=1),
        **dict(zip(("q01", "q05", "q25", "q50", "q75", "q95", "q99"), quantiles)),
    }


def _rotation_stage(
    statistics: FeatureStatistics,
    lower: np.ndarray,
    upper: np.ndarray,
    error: np.ndarray,
    reconstruction: np.ndarray,
    floor_fraction: float,
    activation_floor: int,
) -> np.ndarray:
    live = statistics.activation_count >= activation_floor
    evo_clear = statistics.evo_decoder_norm >= floor_fraction * np.median(statistics.evo_decoder_norm)
    fit_clear = statistics.fit_decoder_norm >= floor_fraction * np.median(statistics.fit_decoder_norm)
    dead = ~live | (~evo_clear & ~fit_clear)
    emergent = ~dead & fit_clear & upper & error & reconstruction
    evolution = ~dead & ~emergent & evo_clear & lower
    second_dead = ~dead & ~emergent & ~evolution & ~(evo_clear & fit_clear)
    return ~(dead | emergent | evolution | second_dead)


def _old_v16_labels(width: int, seed: int) -> np.ndarray | None:
    path = (
        REPO_ROOT / "Reports/step6/drift_matched_rotation"
        / f"taxonomy_all_drift_matched_w{width}_seed{seed}.csv"
    )
    if not path.exists():
        return None
    return pd.read_csv(path)["matched_provisional_class"].astype(str).to_numpy()


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("five-seed taxonomy statistics require the GPU pod")
    started = time.monotonic()
    step6 = load_config(REPO_ROOT / "Code/config/step6.yaml")
    step7 = load_config(REPO_ROOT / "Code/config/step7.yaml")
    q = float(step6["calibration"]["fdr_q"])
    widths = list(map(int, step7["diagnostics"]["selected_working_widths"]))
    real_seeds = list(map(int, step7["training"]["seeds"]))
    null_seeds = list(map(int, step6["calibration"]["shuffled_crosscoder_seeds"]))
    floor_fraction = float(step7["taxonomy"]["decoder_norm_floor_fraction"])
    activation_floor = int(step7["taxonomy"]["activation_floor_count"])
    comfortable_fraction = float(step6["calibration"]["drift_matched"]["comfortable_norm_fraction"])
    report = REPO_ROOT / step6["paths"]["v17_pooled_report_dir"]
    report.mkdir(parents=True, exist_ok=True)

    evo = load_cached_model(REPO_ROOT / step6["paths"]["step4_cache"], "evo")
    fit = load_cached_model(REPO_ROOT / step6["paths"]["step4_cache"], "fit")
    matched = load_cached_model(REPO_ROOT / step6["paths"]["drift_matched_cache"], "drift_matched")
    if evo.shape != fit.shape or evo.shape != matched.shape:
        raise AssertionError("real and matched-drift activation pools are not aligned")

    class_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    effect_rows: list[dict[str, object]] = []
    comfortable_rows: list[dict[str, object]] = []
    null_hashes: dict[str, str] = {}
    real_hashes: dict[str, str] = {}
    for width in widths:
        null_statistics = []
        for seed in null_seeds:
            path = (
                REPO_ROOT / step6["paths"]["drift_matched_crosscoder_checkpoints"]
                / f"w{width}_seed{seed}_nobias.pt"
            )
            null_hashes[str(path.relative_to(REPO_ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
            model = _model(path, int(step6["dimension"]))
            null_statistics.append(feature_statistics(
                model, evo, matched,
                batch_size=int(step7["training"]["evaluation_batch_size"]), device="cuda",
            ))
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
        beta_parameters = {}
        for statistic in ("delta", "cosine", "frequency"):
            beta_parameters[statistic] = _qq_plot(
                nulls[statistic], report / f"w{width}_{statistic}_pooled_beta_qq.png",
                f"IRED w={width} pooled matched-drift {statistic}",
            )
        for seed, values in zip(null_seeds, nulls["reconstruction"]):
            effect_rows.append(_distribution(values, width, "matched_drift_null", seed))

        for seed in real_seeds:
            checkpoint = REPO_ROOT / step7["paths"]["checkpoints"] / f"w{width}_seed{seed}_nobias.pt"
            real_hashes[str(checkpoint.relative_to(REPO_ROOT))] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            model = _model(checkpoint, int(step7["dimension"]))
            statistics = feature_statistics(
                model, evo, fit,
                batch_size=int(step7["training"]["evaluation_batch_size"]), device="cuda",
            )
            del model
            torch.cuda.empty_cache()
            frame = _frame(statistics, width, seed)
            live = statistics.activation_count >= activation_floor
            evo_clear = statistics.evo_decoder_norm >= floor_fraction * np.median(statistics.evo_decoder_norm)
            fit_clear = statistics.fit_decoder_norm >= floor_fraction * np.median(statistics.fit_decoder_norm)
            lower_candidate = live & evo_clear
            upper_candidate = live & fit_clear
            shared_candidate = live & evo_clear & fit_clear

            lower, lower_cutoff, lower_report = _subset(
                statistics.delta, lower_candidate, nulls["delta"], kind="beta", tail="lower", q=q,
            )
            upper, upper_cutoff, upper_report = _subset(
                statistics.delta, upper_candidate, nulls["delta"], kind="beta", tail="upper", q=q,
            )
            tau = min(lower_cutoff, 1.0 - upper_cutoff)
            lower = lower_candidate & (statistics.delta <= tau)
            upper = upper_candidate & (statistics.delta >= 1.0 - tau)
            error, error_cutoff, error_report = _subset(
                statistics.error_ratio, upper_candidate, nulls["error"],
                kind="empirical", tail="lower", q=q,
            )
            reconstruction, reconstruction_cutoff, reconstruction_report = _subset(
                statistics.reconstruction_ratio, upper_candidate, nulls["reconstruction"],
                kind="empirical", tail="lower", q=q,
            )
            rotated, cosine_scaled, cosine_report = _subset(
                (statistics.decoder_cosine + 1.0) / 2.0, shared_candidate, nulls["cosine"],
                kind="beta", tail="lower", q=q,
            )
            frequency, frequency_cutoff, frequency_report = _subset(
                statistics.activation_frequency_change, shared_candidate, nulls["frequency"],
                kind="beta", tail="upper", q=q,
            )
            labels = provisional_classes(
                statistics, lower_delta=lower, upper_delta=upper,
                latent_error_pass=error, latent_reconstruction_pass=reconstruction,
                rotated=rotated, frequency_shifted=frequency,
                floor_fraction=floor_fraction, activation_floor_count=activation_floor,
            )
            frame["lower_delta_tail"] = lower
            frame["upper_delta_tail"] = upper
            frame["latent_error_pass"] = error
            frame["latent_reconstruction_pass"] = reconstruction
            frame["rotation_shift"] = rotated
            frame["activation_frequency_shift"] = frequency
            frame["provisional_class"] = labels
            frame.to_csv(report / f"taxonomy_v17_w{width}_seed{seed}.csv", index=False)
            for label in step7["taxonomy"]["classification_precedence"]:
                count = int((labels == label).sum())
                class_rows.append({
                    "width_multiplier": width, "seed": seed, "calibration": "v17_pooled",
                    "provisional_class": label, "count": count, "fraction": count / len(labels),
                })
            old = _old_v16_labels(width, seed)
            if old is not None:
                for label in step7["taxonomy"]["classification_precedence"]:
                    count = int((old == label).sum())
                    class_rows.append({
                        "width_multiplier": width, "seed": seed,
                        "calibration": "v16_runwise_conjunction", "provisional_class": label,
                        "count": count, "fraction": count / len(old),
                    })
            effect_rows.append(_distribution(
                statistics.reconstruction_ratio[upper_candidate], width, "real_pair", seed
            ))

            stage = _rotation_stage(
                statistics, lower, upper, error, reconstruction, floor_fraction, activation_floor
            )
            comfortable = (
                live
                & (statistics.evo_decoder_norm >= comfortable_fraction * np.median(statistics.evo_decoder_norm))
                & (statistics.fit_decoder_norm >= comfortable_fraction * np.median(statistics.fit_decoder_norm))
            )
            flags = rotated[comfortable & stage]
            frequencies = statistics.activation_frequency[comfortable & stage]
            correlation = (
                spearmanr(frequencies, flags.astype(float)).statistic
                if np.unique(flags).size > 1 else np.nan
            )
            comfortable_rows.append({
                "width_multiplier": width, "seed": seed,
                "comfortable_features": int(comfortable.sum()),
                "comfortable_features_at_rotation_stage": int((comfortable & stage).sum()),
                "rotation_flags_at_stage": int(flags.sum()),
                "rotation_flag_fraction_at_stage": float(flags.mean()),
                "classified_rotated_above_median": int(((labels == "rotated") & comfortable).sum()),
                "classified_rotated_fraction_of_comfortable": float(
                    ((labels == "rotated") & comfortable).sum() / comfortable.sum()
                ),
                "spearman_rotation_flag_vs_activation_frequency": float(correlation),
            })
            threshold_rows.append({
                "dataset": "IRED", "width_multiplier": width, "seed": seed, "fdr_q": q,
                "calibration_rule": "pooled_eligible_drift_matched_null_samples",
                "tau": float(tau), "lower_delta_cutoff": float(lower_cutoff),
                "upper_delta_cutoff": float(upper_cutoff),
                "latent_error_cutoff": float(error_cutoff),
                "latent_reconstruction_cutoff": float(reconstruction_cutoff),
                "decoder_cosine_cutoff": float(2.0 * cosine_scaled - 1.0),
                "activation_frequency_change_cutoff": float(frequency_cutoff),
                "leave_one_null_seed_out_tau": _tau_uncertainty(lower_report, upper_report),
                "lower_delta": lower_report, "upper_delta": upper_report,
                "latent_error": error_report, "latent_reconstruction": reconstruction_report,
                "decoder_cosine": cosine_report, "activation_frequency": frequency_report,
            })

    classes = pd.DataFrame(class_rows)
    classes.to_csv(report / "class_proportions_by_seed.csv", index=False)
    summary = classes.groupby(
        ["width_multiplier", "calibration", "provisional_class"], as_index=False
    ).agg(
        count_mean=("count", "mean"), count_sd=("count", "std"),
        fraction_mean=("fraction", "mean"), fraction_sd=("fraction", "std"),
    )
    summary.to_csv(report / "class_proportions_summary.csv", index=False)
    pd.DataFrame(effect_rows).to_csv(report / "reconstruction_ratio_effect_sizes.csv", index=False)
    pd.DataFrame(comfortable_rows).to_csv(report / "rotation_above_median_by_seed.csv", index=False)
    (report / "thresholds.json").write_text(json.dumps(threshold_rows, indent=2) + "\n")
    manifest = {
        "status": "complete", "plan": "v17", "dataset": "IRED",
        "null": "pooled_drift_matched_shuffled_label",
        "calibration_rule": "one pooled eligible null per width and statistic",
        "widths": widths, "real_seeds": real_seeds, "null_seeds": null_seeds,
        "real_checkpoint_sha256": real_hashes,
        "null_checkpoint_sha256": null_hashes,
        "gpu_wall_hours": (time.monotonic() - started) / 3600.0,
        "partial_results_reported": False,
    }
    (report / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(summary.to_string(index=False), flush=True)
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
