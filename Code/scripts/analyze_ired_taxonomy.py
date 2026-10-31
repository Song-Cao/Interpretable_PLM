#!/usr/bin/env python3
"""Calibrate IRED null thresholds and write the provisional w=1 and w=2 taxonomy."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import beta as beta_distribution

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossfit.activation_cache import load_cached_model, load_cached_pair
from crossfit.crosscoder import CrossCoder
from crossfit.taxonomy import (
    calibrate_beta_tail,
    calibrate_empirical_tail,
    feature_statistics,
    fit_beta_null,
    null_eligible,
    provisional_classes,
)
from crossfit.utils import REPO_ROOT, load_config


def _model(path: Path, dimension: int, device: str) -> tuple[CrossCoder, dict]:
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    model = CrossCoder(
        dimension, int(artifact["width_multiplier"]), int(artifact["nominal_k"]),
        decoder_bias=bool(artifact["decoder_bias"]),
    )
    model.load_state_dict(artifact["state_dict"])
    return model.to(device).eval(), artifact


def _qq_plot(nulls: list[np.ndarray], path: Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(5.5, 5.0))
    for index, values in enumerate(nulls):
        observed = np.sort(np.asarray(values, dtype=float))
        parameters = fit_beta_null(observed)
        probabilities = (np.arange(len(observed)) + 0.5) / len(observed)
        expected = beta_distribution.ppf(probabilities, *parameters)
        axis.plot(expected, observed, ".", ms=2, alpha=0.65, label=f"null seed {index + 1}")
    limits = axis.get_xlim()
    lower = min(limits[0], axis.get_ylim()[0])
    upper = max(limits[1], axis.get_ylim()[1])
    axis.plot([lower, upper], [lower, upper], color="black", lw=1)
    axis.set(xlabel="Fitted Beta quantile", ylabel="Observed Delta", title=title)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _subset_calibration(
    values: np.ndarray,
    candidate: np.ndarray,
    nulls: list[np.ndarray],
    *,
    kind: str,
    tail: str,
    q: float,
) -> tuple[np.ndarray, float, list[dict[str, float]]]:
    passed = np.zeros(len(values), dtype=bool)
    calibrate = calibrate_beta_tail if kind == "beta" else calibrate_empirical_tail
    subset_passed, cutoff, reports = calibrate(values[candidate], nulls, tail=tail, q=q)
    passed[candidate] = subset_passed
    return passed, cutoff, reports


def _leave_one_out_tau(lower_reports: list[dict], upper_reports: list[dict]) -> list[dict]:
    rows = []
    for omitted in range(len(lower_reports)):
        lower = [r["cutoff"] for i, r in enumerate(lower_reports) if i != omitted]
        upper = [r["cutoff"] for i, r in enumerate(upper_reports) if i != omitted]
        lower_cutoff = min(lower) if lower else -np.inf
        upper_cutoff = max(upper) if upper else np.inf
        tau = min(lower_cutoff, 1.0 - upper_cutoff)
        rows.append({"omitted_null_run": omitted, "tau": float(tau)})
    return rows


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("taxonomy statistics require the GPU pod")
    step6 = load_config(REPO_ROOT / "Code/config/step6.yaml")
    step7 = load_config(REPO_ROOT / "Code/config/step7.yaml")
    q = float(step6["calibration"]["fdr_q"])
    floor_fraction = float(step7["taxonomy"]["decoder_norm_floor_fraction"])
    widths = list(map(int, step7["diagnostics"]["selected_working_widths"]))
    seeds = list(map(int, step7["training"]["seeds"]))
    null_seeds = list(map(int, step6["calibration"]["shuffled_crosscoder_seeds"]))
    evo, fit = load_cached_pair(REPO_ROOT / step7["paths"]["cache"])
    shuffled = load_cached_model(REPO_ROOT / step6["paths"]["shuffled_cache"], "shuffled")
    if evo.shape != shuffled.shape:
        raise AssertionError("real and shuffled pools differ")
    report_dir = REPO_ROOT / step7["paths"]["report_dir"]
    calibration_dir = REPO_ROOT / step6["paths"]["calibration_report_dir"]
    report_dir.mkdir(parents=True, exist_ok=True)
    calibration_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    calibrations = []

    for width in widths:
        null_statistics = []
        for seed in null_seeds:
            path = REPO_ROOT / step6["paths"]["shuffled_checkpoints"] / f"w{width}_seed{seed}_nobias.pt"
            model, _ = _model(path, int(step6["dimension"]), "cuda")
            null_statistics.append(feature_statistics(
                model, evo, shuffled,
                batch_size=int(step7["training"]["evaluation_batch_size"]), device="cuda",
            ))
            del model
            torch.cuda.empty_cache()
        eligible = [null_eligible(value, floor_fraction) for value in null_statistics]
        null_delta = [value.delta[mask] for value, mask in zip(null_statistics, eligible)]
        _qq_plot(null_delta, calibration_dir / f"w{width}_delta_beta_qq.png", f"IRED w={width} shuffled-null Delta")

        for seed in seeds:
            path = REPO_ROOT / step7["paths"]["checkpoints"] / f"w{width}_seed{seed}_nobias.pt"
            model, artifact = _model(path, int(step7["dimension"]), "cuda")
            stats = feature_statistics(
                model, evo, fit,
                batch_size=int(step7["training"]["evaluation_batch_size"]), device="cuda",
            )
            del model
            torch.cuda.empty_cache()
            live = stats.activation_count >= int(step7["taxonomy"]["activation_floor_count"])
            evo_floor = floor_fraction * np.median(stats.evo_decoder_norm)
            fit_floor = floor_fraction * np.median(stats.fit_decoder_norm)
            evo_clear = stats.evo_decoder_norm >= evo_floor
            fit_clear = stats.fit_decoder_norm >= fit_floor
            lower_candidate = live & evo_clear
            upper_candidate = live & fit_clear
            shared_candidate = live & evo_clear & fit_clear

            lower_pass, lower_cutoff, lower_reports = _subset_calibration(
                stats.delta, lower_candidate, null_delta, kind="beta", tail="lower", q=q,
            )
            upper_pass, upper_cutoff, upper_reports = _subset_calibration(
                stats.delta, upper_candidate, null_delta, kind="beta", tail="upper", q=q,
            )
            tau = min(lower_cutoff, 1.0 - upper_cutoff)
            lower_pass = lower_candidate & (stats.delta <= tau)
            upper_pass = upper_candidate & (stats.delta >= 1.0 - tau)

            null_error = [value.error_ratio[mask] for value, mask in zip(null_statistics, eligible)]
            null_reconstruction = [
                value.reconstruction_ratio[mask] for value, mask in zip(null_statistics, eligible)
            ]
            error_pass, error_cutoff, error_reports = _subset_calibration(
                stats.error_ratio, upper_candidate, null_error, kind="empirical",
                tail=str(step6["calibration"]["latent_scaling_tail"]), q=q,
            )
            reconstruction_pass, reconstruction_cutoff, reconstruction_reports = _subset_calibration(
                stats.reconstruction_ratio, upper_candidate, null_reconstruction, kind="empirical",
                tail=str(step6["calibration"]["latent_scaling_tail"]), q=q,
            )
            null_cosine = [
                (value.decoder_cosine[mask] + 1.0) / 2.0
                for value, mask in zip(null_statistics, eligible)
            ]
            cosine_pass, cosine_cutoff_scaled, cosine_reports = _subset_calibration(
                (stats.decoder_cosine + 1.0) / 2.0, shared_candidate, null_cosine,
                kind="beta", tail="lower", q=q,
            )
            cosine_cutoff = 2.0 * cosine_cutoff_scaled - 1.0
            null_frequency = [
                value.activation_frequency_change[mask]
                for value, mask in zip(null_statistics, eligible)
            ]
            frequency_pass, frequency_cutoff, frequency_reports = _subset_calibration(
                stats.activation_frequency_change, shared_candidate, null_frequency,
                kind="beta", tail="upper", q=q,
            )
            labels = provisional_classes(
                stats, lower_delta=lower_pass, upper_delta=upper_pass,
                latent_error_pass=error_pass,
                latent_reconstruction_pass=reconstruction_pass,
                rotated=cosine_pass, frequency_shifted=frequency_pass,
                floor_fraction=floor_fraction,
                activation_floor_count=int(step7["taxonomy"]["activation_floor_count"]),
            )
            frame = pd.DataFrame({
                "dataset": str(step7["dataset"]), "width_multiplier": width, "seed": seed,
                "feature": np.arange(len(stats.delta)), "delta": stats.delta,
                "evo_decoder_norm": stats.evo_decoder_norm,
                "fit_decoder_norm": stats.fit_decoder_norm,
                "decoder_cosine": stats.decoder_cosine,
                "activation_count": stats.activation_count,
                "activation_frequency": stats.activation_frequency,
                "self_pair_activation_frequency": stats.self_pair_activation_frequency,
                "activation_frequency_change": stats.activation_frequency_change,
                "error_ratio": stats.error_ratio,
                "reconstruction_ratio": stats.reconstruction_ratio,
                "lower_delta_tail": lower_pass, "upper_delta_tail": upper_pass,
                "latent_error_pass": error_pass,
                "latent_reconstruction_pass": reconstruction_pass,
                "rotation_shift": cosine_pass, "activation_frequency_shift": frequency_pass,
                "provisional_class": labels,
            })
            frame.to_csv(report_dir / f"taxonomy_w{width}_seed{seed}.csv", index=False)
            counts = pd.Series(labels).value_counts()
            for label in step7["taxonomy"]["classification_precedence"]:
                summaries.append({
                    "dataset": str(step7["dataset"]), "width_multiplier": width, "seed": seed,
                    "class": label, "count": int(counts.get(label, 0)),
                    "proportion": float(counts.get(label, 0) / len(labels)),
                    "provisional": True,
                })
            calibrations.append({
                "dataset": str(step7["dataset"]), "width_multiplier": width, "seed": seed,
                "fdr_q": q, "tau": float(tau), "lower_delta_cutoff": float(lower_cutoff),
                "upper_delta_cutoff": float(upper_cutoff),
                "latent_error_cutoff": float(error_cutoff),
                "latent_reconstruction_cutoff": float(reconstruction_cutoff),
                "decoder_cosine_cutoff": float(cosine_cutoff),
                "activation_frequency_change_cutoff": float(frequency_cutoff),
                "leave_one_null_seed_out_tau": _leave_one_out_tau(lower_reports, upper_reports),
                "lower_delta_runs": lower_reports, "upper_delta_runs": upper_reports,
                "latent_error_runs": error_reports,
                "latent_reconstruction_runs": reconstruction_reports,
                "decoder_cosine_runs": cosine_reports,
                "activation_frequency_runs": frequency_reports,
                "real_checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "null_checkpoint_sha256": [
                    hashlib.sha256((REPO_ROOT / step6["paths"]["shuffled_checkpoints"] / f"w{width}_seed{value}_nobias.pt").read_bytes()).hexdigest()
                    for value in null_seeds
                ],
            })
    pd.DataFrame(summaries).to_csv(report_dir / "provisional_class_proportions_by_seed.csv", index=False)
    summary_frame = pd.DataFrame(summaries)
    aggregate = summary_frame.groupby(["dataset", "width_multiplier", "class"], as_index=False).agg(
        count_mean=("count", "mean"), count_sd=("count", "std"),
        proportion_mean=("proportion", "mean"), proportion_sd=("proportion", "std"),
    )
    aggregate["provisional"] = True
    aggregate.to_csv(report_dir / "provisional_class_proportions.csv", index=False)
    (calibration_dir / "thresholds.json").write_text(json.dumps(calibrations, indent=2) + "\n")
    print(aggregate.to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
