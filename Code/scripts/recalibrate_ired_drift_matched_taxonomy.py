#!/usr/bin/env python3
"""Recalibrate every IRED taxonomy threshold against the matched-drift null."""
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
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossfit.activation_cache import load_cached_model
from crossfit.crosscoder import CrossCoder
from crossfit.taxonomy import (
    FeatureStatistics,
    calibrate_beta_tail,
    calibrate_empirical_tail,
    feature_statistics,
    fit_beta_null,
    null_eligible,
    provisional_classes,
)
from crossfit.utils import REPO_ROOT, load_config


def _statistics_from_frame(frame: pd.DataFrame) -> FeatureStatistics:
    return FeatureStatistics(
        delta=frame["delta"].to_numpy(float),
        evo_decoder_norm=frame["evo_decoder_norm"].to_numpy(float),
        fit_decoder_norm=frame["fit_decoder_norm"].to_numpy(float),
        decoder_cosine=frame["decoder_cosine"].to_numpy(float),
        activation_count=frame["activation_count"].to_numpy(float),
        activation_frequency=frame["activation_frequency"].to_numpy(float),
        self_pair_activation_frequency=frame["self_pair_activation_frequency"].to_numpy(float),
        activation_frequency_change=frame["activation_frequency_change"].to_numpy(float),
        error_ratio=frame["error_ratio"].to_numpy(float),
        reconstruction_ratio=frame["reconstruction_ratio"].to_numpy(float),
    )


def _model(path: Path, dimension: int, device: str) -> CrossCoder:
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    model = CrossCoder(
        dimension, int(artifact["width_multiplier"]), int(artifact["nominal_k"]),
        decoder_bias=bool(artifact["decoder_bias"]),
    )
    model.load_state_dict(artifact["state_dict"])
    return model.to(device).eval()


def _subset_calibration(
    values: np.ndarray,
    candidate: np.ndarray,
    nulls: list[np.ndarray],
    *,
    kind: str,
    tail: str,
    q: float,
) -> tuple[np.ndarray, float, list[dict[str, float]]]:
    result = np.zeros(len(values), dtype=bool)
    calibrate = calibrate_beta_tail if kind == "beta" else calibrate_empirical_tail
    passed, cutoff, reports = calibrate(values[candidate], nulls, tail=tail, q=q)
    result[candidate] = passed
    return result, cutoff, reports


def _leave_one_out_tau(lower: list[dict], upper: list[dict]) -> list[dict]:
    rows = []
    for omitted in range(len(lower)):
        lower_cutoffs = [row["cutoff"] for index, row in enumerate(lower) if index != omitted]
        upper_cutoffs = [row["cutoff"] for index, row in enumerate(upper) if index != omitted]
        lower_cutoff = min(lower_cutoffs) if lower_cutoffs else -np.inf
        upper_cutoff = max(upper_cutoffs) if upper_cutoffs else np.inf
        rows.append({
            "omitted_null_run": omitted,
            "tau": float(min(lower_cutoff, 1.0 - upper_cutoff)),
        })
    return rows


def _qq_plot(nulls: list[np.ndarray], path: Path, title: str) -> None:
    figure, axis = plt.subplots(figsize=(5.5, 5.0))
    for index, values in enumerate(nulls):
        observed = np.sort(np.asarray(values, dtype=float))
        parameters = fit_beta_null(observed)
        probabilities = (np.arange(len(observed)) + 0.5) / len(observed)
        expected = beta_distribution.ppf(probabilities, *parameters)
        axis.plot(expected, observed, ".", ms=2, alpha=0.65, label=f"null seed {index + 1}")
    lower = min(axis.get_xlim()[0], axis.get_ylim()[0])
    upper = max(axis.get_xlim()[1], axis.get_ylim()[1])
    axis.plot([lower, upper], [lower, upper], color="black", lw=1)
    axis.set(xlabel="Fitted Beta quantile", ylabel="Observed Delta", title=title)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _distribution_row(values: np.ndarray, *, width: int, source: str, seed: int) -> dict:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    quantiles = np.quantile(finite, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "width_multiplier": width, "source": source, "seed": seed, "n": len(finite),
        "mean": float(finite.mean()), "standard_deviation": float(finite.std(ddof=1)),
        **{
            f"q{label}": float(value)
            for label, value in zip(["01", "05", "25", "50", "75", "95", "99"], quantiles)
        },
    }


def _rotation_stage(
    statistics: FeatureStatistics,
    *,
    lower_delta: np.ndarray,
    upper_delta: np.ndarray,
    latent_error: np.ndarray,
    latent_reconstruction: np.ndarray,
    floor_fraction: float,
    activation_floor: int,
) -> np.ndarray:
    live = statistics.activation_count >= activation_floor
    evo_clear = statistics.evo_decoder_norm >= floor_fraction * np.median(
        statistics.evo_decoder_norm
    )
    fit_clear = statistics.fit_decoder_norm >= floor_fraction * np.median(
        statistics.fit_decoder_norm
    )
    dead = ~live | (~evo_clear & ~fit_clear)
    emergent = ~dead & fit_clear & upper_delta & latent_error & latent_reconstruction
    evolution = ~dead & ~emergent & evo_clear & lower_delta
    second_dead = ~dead & ~emergent & ~evolution & ~(evo_clear & fit_clear)
    return ~(dead | emergent | evolution | second_dead)


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("matched-drift taxonomy recalibration requires the GPU pod")
    step6 = load_config(REPO_ROOT / "Code/config/step6.yaml")
    step7 = load_config(REPO_ROOT / "Code/config/step7.yaml")
    matched = step6["calibration"]["drift_matched"]
    q = float(step6["calibration"]["fdr_q"])
    widths = list(map(int, step7["diagnostics"]["selected_working_widths"]))
    seeds = list(map(int, step7["training"]["seeds"]))
    null_seeds = list(map(int, step6["calibration"]["shuffled_crosscoder_seeds"]))
    floor_fraction = float(step7["taxonomy"]["decoder_norm_floor_fraction"])
    activation_floor = int(step7["taxonomy"]["activation_floor_count"])
    comfortable_fraction = float(matched["comfortable_norm_fraction"])
    shifted_classes = list(map(str, matched["shifted_classes"]))
    report = REPO_ROOT / step6["paths"]["drift_matched_report_dir"]
    report.mkdir(parents=True, exist_ok=True)

    evo = load_cached_model(REPO_ROOT / step6["paths"]["step4_cache"], "evo")
    matched_array = load_cached_model(
        REPO_ROOT / step6["paths"]["drift_matched_cache"], "drift_matched"
    )
    if evo.shape != matched_array.shape:
        raise AssertionError("matched-drift null activations do not align with evo activations")

    calibration_rows = []
    class_rows = []
    headline_rows = []
    effect_rows = []
    comfortable_rows = []
    frequency_rows = []
    for width in widths:
        null_statistics = []
        null_hashes = []
        for null_seed in null_seeds:
            path = (
                REPO_ROOT / step6["paths"]["drift_matched_crosscoder_checkpoints"]
                / f"w{width}_seed{null_seed}_nobias.pt"
            )
            model = _model(path, int(step6["dimension"]), "cuda")
            null_statistics.append(feature_statistics(
                model, evo, matched_array,
                batch_size=int(step7["training"]["evaluation_batch_size"]), device="cuda",
            ))
            null_hashes.append(hashlib.sha256(path.read_bytes()).hexdigest())
            del model
            torch.cuda.empty_cache()
        eligible = [null_eligible(value, floor_fraction) for value in null_statistics]
        null_delta = [value.delta[mask] for value, mask in zip(null_statistics, eligible)]
        null_error = [value.error_ratio[mask] for value, mask in zip(null_statistics, eligible)]
        null_reconstruction = [
            value.reconstruction_ratio[mask] for value, mask in zip(null_statistics, eligible)
        ]
        null_cosine = [
            (value.decoder_cosine[mask] + 1.0) / 2.0
            for value, mask in zip(null_statistics, eligible)
        ]
        null_frequency = [
            value.activation_frequency_change[mask]
            for value, mask in zip(null_statistics, eligible)
        ]
        _qq_plot(
            null_delta, report / f"w{width}_matched_drift_delta_beta_qq.png",
            f"IRED w={width} matched-drift null Delta",
        )
        for null_seed, values in zip(null_seeds, null_reconstruction):
            effect_rows.append(_distribution_row(
                values, width=width, source="matched_drift_null", seed=null_seed,
            ))

        for seed in seeds:
            original_path = (
                REPO_ROOT / step7["paths"]["report_dir"] / f"taxonomy_w{width}_seed{seed}.csv"
            )
            frame = pd.read_csv(original_path)
            statistics = _statistics_from_frame(frame)
            live = statistics.activation_count >= activation_floor
            evo_floor = floor_fraction * np.median(statistics.evo_decoder_norm)
            fit_floor = floor_fraction * np.median(statistics.fit_decoder_norm)
            evo_clear = statistics.evo_decoder_norm >= evo_floor
            fit_clear = statistics.fit_decoder_norm >= fit_floor
            lower_candidate = live & evo_clear
            upper_candidate = live & fit_clear
            shared_candidate = live & evo_clear & fit_clear

            lower_pass, lower_cutoff, lower_reports = _subset_calibration(
                statistics.delta, lower_candidate, null_delta,
                kind="beta", tail="lower", q=q,
            )
            upper_pass, upper_cutoff, upper_reports = _subset_calibration(
                statistics.delta, upper_candidate, null_delta,
                kind="beta", tail="upper", q=q,
            )
            tau = min(lower_cutoff, 1.0 - upper_cutoff)
            lower_pass = lower_candidate & (statistics.delta <= tau)
            upper_pass = upper_candidate & (statistics.delta >= 1.0 - tau)
            error_pass, error_cutoff, error_reports = _subset_calibration(
                statistics.error_ratio, upper_candidate, null_error,
                kind="empirical", tail=str(step6["calibration"]["latent_scaling_tail"]), q=q,
            )
            reconstruction_pass, reconstruction_cutoff, reconstruction_reports = _subset_calibration(
                statistics.reconstruction_ratio, upper_candidate, null_reconstruction,
                kind="empirical", tail=str(step6["calibration"]["latent_scaling_tail"]), q=q,
            )
            cosine_pass, cosine_cutoff_scaled, cosine_reports = _subset_calibration(
                (statistics.decoder_cosine + 1.0) / 2.0, shared_candidate, null_cosine,
                kind="beta", tail="lower", q=q,
            )
            cosine_cutoff = 2.0 * cosine_cutoff_scaled - 1.0
            frequency_pass, frequency_cutoff, frequency_reports = _subset_calibration(
                statistics.activation_frequency_change, shared_candidate, null_frequency,
                kind="beta", tail="upper", q=q,
            )
            labels = provisional_classes(
                statistics, lower_delta=lower_pass, upper_delta=upper_pass,
                latent_error_pass=error_pass,
                latent_reconstruction_pass=reconstruction_pass,
                rotated=cosine_pass, frequency_shifted=frequency_pass,
                floor_fraction=floor_fraction, activation_floor_count=activation_floor,
            )
            original = frame["provisional_class"].astype(str).to_numpy()
            output = frame.copy()
            output["matched_lower_delta_tail"] = lower_pass
            output["matched_upper_delta_tail"] = upper_pass
            output["matched_latent_error_pass"] = error_pass
            output["matched_latent_reconstruction_pass"] = reconstruction_pass
            output["matched_rotation_shift"] = cosine_pass
            output["matched_activation_frequency_shift"] = frequency_pass
            output["matched_provisional_class"] = labels
            output.to_csv(
                report / f"taxonomy_all_drift_matched_w{width}_seed{seed}.csv", index=False
            )

            for null_name, values in (("step_matched", original), ("drift_matched", labels)):
                for label in step7["taxonomy"]["classification_precedence"]:
                    count = int((values == label).sum())
                    class_rows.append({
                        "width_multiplier": width, "seed": seed, "null": null_name,
                        "provisional_class": label, "count": count,
                        "fraction": count / len(values),
                    })
            old_shifted = int(np.isin(original, shifted_classes).sum())
            new_shifted = int(np.isin(labels, shifted_classes).sum())
            headline_rows.append({
                "width_multiplier": width, "seed": seed,
                "step_matched_shifted_count": old_shifted,
                "drift_matched_shifted_count": new_shifted,
                "headline_null": "drift_matched" if new_shifted <= old_shifted else "step_matched",
                "headline_rule": str(matched["headline_rule"]),
            })
            effect_rows.append(_distribution_row(
                statistics.reconstruction_ratio[upper_candidate],
                width=width, source="real_pair", seed=seed,
            ))

            rotation_stage = _rotation_stage(
                statistics, lower_delta=lower_pass, upper_delta=upper_pass,
                latent_error=error_pass, latent_reconstruction=reconstruction_pass,
                floor_fraction=floor_fraction, activation_floor=activation_floor,
            )
            comfortable = (
                live
                & (statistics.evo_decoder_norm >= comfortable_fraction * np.median(
                    statistics.evo_decoder_norm
                ))
                & (statistics.fit_decoder_norm >= comfortable_fraction * np.median(
                    statistics.fit_decoder_norm
                ))
            )
            comfortable_stage = comfortable & rotation_stage
            comfortable_flags = cosine_pass[comfortable_stage]
            comfortable_frequency = statistics.activation_frequency[comfortable_stage]
            correlation = (
                spearmanr(comfortable_frequency, comfortable_flags.astype(float)).statistic
                if np.unique(comfortable_flags).size > 1 else float("nan")
            )
            comfortable_rows.append({
                "width_multiplier": width, "seed": seed,
                "comfortable_norm_fraction_of_median": comfortable_fraction,
                "comfortable_features": int(comfortable.sum()),
                "comfortable_features_at_rotation_stage": int(comfortable_stage.sum()),
                "rotation_flags_at_stage": int(comfortable_flags.sum()),
                "rotation_flag_fraction_at_stage": float(comfortable_flags.mean()),
                "classified_rotated_above_median": int(((labels == "rotated") & comfortable).sum()),
                "classified_rotated_fraction_of_comfortable": float(
                    ((labels == "rotated") & comfortable).sum() / comfortable.sum()
                ),
                "spearman_rotation_flag_vs_activation_frequency": float(correlation),
            })
            bins = pd.qcut(
                comfortable_frequency, q=int(matched["concentration_bins"]),
                labels=False, duplicates="drop",
            )
            for group in sorted(np.unique(bins)):
                mask = np.asarray(bins) == group
                frequency_rows.append({
                    "width_multiplier": width, "seed": seed, "quintile": int(group) + 1,
                    "features": int(mask.sum()), "rotation_flags": int(comfortable_flags[mask].sum()),
                    "rotation_fraction": float(comfortable_flags[mask].mean()),
                    "activation_frequency_median": float(np.median(comfortable_frequency[mask])),
                })
            calibration_rows.append({
                "dataset": str(step7["dataset"]), "width_multiplier": width, "seed": seed,
                "fdr_q": q, "tau": float(tau),
                "lower_delta_cutoff": float(lower_cutoff),
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
                "matched_null_checkpoint_sha256": null_hashes,
            })

    classes = pd.DataFrame(class_rows)
    classes.to_csv(report / "all_threshold_class_proportions_by_seed.csv", index=False)
    classes.groupby(
        ["width_multiplier", "null", "provisional_class"], as_index=False
    ).agg(
        count_mean=("count", "mean"), count_sd=("count", "std"),
        fraction_mean=("fraction", "mean"), fraction_sd=("fraction", "std"),
    ).to_csv(report / "all_threshold_class_proportions_summary.csv", index=False)
    pd.DataFrame(headline_rows).to_csv(report / "headline_null_selection.csv", index=False)
    pd.DataFrame(effect_rows).to_csv(report / "reconstruction_ratio_effect_sizes.csv", index=False)
    pd.DataFrame(comfortable_rows).to_csv(report / "rotation_above_median_by_seed.csv", index=False)
    pd.DataFrame(frequency_rows).to_csv(
        report / "rotation_above_median_frequency_quintiles.csv", index=False
    )
    (report / "all_thresholds.json").write_text(json.dumps(calibration_rows, indent=2) + "\n")
    manifest = {
        "status": "complete", "dataset": str(step7["dataset"]),
        "plan": "v16", "null": "drift_matched_shuffled_label",
        "threshold_scope": list(matched["threshold_scope"]),
        "headline_rule": str(matched["headline_rule"]),
        "widths": widths, "real_seeds": seeds, "null_seeds": null_seeds,
    }
    (report / "all_threshold_recalibration_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(classes.groupby(
        ["width_multiplier", "null", "provisional_class"], as_index=False
    )["fraction"].mean().to_string(index=False), flush=True)
    print(pd.DataFrame(comfortable_rows).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
