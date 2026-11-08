#!/usr/bin/env python3
"""Train and apply the IRED drift-matched rotation null."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import chi2_contingency, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossfit.activation_cache import (
    load_cached_model,
    normalization_scalar,
    normalize_shards_in_place,
    shard_sha256,
)
from crossfit.adaptation import (
    SequenceFitnessAdapter,
    build_lora_adapter,
    fit_adapter_to_drift,
    load_trainable_state,
    save_trainable_state,
)
from crossfit.crosscoder import CrossCoder, crosscoder_diagnostics
from crossfit.crosscoder_training import load_or_fit_crosscoder
from crossfit.datasets import get_split, load_assay
from crossfit.esmc import extract_residue_layer, load_esmc
from crossfit.search import training_pool_folds
from crossfit.taxonomy import (
    FeatureStatistics,
    calibrate_pooled_beta_tail,
    feature_statistics,
    null_eligible,
    provisional_classes,
)
from crossfit.utils import REPO_ROOT, load_config


def _adapter_parameters(step3: dict) -> dict:
    settings = step3["adapter"]
    return {
        "learning_rate": settings["learning_rate"],
        "weight_decay": settings["weight_decay"],
        "batch_size": settings["batch_size"],
        "gradient_accumulation_steps": settings["gradient_accumulation_steps"],
        "gradient_clip_norm": settings["gradient_clip_norm"],
    }


def _new_adapter(step3: dict, seed: int) -> tuple[SequenceFitnessAdapter, object]:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    backbone = load_esmc(
        str(step3["backbone"]["name"]), "cuda",
        repository=str(step3["backbone"]["repository"]),
        expected_revision=str(step3["backbone"]["revision"]),
    )
    adapted = build_lora_adapter(
        backbone, rank=int(step3["adapter"]["rank"]),
        alpha=int(step3["adapter"]["alpha"]), dropout=float(step3["adapter"]["dropout"]),
    )
    return SequenceFitnessAdapter(adapted, int(backbone.embed.embedding_dim)).to("cuda"), backbone._tokenize


def _relative_drift(base: np.ndarray, adapted: np.ndarray) -> float:
    if base.shape != adapted.shape or base.ndim != 3:
        raise ValueError("selected-layer reserve activations must align as N x L x d")
    numerator = np.linalg.norm(adapted.astype(np.float32) - base.astype(np.float32), axis=-1).mean(axis=-1)
    denominator = np.linalg.norm(base.astype(np.float32), axis=-1).mean(axis=-1)
    if np.any(denominator == 0):
        raise ValueError("base reserve activation norm is zero")
    return float(np.mean(numerator / denominator))


def _extract_cache(
    model: SequenceFitnessAdapter,
    sequences: list[str],
    rows: np.ndarray,
    destination: Path,
    *,
    layer: int,
    batch_size: int,
    variants_per_shard: int,
    dimension: int,
    checkpoint_hash: str,
) -> dict:
    temporary = destination.with_name(destination.name + ".incomplete")
    if destination.exists():
        manifest = json.loads((destination / "manifest.json").read_text())
        if manifest.get("status") != "complete":
            raise ValueError("drift-matched cache is not complete")
        return manifest
    if temporary.exists():
        raise FileExistsError(f"inspect incomplete drift-matched cache: {temporary}")
    started = time.monotonic()
    temporary.mkdir(parents=True)
    np.save(temporary / "variant_rows.npy", rows)
    output = temporary / "drift_matched"
    output.mkdir()
    paths, sum_norms, positions = [], 0.0, 0
    model.eval()
    for shard, start in enumerate(range(0, len(sequences), variants_per_shard)):
        values = extract_residue_layer(
            model.backbone, sequences[start : start + variants_per_shard],
            layer=layer, batch_size=batch_size,
        )
        sum_norms += float(np.linalg.norm(values, axis=-1).sum())
        positions += int(np.prod(values.shape[:-1]))
        path = output / f"shard_{shard:04d}.npy"
        np.save(path, values.astype(np.float16))
        paths.append(path)
    scalar = normalization_scalar(sum_norms, positions, dimension)
    realized = normalize_shards_in_place(paths, scalar)
    manifest = {
        "status": "complete", "dataset": "IRED", "adapter": "drift_matched_shuffled_label",
        "analysis_layer": layer, "training_variants": len(sequences), "positions": positions,
        "sequence_length": len(sequences[0]), "dimension": dimension, "dtype": "float16",
        "adapter_checkpoint_sha256": checkpoint_hash,
        "models": {"drift_matched": {
            "raw_mean_norm": sum_norms / positions, "normalization_scalar": scalar,
            "normalized_mean_norm": realized,
            "shards": [
                {"name": path.name, "bytes": path.stat().st_size, "sha256": shard_sha256(path)}
                for path in paths
            ],
        }},
        "gpu_wall_hours": (time.monotonic() - started) / 3600.0,
        "excluded": ["pg_kfold_test", "layer_freeze_reserve", "round_held_out_test", "plate"],
    }
    (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    os.replace(temporary, destination)
    return manifest


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


def _distribution_row(
    values: np.ndarray, *, width: int, source: str, seed: int
) -> dict[str, float | int | str]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    quantiles = np.quantile(finite, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "width_multiplier": width, "source": source, "seed": seed, "n": len(finite),
        "mean": float(finite.mean()), "standard_deviation": float(finite.std(ddof=1)),
        **{f"q{label}": float(value) for label, value in zip(
            ["01", "05", "25", "50", "75", "95", "99"], quantiles
        )},
    }


def _rotation_stage(frame: pd.DataFrame, floor_fraction: float) -> np.ndarray:
    live = frame["activation_count"].to_numpy(float) >= 1
    evo = frame["evo_decoder_norm"].to_numpy(float)
    fit = frame["fit_decoder_norm"].to_numpy(float)
    evo_clear = evo >= floor_fraction * np.median(evo)
    fit_clear = fit >= floor_fraction * np.median(fit)
    dead = ~live | (~evo_clear & ~fit_clear)
    emergent = (
        ~dead & fit_clear & frame["upper_delta_tail"].to_numpy(bool)
        & frame["latent_error_pass"].to_numpy(bool)
        & frame["latent_reconstruction_pass"].to_numpy(bool)
    )
    evolution = ~dead & ~emergent & evo_clear & frame["lower_delta_tail"].to_numpy(bool)
    second_dead = ~dead & ~emergent & ~evolution & ~(evo_clear & fit_clear)
    return ~(dead | emergent | evolution | second_dead)


def _concentration_rows(
    frame: pd.DataFrame,
    flags: np.ndarray,
    *,
    width: int,
    seed: int,
    null_name: str,
    floor_fraction: float,
    bins: int,
) -> tuple[list[dict], list[dict]]:
    stage = _rotation_stage(frame, floor_fraction)
    norm = np.sqrt(
        frame["evo_decoder_norm"].to_numpy(float)
        * frame["fit_decoder_norm"].to_numpy(float)
    )
    frequency = frame["activation_frequency"].to_numpy(float)
    rows, tests = [], []
    for name, values in (("geometric_decoder_norm", norm), ("activation_frequency", frequency)):
        selected_values = values[stage]
        selected_flags = np.asarray(flags, dtype=bool)[stage]
        labels = pd.qcut(selected_values, q=bins, labels=False, duplicates="drop")
        table = []
        for group in sorted(np.unique(labels)):
            mask = np.asarray(labels) == group
            count = int(mask.sum())
            rotated = int(selected_flags[mask].sum())
            rows.append({
                "width_multiplier": width, "seed": seed, "null": null_name,
                "covariate": name, "quintile": int(group) + 1,
                "features_at_rotation_stage": count, "rotation_flags": rotated,
                "rotation_fraction": rotated / count,
                "covariate_median": float(np.median(selected_values[mask])),
            })
            table.append([rotated, count - rotated])
        if np.unique(selected_flags).size < 2:
            correlation = float("nan")
            chi2, pvalue = float("nan"), 1.0
        else:
            correlation = spearmanr(selected_values, selected_flags.astype(float)).statistic
            chi2, pvalue, _, _ = chi2_contingency(np.asarray(table))
        tests.append({
            "width_multiplier": width, "seed": seed, "null": null_name,
            "covariate": name, "spearman_flag_vs_covariate": float(correlation),
            "chi_square": float(chi2), "chi_square_pvalue": float(pvalue),
            "features_at_rotation_stage": int(stage.sum()),
        })
    return rows, tests


def _plot_reconstruction_distributions(rows: list[tuple[int, str, np.ndarray]], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for axis, width in zip(axes, (1, 2)):
        for row_width, source, values in rows:
            if row_width != width:
                continue
            finite = np.sort(np.asarray(values, dtype=float))
            finite = finite[np.isfinite(finite)]
            clipped = finite[(finite >= np.quantile(finite, 0.01)) & (finite <= np.quantile(finite, 0.99))]
            y = np.arange(1, len(clipped) + 1) / len(clipped)
            axis.plot(clipped, y, alpha=0.55, lw=1, label=source)
        axis.set(title=f"w={width}", xlabel="reconstruction ratio (1st-99th pct)")
    axes[0].set_ylabel("empirical CDF")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="center right", frameon=False)
    figure.tight_layout(rect=(0, 0, 0.88, 1))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("drift-matched null training requires the GPU pod")
    started = time.monotonic()
    step2 = load_config(REPO_ROOT / "Code/config/step2.yaml")
    step3 = load_config(REPO_ROOT / "Code/config/step3.yaml")
    step4 = load_config(REPO_ROOT / "Code/config/step4.yaml")
    step6 = load_config(REPO_ROOT / "Code/config/step6.yaml")
    step7 = load_config(REPO_ROOT / "Code/config/step7.yaml")
    matched = step6["calibration"]["drift_matched"]
    widths = list(map(int, step7["diagnostics"]["selected_working_widths"]))
    seeds = list(map(int, step6["calibration"]["shuffled_crosscoder_seeds"]))
    report = REPO_ROOT / step6["paths"]["drift_matched_report_dir"]
    report.mkdir(parents=True, exist_ok=True)

    gate1 = json.loads((REPO_ROOT / matched["target_source"]).read_text())
    target_drift = float(gate1[matched["target_field"]])
    layer = int(matched["analysis_layer"])
    frame = load_assay("IRED", with_sequences=True)
    train_pool, _ = get_split("IRED", "pg_kfold", fold=int(step4["source_fold"]))
    folds = training_pool_folds(
        train_pool, int(step3["selection"]["cv_folds"]), int(step3["selection"]["cv_seed"])
    )
    final_fold = int(step3["gate1"]["final_adapter_cv_fold"])
    adapter_train = folds[final_fold][0]
    rng = np.random.default_rng(int(step3["adapter"]["shuffled_label_seed"]))
    permutation = rng.permutation(len(adapter_train))
    target = frame["y"].to_numpy(np.float32)
    censoring = frame["censored"].astype(str).to_numpy()
    sequences = frame["mutant_sequence"].astype(str).tolist()
    shuffled_target = target[adapter_train][permutation]
    shuffled_censoring = censoring[adapter_train][permutation]
    reserve = load_assay("IRED", with_sequences=True, include_reserve=True)
    reserve = reserve.loc[reserve["is_layer_freeze_reserve"]].reset_index(drop=True)
    reserve_sequences = reserve["mutant_sequence"].astype(str).tolist()
    base_reserve = np.load(
        REPO_ROOT / step2["paths"]["representations"] / "reserve_all_layers_base.npy",
        mmap_mode="r",
    )[:, layer]
    checkpoint = REPO_ROOT / step6["paths"]["drift_matched_adapter_checkpoint"]
    training_manifest_path = report / "adapter_matching_manifest.json"
    model, tokenize = _new_adapter(step3, int(step3["adapter"]["seed"]) + final_fold)
    if checkpoint.exists() and training_manifest_path.exists():
        load_trainable_state(model, checkpoint)
        adapter_manifest = json.loads(training_manifest_path.read_text())
        resumed_adapter = True
    else:
        if checkpoint.exists() or training_manifest_path.exists():
            raise RuntimeError("partial drift-matched adapter artifacts found; archive before rerun")

        def measure(value: SequenceFitnessAdapter) -> float:
            activations = extract_residue_layer(
                value.backbone, reserve_sequences, layer=layer,
                batch_size=int(step3["extraction"]["reserve_batch_size"]),
            )
            return _relative_drift(base_reserve, activations)

        adapter_started = time.monotonic()
        fit = fit_adapter_to_drift(
            model, [sequences[index] for index in adapter_train], shuffled_target,
            shuffled_censoring, backbone_tokenize=tokenize,
            objective_config=step2["objective"], adapter_config=_adapter_parameters(step3),
            drift_measure=measure, target_drift=target_drift,
            max_epochs=int(matched["maximum_epochs"]),
            seed=int(step3["adapter"]["seed"]) + final_fold, device="cuda",
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary_checkpoint = checkpoint.with_suffix(".pt.tmp")
        save_trainable_state(fit.model, temporary_checkpoint)
        os.replace(temporary_checkpoint, checkpoint)
        adapter_manifest = {
            "status": "complete", "dataset": "IRED", "null": "drift_matched_shuffled_label",
            "target_drift": target_drift, "selected_drift": fit.selected_drift,
            "absolute_mismatch": abs(fit.selected_drift - target_drift),
            "relative_mismatch": abs(fit.selected_drift - target_drift) / target_drift,
            "selected_epoch": fit.selected_epoch + 1,
            "selected_optimizer_steps": fit.selected_optimizer_steps,
            "executed_optimizer_steps": fit.executed_optimizer_steps,
            "matching_rule": str(matched["matching_rule"]),
            "optimizer_schedule": str(matched["optimizer_schedule"]),
            "learning_rate": float(step3["adapter"]["learning_rate"]),
            "weight_decay": float(step3["adapter"]["weight_decay"]),
            "rank": int(step3["adapter"]["rank"]),
            "trace": fit.trace,
            "checkpoint": str(checkpoint.relative_to(REPO_ROOT)),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "training_gpu_wall_hours": (time.monotonic() - adapter_started) / 3600.0,
        }
        training_manifest_path.write_text(json.dumps(adapter_manifest, indent=2) + "\n")
        model = fit.model
        resumed_adapter = False

    training = frame.iloc[train_pool].reset_index(drop=True)
    cache_rows = training["variant_row"].to_numpy(np.int64)
    reference_rows = np.load(REPO_ROOT / step6["paths"]["step4_cache"] / "variant_rows.npy")
    if not np.array_equal(cache_rows, reference_rows):
        raise AssertionError("drift-matched and real-pair cache rows are not aligned")
    cache = REPO_ROOT / step6["paths"]["drift_matched_cache"]
    cache_manifest = _extract_cache(
        model, training["mutant_sequence"].astype(str).tolist(), cache_rows, cache,
        layer=layer, batch_size=int(step4["extraction_batch_size"]),
        variants_per_shard=int(step4["variants_per_shard"]),
        dimension=int(step6["dimension"]),
        checkpoint_hash=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()

    evo_array = load_cached_model(REPO_ROOT / step6["paths"]["step4_cache"], "evo")
    matched_array = load_cached_model(cache, "drift_matched")
    step_array = load_cached_model(REPO_ROOT / step6["paths"]["shuffled_cache"], "shuffled")
    if evo_array.shape != matched_array.shape or evo_array.shape != step_array.shape:
        raise AssertionError("all null activation pools must align")
    evo = torch.as_tensor(evo_array, dtype=torch.float32, device="cuda")
    matched_tensor = torch.as_tensor(matched_array, dtype=torch.float32, device="cuda")
    null_runs, crosscoder_rows = {}, []
    for width in widths:
        null_runs[width] = []
        for seed in seeds:
            path = (
                REPO_ROOT / step6["paths"]["drift_matched_crosscoder_checkpoints"]
                / f"w{width}_seed{seed}_nobias.pt"
            )
            run_started = time.monotonic()
            crosscoder, trace, resumed = load_or_fit_crosscoder(
                path, dimension=int(step6["dimension"]), width_multiplier=width,
                k=int(step6["batch_topk"]["k"]), decoder_bias=bool(step6["decoder_bias"]),
                evo=evo, fit=matched_tensor, training=step6["training"], seed=seed, device="cuda",
            )
            stats = feature_statistics(
                crosscoder, evo, matched_tensor,
                batch_size=int(step7["training"]["evaluation_batch_size"]), device="cuda",
            )
            null_runs[width].append(stats)
            diagnostics = crosscoder_diagnostics(
                crosscoder, evo, matched_tensor,
                batch_size=int(step7["training"]["evaluation_batch_size"]), device="cuda",
            )
            crosscoder_rows.append({
                "width_multiplier": width, "seed": seed,
                "dead_feature_fraction": diagnostics.dead_feature_fraction,
                "evo_fraction_variance_explained": diagnostics.evo_fraction_variance_explained,
                "matched_fraction_variance_explained": diagnostics.fit_fraction_variance_explained,
                "realized_l0": diagnostics.realized_l0, "epochs": len(trace),
                "checkpoint": str(path.relative_to(REPO_ROOT)),
                "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "loaded_complete_checkpoint": resumed,
                "run_wall_hours": (time.monotonic() - run_started) / 3600.0,
            })
            del crosscoder
            torch.cuda.empty_cache()
    pd.DataFrame(crosscoder_rows).to_csv(report / "drift_matched_crosscoder_runs.csv", index=False)

    floor = float(step7["taxonomy"]["decoder_norm_floor_fraction"])
    q = float(step6["calibration"]["fdr_q"])
    reconstruction_rows, reconstruction_plot = [], []
    matched_thresholds, rotation_rows, class_rows = [], [], []
    concentration_rows, concentration_tests = [], []
    for width in widths:
        step_stats = []
        for seed in seeds:
            path = (
                REPO_ROOT / step6["paths"]["shuffled_checkpoints"]
                / f"w{width}_seed{seed}_nobias.pt"
            )
            artifact = torch.load(path, map_location="cpu", weights_only=True)
            crosscoder = CrossCoder(
                int(step6["dimension"]), width, int(artifact["nominal_k"]),
                decoder_bias=bool(artifact["decoder_bias"]),
            )
            crosscoder.load_state_dict(artifact["state_dict"])
            stats = feature_statistics(
                crosscoder, evo_array, step_array,
                batch_size=int(step7["training"]["evaluation_batch_size"]), device="cuda",
            )
            step_stats.append(stats)
            eligible = null_eligible(stats, floor)
            values = stats.reconstruction_ratio[eligible]
            reconstruction_rows.append(_distribution_row(
                values, width=width, source="step_matched_null", seed=seed
            ))
            reconstruction_plot.append((width, f"null seed {seed}", values))
            del crosscoder
            torch.cuda.empty_cache()
        matched_eligible = [null_eligible(value, floor) for value in null_runs[width]]
        matched_cosines = [
            (value.decoder_cosine[mask] + 1.0) / 2.0
            for value, mask in zip(null_runs[width], matched_eligible)
        ]
        for seed in seeds:
            taxonomy_path = REPO_ROOT / step7["paths"]["report_dir"] / f"taxonomy_w{width}_seed{seed}.csv"
            frame_real = pd.read_csv(taxonomy_path)
            stats_real = _statistics_from_frame(frame_real)
            live = stats_real.activation_count >= int(step7["taxonomy"]["activation_floor_count"])
            evo_clear = stats_real.evo_decoder_norm >= floor * np.median(stats_real.evo_decoder_norm)
            fit_clear = stats_real.fit_decoder_norm >= floor * np.median(stats_real.fit_decoder_norm)
            shared = live & evo_clear & fit_clear
            flags = np.zeros(len(frame_real), dtype=bool)
            passed, cutoff_scaled, reports = calibrate_pooled_beta_tail(
                (stats_real.decoder_cosine[shared] + 1.0) / 2.0,
                matched_cosines, tail="lower", q=q,
            )
            flags[shared] = passed
            cutoff = 2.0 * cutoff_scaled - 1.0
            labels = provisional_classes(
                stats_real,
                lower_delta=frame_real["lower_delta_tail"].to_numpy(bool),
                upper_delta=frame_real["upper_delta_tail"].to_numpy(bool),
                latent_error_pass=frame_real["latent_error_pass"].to_numpy(bool),
                latent_reconstruction_pass=frame_real["latent_reconstruction_pass"].to_numpy(bool),
                rotated=flags,
                frequency_shifted=frame_real["activation_frequency_shift"].to_numpy(bool),
                floor_fraction=floor,
                activation_floor_count=int(step7["taxonomy"]["activation_floor_count"]),
            )
            original = frame_real["provisional_class"].to_numpy(str)
            updated = frame_real.copy()
            updated["drift_matched_rotation_shift"] = flags
            updated["drift_matched_provisional_class"] = labels
            updated.to_csv(
                report / f"taxonomy_rotation_drift_matched_w{width}_seed{seed}.csv",
                index=False,
            )
            for label in (
                "dead", "emergent_candidate", "evolution_only", "rotated",
                "re_weighted", "conserved",
            ):
                class_rows.append({
                    "width_multiplier": width,
                    "seed": seed,
                    "provisional_class": label,
                    "count": int((labels == label).sum()),
                    "fraction": float(np.mean(labels == label)),
                })
            rotation_rows.append({
                "width_multiplier": width, "seed": seed,
                "original_shuffled_rotated_count": int((original == "rotated").sum()),
                "original_shuffled_rotated_fraction": float(np.mean(original == "rotated")),
                "drift_matched_rotated_count": int((labels == "rotated").sum()),
                "drift_matched_rotated_fraction": float(np.mean(labels == "rotated")),
                "features": len(labels), "drift_matched_decoder_cosine_cutoff": cutoff,
            })
            matched_thresholds.append({
                "width_multiplier": width, "seed": seed,
                "decoder_cosine_cutoff": cutoff, "null_runs": reports,
            })
            for null_name, null_flags in (
                ("original_step_matched", frame_real["rotation_shift"].to_numpy(bool)),
                ("drift_matched", flags),
            ):
                rows_out, tests_out = _concentration_rows(
                    frame_real, null_flags, width=width, seed=seed, null_name=null_name,
                    floor_fraction=floor, bins=int(matched["concentration_bins"]),
                )
                concentration_rows.extend(rows_out)
                concentration_tests.extend(tests_out)
            real_values = stats_real.reconstruction_ratio[live & fit_clear]
            reconstruction_rows.append(_distribution_row(
                real_values, width=width, source="real_pair", seed=seed
            ))
            reconstruction_plot.append((width, f"real seed {seed}", real_values))

    pd.DataFrame(reconstruction_rows).to_csv(
        report / "reconstruction_ratio_distributions.csv", index=False
    )
    _plot_reconstruction_distributions(
        reconstruction_plot, report / "reconstruction_ratio_ecdf.png"
    )
    rotation_frame = pd.DataFrame(rotation_rows)
    rotation_frame.to_csv(report / "rotation_null_comparison_by_seed.csv", index=False)
    rotation_summary = rotation_frame.groupby("width_multiplier", as_index=False).agg(
        original_rotated_fraction_mean=("original_shuffled_rotated_fraction", "mean"),
        original_rotated_fraction_sd=("original_shuffled_rotated_fraction", "std"),
        drift_matched_rotated_fraction_mean=("drift_matched_rotated_fraction", "mean"),
        drift_matched_rotated_fraction_sd=("drift_matched_rotated_fraction", "std"),
        drift_matched_cosine_cutoff_mean=("drift_matched_decoder_cosine_cutoff", "mean"),
    )
    rotation_summary.to_csv(report / "rotation_null_comparison_summary.csv", index=False)
    class_frame = pd.DataFrame(class_rows)
    class_frame.to_csv(report / "drift_matched_class_proportions_by_seed.csv", index=False)
    class_frame.groupby(
        ["width_multiplier", "provisional_class"], as_index=False
    ).agg(
        fraction_mean=("fraction", "mean"), fraction_sd=("fraction", "std"),
        count_mean=("count", "mean"),
    ).to_csv(report / "drift_matched_class_proportions_summary.csv", index=False)
    pd.DataFrame(concentration_rows).to_csv(report / "rotation_concentration_by_quintile.csv", index=False)
    pd.DataFrame(concentration_tests).to_csv(report / "rotation_concentration_tests.csv", index=False)
    (report / "rotation_thresholds.json").write_text(
        json.dumps(matched_thresholds, indent=2) + "\n"
    )
    audit = {
        "status": "complete", "dataset": "IRED",
        "step_matched_null_widths": widths, "real_widths": widths,
        "matched_width_calibration": True,
        "step_matched_null_checkpoint_pattern": "Models/step6/IRED/shuffled/w{width}_seed{seed}_nobias.pt",
        "drift_matched_null_checkpoint_pattern": "Models/step6/IRED/drift_matched_rotation/crosscoders/w{width}_seed{seed}_nobias.pt",
        "real_adapter_drift": target_drift,
        "drift_matched_adapter_drift": float(adapter_manifest["selected_drift"]),
        "drift_match_relative_error": float(adapter_manifest["relative_mismatch"]),
        "adapter_resumed_complete_checkpoint": resumed_adapter,
        "cache_gpu_wall_hours": float(cache_manifest["gpu_wall_hours"]),
        "invocation_gpu_wall_hours": (time.monotonic() - started) / 3600.0,
        "threshold_scope": list(matched["threshold_scope"]),
        "initial_rotation_only_audit": True,
    }
    (report / "audit_manifest.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"audit": audit, "rotation_summary": rotation_summary.to_dict("records")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
