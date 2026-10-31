#!/usr/bin/env python3
"""Run the v14 IRED raw-rank, feature-death, and resampling diagnostics."""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossfit.activation_cache import load_cached_pair
from crossfit.crosscoder import CrossCoder, crosscoder_diagnostics, reconstruction_loss
from crossfit.dictionary_diagnostics import (
    activation_counts,
    covariance_spectrum,
    death_timing,
    participation_ratio,
    resample_dead_features,
)
from crossfit.utils import REPO_ROOT, load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _atomic_torch_save(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _train_epoch(model, evo, fit, optimizer, generator, train) -> float:
    model.train()
    order = torch.randperm(len(evo), generator=generator, device=evo.device)
    losses = []
    for start in range(0, len(order), int(train["batch_size"])):
        indices = order[start : start + int(train["batch_size"])]
        left = evo.index_select(0, indices)
        right = fit.index_select(0, indices)
        code, left_hat, right_hat = model(left, right)
        loss = reconstruction_loss(left, right, left_hat, right_hat)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(train["gradient_clip_norm"])
        )
        optimizer.step()
        losses.append(float(loss.detach()))
    return float(np.mean(losses))


def _new_model(config, width: int, seed: int, device: str) -> CrossCoder:
    _set_seed(seed)
    return CrossCoder(
        int(config["dimension"]),
        width,
        int(config["batch_topk"]["k"]),
        decoder_bias=bool(config["decoder_bias"]),
    ).to(device)


def _new_optimizer(model, train):
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(train["learning_rate"]),
        weight_decay=float(train["weight_decay"]),
    )


def _counts(model, evo, fit, config) -> np.ndarray:
    return activation_counts(
        model,
        evo,
        fit,
        batch_size=int(config["training"]["evaluation_batch_size"]),
        device=evo.device,
    )


def _assert_exact_reproduction(model, trace, authoritative_path: Path) -> None:
    artifact = torch.load(authoritative_path, map_location="cpu", weights_only=True)
    expected_trace = np.asarray(artifact["trace"], dtype=np.float64)
    if not np.array_equal(np.asarray(trace, dtype=np.float64), expected_trace):
        raise AssertionError(f"loss trace failed exact reproduction: {authoritative_path}")
    actual = model.state_dict()
    for name, expected in artifact["state_dict"].items():
        if not torch.equal(actual[name].detach().cpu(), expected):
            raise AssertionError(
                f"state tensor {name} failed exact reproduction: {authoritative_path}"
            )


def _run_death_timing(config, evo, fit, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = config["training"]
    run_rows = []
    epoch_rows = []
    counts_dir = output_dir / "death_timing"
    counts_dir.mkdir(parents=True, exist_ok=True)
    for width in map(int, config["width_multipliers"]):
        for seed in map(int, train["seeds"]):
            name = f"w{width}_seed{seed}"
            counts_path = counts_dir / f"{name}.npz"
            if counts_path.exists():
                saved = np.load(counts_path)
                counts_by_epoch = saved["activation_counts"]
                trace = saved["loss_trace"].tolist()
                reproduced = bool(saved["exact_reproduction"].item())
                if not reproduced or counts_by_epoch.shape[0] != int(train["epochs"]) + 1:
                    raise AssertionError(f"incomplete death-timing artifact: {counts_path}")
                resumed = True
            else:
                run_started = time.monotonic()
                model = _new_model(config, width, seed, "cuda")
                optimizer = _new_optimizer(model, train)
                generator = torch.Generator(device="cuda").manual_seed(seed)
                count_rows = [_counts(model, evo, fit, config)]
                trace = []
                for epoch in range(1, int(train["epochs"]) + 1):
                    trace.append(_train_epoch(model, evo, fit, optimizer, generator, train))
                    count_rows.append(_counts(model, evo, fit, config))
                    print(
                        json.dumps(
                            {
                                "diagnostic": "death_timing",
                                "width_multiplier": width,
                                "seed": seed,
                                "epoch": epoch,
                                "loss": trace[-1],
                                "live_features": int((count_rows[-1] > 0).sum()),
                            }
                        ),
                        flush=True,
                    )
                authoritative = (
                    REPO_ROOT
                    / config["paths"]["checkpoints"]
                    / f"w{width}_seed{seed}_nobias.pt"
                )
                _assert_exact_reproduction(model, trace, authoritative)
                counts_by_epoch = np.stack(count_rows)
                temporary = counts_path.with_suffix(".npz.tmp")
                with temporary.open("wb") as handle:
                    np.savez_compressed(
                        handle,
                        activation_counts=counts_by_epoch,
                        loss_trace=np.asarray(trace),
                        exact_reproduction=np.asarray(True),
                        wall_hours=np.asarray((time.monotonic() - run_started) / 3600.0),
                    )
                os.replace(temporary, counts_path)
                resumed = False
                del model, optimizer
                torch.cuda.empty_cache()
            timing = death_timing(counts_by_epoch)
            run_rows.append(
                {
                    "dataset": config["dataset"],
                    "width_multiplier": width,
                    "seed": seed,
                    "features": timing.features,
                    "initially_dead": timing.features - timing.live_at_initialization,
                    "never_active": timing.never_active,
                    "crowded_out": timing.crowded_out,
                    "revived": timing.revived,
                    "live_at_initialization": timing.live_at_initialization,
                    "live_at_final": timing.live_at_final,
                    "final_dead_fraction": 1.0 - timing.live_at_final / timing.features,
                    "exact_reproduction": True,
                    "loaded_complete_artifact": resumed,
                    "counts_artifact": str(counts_path.relative_to(REPO_ROOT)),
                    "counts_sha256": _sha256(counts_path),
                }
            )
            active = counts_by_epoch > 0
            for epoch in range(counts_by_epoch.shape[0]):
                previous = active[epoch - 1] if epoch else np.zeros(active.shape[1], dtype=bool)
                epoch_rows.append(
                    {
                        "dataset": config["dataset"],
                        "width_multiplier": width,
                        "seed": seed,
                        "epoch": epoch,
                        "live_features": int(active[epoch].sum()),
                        "live_fraction": float(active[epoch].mean()),
                        "newly_inactive_since_previous": int((previous & ~active[epoch]).sum()),
                        "newly_active_since_previous": int((~previous & active[epoch]).sum()),
                    }
                )
    return pd.DataFrame(run_rows), pd.DataFrame(epoch_rows)


def _run_resampling(config, evo, fit, output_dir: Path) -> tuple[dict, pd.DataFrame]:
    registered = config["diagnostics"]["dead_feature_resampling"]
    width = int(registered["width_multiplier"])
    seed = int(registered["seed"])
    schedule = set(map(int, registered["epochs"]))
    path = (
        REPO_ROOT
        / config["paths"]["diagnostic_checkpoints"]
        / f"w{width}_seed{seed}_resampled_nobias.pt"
    )
    if path.exists():
        artifact = torch.load(path, map_location="cpu", weights_only=True)
        if artifact.get("status") != "complete":
            raise AssertionError(f"incomplete resampling checkpoint: {path}")
        model = _new_model(config, width, seed, "cuda")
        model.load_state_dict(artifact["state_dict"])
        counts_by_epoch = np.asarray(artifact["activation_counts"])
        trace = list(artifact["trace"])
        events = list(artifact["resampling_events"])
        wall_hours = float(artifact["wall_hours"])
        resumed = True
    else:
        started = time.monotonic()
        model = _new_model(config, width, seed, "cuda")
        optimizer = _new_optimizer(model, config["training"])
        order_generator = torch.Generator(device="cuda").manual_seed(seed)
        resample_generator = torch.Generator(device="cuda").manual_seed(
            int(registered["resampling_rng_seed"])
        )
        count_rows = [_counts(model, evo, fit, config)]
        trace = []
        events = []
        for epoch in range(1, int(config["training"]["epochs"]) + 1):
            trace.append(
                _train_epoch(
                    model, evo, fit, optimizer, order_generator, config["training"]
                )
            )
            current = _counts(model, evo, fit, config)
            count_rows.append(current)
            if epoch in schedule:
                dead_mask = current == 0
                resampled = resample_dead_features(
                    model, optimizer, dead_mask, generator=resample_generator
                )
                events.append(
                    {
                        "epoch": epoch,
                        "resampled_features": resampled,
                        "dead_fraction_before_resampling": float(dead_mask.mean()),
                    }
                )
            print(
                json.dumps(
                    {
                        "diagnostic": "dead_feature_resampling",
                        "width_multiplier": width,
                        "seed": seed,
                        "epoch": epoch,
                        "loss": trace[-1],
                        "live_features_before_resampling": int((current > 0).sum()),
                        "resampled": events[-1]["resampled_features"]
                        if epoch in schedule
                        else 0,
                    }
                ),
                flush=True,
            )
        counts_by_epoch = np.stack(count_rows)
        wall_hours = (time.monotonic() - started) / 3600.0
        artifact = {
            "status": "complete",
            "format": "crossfit_crosscoder_resampling_diagnostic_v1",
            "width_multiplier": width,
            "seed": seed,
            "resampling_rng_seed": int(registered["resampling_rng_seed"]),
            "resampling_epochs": sorted(schedule),
            "state_dict": {name: value.detach().cpu() for name, value in model.state_dict().items()},
            "trace": trace,
            "activation_counts": counts_by_epoch,
            "resampling_events": events,
            "wall_hours": wall_hours,
        }
        _atomic_torch_save(artifact, path)
        resumed = False
        del optimizer
    diagnostics = crosscoder_diagnostics(
        model,
        evo,
        fit,
        batch_size=int(config["training"]["evaluation_batch_size"]),
        device="cuda",
    )
    baseline_runs = pd.read_csv(output_dir.parent / "width_sweep_runs.csv")
    baseline_same = baseline_runs[
        (baseline_runs["width_multiplier"] == width) & (baseline_runs["seed"] == seed)
    ].iloc[0]
    baseline_w2_mean = float(
        baseline_runs.loc[
            baseline_runs["width_multiplier"] == 2, "dead_feature_fraction"
        ].mean()
    )
    reduction = float(baseline_same["dead_feature_fraction"]) - diagnostics.dead_feature_fraction
    substantial = reduction >= float(registered["minimum_absolute_dead_fraction_reduction"])
    competitive = diagnostics.dead_feature_fraction <= baseline_w2_mean
    result = {
        "dataset": config["dataset"],
        "width_multiplier": width,
        "seed": seed,
        "resampling_epochs": sorted(schedule),
        "baseline_same_seed_dead_fraction": float(baseline_same["dead_feature_fraction"]),
        "resampled_dead_fraction": diagnostics.dead_feature_fraction,
        "absolute_dead_fraction_reduction": reduction,
        "substantial_reduction_threshold": float(
            registered["minimum_absolute_dead_fraction_reduction"]
        ),
        "substantial_reduction": substantial,
        "baseline_w2_mean_dead_fraction": baseline_w2_mean,
        "no_worse_than_baseline_w2_mean": competitive,
        "qualifies_as_working_width": substantial and competitive,
        "participation_ratio_effective_rank": diagnostics.participation_ratio,
        "evo_fraction_variance_explained": diagnostics.evo_fraction_variance_explained,
        "fit_fraction_variance_explained": diagnostics.fit_fraction_variance_explained,
        "realized_l0": diagnostics.realized_l0,
        "final_reconstruction_loss": trace[-1],
        "checkpoint": str(path.relative_to(REPO_ROOT)),
        "checkpoint_sha256": _sha256(path),
        "loaded_complete_checkpoint": resumed,
        "gpu_wall_hours": wall_hours,
        "resampling_events": events,
    }
    rows = []
    for epoch, values in enumerate(counts_by_epoch):
        rows.append(
            {
                "dataset": config["dataset"],
                "width_multiplier": width,
                "seed": seed,
                "epoch": epoch,
                "live_features": int((values > 0).sum()),
                "live_fraction": float((values > 0).mean()),
                "resampling_follows_epoch": epoch in schedule,
            }
        )
    del model
    torch.cuda.empty_cache()
    return result, pd.DataFrame(rows)


def _run_raw_spectra(config, evo, fit) -> tuple[pd.DataFrame, dict]:
    batch_size = int(config["diagnostics"]["raw_pca_batch_size"])
    rows = []
    summary = {}
    for model_name, values in (("evo", evo), ("fit", fit)):
        eigenvalues = covariance_spectrum(
            values, batch_size=batch_size, device=values.device
        )
        cumulative = np.cumsum(eigenvalues) / eigenvalues.sum()
        for component, (eigenvalue, explained) in enumerate(
            zip(eigenvalues, cumulative), start=1
        ):
            rows.append(
                {
                    "dataset": config["dataset"],
                    "model": model_name,
                    "component": component,
                    "eigenvalue": eigenvalue,
                    "explained_variance_fraction": eigenvalue / eigenvalues.sum(),
                    "cumulative_explained_variance": explained,
                }
            )
        summary[model_name] = {
            "participation_ratio_effective_rank": participation_ratio(eigenvalues),
            "components_for_variance": {
                str(threshold): int(np.searchsorted(cumulative, threshold) + 1)
                for threshold in config["diagnostics"]["pca_variance_thresholds"]
            },
        }
    return pd.DataFrame(rows), summary


def _plot_diagnostics(output_dir: Path, spectra, epochs, resampling) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for model_name, group in spectra.groupby("model"):
        axes[0].plot(
            group["component"], group["cumulative_explained_variance"], label=model_name
        )
    axes[0].set(
        xlabel="PCA component", ylabel="cumulative explained variance", xscale="log"
    )
    axes[0].legend()
    mean = epochs.groupby(["width_multiplier", "epoch"])["live_fraction"].mean().reset_index()
    for width, group in mean.groupby("width_multiplier"):
        axes[1].plot(group["epoch"], group["live_fraction"], label=f"baseline w={width}")
    axes[1].plot(
        resampling["epoch"],
        resampling["live_fraction"],
        linestyle="--",
        linewidth=2,
        label="w=4 resampled",
    )
    axes[1].set(xlabel="epoch (0 = initialization)", ylabel="live-feature fraction")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_dir / "rank_and_feature_death.png", dpi=180)
    plt.close(figure)


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("IRED dictionary diagnostics require the GPU pod")
    started = time.monotonic()
    config = load_config(REPO_ROOT / "Code/config/step7.yaml")
    output_dir = REPO_ROOT / config["paths"]["report_dir"] / "part_a"
    output_dir.mkdir(parents=True, exist_ok=True)
    evo_array, fit_array = load_cached_pair(REPO_ROOT / config["paths"]["cache"])
    evo = torch.as_tensor(evo_array, dtype=torch.float32, device="cuda")
    fit = torch.as_tensor(fit_array, dtype=torch.float32, device="cuda")
    del evo_array, fit_array

    spectra, raw_summary = _run_raw_spectra(config, evo, fit)
    death_runs, death_epochs = _run_death_timing(config, evo, fit, output_dir)
    resampling_result, resampling_epochs = _run_resampling(
        config, evo, fit, output_dir
    )
    default_widths = list(map(int, config["diagnostics"]["default_working_widths"]))
    working_widths = (
        [1, int(resampling_result["width_multiplier"])]
        if resampling_result["qualifies_as_working_width"]
        else default_widths
    )

    spectra.to_csv(output_dir / "raw_activation_pca_spectrum.csv", index=False)
    death_runs.to_csv(output_dir / "death_timing_runs.csv", index=False)
    death_epochs.to_csv(output_dir / "death_timing_epochs.csv", index=False)
    resampling_epochs.to_csv(output_dir / "w4_resampling_epochs.csv", index=False)
    (output_dir / "raw_activation_pca_summary.json").write_text(
        json.dumps(raw_summary, indent=2) + "\n"
    )
    (output_dir / "w4_resampling.json").write_text(
        json.dumps(resampling_result, indent=2) + "\n"
    )
    _plot_diagnostics(output_dir, spectra, death_epochs, resampling_epochs)
    manifest = {
        "status": "complete",
        "dataset": config["dataset"],
        "positions": len(evo),
        "dimension": int(config["dimension"]),
        "nominal_k": int(config["batch_topk"]["k"]),
        "working_widths": working_widths,
        "raw_activation_summary": raw_summary,
        "resampling_result": resampling_result,
        "all_baseline_runs_exactly_reproduced": bool(death_runs["exact_reproduction"].all()),
        "gpu_wall_hours_this_invocation": (time.monotonic() - started) / 3600.0,
        "partial_results_reported": False,
    }
    (output_dir / "part_a_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
