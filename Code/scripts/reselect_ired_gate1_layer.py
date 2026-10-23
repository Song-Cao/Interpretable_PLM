#!/usr/bin/env python3
"""Re-select the IRED Gate 1 layer from cached reserve activations."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossfit.gate1 import (
    activation_norms_by_layer,
    paired_task_gain_test,
    select_analysis_layer,
)
from crossfit.utils import REPO_ROOT, load_config


def _archive_previous(report_dir: Path) -> Path:
    archive = REPO_ROOT / "Reports/archive/step3_gate1_depth29_pre_cka_floor"
    archive.mkdir(parents=True, exist_ok=True)
    for name in ("gate1_report.json", "gate1_by_layer.csv", "gate1_drift_by_depth.png"):
        source = report_dir / name
        destination = archive / name
        if destination.exists():
            continue
        shutil.copy2(source, destination)
    return archive


def _sensitivity(frame: pd.DataFrame, selected: int, offsets: list[int]) -> list[dict]:
    output = []
    for offset in offsets:
        layer = selected + int(offset)
        if layer < 0 or layer >= len(frame):
            output.append({"offset": int(offset), "available": False})
            continue
        row = frame.iloc[layer]
        output.append({
            "offset": int(offset),
            "available": True,
            "layer": int(layer),
            "drift_gap": float(row["drift_gap"]),
            "drift_gap_95_ci": [
                float(row["drift_gap_lower"]), float(row["drift_gap_upper"])
            ],
            "real_cka": float(row["real_cka"]),
            "activation_norm_ratio": float(row["activation_norm_ratio"]),
        })
    return output


def main() -> int:
    step2 = load_config(REPO_ROOT / "Code/config/step2.yaml")
    config = load_config(REPO_ROOT / "Code/config/step3.yaml")
    gate = config["gate1"]
    report_dir = REPO_ROOT / config["paths"]["reports"]
    archive = _archive_previous(report_dir)

    frame = pd.read_csv(report_dir / "gate1_by_layer.csv")
    base = np.load(
        REPO_ROOT / step2["paths"]["representations"] / "reserve_all_layers_base.npy",
        mmap_mode="r",
    )
    adapted = np.load(
        REPO_ROOT / config["paths"]["cache"] / "reserve_all_layers_adapted.npy",
        mmap_mode="r",
    )
    base_norm, adapted_norm, norm_ratio = activation_norms_by_layer(base, adapted)
    if len(frame) != len(norm_ratio):
        raise AssertionError("per-layer table and reserve activations have different depths")
    frame["base_mean_activation_norm"] = base_norm
    frame["real_mean_activation_norm"] = adapted_norm
    frame["activation_norm_ratio"] = norm_ratio

    selected, eligible, near_max, threshold = select_analysis_layer(
        frame["drift_gap"].to_numpy(), frame["real_cka"].to_numpy(),
        cka_floor=float(gate["real_cka_floor"]),
        exclude_first=int(gate["exclude_first_layers"]),
        exclude_last=int(gate["exclude_last_layers"]),
        near_max_fraction=float(gate["near_maximum_drift_gap_fraction"]),
    )
    frame["selection_eligible"] = eligible
    frame["within_near_maximum"] = near_max
    frame["selected"] = frame["layer"].to_numpy(int) == selected
    frame.to_csv(report_dir / "gate1_by_layer.csv", index=False)

    cv = pd.read_csv(report_dir / "gate1_task_gain_cv.csv")
    task_test = paired_task_gain_test(cv["task_gain"].to_numpy())
    alpha = float(gate["task_gain_alpha"])
    row = frame.iloc[selected]
    drift_ratio = float(row["real_relative_drift"] / row["shuffled_relative_drift"])
    ratio_window = list(map(float, gate["drift_ratio_window"]))
    drift_separation = float(row["drift_gap_lower"]) > 0.0
    task_gain_established = (
        task_test["mean"] > 0.0 and task_test["p_value_two_sided"] < alpha
    )
    previous = json.loads((report_dir / "gate1_report.json").read_text())
    result = {
        **previous,
        "task_gain_test": {
            "method": str(gate["task_gain_test"]),
            "alpha": alpha,
            **task_test,
            "established": bool(task_gain_established),
        },
        "selected_layer": selected,
        "selected_layer_frozen": False,
        "layer_selection_rule": (
            "earliest mid-network layer with real CKA at least 0.5 and drift gap "
            "at least 90% of the eligible maximum"
        ),
        "layer_selection_parameters": {
            "real_cka_floor": float(gate["real_cka_floor"]),
            "exclude_first_layers": int(gate["exclude_first_layers"]),
            "exclude_last_layers": int(gate["exclude_last_layers"]),
            "near_maximum_drift_gap_fraction": float(
                gate["near_maximum_drift_gap_fraction"]
            ),
            "eligible_maximum_drift_gap": float(frame.loc[eligible, "drift_gap"].max()),
            "near_maximum_threshold": threshold,
        },
        "selected_real_relative_drift": float(row["real_relative_drift"]),
        "selected_shuffled_relative_drift": float(row["shuffled_relative_drift"]),
        "selected_real_to_shuffled_drift_ratio": drift_ratio,
        "matched_drift_null_required": not ratio_window[0] <= drift_ratio <= ratio_window[1],
        "selected_drift_gap_95_ci": [
            float(row["drift_gap_lower"]), float(row["drift_gap_upper"])
        ],
        "selected_real_cka": float(row["real_cka"]),
        "selected_shuffled_cka": float(row["shuffled_cka"]),
        "selected_activation_norm_ratio": float(row["activation_norm_ratio"]),
        "sensitivity": _sensitivity(
            frame, selected, list(map(int, gate["sensitivity_offsets"]))
        ),
        "gate1_pass": bool(task_gain_established and drift_separation),
        "gate_failure_reasons": ([] if task_gain_established else ["task_gain_not_established"]),
        "superseded_layer_29_result": {
            "reason": "real CKA 0.06415 indicates final-layer collapse",
            "archive": str(archive.relative_to(REPO_ROOT)),
            "full_selected_layer_cache_is_superseded": True,
        },
        "run_complete": True,
    }
    result.pop("selected_cache_shapes", None)
    (report_dir / "gate1_report.json").write_text(json.dumps(result, indent=2) + "\n")

    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(frame["layer"], frame["real_relative_drift"], label="real")
    axes[0].plot(frame["layer"], frame["shuffled_relative_drift"], label="shuffled")
    axes[0].set(xlabel="layer", ylabel="relative drift")
    axes[0].legend()
    axes[1].plot(frame["layer"], frame["drift_gap"], label="drift gap")
    axes[1].fill_between(
        frame["layer"], frame["drift_gap_lower"], frame["drift_gap_upper"], alpha=0.2
    )
    axes[1].axhline(threshold, color="grey", linestyle=":", label="90% threshold")
    axes[1].set(xlabel="layer", ylabel="real - shuffled drift")
    axes[1].legend()
    axes[2].plot(frame["layer"], frame["real_cka"], label="real CKA")
    axes[2].plot(frame["layer"], frame["activation_norm_ratio"], label="norm ratio")
    axes[2].axhline(float(gate["real_cka_floor"]), color="grey", linestyle=":")
    axes[2].set(xlabel="layer", ylabel="value")
    axes[2].legend()
    for axis in axes:
        axis.axvline(selected, color="black", linestyle="--")
    figure.tight_layout()
    figure.savefig(report_dir / "gate1_drift_by_depth.png", dpi=180)
    plt.close(figure)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
