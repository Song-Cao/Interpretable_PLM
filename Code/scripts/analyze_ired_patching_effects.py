#!/usr/bin/env python3
"""Summarize IRED patching distributions and spatial-causal associations."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossfit.patching_analysis import quantile_rows, spatial_effect_correlations
from crossfit.utils import REPO_ROOT, load_config


def main() -> int:
    config = load_config(REPO_ROOT / "Code/config/step8.yaml")
    analysis = config["patching_effect_analysis"]
    report = REPO_ROOT / config["paths"]["reports"]
    effects = pd.read_csv(report / "v18_causal_patching_variant_effects.csv")
    features = pd.read_csv(report / "v18_causal_patching_feature_effects.csv")
    controls = pd.read_csv(report / "v18_causal_patching_control_effects.csv")
    quantiles = list(map(float, analysis["quantiles"]))

    rows = []
    rows.extend(quantile_rows(
        features["mean_standardized_effect"], quantiles,
        population="rotated_feature_mean_signed_effect", sampling_unit="feature",
    ))
    rows.extend(quantile_rows(
        features["mean_absolute_standardized_effect"], quantiles,
        population="rotated_feature_mean_absolute_effect", sampling_unit="feature",
    ))
    rows.extend(quantile_rows(
        effects["standardized_effect"].abs(), quantiles,
        population="rotated_candidate_absolute_effect", sampling_unit="feature_variant",
    ))
    for control in ("dead_feature", "random_live_feature"):
        selected = controls.loc[controls["control"] == control, "standardized_effect"]
        rows.extend(quantile_rows(
            selected, quantiles, population=f"{control}_signed_effect",
            sampling_unit="variant",
        ))
        rows.extend(quantile_rows(
            selected.abs(), quantiles, population=f"{control}_absolute_effect",
            sampling_unit="variant",
        ))
    pd.DataFrame(rows).to_csv(report / "v18_causal_patching_effect_quantiles.csv", index=False)

    features.nlargest(
        int(analysis["top_feature_count"]), "mean_absolute_standardized_effect"
    ).to_csv(report / "v18_causal_patching_top_features.csv", index=False)

    seed = int(config["primary_crosscoder_seed"])
    profile_path = (
        REPO_ROOT / config["paths"]["delta_z_cache"] / "v18_semantic_profiles"
        / f"w1_seed{seed}_residue_mean.npy"
    )
    profile = np.load(profile_path)
    shells = pd.read_csv(REPO_ROOT / "Reports/step8/ired_functional_shells.csv")
    shell_labels = shells["shell"].astype(str).to_numpy()
    masks = {
        "catalytic": shell_labels == "catalytic",
        "first_shell": shell_labels == "first_shell",
        "second_shell": shell_labels == "second_shell",
        "distal": np.isin(
            shell_labels, ["distal_functional_network", "generic_distal"]
        ),
    }
    feature_ids = features["feature"].astype(int).to_numpy()
    shell_profiles = {
        name: profile[mask][:, feature_ids].mean(axis=0) for name, mask in masks.items()
    }
    linked = features[[
        "feature", "mean_standardized_effect", "mean_absolute_standardized_effect",
    ]].copy()
    for name, values in shell_profiles.items():
        linked[f"mean_activation_{name}"] = values
    linked.to_csv(report / "v18_causal_feature_shell_profiles.csv", index=False)

    correlations = spatial_effect_correlations(
        features["mean_absolute_standardized_effect"].to_numpy(float),
        shell_profiles,
        permutations=int(analysis["permutations"]),
        seed=int(analysis["permutation_seed"]),
        fdr_q=float(analysis["fdr_q"]),
    )
    for row in correlations:
        row["position_count"] = int(masks[str(row["shell"])].sum())
        row["effect_definition"] = "mean absolute standardized prediction effect"
        row["profile_definition"] = "mean static latent activation over shell positions"
    pd.DataFrame(correlations).to_csv(
        report / "v18_causal_spatial_effect_correlations.csv", index=False
    )
    manifest = {
        "status": "complete",
        "dataset": "IRED",
        "representation_layer": int(config["analysis_layer"]),
        "crosscoder_seed": seed,
        "features": int(len(features)),
        "candidate_feature_variant_effects": int(len(effects)),
        "control_variant_effects": int(len(controls)),
        "residue_profile_path": str(profile_path.relative_to(REPO_ROOT)),
        "profile_tensor_committed": False,
        "permutations": int(analysis["permutations"]),
        "permutation_seed": int(analysis["permutation_seed"]),
        "fdr_q": float(analysis["fdr_q"]),
    }
    (report / "v18_causal_effect_analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(pd.DataFrame(correlations).to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
