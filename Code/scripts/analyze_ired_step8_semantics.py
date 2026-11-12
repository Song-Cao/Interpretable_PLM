#!/usr/bin/env python3
"""Run IRED residue-level class semantics and write a causal-patching budget only."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossfit.crosscoder import CrossCoder
from crossfit.cache_provenance import verify_ired_esmc_cache
from crossfit.datasets import get_split, load_assay
from crossfit.semantic_enrichment import (
    benjamini_hochberg_adjusted,
    blocked_class_permutation,
    profile_enrichment,
    top_position_shell_statistic,
)
from crossfit.utils import REPO_ROOT, load_config


def _load_model(path: Path, dimension: int) -> CrossCoder:
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    model = CrossCoder(
        dimension, int(artifact["width_multiplier"]), int(artifact["nominal_k"]),
        decoder_bias=bool(artifact["decoder_bias"]),
    )
    model.load_state_dict(artifact["state_dict"])
    return model.cuda().eval()


@torch.inference_mode()
def _residue_profile(
    model: CrossCoder,
    evo: np.ndarray,
    fit: np.ndarray,
    evo_scalar: float,
    fit_scalar: float,
) -> np.ndarray:
    total = np.zeros((evo.shape[1], model.features), dtype=np.float64)
    for index in range(len(evo)):
        left = torch.as_tensor(
            np.asarray(evo[index], dtype=np.float32) * evo_scalar, device="cuda"
        )
        right = torch.as_tensor(
            np.asarray(fit[index], dtype=np.float32) * fit_scalar, device="cuda"
        )
        total += model.encode(left, right).cpu().numpy()
    return (total / len(evo)).astype(np.float32)


def _tracks(length: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    shells = pd.read_csv(REPO_ROOT / "Reports/step8/ired_functional_shells.csv")
    if shells["campaign_position"].tolist() != list(range(1, length + 1)):
        raise AssertionError("functional shell rows are not in campaign-position order")
    shell_names = shells["shell"].astype(str).to_numpy()
    claim = {
        f"shell_{name}": shell_names == name
        for name in (
            "catalytic", "first_shell", "second_shell",
            "distal_functional_network", "generic_distal",
        )
    }
    annotations = pd.read_csv(REPO_ROOT / "Reports/step8/ired_annotation_tracks.csv")
    for name in (
        "substrate_direct_contact", "NADPH_direct_contact",
        "NADPH_proximal_or_functionally_coupled",
    ):
        positions = annotations.loc[annotations["track"] == name, "campaign_position"].astype(int)
        mask = np.zeros(length, dtype=bool)
        mask[positions.to_numpy() - 1] = True
        claim[name] = mask
    contacts = pd.read_csv(REPO_ROOT / "Reports/step8/ired_crystallographic_contacts.csv")
    mask = np.zeros(length, dtype=bool)
    mask[np.unique(contacts["campaign_position"].astype(int)) - 1] = True
    claim["crystallographic_ketone_analogue_proxy"] = mask
    control_positions = annotations.loc[
        annotations["track"] == "evolutionary_peak_residue", "campaign_position"
    ].astype(int)
    control_mask = np.zeros(length, dtype=bool)
    control_mask[control_positions.to_numpy() - 1] = True
    return claim, {"evolutionary_peak_residue": control_mask}, shell_names


def _bootstrap_effects(
    scores: list[np.ndarray], labels: list[np.ndarray], target: str, comparison: str,
    *, samples: int, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    prepared = []
    for values, groups in zip(scores, labels):
        keep = (groups == target) | (groups == comparison)
        selected = values[keep]
        scale = selected.std(axis=0, ddof=1)
        selected = (selected - selected.mean(axis=0)) / np.where(scale > 0, scale, 1.0)
        prepared.append((selected, groups[keep]))
    draws = np.empty((samples, scores[0].shape[1]), dtype=float)
    for draw in range(samples):
        effects = []
        for _ in range(len(prepared)):
            values, groups = prepared[int(rng.integers(0, len(prepared)))]
            left = values[groups == target]
            right = values[groups == comparison]
            effects.append(
                left[rng.integers(0, len(left), len(left))].mean(axis=0)
                - right[rng.integers(0, len(right), len(right))].mean(axis=0)
            )
        draws[draw] = np.mean(effects, axis=0)
    return np.quantile(draws, 0.025, axis=0), np.quantile(draws, 0.975, axis=0)


def _select_target(taxonomies: list[pd.DataFrame], shifted: list[str]) -> tuple[str, pd.DataFrame]:
    rows = []
    for label in shifted:
        fractions = np.asarray([
            np.mean(frame["provisional_class"].astype(str).to_numpy() == label)
            for frame in taxonomies
        ])
        rows.append({
            "class": label, "populated_all_five_seeds": bool(np.all(fractions > 0)),
            "fraction_mean": float(fractions.mean()), "fraction_sd": float(fractions.std(ddof=1)),
            "coefficient_of_variation": (
                float(fractions.std(ddof=1) / fractions.mean()) if fractions.mean() > 0 else np.inf
            ),
        })
    frame = pd.DataFrame(rows)
    eligible = frame.loc[frame["populated_all_five_seeds"]]
    if eligible.empty:
        raise RuntimeError("no shifted class is populated in every real crosscoder seed")
    maximum = eligible["fraction_mean"].max()
    tied = eligible.loc[np.isclose(eligible["fraction_mean"], maximum)]
    target = str(tied.sort_values("coefficient_of_variation").iloc[0]["class"])
    frame["selected_target"] = frame["class"] == target
    return target, frame


def _mutation_response(
    tracks: dict[str, np.ndarray], taxonomy: pd.DataFrame, shell_names: np.ndarray,
    config: dict, target: str, comparison: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    cache = REPO_ROOT / config["paths"]["delta_z_cache"] / "w1"
    delta = sparse.load_npz(cache / "delta_z.npz").tocsr()
    rows = np.load(cache.parent / "variant_rows.npy").astype(int)
    frame = load_assay("IRED", with_sequences=True, include_reserve=True).set_index("variant_row").loc[rows]
    distal_shells = {"distal_functional_network", "generic_distal"}

    def positions(value: object) -> tuple[int, ...]:
        return tuple(int(item) for item in str(value).split(";") if item and str(item) != "nan")

    position_sets = [positions(value) for value in frame["mutated_positions"]]
    distal = np.asarray([
        bool(items) and all(shell_names[item - 1] in distal_shells for item in items)
        for items in position_sets
    ])
    groups: dict[tuple[int, ...], list[int]] = {}
    for index, (keep, items) in enumerate(zip(distal, position_sets)):
        if keep:
            groups.setdefault(items, []).append(index)
    labels = taxonomy["provisional_class"].astype(str).to_numpy()
    selected = (labels == target) | (labels == comparison)
    selected_labels = labels[selected]
    positive = np.abs(delta.data)
    epsilon = (float(positive.mean()) if len(positive) else 1.0) * 1e-6
    group_effects = []
    length = len(shell_names)
    for variants in groups.values():
        feature_scores = []
        all_rows = np.concatenate([
            np.arange(variant * length, (variant + 1) * length) for variant in variants
        ])
        total = np.asarray(np.abs(delta[all_rows]).mean(axis=0)).ravel()
        for mask in tracks.values():
            inside_rows = np.concatenate([
                variant * length + np.flatnonzero(mask) for variant in variants
            ])
            inside = np.asarray(np.abs(delta[inside_rows]).mean(axis=0)).ravel()
            outside_count = len(all_rows) - len(inside_rows)
            outside = (total * len(all_rows) - inside * len(inside_rows)) / outside_count
            feature_scores.append(np.log2((inside + epsilon) / (outside + epsilon)))
        values = np.stack(feature_scores, axis=1)[selected]
        scale = values.std(axis=0, ddof=1)
        values = (values - values.mean(axis=0)) / np.where(scale > 0, scale, 1.0)
        group_effects.append(
            values[selected_labels == target].mean(axis=0)
            - values[selected_labels == comparison].mean(axis=0)
        )
    effects = np.stack(group_effects)
    rng = np.random.default_rng(int(config["semantic_test"]["bootstrap_seed"]))
    samples = int(config["semantic_test"]["bootstrap_samples"])
    draw = np.empty((samples, effects.shape[1]), dtype=float)
    for index in range(samples):
        draw[index] = effects[rng.integers(0, len(effects), len(effects))].mean(axis=0)
    point = effects.mean(axis=0)
    pvalue = 2.0 * np.minimum((draw <= 0).mean(axis=0), (draw >= 0).mean(axis=0))
    pvalue = np.maximum(pvalue, 1.0 / samples)
    names = list(tracks)
    output = pd.DataFrame({
        "track": names, "target_class": target, "comparison_class": comparison,
        "grouped_effect_z": point, "ci_lower": np.quantile(draw, 0.025, axis=0),
        "ci_upper": np.quantile(draw, 0.975, axis=0), "p_value": pvalue,
        "fdr_q_value": benjamini_hochberg_adjusted(pvalue),
    })
    details = {
        "definition": "mean absolute mutant-minus-WT latent response",
        "distal_variant_definition": "all mutated positions in distal functional network or generic distal shell",
        "distal_variants": int(distal.sum()), "mutated_position_background_groups": len(groups),
        "resampling_unit": "exact mutated-position set; all variants sharing a set remain together",
        "crosscoder_seed": int(config["primary_crosscoder_seed"]),
    }
    return output, details


def _variant_subsample(config: dict, count: int) -> pd.DataFrame:
    frame = load_assay("IRED", with_sequences=True)
    train, _ = get_split("IRED", "pg_kfold", fold=0)
    candidates = frame.iloc[train].copy()
    candidates["fitness_quartile"] = pd.qcut(candidates["y"], 4, labels=False, duplicates="drop")
    candidates["stratum"] = (
        candidates["n_mutations"].clip(upper=5).astype(str) + "_" + candidates["fitness_quartile"].astype(str)
    )
    rng = np.random.default_rng(int(config["causal_budget"]["variant_subsample_seed"]))
    chosen = []
    groups = list(candidates.groupby("stratum", sort=True))
    base = count // len(groups)
    for _, group in groups:
        take = min(base, len(group))
        chosen.extend(rng.choice(group.index.to_numpy(), take, replace=False).tolist())
    remaining = candidates.index.difference(chosen).to_numpy()
    if len(chosen) < count:
        chosen.extend(rng.choice(remaining, count - len(chosen), replace=False).tolist())
    return candidates.loc[chosen, [
        "variant_row", "mutations", "n_mutations", "y", "fitness_quartile", "stratum",
    ]].sort_values(["stratum", "variant_row"]).reset_index(drop=True)


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("five-seed residue profiles require the GPU pod")
    started = time.monotonic()
    config = load_config(REPO_ROOT / "Code/config/step8.yaml")
    step3 = load_config(REPO_ROOT / "Code/config/step3.yaml")
    provenance = load_config(REPO_ROOT / config["paths"]["provenance_config"])
    reference_verification = verify_ired_esmc_cache(
        REPO_ROOT / config["paths"]["campaign_manifest"],
        step3=step3,
        verification=provenance["verification"],
        adapter_checkpoint=REPO_ROOT / config["paths"]["step3_adapter"],
    )
    step4 = json.loads((REPO_ROOT / config["paths"]["step4_manifest"]).read_text())
    evo_scalar = float(step4["models"]["evo"]["normalization_scalar"])
    fit_scalar = float(step4["models"]["fit"]["normalization_scalar"])
    evo = np.load(REPO_ROOT / config["paths"]["campaign_base"], mmap_mode="r")
    fit = np.load(REPO_ROOT / config["paths"]["campaign_fit"], mmap_mode="r")
    if evo.shape != fit.shape:
        raise AssertionError("campaign activations are not aligned")
    report = REPO_ROOT / config["paths"]["reports"]
    profile_cache = REPO_ROOT / config["paths"]["delta_z_cache"] / "v18_semantic_profiles"
    profile_cache.mkdir(parents=True, exist_ok=True)
    taxonomy_dir = REPO_ROOT / "Reports/step6/v17_pooled_null"
    seeds = list(map(int, load_config(REPO_ROOT / "Code/config/step7.yaml")["training"]["seeds"]))
    taxonomies = [pd.read_csv(taxonomy_dir / f"taxonomy_v17_w1_seed{seed}.csv") for seed in seeds]
    target, selection = _select_target(taxonomies, list(config["semantic_test"]["shifted_classes"]))
    comparison = str(config["semantic_test"]["comparison_class"])
    selection.to_csv(report / "v18_target_class_selection.csv", index=False)
    claim_tracks, control_tracks, shells = _tracks(evo.shape[1])

    profiles, labels, checkpoint_hashes = [], [], {}
    for seed, taxonomy in zip(seeds, taxonomies):
        checkpoint = REPO_ROOT / config["paths"]["crosscoder_checkpoints"] / f"w1_seed{seed}_nobias.pt"
        checkpoint_hashes[str(checkpoint.relative_to(REPO_ROOT))] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        path = profile_cache / f"w1_seed{seed}_residue_mean.npy"
        if path.exists():
            profile = np.load(path)
        else:
            model = _load_model(checkpoint, int(config["dimension"]))
            profile = _residue_profile(model, evo, fit, evo_scalar, fit_scalar)
            temporary = path.with_suffix(".npy.tmp")
            with temporary.open("wb") as stream:
                np.save(stream, profile)
            os.replace(temporary, path)
            del model
            torch.cuda.empty_cache()
        if profile.shape != (evo.shape[1], len(taxonomy)):
            raise AssertionError("semantic profile and taxonomy feature axes differ")
        profiles.append(profile)
        labels.append(taxonomy["provisional_class"].astype(str).to_numpy())

    score_sets = [profile_enrichment(profile, claim_tracks)[1] for profile in profiles]
    names = list(claim_tracks)
    observed, pvalues, seed_effects = blocked_class_permutation(
        score_sets, labels, target=target, comparison=comparison,
        permutations=int(config["semantic_test"]["permutations"]),
        seed=int(config["semantic_test"]["permutation_seed"]),
    )
    lower, upper = _bootstrap_effects(
        score_sets, labels, target, comparison,
        samples=int(config["semantic_test"]["bootstrap_samples"]),
        seed=int(config["semantic_test"]["bootstrap_seed"]),
    )
    rows = []
    adjusted = benjamini_hochberg_adjusted(pvalues)
    for index, name in enumerate(names):
        row = {
            "track": name, "target_class": target, "comparison_class": comparison,
            "position_count": int(np.sum(claim_tracks[name])),
            "mean_standardized_effect": observed[index], "bootstrap_ci_lower": lower[index],
            "bootstrap_ci_upper": upper[index], "permutation_p_value": pvalues[index],
            "fdr_q_value": adjusted[index], "fdr_significant_q05": adjusted[index] <= 0.05,
        }
        for seed_index, seed in enumerate(seeds):
            row[f"effect_seed_{seed}"] = seed_effects[seed_index, index]
        rows.append(row)
    semantic = pd.DataFrame(rows)
    semantic.to_csv(report / "v18_class_semantic_tests.csv", index=False)

    control_scores = [profile_enrichment(profile, control_tracks)[1] for profile in profiles]
    control_observed, control_p, control_seed = blocked_class_permutation(
        control_scores, labels, target=target, comparison=comparison,
        permutations=int(config["semantic_test"]["permutations"]),
        seed=int(config["semantic_test"]["permutation_seed"]) + 1,
    )
    pd.DataFrame([{
        "track": "evolutionary_peak_residue", "role": "positive_control_only",
        "position_count": int(np.sum(control_tracks["evolutionary_peak_residue"])),
        "target_class": target, "comparison_class": comparison,
        "mean_standardized_effect": control_observed[0], "permutation_p_value": control_p[0],
        **{f"effect_seed_{seed}": control_seed[index, 0] for index, seed in enumerate(seeds)},
    }]).to_csv(report / "v18_positive_control_test.csv", index=False)

    spatial_statistic, spatial_p, spatial_rows = top_position_shell_statistic(
        profiles, labels, shells, target=target, comparison=comparison,
        permutations=int(config["semantic_test"]["permutations"]),
        seed=int(config["semantic_test"]["permutation_seed"]) + 2,
    )
    pd.DataFrame(spatial_rows).to_csv(report / "v18_top_position_shell_counts.csv", index=False)
    spatial = {
        "target_class": target, "comparison_class": comparison,
        "statistic_mean_total_variation": spatial_statistic,
        "blocked_permutation_p_value": spatial_p,
        "top_positions_per_feature": int(config["semantic_test"]["top_activating_positions_per_feature"]),
    }
    (report / "v18_spatial_structure_test.json").write_text(json.dumps(spatial, indent=2) + "\n")

    mutation, mutation_details = _mutation_response(
        claim_tracks, taxonomies[0], shells, config, target, comparison
    )
    mutation.to_csv(report / "v18_mutation_responsive_tests.csv", index=False)
    (report / "v18_mutation_responsive_manifest.json").write_text(
        json.dumps(mutation_details, indent=2) + "\n"
    )

    primary = taxonomies[0]
    evo_median = primary["evo_decoder_norm"].median()
    fit_median = primary["fit_decoder_norm"].median()
    causal_features = primary.loc[
        (primary["provisional_class"] == target)
        & (primary["evo_decoder_norm"] >= evo_median)
        & (primary["fit_decoder_norm"] >= fit_median), "feature"
    ].astype(int)
    variant_count = int(config["causal_budget"]["proposed_variant_count"])
    variants = _variant_subsample(config, variant_count)
    variants.to_csv(report / "v18_causal_variant_subsample.csv", index=False)
    controls = list(config["causal_budget"]["controls"])
    budget = {
        "status": "planned_not_run", "target_class": target,
        "crosscoder_width": 1, "crosscoder_seed": seeds[0],
        "target_features_total": int((primary["provisional_class"] == target).sum()),
        "target_features_above_both_decoder_medians": int(len(causal_features)),
        "target_feature_ids": causal_features.tolist(), "variant_count": variant_count,
        "variant_source": "pg_kfold fold-0 training pool only",
        "variant_sampling": "fixed-seed stratification by clipped mutation count and fitness quartile",
        "control_passes": controls,
        "candidate_partial_forward_passes": int(len(causal_features) * variant_count),
        "control_partial_forward_passes": int(len(controls) * variant_count),
        "planned_partial_forward_passes_total": int((len(causal_features) + len(controls)) * variant_count),
        "baseline_forward_passes_not_counted_as_partial": variant_count,
        "patching_started": False,
        "approval_required_before_execution": True,
    }
    (report / "v18_causal_patching_budget.json").write_text(json.dumps(budget, indent=2) + "\n")
    manifest = {
        "status": "complete", "dataset": "IRED", "plan": "v18",
        "representation_layer": int(config["analysis_layer"]),
        "cache_reference_verification": reference_verification,
        "target_class": target, "comparison_class": comparison,
        "real_crosscoder_seeds": seeds, "checkpoint_sha256": checkpoint_hashes,
        "variants_in_static_residue_average": int(len(evo)),
        "permutation_unit": "feature labels within each independently initialized dictionary seed",
        "fdr_scope": "claim-eligible annotation families; positive control separate",
        "gpu_wall_hours": (time.monotonic() - started) / 3600.0,
        "causal_patching_run": False, "oracle_validation_run": False,
    }
    (report / "v18_semantic_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(semantic.to_string(index=False), flush=True)
    print(json.dumps(spatial, indent=2), flush=True)
    print(json.dumps(budget, indent=2), flush=True)
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
