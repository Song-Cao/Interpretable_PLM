#!/usr/bin/env python3
"""Complete IRED v20 domain, long-range, atlas, and subtype analyses on CPU."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossfit.datasets import load_assay
from crossfit.feature_atlas import (
    activation_weighted_profiles,
    empirical_reference_qvalues,
    feature_seed_replication,
    paired_label_permutation,
    select_profile_subtypes,
    subtype_effect_test,
)
from crossfit.patching_analysis import spatial_effect_correlations
from crossfit.semantic_enrichment import (
    benjamini_hochberg_adjusted,
    blocked_class_permutation,
    profile_enrichment,
)
from crossfit.utils import REPO_ROOT, load_config


def _load_inputs(config: dict) -> tuple[list[int], list[np.ndarray], list[pd.DataFrame]]:
    seeds = list(map(int, load_config(REPO_ROOT / "Code/config/step7.yaml")["training"]["seeds"]))
    cache = REPO_ROOT / config["paths"]["delta_z_cache"] / "v18_semantic_profiles"
    taxonomy = REPO_ROOT / "Reports/step6/v17_pooled_null"
    profiles = [np.load(cache / f"w1_seed{seed}_residue_mean.npy") for seed in seeds]
    taxonomies = [pd.read_csv(taxonomy / f"taxonomy_v17_w1_seed{seed}.csv") for seed in seeds]
    for seed, profile, table in zip(seeds, profiles, taxonomies, strict=True):
        if profile.shape != (290, 960) or len(table) != 960:
            raise AssertionError(f"seed {seed}: residue profile and taxonomy are not aligned")
    return seeds, profiles, taxonomies


def _annotation_tracks(config: dict, partition: pd.DataFrame) -> dict[str, np.ndarray]:
    tracks: dict[str, np.ndarray] = {}
    categorical = {
        "domain": "structural_domain",
        "interface": "dimer_interface",
        "rsa_class": "core_surface",
        "secondary_structure": "secondary_structure",
        "shell": "shell",
    }
    for prefix, column in categorical.items():
        values = partition[column].astype(str).to_numpy()
        for category in sorted(np.unique(values)):
            tracks[f"{prefix}_{category}"] = (values == category).astype(float)
    tracks["relative_sasa"] = partition["mean_relative_sasa"].to_numpy(float)
    tracks["crystallographic_NADP_contact"] = partition[
        "crystallographic_NADP_contact"
    ].to_numpy(float)
    tracks["crystallographic_ketone_analogue_contact"] = partition[
        "crystallographic_ketone_analogue_contact"
    ].to_numpy(float)
    annotations = pd.read_csv(REPO_ROOT / "Reports/step8/ired_annotation_tracks.csv")
    for name in config["annotation_tracks"]:
        mask = np.zeros(len(partition), dtype=float)
        positions = annotations.loc[annotations["track"] == name, "campaign_position"].astype(int)
        mask[positions.to_numpy() - 1] = 1.0
        tracks[name] = mask
    return tracks


def _class_bootstrap(
    score_sets: list[np.ndarray], labels: list[np.ndarray], *, target: str,
    comparison: str, samples: int, seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    prepared = []
    for scores, groups in zip(score_sets, labels, strict=True):
        keep = (groups == target) | (groups == comparison)
        values = scores[keep]
        scale = values.std(axis=0, ddof=1)
        values = (values - values.mean(axis=0)) / np.where(scale > 0, scale, 1.0)
        prepared.append((values, groups[keep]))
    rng = np.random.default_rng(seed)
    draws = np.empty((samples, score_sets[0].shape[1]), dtype=float)
    for draw in range(samples):
        seed_effects = []
        for _ in prepared:
            values, groups = prepared[int(rng.integers(0, len(prepared)))]
            left = values[groups == target]
            right = values[groups == comparison]
            seed_effects.append(
                left[rng.integers(0, len(left), len(left))].mean(axis=0)
                - right[rng.integers(0, len(right), len(right))].mean(axis=0)
            )
        draws[draw] = np.mean(seed_effects, axis=0)
    return np.quantile(draws, 0.025, axis=0), np.quantile(draws, 0.975, axis=0)


def _domain_semantics(
    config: dict, profiles: list[np.ndarray], labels: list[np.ndarray],
    tracks: dict[str, np.ndarray], seeds: list[int], report: Path,
) -> pd.DataFrame:
    names = [
        "domain_cofactor_binding_domain", "domain_substrate_catalytic_region",
        "domain_unassigned", "interface_True", "interface_False",
        "rsa_class_core", "rsa_class_surface",
    ]
    masks = {name: tracks[name].astype(bool) for name in names}
    scores = [profile_enrichment(profile, masks)[1] for profile in profiles]
    target, comparison = "rotated", str(config["semantic_test"]["comparison_class"])
    observed, pvalues, by_seed = blocked_class_permutation(
        scores, labels, target=target, comparison=comparison,
        permutations=int(config["v20_analysis"]["permutations"]),
        seed=int(config["v20_analysis"]["permutation_seed"]),
    )
    lower, upper = _class_bootstrap(
        scores, labels, target=target, comparison=comparison,
        samples=int(config["v20_analysis"]["bootstrap_samples"]),
        seed=int(config["v20_analysis"]["bootstrap_seed"]),
    )
    adjusted = benjamini_hochberg_adjusted(pvalues)
    rows = []
    for index, name in enumerate(names):
        row = {
            "track": name, "target_class": target, "comparison_class": comparison,
            "position_count": int(masks[name].sum()),
            "mean_standardized_effect": float(observed[index]),
            "bootstrap_ci_lower": float(lower[index]),
            "bootstrap_ci_upper": float(upper[index]),
            "permutation_p_value": float(pvalues[index]),
            "fdr_q_value": float(adjusted[index]),
            "fdr_significant_q05": bool(adjusted[index] <= config["v20_analysis"]["fdr_q"]),
        }
        row.update({f"effect_seed_{seed}": float(by_seed[s, index]) for s, seed in enumerate(seeds)})
        rows.append(row)
    output = pd.DataFrame(rows)
    output.to_csv(report / "v20_domain_class_enrichment.csv", index=False)
    return output


def _domain_causal_linkage(
    config: dict, profile: np.ndarray, taxonomy: pd.DataFrame,
    tracks: dict[str, np.ndarray], report: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    effects = pd.read_csv(report / "v18_causal_patching_feature_effects.csv")
    rotated = taxonomy.loc[taxonomy["provisional_class"] == "rotated", "feature"].astype(int)
    effects = effects.loc[effects["feature"].isin(rotated)].sort_values("feature")
    feature_ids = effects["feature"].to_numpy(int)
    category_names = [
        "domain_cofactor_binding_domain", "domain_substrate_catalytic_region",
        "domain_unassigned", "interface_True", "interface_False",
        "rsa_class_core", "rsa_class_surface",
    ]
    mass = profile[:, feature_ids]
    total = mass.sum(axis=0)
    fractions = {
        name: (mass * tracks[name][:, None]).sum(axis=0) / total for name in category_names
    }
    rows = spatial_effect_correlations(
        effects["mean_absolute_standardized_effect"].to_numpy(float), fractions,
        permutations=int(config["v20_analysis"]["permutations"]),
        seed=int(config["v20_analysis"]["permutation_seed"]) + 1,
        fdr_q=float(config["v20_analysis"]["fdr_q"]),
    )
    linkage = pd.DataFrame(rows).rename(columns={"shell": "domain_track"})
    linkage.to_csv(report / "v20_causal_domain_fraction_correlations.csv", index=False)
    fraction_table = effects[["feature", "mean_absolute_standardized_effect"]].copy()
    for name, values in fractions.items():
        fraction_table[f"fraction_{name}"] = values
    fraction_table.to_csv(report / "v20_causal_feature_domain_fractions.csv", index=False)
    return linkage, fraction_table


def _parse_positions(value: object) -> tuple[int, ...]:
    return tuple(int(item) for item in str(value).split(";") if item and str(item) != "nan")


def _long_range_response(
    config: dict, taxonomy: pd.DataFrame, partition: pd.DataFrame,
    tracks: dict[str, np.ndarray], report: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cache = REPO_ROOT / config["paths"]["delta_z_cache"]
    delta = sparse.load_npz(cache / "w1/delta_z.npz").tocsr()
    rows = np.load(cache / "variant_rows.npy").astype(int)
    variants = load_assay("IRED", with_sequences=True, include_reserve=True).set_index("variant_row").loc[rows]
    length = len(partition)
    if delta.shape[0] != len(variants) * length:
        raise AssertionError("delta-z rows are not variant-by-position aligned")
    inside = np.zeros(length, dtype=bool)
    for name in config["v20_analysis"]["long_range_inside_tracks"]:
        inside |= tracks[name].astype(bool)
    outside = partition["shell"].astype(str).to_numpy() == str(
        config["v20_analysis"]["long_range_comparison_shell"]
    )
    eligible_shells = set(map(str, config["v20_analysis"]["long_range_eligible_mutation_shells"]))
    shell = partition["shell"].astype(str).to_numpy()
    position_sets = [_parse_positions(value) for value in variants["mutated_positions"]]
    eligible = np.asarray([
        bool(items) and all(shell[position - 1] in eligible_shells for position in items)
        for items in position_sets
    ])
    feature_sets = {
        "all_live": taxonomy.loc[taxonomy["provisional_class"] != "dead", "feature"].to_numpy(int),
        "rotated": taxonomy.loc[taxonomy["provisional_class"] == "rotated", "feature"].to_numpy(int),
        "conserved": taxonomy.loc[taxonomy["provisional_class"] == "conserved", "feature"].to_numpy(int),
    }
    per_variant = []
    inside_positions, outside_positions = np.flatnonzero(inside), np.flatnonzero(outside)
    for variant_index in np.flatnonzero(eligible):
        for feature_set, feature_ids in feature_sets.items():
            inside_rows = variant_index * length + inside_positions
            outside_rows = variant_index * length + outside_positions
            inside_mean = float(abs(delta[inside_rows][:, feature_ids]).sum()) / (
                len(inside_rows) * len(feature_ids)
            )
            outside_mean = float(abs(delta[outside_rows][:, feature_ids]).sum()) / (
                len(outside_rows) * len(feature_ids)
            )
            per_variant.append({
                "variant_row": int(rows[variant_index]),
                "mutations": str(variants.iloc[variant_index]["mutations"]),
                "mutated_positions": str(variants.iloc[variant_index]["mutated_positions"]),
                "feature_set": feature_set, "inside_mean_abs_delta_z": inside_mean,
                "distal_mean_abs_delta_z": outside_mean,
                "paired_difference": inside_mean - outside_mean,
            })
    detail = pd.DataFrame(per_variant)
    detail.to_csv(report / "v20_long_range_response_by_variant.csv", index=False)
    summaries = []
    for index, (name, group) in enumerate(detail.groupby("feature_set", sort=False)):
        result = paired_label_permutation(
            group["inside_mean_abs_delta_z"].to_numpy(float),
            group["distal_mean_abs_delta_z"].to_numpy(float),
            permutations=int(config["v20_analysis"]["permutations"]),
            seed=int(config["v20_analysis"]["permutation_seed"]) + 10 + index,
            bootstrap_samples=int(config["v20_analysis"]["bootstrap_samples"]),
            bootstrap_seed=int(config["v20_analysis"]["bootstrap_seed"]) + 10 + index,
        )
        result.update({
            "feature_set": name, "inside_position_count": int(inside.sum()),
            "distal_position_count": int(outside.sum()),
            "eligible_variant_rule": "all substitutions in distal-functional or generic-distal shells",
        })
        summaries.append(result)
    summary = pd.DataFrame(summaries)
    summary["fdr_q_value"] = benjamini_hochberg_adjusted(summary["permutation_p_value"].to_numpy())
    summary.to_csv(report / "v20_long_range_response.csv", index=False)
    return summary, detail


def _backfill_mutation_group_records(
    config: dict, taxonomy: pd.DataFrame, partition: pd.DataFrame,
    report: Path,
) -> pd.DataFrame:
    cache = REPO_ROOT / config["paths"]["delta_z_cache"]
    delta = sparse.load_npz(cache / "w1/delta_z.npz").tocsr()
    rows = np.load(cache / "variant_rows.npy").astype(int)
    variants = load_assay("IRED", with_sequences=True, include_reserve=True).set_index("variant_row").loc[rows]
    shell = partition["shell"].astype(str).to_numpy()
    allowed = {"distal_functional_network", "generic_distal"}
    position_sets = [_parse_positions(value) for value in variants["mutated_positions"]]
    groups: dict[tuple[int, ...], list[int]] = {}
    for index, items in enumerate(position_sets):
        if items and all(shell[position - 1] in allowed for position in items):
            groups.setdefault(items, []).append(index)
    track_masks = {
        f"shell_{name}": shell == name for name in (
            "catalytic", "first_shell", "second_shell", "distal_functional_network",
            "generic_distal",
        )
    }
    labels = taxonomy["provisional_class"].astype(str).to_numpy()
    keep = (labels == "rotated") | (labels == "conserved")
    selected_labels = labels[keep]
    length = len(shell)
    positive = np.abs(delta.data)
    epsilon = (float(positive.mean()) if len(positive) else 1.0) * 1e-6
    output = []
    for group_id, (positions, indices) in enumerate(sorted(groups.items())):
        all_rows = np.concatenate([
            np.arange(index * length, (index + 1) * length) for index in indices
        ])
        total = np.asarray(np.abs(delta[all_rows]).mean(axis=0)).ravel()
        scores = []
        for mask in track_masks.values():
            inside_rows = np.concatenate([index * length + np.flatnonzero(mask) for index in indices])
            inside = np.asarray(np.abs(delta[inside_rows]).mean(axis=0)).ravel()
            outside_count = len(all_rows) - len(inside_rows)
            outside = (total * len(all_rows) - inside * len(inside_rows)) / outside_count
            scores.append(np.log2((inside + epsilon) / (outside + epsilon)))
        values = np.stack(scores, axis=1)[keep]
        scale = values.std(axis=0, ddof=1)
        values = (values - values.mean(axis=0)) / np.where(scale > 0, scale, 1.0)
        effect = values[selected_labels == "rotated"].mean(axis=0) - values[
            selected_labels == "conserved"
        ].mean(axis=0)
        for track, value in zip(track_masks, effect, strict=True):
            output.append({
                "mutated_position_group_id": group_id,
                "mutated_positions": ";".join(map(str, positions)),
                "variants_in_group": len(indices), "track": track,
                "rotated_minus_conserved_standardized_effect": float(value),
            })
    frame = pd.DataFrame(output)
    frame.to_csv(report / "v20_mutation_responsive_group_effects.csv", index=False)
    return frame


def _atlas(
    config: dict, seeds: list[int], profiles: list[np.ndarray],
    taxonomies: list[pd.DataFrame], tracks: dict[str, np.ndarray],
    partition: pd.DataFrame, report: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    labels = [table["provisional_class"].astype(str).to_numpy() for table in taxonomies]
    live_masks = [group != "dead" for group in labels]
    primary_ids = np.flatnonzero(live_masks[0])
    primary_profile = profiles[0][:, primary_ids]
    positive_mass = primary_profile.sum(axis=0) > 0
    track_names, defined_values = activation_weighted_profiles(primary_profile[:, positive_mass], tracks)
    values = np.full((len(primary_ids), len(track_names)), np.nan, dtype=float)
    values[positive_mass] = defined_values
    defined_p, defined_q = empirical_reference_qvalues(
        defined_values, labels[0][primary_ids][positive_mass],
        reference_class=str(config["v20_analysis"]["atlas_reference_class"]),
    )
    pvalues = np.ones_like(values)
    qvalues = np.ones_like(values)
    pvalues[positive_mass] = defined_p
    qvalues[positive_mass] = defined_q
    replication, match_rows = feature_seed_replication(
        profiles, labels, live_masks=live_masks,
        minimum_cosine=float(config["v20_analysis"]["feature_match_min_cosine"]),
    )
    match_table = pd.DataFrame(match_rows)
    match_table["matched_seed"] = match_table["seed_index"].map(
        {index: seed for index, seed in enumerate(seeds)}
    )
    match_table.to_csv(report / "v20_feature_seed_matches.csv", index=False)
    taxonomy = taxonomies[0].set_index("feature").loc[primary_ids].reset_index()
    atlas = taxonomy[[
        "feature", "provisional_class", "evo_decoder_norm", "fit_decoder_norm", "delta",
        "decoder_cosine", "activation_frequency",
    ]].rename(columns={"provisional_class": "outcome_class"})
    atlas.insert(1, "dictionary_width", 1)
    atlas.insert(2, "crosscoder_seed", seeds[0])
    atlas["class_seed_replication"] = replication
    atlas["accepted_seed_matches"] = match_table.groupby("primary_feature")[
        "accepted_match"
    ].sum().reindex(primary_ids, fill_value=0).to_numpy() + 1
    atlas["annotation_profile_defined"] = positive_mass
    top_k = int(config["v20_analysis"]["atlas_top_positions"])
    top_positions = []
    for column in range(primary_profile.shape[1]):
        if not positive_mass[column]:
            top_positions.append("")
            continue
        order = np.argsort(primary_profile[:, column], kind="stable")[-top_k:][::-1]
        top_positions.append(";".join(
            f"{position + 1}:{partition.iloc[position]['campaign_residue']}:"
            f"{partition.iloc[position]['structural_domain']}:{partition.iloc[position]['shell']}"
            for position in order
        ))
    atlas["top_activating_positions"] = top_positions
    for column, name in enumerate(track_names):
        atlas[f"annotation_{name}_weighted_mean"] = values[:, column]
        atlas[f"annotation_{name}_p"] = pvalues[:, column]
        atlas[f"annotation_{name}_q"] = qvalues[:, column]
    causal = pd.read_csv(report / "v18_causal_patching_feature_effects.csv").rename(columns={
        "bootstrap_ci_lower": "causal_signed_ci_lower",
        "bootstrap_ci_upper": "causal_signed_ci_upper",
    })
    atlas = atlas.merge(causal[[
        "feature", "mean_standardized_effect", "mean_absolute_standardized_effect",
        "causal_signed_ci_lower", "causal_signed_ci_upper",
    ]], on="feature", how="left")
    atlas = atlas.rename(columns={
        "mean_standardized_effect": "causal_mean_signed_effect",
        "mean_absolute_standardized_effect": "causal_mean_absolute_effect",
    })

    rotated_mask = atlas["outcome_class"].to_numpy() == "rotated"
    if not np.all(positive_mass[rotated_mask]):
        raise AssertionError("a rotated feature has no activation mass under registered semantics")
    rotated_values = values[rotated_mask]
    subtype, selection_rows, selected_k = select_profile_subtypes(
        rotated_values,
        candidates=config["v20_analysis"]["subtype_cluster_candidates"],
        seed=int(config["v20_analysis"]["subtype_cluster_seed"]),
        n_init=int(config["v20_analysis"]["subtype_cluster_n_init"]),
    )
    atlas["rotated_subtype"] = pd.Series(pd.NA, index=atlas.index, dtype="Int64")
    atlas.loc[rotated_mask, "rotated_subtype"] = subtype
    pd.DataFrame(selection_rows).assign(selected=lambda x: x["clusters"] == selected_k).to_csv(
        report / "v20_rotated_subtype_cluster_selection.csv", index=False
    )

    rng = np.random.default_rng(int(config["v20_analysis"]["permutation_seed"]) + 30)
    enrichment_rows = []
    permutations = int(config["v20_analysis"]["permutations"])
    for cluster in range(selected_k):
        member = subtype == cluster
        for column, name in enumerate(track_names):
            observed = float(rotated_values[member, column].mean() - rotated_values[~member, column].mean())
            exceed = 0
            for _ in range(permutations):
                shuffled = rng.permutation(member)
                null = rotated_values[shuffled, column].mean() - rotated_values[~shuffled, column].mean()
                exceed += abs(null) >= abs(observed)
            enrichment_rows.append({
                "subtype": cluster, "track": name, "features": int(member.sum()),
                "mean_profile": float(rotated_values[member, column].mean()),
                "difference_from_other_rotated": observed,
                "permutation_p_value": (exceed + 1.0) / (permutations + 1.0),
            })
    subtype_profiles = pd.DataFrame(enrichment_rows)
    subtype_profiles["fdr_q_value"] = benjamini_hochberg_adjusted(
        subtype_profiles["permutation_p_value"].to_numpy()
    )
    subtype_profiles.to_csv(report / "v20_rotated_subtype_annotation_profiles.csv", index=False)

    rotated_atlas = atlas.loc[rotated_mask].copy()
    subtype_summary_rows = []
    for cluster, group in rotated_atlas.groupby("rotated_subtype", sort=True):
        effects = group["causal_mean_absolute_effect"].dropna().to_numpy(float)
        significant = subtype_profiles.loc[
            (subtype_profiles["subtype"] == int(cluster))
            & (subtype_profiles["fdr_q_value"] <= float(config["v20_analysis"]["fdr_q"]))
        ].sort_values("fdr_q_value")
        subtype_summary_rows.append({
            "subtype": int(cluster), "features": len(group),
            "features_with_causal_effect": len(effects),
            "causal_mean_absolute_effect_mean": float(effects.mean()) if len(effects) else np.nan,
            "causal_mean_absolute_effect_median": float(np.median(effects)) if len(effects) else np.nan,
            "enriched_tracks_q05": ";".join(significant["track"].astype(str)),
            "cofactor_domain_occupancy_mean": float(group[
                "annotation_domain_cofactor_binding_domain_weighted_mean"
            ].mean()),
            "substrate_domain_occupancy_mean": float(group[
                "annotation_domain_substrate_catalytic_region_weighted_mean"
            ].mean()),
            "interface_occupancy_mean": float(group[
                "annotation_interface_True_weighted_mean"
            ].mean()),
            "core_occupancy_mean": float(group[
                "annotation_rsa_class_core_weighted_mean"
            ].mean()),
        })
    subtype_summary = pd.DataFrame(subtype_summary_rows)
    subtype_summary.to_csv(report / "v20_rotated_subtype_summary.csv", index=False)
    effect_test = subtype_effect_test(
        rotated_atlas["causal_mean_absolute_effect"].to_numpy(float),
        rotated_atlas["rotated_subtype"].to_numpy(dtype=int),
        permutations=permutations,
        seed=int(config["v20_analysis"]["permutation_seed"]) + 31,
    )
    (report / "v20_rotated_subtype_causal_test.json").write_text(
        json.dumps(effect_test, indent=2) + "\n"
    )
    atlas.to_csv(report / "v20_ired_feature_atlas.csv", index=False)
    return atlas, subtype_summary, subtype_profiles, effect_test


def main() -> int:
    started = time.monotonic()
    config = load_config(REPO_ROOT / "Code/config/step8.yaml")
    report = REPO_ROOT / config["paths"]["reports"]
    report.mkdir(parents=True, exist_ok=True)
    partition = pd.read_csv(report / "ired_domain_partition.csv")
    if partition["campaign_position"].tolist() != list(range(1, 291)):
        raise AssertionError("domain partition is not in campaign-position order")
    seeds, profiles, taxonomies = _load_inputs(config)
    labels = [table["provisional_class"].astype(str).to_numpy() for table in taxonomies]
    tracks = _annotation_tracks(config, partition)
    domain = _domain_semantics(config, profiles, labels, tracks, seeds, report)
    linkage, _ = _domain_causal_linkage(config, profiles[0], taxonomies[0], tracks, report)
    long_range, detail = _long_range_response(config, taxonomies[0], partition, tracks, report)
    backfill = _backfill_mutation_group_records(config, taxonomies[0], partition, report)
    atlas, subtypes, _, causal_test = _atlas(
        config, seeds, profiles, taxonomies, tracks, partition, report
    )
    manifest = {
        "status": "complete_except_oracles", "dataset": "IRED", "analysis_layer": 26,
        "crosscoder_width": 1, "crosscoder_seeds": seeds,
        "domain_tests": len(domain), "domain_linkage_tests": len(linkage),
        "long_range_variants": int(detail["variant_row"].nunique()),
        "mutation_background_records_backfilled": len(backfill),
        "atlas_live_features": len(atlas), "rotated_subtypes": len(subtypes),
        "subtype_causal_test": causal_test,
        "cpu_wall_hours": (time.monotonic() - started) / 3600.0,
        "gpu_hours": 0.0,
        "oracle_status": "pending separate validated CPU inference",
    }
    (report / "v20_cpu_analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
