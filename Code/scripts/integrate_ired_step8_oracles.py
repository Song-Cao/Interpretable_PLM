#!/usr/bin/env python3
"""Validate IRED CPU oracles and add their feature correlations to the atlas."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossfit.datasets import load_assay
from crossfit.utils import REPO_ROOT, load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_mutations(value: str) -> list[tuple[str, int, str]]:
    output = []
    for item in str(value).replace("-", ":").split(":"):
        item = item.strip()
        if not item or item == "WT":
            continue
        output.append((item[0], int(item[1:-1]), item[-1]))
    return output


def _thermompnn(
    raw_path: Path, variants: pd.DataFrame, report: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(raw_path)
    raw = raw.loc[(raw["position"].between(0, 288)) & (raw["wildtype"] != raw["mutation"])].copy()
    raw["campaign_position"] = raw["position"].astype(int) + 2
    raw["mutation_id"] = (
        raw["wildtype"].astype(str) + raw["campaign_position"].astype(str) + raw["mutation"].astype(str)
    )
    single = raw[["mutation_id", "campaign_position", "wildtype", "mutation", "ddG_pred"]].rename(
        columns={"ddG_pred": "thermompnn_ddg"}
    ).sort_values(["campaign_position", "mutation"])
    single.to_csv(report / "v20_thermompnn_single_mutation_scores.csv", index=False)
    lookup = {
        (str(row.wildtype), int(row.campaign_position), str(row.mutation)): float(row.thermompnn_ddg)
        for row in single.itertuples()
    }
    scores = []
    for row in variants.itertuples():
        mutations = _parse_mutations(str(row.mutations))
        missing = [f"{left}{position}{right}" for left, position, right in mutations if (
            left, position, right
        ) not in lookup]
        scores.append({
            "variant_row": int(row.variant_row), "mutations": str(row.mutations),
            "thermompnn_ddg_additive": (
                float(sum(lookup[item] for item in mutations)) if not missing else np.nan
            ),
            "thermompnn_supported": not missing,
            "unsupported_mutations": ";".join(missing),
        })
    variant_scores = pd.DataFrame(scores)
    variant_scores.to_csv(report / "v20_thermompnn_variant_scores.csv", index=False)
    expected = {
        "T241A": "stabilizing", "S68P": "destabilizing", "P203Q": "destabilizing",
        "H244Y": "destabilizing", "A72T": "destabilizing", "G37C": "destabilizing",
    }
    anchors = []
    for mutation, direction in expected.items():
        row = single.loc[single["mutation_id"] == mutation]
        if len(row) != 1:
            raise AssertionError(f"ThermoMPNN anchor {mutation} is missing or duplicated")
        value = float(row.iloc[0]["thermompnn_ddg"])
        observed = "stabilizing" if value < 0 else "destabilizing"
        anchors.append({
            "mutation": mutation, "experimental_direction": direction,
            "thermompnn_ddg": value, "predicted_direction": observed,
            "sign_agrees": observed == direction,
        })
    validation = pd.DataFrame(anchors)
    validation.to_csv(report / "v20_thermompnn_anchor_validation.csv", index=False)
    return single, variant_scores, validation


def _load_soluprot(path: Path, variants: pd.DataFrame, report: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw["variant_row"] = raw["fa_id"].astype(str).str.replace("variant_", "", regex=False).astype(int)
    output = variants[["variant_row", "mutations"]].merge(
        raw[["variant_row", "soluble"]], on="variant_row", how="left", validate="one_to_one"
    ).rename(columns={"soluble": "soluprot_score"})
    if output["soluprot_score"].isna().any():
        raise AssertionError("SoluProt output does not cover every curated campaign variant")
    output.to_csv(report / "v20_soluprot_variant_scores.csv", index=False)
    return output


def _pooled_phi(config: dict, variant_rows: np.ndarray) -> np.ndarray:
    cache = REPO_ROOT / config["paths"]["delta_z_cache"]
    delta = sparse.load_npz(cache / "w1/delta_z.npz").tocsr()
    wild_type = np.load(cache / "w1/wild_type_codes.npy").astype(np.float64)
    length, features = wild_type.shape
    if delta.shape != (len(variant_rows) * length, features):
        raise AssertionError("delta-z cache is not aligned to the registered variant rows")
    pooled = np.empty((len(variant_rows), features), dtype=np.float64)
    baseline = wild_type.sum(axis=0)
    for index in range(len(variant_rows)):
        pooled[index] = baseline + np.asarray(
            delta[index * length : (index + 1) * length].sum(axis=0)
        ).ravel()
    return pooled


def _rank_pair_bootstrap(
    x: np.ndarray, y: np.ndarray, *, samples: int, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bootstrap Spearman correlations by resampling pre-ranked variant pairs."""
    values = np.asarray(x, dtype=float)
    target = np.asarray(y, dtype=float)
    keep = np.isfinite(target)
    values, target = values[keep], target[keep]
    xrank = np.apply_along_axis(rankdata, 0, values).astype(np.float32)
    yrank = rankdata(target).astype(np.float32)
    observed = np.asarray([
        spearmanr(values[:, column], target).statistic for column in range(values.shape[1])
    ])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(target), size=(samples, len(target)))
    draws = np.empty((samples, values.shape[1]), dtype=np.float32)
    sampled_y = yrank[indices]
    y_center = sampled_y - sampled_y.mean(axis=1, keepdims=True)
    y_scale = np.sqrt((y_center * y_center).sum(axis=1))
    for column in range(values.shape[1]):
        sampled_x = xrank[indices, column]
        x_center = sampled_x - sampled_x.mean(axis=1, keepdims=True)
        denominator = np.sqrt((x_center * x_center).sum(axis=1)) * y_scale
        draws[:, column] = (x_center * y_center).sum(axis=1) / np.maximum(
            denominator, np.finfo(np.float32).tiny
        )
    return observed, np.quantile(draws, 0.025, axis=0), np.quantile(draws, 0.975, axis=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thermompnn-raw", type=Path, required=True)
    parser.add_argument("--soluprot", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(REPO_ROOT / "Code/config/step8.yaml")
    report = REPO_ROOT / config["paths"]["reports"]
    variants = load_assay("IRED", with_sequences=True, include_reserve=True)
    rows = np.load(REPO_ROOT / config["paths"]["delta_z_cache"] / "variant_rows.npy").astype(int)
    variants = variants.set_index("variant_row").loc[rows].reset_index()
    _, thermo, anchors = _thermompnn(args.thermompnn_raw, variants, report)
    soluprot = _load_soluprot(args.soluprot, variants, report)
    oracle = thermo.merge(soluprot, on=["variant_row", "mutations"], validate="one_to_one")
    oracle.to_csv(report / "v20_oracle_variant_scores.csv", index=False)
    supported = oracle["thermompnn_supported"].to_numpy(bool)
    agreement = pd.DataFrame([{
        "oracle_a": "ThermoMPNN_additive_ddG", "oracle_b": "SoluProt_exploratory",
        "variants": int(supported.sum()),
        "spearman_rho": float(spearmanr(
            oracle.loc[supported, "thermompnn_ddg_additive"],
            oracle.loc[supported, "soluprot_score"],
        ).statistic),
    }])
    agreement.to_csv(report / "v20_oracle_agreement_matrix.csv", index=False)

    atlas_path = report / "v20_ired_feature_atlas.csv"
    atlas = pd.read_csv(atlas_path)
    feature_ids = atlas["feature"].to_numpy(int)
    phi = _pooled_phi(config, rows)[:, feature_ids]
    samples = int(config["v20_analysis"]["oracle_bootstrap_samples"])
    seed = int(config["v20_analysis"]["oracle_bootstrap_seed"])
    for index, (name, scores) in enumerate((
        ("thermompnn", oracle["thermompnn_ddg_additive"].to_numpy(float)),
        ("soluprot", oracle["soluprot_score"].to_numpy(float)),
    )):
        point, lower, upper = _rank_pair_bootstrap(
            phi, scores, samples=samples, seed=seed + index
        )
        atlas[f"oracle_{name}_spearman"] = point
        atlas[f"oracle_{name}_bootstrap_ci_lower"] = lower
        atlas[f"oracle_{name}_bootstrap_ci_upper"] = upper
    atlas.to_csv(atlas_path, index=False)
    manifest = {
        "status": "complete", "dataset": "IRED", "analysis_layer": 26,
        "thermompnn_repository": "https://github.com/Kuhlman-Lab/ThermoMPNN",
        "thermompnn_revision": "370f76ec62bd929f7425e311d8df04a0d094990f",
        "thermompnn_raw_sha256": _sha256(args.thermompnn_raw),
        "thermompnn_aggregation": "additive sum of official point-mutation ddG scores",
        "thermompnn_anchor_signs_passed": int(anchors["sign_agrees"].sum()),
        "thermompnn_anchor_signs_total": len(anchors),
        "thermompnn_supported_variants": int(supported.sum()),
        "thermompnn_unsupported_variants": int((~supported).sum()),
        "soluprot_source": "official SoluProt 1.0.1.0 standalone no-TMHMM model",
        "soluprot_status": "exploratory_no_IRED_anchor_and_not_designed_for_point_mutation_effects",
        "soluprot_variants": len(soluprot),
        "feature_pooling": "sum over residue codes",
        "oracle_correlation_bootstrap": "variant-pair bootstrap on pre-ranked values",
        "oracle_correlation_bootstrap_samples": samples,
        "gpu_hours": 0.0,
    }
    (report / "v20_oracle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    cpu_manifest_path = report / "v20_cpu_analysis_manifest.json"
    if cpu_manifest_path.exists():
        cpu_manifest = json.loads(cpu_manifest_path.read_text())
        cpu_manifest["status"] = "complete"
        cpu_manifest["oracle_status"] = "complete"
        cpu_manifest["oracle_manifest"] = str(
            (report / "v20_oracle_manifest.json").relative_to(REPO_ROOT)
        )
        cpu_manifest_path.write_text(json.dumps(cpu_manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
