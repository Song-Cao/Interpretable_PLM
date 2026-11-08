#!/usr/bin/env python3
"""Run approved error-preserving causal patches for IRED rotated features."""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossfit.adaptation import SequenceFitnessAdapter, build_lora_adapter, load_trainable_state
from crossfit.crosscoder import CrossCoder
from crossfit.datasets import get_split, load_assay
from crossfit.esmc import load_esmc
from crossfit.utils import REPO_ROOT, load_config


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _crosscoder(path: Path, dimension: int) -> CrossCoder:
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    model = CrossCoder(
        dimension, int(artifact["width_multiplier"]), int(artifact["nominal_k"]),
        decoder_bias=bool(artifact["decoder_bias"]),
    )
    model.load_state_dict(artifact["state_dict"])
    return model.cuda().eval()


def _adapter(step3: dict, dimension: int) -> SequenceFitnessAdapter:
    backbone = load_esmc(
        str(step3["backbone"]["name"]), "cuda",
        repository=str(step3["backbone"]["repository"]),
        expected_revision=str(step3["backbone"]["revision"]),
    )
    adapted = build_lora_adapter(
        backbone, rank=int(step3["adapter"]["rank"]),
        alpha=int(step3["adapter"]["alpha"]), dropout=float(step3["adapter"]["dropout"]),
    )
    return SequenceFitnessAdapter(adapted, dimension).cuda().eval()


def _sample_variants(config: dict, count: int) -> pd.DataFrame:
    frame = load_assay("IRED", with_sequences=True)
    train, _ = get_split("IRED", "pg_kfold", fold=0)
    candidates = frame.iloc[train].copy()
    candidates["fitness_quartile"] = pd.qcut(candidates["y"], 4, labels=False, duplicates="drop")
    candidates["stratum"] = (
        candidates["n_mutations"].clip(upper=5).astype(str)
        + "_" + candidates["fitness_quartile"].astype(str)
    )
    rng = np.random.default_rng(int(config["causal_budget"]["variant_subsample_seed"]))
    chosen: list[int] = []
    groups = list(candidates.groupby("stratum", sort=True))
    base = count // len(groups)
    for _, group in groups:
        take = min(base, len(group))
        chosen.extend(rng.choice(group.index.to_numpy(), take, replace=False).tolist())
    remaining = candidates.index.difference(chosen).to_numpy()
    if len(chosen) < count:
        chosen.extend(rng.choice(remaining, count - len(chosen), replace=False).tolist())
    columns = [
        "variant_row", "sequence", "mutations", "n_mutations", "y",
        "fitness_quartile", "stratum",
    ]
    return candidates.loc[chosen, columns].sort_values(["stratum", "variant_row"]).reset_index(drop=True)


def _native_encoder(wrapped: SequenceFitnessAdapter):
    """Return the adapted native ESM-C encoder containing the LoRA-wrapped blocks."""
    compatibility_wrapper = wrapped.backbone.base_model.model
    return compatibility_wrapper.model.esmc


@torch.inference_mode()
def _downstream_predictions(
    wrapped: SequenceFitnessAdapter, hidden: torch.Tensor, analysis_layer: int,
) -> torch.Tensor:
    encoder = _native_encoder(wrapped)
    value = hidden
    for block in encoder.transformer.blocks[analysis_layer + 1 :]:
        value, _ = block(value, None, output_attentions=False)
    value = encoder.transformer.norm(value)
    return wrapped.head(value[:, 1:-1].float().mean(dim=1)).squeeze(-1)


@torch.inference_mode()
def _variant_patch_predictions(
    wrapped: SequenceFitnessAdapter,
    crosscoder: CrossCoder,
    sequence: str,
    evo_raw: np.ndarray,
    cached_fit_raw: np.ndarray,
    *,
    evo_scalar: float,
    fit_scalar: float,
    analysis_layer: int,
    feature_ids: list[int | None],
    feature_batch_size: int,
) -> tuple[float, np.ndarray, float]:
    tokens = wrapped.backbone._tokenize([sequence])
    output = wrapped.backbone(sequence_tokens=tokens)
    full_hidden = output.hidden_states[analysis_layer].detach()
    current_fit = full_hidden[0, 1:-1].float()
    cached_fit = torch.as_tensor(cached_fit_raw, dtype=torch.float32, device="cuda")
    cache_max_difference = float(torch.max(torch.abs(current_fit - cached_fit)))

    evo = torch.as_tensor(evo_raw, dtype=torch.float32, device="cuda") * evo_scalar
    fit = cached_fit * fit_scalar
    code = crosscoder.encode(evo, fit)
    residual = fit - crosscoder.fit_decoder(code)
    baseline_hidden = full_hidden.clone()
    baseline_hidden[:, 1:-1] = (fit / fit_scalar).to(full_hidden.dtype).unsqueeze(0)
    baseline = float(_downstream_predictions(wrapped, baseline_hidden, analysis_layer).item())

    predictions: list[float] = []
    for start in range(0, len(feature_ids), feature_batch_size):
        chunk = feature_ids[start : start + feature_batch_size]
        batch_code = code.unsqueeze(0).expand(len(chunk), -1, -1).clone()
        for index, feature in enumerate(chunk):
            if feature is not None:
                batch_code[index, :, int(feature)] = 0.0
        patched = crosscoder.fit_decoder(batch_code) + residual.unsqueeze(0)
        hidden = full_hidden.expand(len(chunk), -1, -1).clone()
        hidden[:, 1:-1] = (patched / fit_scalar).to(hidden.dtype)
        predictions.extend(
            _downstream_predictions(wrapped, hidden, analysis_layer).float().cpu().numpy().tolist()
        )
    return baseline, np.asarray(predictions, dtype=float), cache_max_difference


def _bootstrap(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=float)
    for index in range(samples):
        draws[index] = values[rng.integers(0, len(values), len(values))].mean()
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("causal patching requires the GPU pod")
    started = time.monotonic()
    config = load_config(REPO_ROOT / "Code/config/step8.yaml")
    step3 = load_config(REPO_ROOT / "Code/config/step3.yaml")
    report = REPO_ROOT / config["paths"]["reports"]
    dimension = int(config["dimension"])
    layer = int(config["analysis_layer"])
    seed = int(config["primary_crosscoder_seed"])
    taxonomy_path = REPO_ROOT / "Reports/step6/v17_pooled_null" / f"taxonomy_v17_w1_seed{seed}.csv"
    taxonomy = pd.read_csv(taxonomy_path)
    evo_median = float(taxonomy["evo_decoder_norm"].median())
    fit_median = float(taxonomy["fit_decoder_norm"].median())
    candidates = taxonomy.loc[
        (taxonomy["provisional_class"] == "rotated")
        & (taxonomy["evo_decoder_norm"] >= evo_median)
        & (taxonomy["fit_decoder_norm"] >= fit_median), "feature"
    ].astype(int).tolist()
    if len(candidates) != 116:
        raise AssertionError(f"approved rotated candidate count changed: {len(candidates)}")
    dead = taxonomy.loc[
        taxonomy["provisional_class"] == "dead", "feature"
    ].astype(int).tolist()
    if not dead:
        raise RuntimeError("no dead feature is available for the hygiene control")
    dead_feature = dead[0]
    random_pool = taxonomy.loc[
        (taxonomy["provisional_class"] != "rotated")
        & (taxonomy["provisional_class"] != "dead")
        & (taxonomy["evo_decoder_norm"] >= evo_median)
        & (taxonomy["fit_decoder_norm"] >= fit_median), "feature"
    ].astype(int).to_numpy()
    rng = np.random.default_rng(int(config["causal_budget"]["random_feature_seed"]))
    random_feature = int(rng.choice(random_pool))

    variants = _sample_variants(config, int(config["causal_budget"]["proposed_variant_count"]))
    variants.drop(columns="sequence").to_csv(report / "v17_causal_variant_subsample.csv", index=False)
    all_rows = np.load(REPO_ROOT / config["paths"]["campaign_rows"]).astype(int)
    row_to_cache = {int(row): index for index, row in enumerate(all_rows)}
    evo_cache = np.load(REPO_ROOT / config["paths"]["campaign_base"], mmap_mode="r")
    fit_cache = np.load(REPO_ROOT / config["paths"]["campaign_fit"], mmap_mode="r")
    step4 = json.loads((REPO_ROOT / config["paths"]["step4_manifest"]).read_text())
    evo_scalar = float(step4["models"]["evo"]["normalization_scalar"])
    fit_scalar = float(step4["models"]["fit"]["normalization_scalar"])

    checkpoint = REPO_ROOT / config["paths"]["step3_adapter"]
    verification = json.loads((REPO_ROOT / config["paths"]["gate1_checkpoint_verification"]).read_text())
    if _sha256(checkpoint) != verification["real"]["sha256"]:
        raise AssertionError("adapter hash differs from the Gate 1 artifact")
    wrapped = _adapter(step3, dimension)
    load_trainable_state(wrapped, checkpoint)
    wrapped.eval()
    crosscoder_path = REPO_ROOT / config["paths"]["crosscoder_checkpoints"] / f"w1_seed{seed}_nobias.pt"
    crosscoder = _crosscoder(crosscoder_path, dimension)

    controls: list[int | None] = [None, dead_feature, random_feature]
    baselines = []
    control_predictions = []
    cache_differences = []
    for row in variants.itertuples(index=False):
        cache_index = row_to_cache[int(row.variant_row)]
        baseline, predictions, difference = _variant_patch_predictions(
            wrapped, crosscoder, str(row.sequence), evo_cache[cache_index], fit_cache[cache_index],
            evo_scalar=evo_scalar, fit_scalar=fit_scalar, analysis_layer=layer,
            feature_ids=controls, feature_batch_size=len(controls),
        )
        baselines.append(baseline)
        control_predictions.append(predictions)
        cache_differences.append(difference)
    baselines_array = np.asarray(baselines)
    control_array = np.stack(control_predictions)
    baseline_sd = float(baselines_array.std(ddof=1))
    standardized = (control_array - baselines_array[:, None]) / baseline_sd
    hygiene = {
        "status": "passed",
        "variant_count": len(variants), "baseline_prediction_sd": baseline_sd,
        "no_op_max_absolute_raw_effect": float(np.max(np.abs(control_array[:, 0] - baselines_array))),
        "dead_feature": dead_feature,
        "dead_feature_max_absolute_raw_effect": float(np.max(np.abs(control_array[:, 1] - baselines_array))),
        "dead_feature_mean_standardized_effect": float(standardized[:, 1].mean()),
        "random_live_feature": random_feature,
        "random_feature_mean_standardized_effect": float(standardized[:, 2].mean()),
        "random_feature_mean_absolute_standardized_effect": float(np.mean(np.abs(standardized[:, 2]))),
        "maximum_cached_vs_reextracted_layer_absolute_difference": float(max(cache_differences)),
        "tolerances": {
            "no_op_absolute": float(config["causal_budget"]["no_op_absolute_tolerance"]),
            "dead_absolute": float(config["causal_budget"]["dead_feature_absolute_tolerance"]),
            "random_standardized_mean_absolute": float(
                config["causal_budget"]["random_feature_standardized_mean_absolute_tolerance"]
            ),
        },
    }
    if hygiene["no_op_max_absolute_raw_effect"] > hygiene["tolerances"]["no_op_absolute"]:
        hygiene["status"] = "failed_no_op"
    elif hygiene["dead_feature_max_absolute_raw_effect"] > hygiene["tolerances"]["dead_absolute"]:
        hygiene["status"] = "failed_dead_feature"
    elif abs(hygiene["random_feature_mean_standardized_effect"]) > hygiene["tolerances"]["random_standardized_mean_absolute"]:
        hygiene["status"] = "failed_random_feature"
    (report / "v17_causal_patching_hygiene.json").write_text(json.dumps(hygiene, indent=2) + "\n")
    print(json.dumps(hygiene, indent=2), flush=True)
    if hygiene["status"] != "passed":
        raise RuntimeError(f"causal patching hygiene failed: {hygiene['status']}")

    result_rows = []
    feature_values = {feature: [] for feature in candidates}
    for variant_index, row in enumerate(variants.itertuples(index=False)):
        cache_index = row_to_cache[int(row.variant_row)]
        baseline, predictions, _ = _variant_patch_predictions(
            wrapped, crosscoder, str(row.sequence), evo_cache[cache_index], fit_cache[cache_index],
            evo_scalar=evo_scalar, fit_scalar=fit_scalar, analysis_layer=layer,
            feature_ids=[int(value) for value in candidates], feature_batch_size=16,
        )
        effects = (predictions - baseline) / baseline_sd
        for feature, prediction, effect in zip(candidates, predictions, effects):
            feature_values[feature].append(float(effect))
            result_rows.append({
                "dataset": "IRED", "crosscoder_seed": seed, "feature": feature,
                "variant_row": int(row.variant_row), "mutations": str(row.mutations),
                "n_mutations": int(row.n_mutations), "fitness": float(row.y),
                "stratum": str(row.stratum), "baseline_prediction": baseline,
                "ablated_prediction": float(prediction), "raw_effect": float(prediction - baseline),
                "standardized_effect": float(effect),
            })
    effects = pd.DataFrame(result_rows)
    effects.to_csv(report / "v17_causal_patching_variant_effects.csv", index=False)
    summaries = []
    bootstrap_samples = int(config["semantic_test"]["bootstrap_samples"])
    bootstrap_seed = int(config["semantic_test"]["bootstrap_seed"]) + 100
    for feature in candidates:
        values = np.asarray(feature_values[feature])
        lower, upper = _bootstrap(values, bootstrap_samples, bootstrap_seed + feature)
        summaries.append({
            "feature": feature, "variants": len(values),
            "mean_standardized_effect": float(values.mean()),
            "mean_absolute_standardized_effect": float(np.abs(values).mean()),
            "standardized_effect_sd": float(values.std(ddof=1)),
            "bootstrap_ci_lower": lower, "bootstrap_ci_upper": upper,
        })
    pd.DataFrame(summaries).to_csv(report / "v17_causal_patching_feature_effects.csv", index=False)
    gpu_hours = (time.monotonic() - started) / 3600.0
    manifest = {
        "status": "complete", "dataset": "IRED", "target_class": "rotated",
        "candidate_features": len(candidates), "variants": len(variants),
        "candidate_partial_forward_passes": len(candidates) * len(variants),
        "control_partial_forward_passes": len(controls) * len(variants),
        "partial_forward_passes_total": (len(candidates) + len(controls)) * len(variants),
        "patch_location": "inside E_fit at layer 26",
        "residual_rule": "fit_decoder(ablated_code) + original_fit_residual",
        "effect_standardization": str(config["causal_budget"]["standardization"]),
        "adapter_checkpoint": str(checkpoint.relative_to(REPO_ROOT)),
        "adapter_sha256": _sha256(checkpoint),
        "crosscoder_checkpoint": str(crosscoder_path.relative_to(REPO_ROOT)),
        "crosscoder_sha256": _sha256(crosscoder_path),
        "taxonomy": str(taxonomy_path.relative_to(REPO_ROOT)),
        "gpu_wall_hours": gpu_hours, "partial_results_reported": False,
    }
    (report / "v17_causal_patching_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
