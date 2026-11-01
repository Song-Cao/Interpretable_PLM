#!/usr/bin/env python3
"""Run the IRED phi expressivity ladder, auxiliary arms, and sample-efficiency curve."""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossfit.datasets import get_split, load_assay, load_plate_dataset
from crossfit.evaluation import bootstrap_metric_intervals, combine_deep_ensemble
from crossfit.feature_pooling import select_feature_set
from crossfit.models import build_model
from crossfit.persistence import prediction_frame, write_prediction_csv, write_prediction_parquet
from crossfit.search import training_pool_folds
from crossfit.training import fit_torch_fixed_epochs, fit_torch_predictor, predict_torch, seed_everything
from crossfit.utils import REPO_ROOT, load_config


def _phi(config: dict, frame: pd.DataFrame, arm: str):
    parts = arm.split("_")
    width = int(parts[1][1:])
    operator = parts[2]
    feature_set = "_".join(parts[3:])
    root = REPO_ROOT / config["paths"]["phi_cache"] / f"w{width}"
    rows = np.load(REPO_ROOT / config["paths"]["phi_cache"] / "campaign_rows.npy")
    lookup = {int(row): index for index, row in enumerate(rows)}
    indices = np.asarray([lookup[int(row)] for row in frame["variant_row"]], dtype=int)
    campaign = np.asarray(np.load(root / f"campaign_{operator}.npy", mmap_mode="r")[indices], dtype=np.float32)
    plate = np.asarray(np.load(root / f"plate_{operator}.npy", mmap_mode="r"), dtype=np.float32)
    labels = pd.read_csv(root / "taxonomy.csv")["provisional_class"].to_numpy(str)
    mask = select_feature_set(labels, feature_set, config)
    return campaign[:, mask], plate[:, mask]


def _estimator(level: str, parameters: dict, seed: int):
    if level == "L0":
        return make_pipeline(StandardScaler(), Ridge(alpha=float(parameters["alpha"])))
    if level == "L1":
        return make_pipeline(
            StandardScaler(),
            SplineTransformer(n_knots=int(parameters["n_knots"]), degree=3, include_bias=False),
            Ridge(alpha=float(parameters["alpha"])),
        )
    if level == "L1.5":
        return RandomForestRegressor(**parameters, random_state=seed, n_jobs=-1)
    raise KeyError(level)


def _fit_sklearn_cv(
    arm: str,
    level: str,
    configurations: list[dict],
    features: np.ndarray,
    target: np.ndarray,
    train_pool: np.ndarray,
    frame: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, int, dict]:
    folds = training_pool_folds(
        train_pool, int(config["selection"]["cv_folds"]), int(config["selection"]["cv_seed"])
    )
    cache = REPO_ROOT / config["paths"]["phi_cache"] / "ladder_predictions" / arm / level
    models = REPO_ROOT / config["paths"]["models"] / "ladder_search" / arm / level
    rows = []
    for config_id, parameters in enumerate(configurations):
        for fold, (train, validation) in enumerate(folds):
            seed = int(config["selection"]["sampling_seed"]) + 1009 * config_id + fold
            prediction_path = cache / f"config_{config_id:03d}_fold_{fold:02d}.parquet"
            checkpoint = models / f"config_{config_id:03d}_fold_{fold:02d}.joblib"
            if prediction_path.exists() and checkpoint.exists():
                prediction = pd.read_parquet(prediction_path)["prediction"].to_numpy(float)
                resumed = True
            else:
                model = _estimator(level, parameters, seed)
                model.fit(features[train], target[train])
                prediction = model.predict(features[validation])
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(model, checkpoint)
                write_prediction_parquet(
                    prediction_frame(
                        dataset="IRED", step=9, arm=f"{arm}_{level}", config_id=config_id,
                        fold=fold, seed=seed, condition="cv_validation",
                        variant_rows=frame.iloc[validation]["variant_row"],
                        targets=target[validation], predictions=prediction,
                    ),
                    prediction_path,
                )
                resumed = False
            rows.append({
                "arm": arm, "level": level, "config_id": config_id, "fold": fold,
                "validation_spearman": float(spearmanr(prediction, target[validation]).statistic),
                "checkpoint": str(checkpoint.relative_to(REPO_ROOT)),
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "resumed_complete_fit": resumed, **parameters,
            })
    search = pd.DataFrame(rows)
    means = search.groupby("config_id")["validation_spearman"].mean()
    selected_id = int(means.idxmax())
    return search, selected_id, configurations[selected_id]


def _final_sklearn(
    arm: str,
    level: str,
    parameters: dict,
    features: np.ndarray,
    plate_features: np.ndarray,
    target: np.ndarray,
    train_pool: np.ndarray,
    test: np.ndarray,
    round_train: np.ndarray,
    round_test: np.ndarray,
    frame: pd.DataFrame,
    plate: pd.DataFrame,
    config: dict,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    members = {"pg_kfold_test": [], "round_held_out": [], "plate": []}
    root = REPO_ROOT / config["paths"]["models"] / "ladder" / arm / level
    for seed in map(int, config["ensemble_seeds"]):
        pg_path, round_path = root / f"seed_{seed}.joblib", root / f"seed_{seed}_round.joblib"
        if pg_path.exists() and round_path.exists():
            pg, round_model = joblib.load(pg_path), joblib.load(round_path)
        else:
            pg, round_model = _estimator(level, parameters, seed), _estimator(level, parameters, seed)
            pg.fit(features[train_pool], target[train_pool])
            round_model.fit(features[round_train], target[round_train])
            root.mkdir(parents=True, exist_ok=True)
            joblib.dump(pg, pg_path)
            joblib.dump(round_model, round_path)
        predictions = {
            "pg_kfold_test": pg.predict(features[test]),
            "round_held_out": round_model.predict(features[round_test]),
            "plate": pg.predict(plate_features),
        }
        for condition, prediction in predictions.items():
            members[condition].append(prediction)
            indices = test if condition == "pg_kfold_test" else round_test
            variant_rows = np.arange(len(plate)) if condition == "plate" else frame.iloc[indices]["variant_row"]
            values = plate["1/s"].to_numpy(float) if condition == "plate" else target[indices]
            rows.append(prediction_frame(
                dataset="IRED", step=9, arm=f"{arm}_{level}", config_id="selected",
                fold="final", seed=seed, condition=condition, variant_rows=variant_rows,
                targets=values, predictions=prediction,
            ))
    ensembles = {}
    for condition, values in members.items():
        combined = combine_deep_ensemble(np.stack(values))
        ensembles[condition] = combined.mean
        indices = test if condition == "pg_kfold_test" else round_test
        variant_rows = np.arange(len(plate)) if condition == "plate" else frame.iloc[indices]["variant_row"]
        targets = plate["1/s"].to_numpy(float) if condition == "plate" else target[indices]
        rows.append(prediction_frame(
            dataset="IRED", step=9, arm=f"{arm}_{level}", config_id="selected",
            fold="final", seed="ensemble", condition=condition, variant_rows=variant_rows,
            targets=targets, predictions=combined.mean, uncertainty=combined.standard_deviation,
        ))
    return pd.concat(rows, ignore_index=True), ensembles


def _metrics(prediction: np.ndarray, target: np.ndarray, config: dict) -> dict:
    return bootstrap_metric_intervals(
        prediction, target, samples=int(config["evaluation"]["bootstrap_samples"]),
        seed=int(config["evaluation"]["bootstrap_seed"]),
        confidence_level=float(config["evaluation"]["confidence_level"]), rank_only=True,
    )


def _interaction_gate(model: RandomForestRegressor, features: np.ndarray, config: dict) -> dict:
    top = np.argsort(model.feature_importances_)[-2:]
    sample = features[: min(256, len(features))].copy()
    baseline = model.predict(sample)
    additive = np.zeros_like(baseline)
    reference = sample.copy()
    for feature in top:
        values = []
        for value in sample[:, feature]:
            modified = reference.copy()
            modified[:, feature] = value
            values.append(model.predict(modified).mean())
        additive += np.asarray(values) - np.mean(values)
    centered = baseline - baseline.mean()
    residual = centered - additive
    observed = float(np.var(residual) / max(np.var(centered), 1e-12))
    rng = np.random.default_rng(int(config["selection"]["sampling_seed"]) + 77)
    null = []
    for _ in range(int(config["expressivity"]["interaction_permutations"])):
        permuted = sample.copy()
        permuted[:, top[1]] = rng.permutation(permuted[:, top[1]])
        prediction = model.predict(permuted)
        null.append(float(np.var(prediction - prediction.mean() - additive) / max(np.var(prediction), 1e-12)))
    pvalue = (1 + np.sum(np.asarray(null) >= observed)) / (len(null) + 1)
    return {
        "method": "Friedman-H-style additive residual with feature-permutation null",
        "top_feature_indices": top.tolist(), "observed_h_squared": observed,
        "permutation_pvalue": float(pvalue), "permutations": len(null),
        "passes": bool(pvalue < float(config["expressivity"]["interaction_alpha"])),
    }


def main() -> int:
    started = time.monotonic()
    config = load_config(REPO_ROOT / "Code/config/step9.yaml")
    step2 = load_config(REPO_ROOT / "Code/config/step2.yaml")
    frame = load_assay("IRED", with_sequences=True)
    plate = load_plate_dataset("IRED")
    target = frame["y"].to_numpy(np.float32)
    censoring = frame["censored"].astype(str).to_numpy()
    train_pool, test = get_split("IRED", "pg_kfold", fold=int(config["evaluation"]["primary_test_fold"]))
    round_train, round_test = get_split("IRED", "round_held_out")
    report = REPO_ROOT / config["paths"]["reports_step9"]
    feature_results = pd.read_csv(report / "ired_phi_feature_set_results.csv")
    complete = feature_results.loc[feature_results["status"] == "complete"].copy()
    headline = str(complete.sort_values("validation_spearman", ascending=False).iloc[0]["arm"])
    headline_x, headline_plate = _phi(config, frame, headline)

    alpha_grid = [{"alpha": value} for value in [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]]
    rf_space = config["selection"]["random_forest"]
    rf_grid = [
        {"n_estimators": n, "max_depth": depth, "min_samples_leaf": leaf, "max_features": features}
        for n, depth, leaf, features in itertools.product(
            rf_space["n_estimators"], rf_space["max_depth"], rf_space["min_samples_leaf"],
            rf_space["max_features"],
        )
    ]
    rng = np.random.default_rng(int(config["selection"]["sampling_seed"]) + 811)
    rf_grid = [rf_grid[index] for index in rng.permutation(len(rf_grid))[:20]]

    timing_started = time.perf_counter()
    for index in range(10):
        sample = train_pool[: min(512, len(train_pool))]
        model = _estimator("L0" if index < 5 else "L1.5", alpha_grid[index % len(alpha_grid)] if index < 5 else rf_grid[index], index)
        model.fit(headline_x[sample], target[sample])
    seconds = time.perf_counter() - timing_started
    projected_fits = len(complete) * (len(alpha_grid) * 5 + 10) + 20 * 5 + 3 * 5
    projected_hours = seconds / 10 * projected_fits / 3600
    budget = {
        "representative_fits": 10, "representative_seconds": seconds,
        "projected_total_fits": projected_fits, "projected_gpu_wall_hours": projected_hours,
        "cap_hours": 3.0, "random_search_count": 20,
        "reduced": False if projected_hours <= 3.0 else True,
    }
    (report / "step9_ladder_budget.json").write_text(json.dumps(budget, indent=2) + "\n")
    if projected_hours > 4.0:
        raise RuntimeError("projected Step 9 ladder exceeds the hard four-hour cap")

    rows = []
    selected_rf = None
    ordered_arms = [headline] + [str(value) for value in complete["arm"] if str(value) != headline]
    for arm in ordered_arms:
        features, plate_features = _phi(config, frame, str(arm))
        levels = ["L0", "L1.5"]
        for level in levels:
            candidates = alpha_grid if level == "L0" else (rf_grid if arm == headline else [selected_rf])
            search, selected_id, selected = _fit_sklearn_cv(
                str(arm), level, candidates, features, target, train_pool, frame, config
            )
            if arm == headline and level == "L1.5":
                selected_rf = selected
            search.to_csv(report / f"{arm}_{level}_search.csv", index=False)
            predictions, ensemble = _final_sklearn(
                str(arm), level, selected, features, plate_features, target, train_pool, test,
                round_train, round_test, frame, plate, config,
            )
            write_prediction_csv(predictions, report / f"{arm}_{level}_predictions.csv")
            rows.append({
                "arm": arm, "level": level, "features": features.shape[1],
                "validation_spearman": float(search.groupby("config_id")["validation_spearman"].mean().loc[selected_id]),
                "selected_configuration": json.dumps(selected),
                "pg_kfold_test_spearman": float(spearmanr(ensemble["pg_kfold_test"], target[test]).statistic),
                "pg_kfold_test_r2": float(r2_score(target[test], ensemble["pg_kfold_test"])),
                "round_held_out_spearman": float(spearmanr(ensemble["round_held_out"], target[round_test]).statistic),
                "round_held_out_r2": float(r2_score(target[round_test], ensemble["round_held_out"])),
                "plate_spearman": float(spearmanr(ensemble["plate"], plate["1/s"]).statistic),
                "plate_r2": float(r2_score(plate["1/s"], ensemble["plate"])),
                "pg_metrics": json.dumps(_metrics(ensemble["pg_kfold_test"], target[test], config)),
                "round_metrics": json.dumps(_metrics(ensemble["round_held_out"], target[round_test], config)),
                "plate_metrics": json.dumps(_metrics(ensemble["plate"], plate["1/s"].to_numpy(float), config)),
            })

    spline_grid = [
        {"n_knots": knots, "alpha": alpha}
        for knots, alpha in itertools.product(config["expressivity"]["spline_knots"], [0.01, 0.1, 1.0])
    ]
    search, selected_id, selected = _fit_sklearn_cv(
        headline, "L1", spline_grid, headline_x, target, train_pool, frame, config
    )
    search.to_csv(report / f"{headline}_L1_search.csv", index=False)
    predictions, l1_ensemble = _final_sklearn(
        headline, "L1", selected, headline_x, headline_plate, target, train_pool, test,
        round_train, round_test, frame, plate, config,
    )
    write_prediction_csv(predictions, report / f"{headline}_L1_predictions.csv")
    rows.append({
        "arm": headline, "level": "L1", "features": headline_x.shape[1],
        "validation_spearman": float(search.groupby("config_id")["validation_spearman"].mean().loc[selected_id]),
        "selected_configuration": json.dumps(selected),
        "pg_kfold_test_spearman": float(spearmanr(l1_ensemble["pg_kfold_test"], target[test]).statistic),
        "pg_kfold_test_r2": float(r2_score(target[test], l1_ensemble["pg_kfold_test"])),
        "round_held_out_spearman": float(spearmanr(l1_ensemble["round_held_out"], target[round_test]).statistic),
        "round_held_out_r2": float(r2_score(target[round_test], l1_ensemble["round_held_out"])),
        "plate_spearman": float(spearmanr(l1_ensemble["plate"], plate["1/s"]).statistic),
        "plate_r2": float(r2_score(plate["1/s"], l1_ensemble["plate"])),
        "pg_metrics": json.dumps(_metrics(l1_ensemble["pg_kfold_test"], target[test], config)),
        "round_metrics": json.dumps(_metrics(l1_ensemble["round_held_out"], target[round_test], config)),
        "plate_metrics": json.dumps(_metrics(l1_ensemble["plate"], plate["1/s"].to_numpy(float), config)),
    })

    headline_rf = _estimator("L1.5", selected_rf, 0)
    headline_rf.fit(headline_x[train_pool], target[train_pool])
    interaction = _interaction_gate(headline_rf, headline_x[train_pool], config)
    (report / "interaction_gate.json").write_text(json.dumps(interaction, indent=2) + "\n")
    if interaction["passes"]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ranks = [{"rank": rank, "learning_rate": 3e-4, "weight_decay": 0.01, "batch_size": 32} for rank in config["expressivity"]["factorization_rank"]]
        # The gated L2 shares the existing validation-Spearman trainer; predictions are persisted.
        from run_ired_part_b_prediction import _fit_cv, _fit_final_members
        search, selected_id, selected_l2, epochs, cv_predictions = _fit_cv(
            f"{headline}_L2", "low_rank_multiplicative", ranks, headline_x, target,
            censoring, train_pool, frame, config, step2["objective"], device=device,
            persist_search=True,
        )
        write_prediction_csv(cv_predictions, report / f"{headline}_L2_cv_predictions.csv")
        predictions, ensemble = _fit_final_members(
            f"{headline}_L2", "low_rank_multiplicative", selected_l2, epochs, headline_x,
            headline_plate, target, censoring, frame, train_pool, test, round_train,
            round_test, plate, config, step2["objective"], device=device,
        )
        write_prediction_csv(predictions, report / f"{headline}_L2_predictions.csv")
    else:
        ensemble = None

    l4_predictions = pd.read_csv(report / f"{headline}_predictions.csv")
    l4_by_condition = {
        condition: l4_predictions[
            (l4_predictions["seed"].astype(str) == "ensemble")
            & (l4_predictions["condition"] == condition)
        ]
        for condition in ("pg_kfold_test", "round_held_out", "plate")
    }
    rows.append({
        "arm": headline, "level": "L4", "features": headline_x.shape[1],
        "validation_spearman": float(
            complete.loc[complete["arm"] == headline, "validation_spearman"].iloc[0]
        ),
        "selected_configuration": "Step 9 phi-selected MLP",
        "pg_kfold_test_spearman": float(spearmanr(
            l4_by_condition["pg_kfold_test"]["prediction"],
            l4_by_condition["pg_kfold_test"]["target"],
        ).statistic),
        "pg_kfold_test_r2": float(r2_score(
            l4_by_condition["pg_kfold_test"]["target"],
            l4_by_condition["pg_kfold_test"]["prediction"],
        )),
        "round_held_out_spearman": float(spearmanr(
            l4_by_condition["round_held_out"]["prediction"],
            l4_by_condition["round_held_out"]["target"],
        ).statistic),
        "round_held_out_r2": float(r2_score(
            l4_by_condition["round_held_out"]["target"],
            l4_by_condition["round_held_out"]["prediction"],
        )),
        "plate_spearman": float(spearmanr(
            l4_by_condition["plate"]["prediction"], l4_by_condition["plate"]["target"]
        ).statistic),
        "plate_r2": float(r2_score(
            l4_by_condition["plate"]["target"], l4_by_condition["plate"]["prediction"]
        )),
        "pg_metrics": json.dumps(_metrics(
            l4_by_condition["pg_kfold_test"]["prediction"].to_numpy(float),
            l4_by_condition["pg_kfold_test"]["target"].to_numpy(float), config,
        )),
        "round_metrics": json.dumps(_metrics(
            l4_by_condition["round_held_out"]["prediction"].to_numpy(float),
            l4_by_condition["round_held_out"]["target"].to_numpy(float), config,
        )),
        "plate_metrics": json.dumps(_metrics(
            l4_by_condition["plate"]["prediction"].to_numpy(float),
            l4_by_condition["plate"]["target"].to_numpy(float), config,
        )),
    })
    ladder = pd.DataFrame(rows)
    l4 = l4_by_condition["pg_kfold_test"]
    l0 = ladder[(ladder["arm"] == headline) & (ladder["level"] == "L0")].iloc[0]
    l4_r2 = float(r2_score(l4["target"], l4["prediction"]))
    gap = l4_r2 - float(l0["pg_kfold_test_r2"])
    ladder.to_csv(report / "ired_expressivity_ladder.csv", index=False)
    manifest = {
        "status": "complete", "headline_arm": headline,
        "expressivity_gap_r2_L4_minus_L0": gap, "L4_test_r2": l4_r2,
        "L0_test_r2": float(l0["pg_kfold_test_r2"]), "interaction_gate": interaction,
        "gpu_wall_hours": (time.monotonic() - started) / 3600.0,
        "plate_labels_used_for_selection": False, "test_labels_used_for_selection": False,
    }
    (report / "step9_ladder_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
