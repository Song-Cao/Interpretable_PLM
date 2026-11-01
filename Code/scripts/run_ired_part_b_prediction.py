#!/usr/bin/env python3
"""Fit IRED dense probes, evaluate Gate 2, and run Step 9 feature predictors."""
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
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossfit.datasets import get_split, load_assay, load_plate_dataset, wild_type_fitness
from crossfit.evaluation import bootstrap_metric_intervals, combine_deep_ensemble
from crossfit.feature_pooling import select_feature_set
from crossfit.models import build_model
from crossfit.persistence import prediction_frame, write_prediction_csv, write_prediction_parquet
from crossfit.search import training_pool_folds
from crossfit.training import (
    FeatureNormalizer,
    fit_torch_fixed_epochs,
    fit_torch_predictor,
    predict_torch,
    seed_everything,
)
from crossfit.utils import REPO_ROOT, load_config


def _configurations(family: str, config: dict) -> list[dict]:
    selection = config["selection"]
    if family == "ridge":
        values = [
            {"learning_rate": lr, "weight_decay": wd, "batch_size": bs}
            for lr, wd, bs in itertools.product(
                selection["ridge_learning_rate"], selection["ridge_weight_decay"],
                selection["ridge_batch_size"],
            )
        ]
    elif family == "mlp":
        values = [
            {"hidden_dims": hidden, "dropout": dropout, "learning_rate": lr,
             "weight_decay": wd, "batch_size": 32}
            for hidden, dropout, lr, wd in itertools.product(
                selection["mlp_hidden_dims"], selection["mlp_dropout"],
                selection["ridge_learning_rate"], selection["ridge_weight_decay"],
            )
        ]
    elif family == "random_forest":
        space = selection["random_forest"]
        values = [
            {"n_estimators": n, "max_depth": depth, "min_samples_leaf": leaf,
             "max_features": features}
            for n, depth, leaf, features in itertools.product(
                space["n_estimators"], space["max_depth"], space["min_samples_leaf"],
                space["max_features"],
            )
        ]
    else:
        raise KeyError(family)
    rng = np.random.default_rng(int(selection["sampling_seed"]) + sum(map(ord, family)))
    order = rng.permutation(len(values))
    count = min(int(selection["configurations_per_family"]), len(values))
    return [values[index] for index in order[:count]]


def _model(family: str, input_dim: int, parameters: dict, seed: int):
    if family == "random_forest":
        return RandomForestRegressor(**parameters, random_state=seed, n_jobs=-1)
    architecture = (
        "ridge" if family == "ridge"
        else "low_rank_multiplicative" if family == "low_rank_multiplicative"
        else "mlp"
    )
    model_parameters = {
        key: value for key, value in parameters.items()
        if key not in {"learning_rate", "weight_decay", "batch_size"}
    }
    return build_model(architecture, (input_dim,), model_parameters)


def _torch_checkpoint(fit, path: Path, family: str, parameters: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save({
        "family": family, "parameters": parameters, "state_dict": fit.model.state_dict(),
        "residual_scale": fit.loss.residual_scale.state_dict(),
        "normalizer_mean": fit.normalizer.mean, "normalizer_scale": fit.normalizer.scale,
        "epochs": int(getattr(fit, "epochs", getattr(fit, "best_epoch", -1) + 1)),
    }, temporary)
    os.replace(temporary, path)


def _load_torch_checkpoint(path: Path, family: str, input_dim: int, parameters: dict, device: str):
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if artifact["family"] != family or artifact["parameters"] != parameters:
        raise AssertionError(f"checkpoint identity mismatch: {path}")
    model = _model(family, input_dim, parameters, 0).to(device)
    model.load_state_dict(artifact["state_dict"])
    normalizer = FeatureNormalizer(
        mean=np.asarray(artifact["normalizer_mean"]), scale=np.asarray(artifact["normalizer_scale"])
    )
    return model, normalizer, int(artifact["epochs"])


def _fit_cv(
    arm: str,
    family: str,
    configurations: list[dict],
    features: np.ndarray,
    target: np.ndarray,
    censoring: np.ndarray,
    train_pool: np.ndarray,
    frame: pd.DataFrame,
    config: dict,
    objective: dict,
    *,
    device: str,
    persist_search: bool,
    pca_variance: float | None = None,
) -> tuple[pd.DataFrame, int, dict, int, pd.DataFrame]:
    folds = training_pool_folds(
        train_pool, int(config["selection"]["cv_folds"]), int(config["selection"]["cv_seed"])
    )
    rows = []
    prediction_root = REPO_ROOT / config["paths"]["phi_cache"] / "search_predictions" / arm
    checkpoint_root = REPO_ROOT / config["paths"]["models"] / "search" / arm
    for config_id, parameters in enumerate(configurations):
        for fold, (train, validation) in enumerate(folds):
            seed = int(config["selection"]["sampling_seed"]) + config_id * 101 + fold
            if pca_variance is None:
                train_features = features[train]
                validation_features = features[validation]
            else:
                fold_pca = PCA(n_components=pca_variance, svd_solver="full")
                train_features = fold_pca.fit_transform(features[train])
                validation_features = fold_pca.transform(features[validation])
            suffix = "joblib" if family == "random_forest" else "pt"
            checkpoint = checkpoint_root / f"config_{config_id:03d}_fold_{fold:02d}.{suffix}"
            prediction_path = prediction_root / f"config_{config_id:03d}_fold_{fold:02d}.parquet"
            if checkpoint.exists() and prediction_path.exists():
                held_out = pd.read_parquet(prediction_path)
                expected_rows = frame.iloc[validation]["variant_row"].to_numpy(np.int64)
                if not np.array_equal(held_out["variant_row"].to_numpy(np.int64), expected_rows):
                    raise AssertionError(f"cached held-out rows differ: {prediction_path}")
                prediction = held_out["prediction"].to_numpy(float)
                if family == "random_forest":
                    best_epoch = 0
                else:
                    artifact = torch.load(checkpoint, map_location="cpu", weights_only=False)
                    if artifact["family"] != family or artifact["parameters"] != parameters:
                        raise AssertionError(f"checkpoint identity mismatch: {checkpoint}")
                    best_epoch = int(artifact["epochs"])
                rows.append({
                    "arm": arm, "family": family, "config_id": config_id, "fold": fold,
                    "validation_spearman": float(spearmanr(prediction, target[validation]).statistic),
                    "best_epoch": best_epoch, "checkpoint": str(checkpoint.relative_to(REPO_ROOT)),
                    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    "resumed_complete_fit": True, **parameters,
                })
                continue
            if family == "random_forest":
                fitted = _model(family, train_features.shape[1], parameters, seed)
                fitted.fit(train_features, target[train])
                prediction = fitted.predict(validation_features)
                best_epoch = 0
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                joblib.dump(fitted, checkpoint)
            else:
                seed_everything(seed)
                fitted = fit_torch_predictor(
                    _model(family, train_features.shape[1], parameters, seed),
                    train_features, target[train], censoring[train],
                    validation_features, target[validation], censoring[validation],
                    objective_config=objective, optimization_config=config["optimization"],
                    parameters=parameters, seed=seed, device=device,
                )
                prediction = predict_torch(
                    fitted.model, fitted.normalizer, validation_features, device=device
                )
                best_epoch = int(fitted.best_epoch) + 1
                _torch_checkpoint(fitted, checkpoint, family, parameters)
            prediction_rows = prediction_frame(
                dataset="IRED", step=9 if arm.startswith("phi_") else 5, arm=arm,
                config_id=config_id, fold=fold, seed=seed, condition="cv_validation",
                variant_rows=frame.iloc[validation]["variant_row"], targets=target[validation],
                predictions=prediction,
            )
            if persist_search:
                write_prediction_parquet(
                    prediction_rows,
                    prediction_path,
                )
            rows.append({
                "arm": arm, "family": family, "config_id": config_id, "fold": fold,
                "validation_spearman": float(spearmanr(prediction, target[validation]).statistic),
                "best_epoch": best_epoch, "checkpoint": str(checkpoint.relative_to(REPO_ROOT)),
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                "resumed_complete_fit": False,
                **parameters,
            })
    search = pd.DataFrame(rows)
    means = search.groupby("config_id")["validation_spearman"].mean()
    selected_id = int(means.idxmax())
    selected = configurations[selected_id]
    selected_epochs = 1 if family == "random_forest" else max(
        1, int(round(search.loc[search["config_id"] == selected_id, "best_epoch"].median()))
    )
    selected_predictions = []
    for fold in range(len(folds)):
        path = prediction_root / f"config_{selected_id:03d}_fold_{fold:02d}.parquet"
        selected_predictions.append(pd.read_parquet(path))
    return search, selected_id, selected, selected_epochs, pd.concat(selected_predictions, ignore_index=True)


def _fit_final_members(
    arm: str,
    family: str,
    parameters: dict,
    epochs: int,
    campaign_features: np.ndarray,
    plate_features: np.ndarray,
    target: np.ndarray,
    censoring: np.ndarray,
    frame: pd.DataFrame,
    train_pool: np.ndarray,
    test: np.ndarray,
    round_train: np.ndarray,
    round_test: np.ndarray,
    plate: pd.DataFrame,
    config: dict,
    objective: dict,
    *,
    device: str,
    pca_variance: float | None = None,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows = []
    members = {"pg_kfold_test": [], "round_held_out": [], "plate": []}
    model_dir = REPO_ROOT / config["paths"]["models"] / arm
    if pca_variance is None:
        pg_train_features = campaign_features[train_pool]
        pg_test_features = campaign_features[test]
        pg_plate_features = plate_features
        round_train_features = campaign_features[round_train]
        round_test_features = campaign_features[round_test]
    else:
        pg_pca = PCA(n_components=pca_variance, svd_solver="full")
        pg_train_features = pg_pca.fit_transform(campaign_features[train_pool])
        pg_test_features = pg_pca.transform(campaign_features[test])
        pg_plate_features = pg_pca.transform(plate_features)
        round_pca = PCA(n_components=pca_variance, svd_solver="full")
        round_train_features = round_pca.fit_transform(campaign_features[round_train])
        round_test_features = round_pca.transform(campaign_features[round_test])
    for seed in map(int, config["ensemble_seeds"]):
        pg_suffix = "joblib" if family == "random_forest" else "pt"
        pg_path = model_dir / f"seed_{seed}.{pg_suffix}"
        round_path = model_dir / f"seed_{seed}_round.{pg_suffix}"
        if family == "random_forest":
            if pg_path.exists() and round_path.exists():
                pg_fit, round_fit = joblib.load(pg_path), joblib.load(round_path)
            else:
                pg_fit = _model(family, pg_train_features.shape[1], parameters, seed)
                round_fit = _model(family, round_train_features.shape[1], parameters, seed)
                pg_fit.fit(pg_train_features, target[train_pool])
                round_fit.fit(round_train_features, target[round_train])
                model_dir.mkdir(parents=True, exist_ok=True)
                joblib.dump(pg_fit, pg_path)
                joblib.dump(round_fit, round_path)
            predictions = {
                "pg_kfold_test": pg_fit.predict(pg_test_features),
                "round_held_out": round_fit.predict(round_test_features),
                "plate": pg_fit.predict(pg_plate_features),
            }
        else:
            if pg_path.exists() and round_path.exists():
                pg_model, pg_normalizer, pg_epochs = _load_torch_checkpoint(
                    pg_path, family, pg_train_features.shape[1], parameters, device
                )
                round_model, round_normalizer, round_epochs = _load_torch_checkpoint(
                    round_path, family, round_train_features.shape[1], parameters, device
                )
                if pg_epochs != epochs or round_epochs != epochs:
                    raise AssertionError(f"final checkpoint epoch policy differs: {arm} seed {seed}")
                pg_fit = type("LoadedFit", (), {"model": pg_model, "normalizer": pg_normalizer})()
                round_fit = type("LoadedFit", (), {"model": round_model, "normalizer": round_normalizer})()
            else:
                seed_everything(seed)
                pg_fit = fit_torch_fixed_epochs(
                    _model(family, pg_train_features.shape[1], parameters, seed),
                    pg_train_features, target[train_pool], censoring[train_pool],
                    objective_config=objective, optimization_config=config["optimization"],
                    parameters=parameters, epochs=epochs, seed=seed, device=device,
                )
                seed_everything(seed)
                round_fit = fit_torch_fixed_epochs(
                    _model(family, round_train_features.shape[1], parameters, seed),
                    round_train_features, target[round_train], censoring[round_train],
                    objective_config=objective, optimization_config=config["optimization"],
                    parameters=parameters, epochs=epochs, seed=seed, device=device,
                )
                _torch_checkpoint(pg_fit, pg_path, family, parameters)
                _torch_checkpoint(round_fit, round_path, family, parameters)
            predictions = {
                "pg_kfold_test": predict_torch(
                    pg_fit.model, pg_fit.normalizer, pg_test_features, device=device
                ),
                "round_held_out": predict_torch(
                    round_fit.model, round_fit.normalizer, round_test_features, device=device
                ),
                "plate": predict_torch(
                    pg_fit.model, pg_fit.normalizer, pg_plate_features, device=device
                ),
            }
        for condition, prediction in predictions.items():
            if condition == "pg_kfold_test":
                variant_rows, values = frame.iloc[test]["variant_row"], target[test]
            elif condition == "round_held_out":
                variant_rows, values = frame.iloc[round_test]["variant_row"], target[round_test]
            else:
                variant_rows, values = np.arange(len(plate)), plate["1/s"].to_numpy(float)
            rows.append(prediction_frame(
                dataset="IRED", step=9 if arm.startswith("phi_") else 5, arm=arm,
                config_id="selected", fold="final", seed=seed, condition=condition,
                variant_rows=variant_rows, targets=values, predictions=prediction,
            ))
            members[condition].append(prediction)
    combined = {}
    for condition, values in members.items():
        ensemble = combine_deep_ensemble(np.stack(values))
        combined[condition] = ensemble.mean
        if condition == "pg_kfold_test":
            variant_rows, targets = frame.iloc[test]["variant_row"], target[test]
        elif condition == "round_held_out":
            variant_rows, targets = frame.iloc[round_test]["variant_row"], target[round_test]
        else:
            variant_rows, targets = np.arange(len(plate)), plate["1/s"].to_numpy(float)
        rows.append(prediction_frame(
            dataset="IRED", step=9 if arm.startswith("phi_") else 5, arm=arm,
            config_id="selected", fold="final", seed="ensemble", condition=condition,
            variant_rows=variant_rows, targets=targets, predictions=ensemble.mean,
            uncertainty=ensemble.standard_deviation,
        ))
    return pd.concat(rows, ignore_index=True), combined


def _bootstrap(prediction: np.ndarray, target: np.ndarray, config: dict, plate: bool) -> dict:
    evaluation = config["evaluation"]
    return bootstrap_metric_intervals(
        prediction, target, samples=int(evaluation["bootstrap_samples"]),
        seed=int(evaluation["bootstrap_seed"]),
        confidence_level=float(evaluation["confidence_level"]),
        top_k=() if plate else [24, 48, 96],
        wild_type_fitness=None if plate else wild_type_fitness("IRED"), rank_only=plate,
    )


def _paired_spearman_interval(
    dense: np.ndarray, sparse: np.ndarray, target: np.ndarray, config: dict
) -> dict:
    rng = np.random.default_rng(int(config["evaluation"]["paired_bootstrap_seed"]))
    draws = []
    for _ in range(int(config["evaluation"]["paired_bootstrap_samples"])):
        indices = rng.integers(0, len(target), len(target))
        difference = (
            spearmanr(dense[indices], target[indices]).statistic
            - spearmanr(sparse[indices], target[indices]).statistic
        )
        if np.isfinite(difference):
            draws.append(float(difference))
    alpha = (1.0 - float(config["evaluation"]["confidence_level"])) / 2.0
    point = float(spearmanr(dense, target).statistic - spearmanr(sparse, target).statistic)
    return {
        "dense_minus_sparse": point, "lower": float(np.quantile(draws, alpha)),
        "upper": float(np.quantile(draws, 1.0 - alpha)),
    }


def _campaign_arrays(config: dict, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    rows = np.load(REPO_ROOT / config["paths"]["campaign_rows"])
    lookup = {int(row): index for index, row in enumerate(rows)}
    indices = np.asarray([lookup[int(row)] for row in frame["variant_row"]], dtype=int)
    evo = np.asarray(np.load(REPO_ROOT / config["paths"]["campaign_base"], mmap_mode="r")[indices], dtype=np.float32)
    fit = np.asarray(np.load(REPO_ROOT / config["paths"]["campaign_fit"], mmap_mode="r")[indices], dtype=np.float32)
    return evo, fit


def _phi(config: dict, frame: pd.DataFrame, width: int, operator: str, feature_set: str):
    root = REPO_ROOT / config["paths"]["phi_cache"] / f"w{width}"
    rows = np.load(REPO_ROOT / config["paths"]["phi_cache"] / "campaign_rows.npy")
    lookup = {int(row): index for index, row in enumerate(rows)}
    indices = np.asarray([lookup[int(row)] for row in frame["variant_row"]], dtype=int)
    campaign = np.asarray(np.load(root / f"campaign_{operator}.npy", mmap_mode="r")[indices], dtype=np.float32)
    plate = np.asarray(np.load(root / f"plate_{operator}.npy", mmap_mode="r"), dtype=np.float32)
    labels = pd.read_csv(root / "taxonomy.csv")["provisional_class"].to_numpy(str)
    mask = select_feature_set(labels, feature_set, config)
    if not mask.any():
        return None, None, 0
    return campaign[:, mask], plate[:, mask], int(mask.sum())


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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    report5 = REPO_ROOT / config["paths"]["reports_step5"]
    report7 = REPO_ROOT / config["paths"]["reports_step7"]
    report9 = REPO_ROOT / config["paths"]["reports_step9"]
    for path in (report5, report7, report9):
        path.mkdir(parents=True, exist_ok=True)

    evo, fit = _campaign_arrays(config, frame)
    plate_root = REPO_ROOT / config["paths"]["plate_cache"]
    plate_evo = np.asarray(np.load(plate_root / "base.npy"), dtype=np.float32)
    plate_fit = np.asarray(np.load(plate_root / "fit.npy"), dtype=np.float32)
    dense_inputs = {
        "fit_mean": (fit.mean(axis=1), plate_fit.mean(axis=1)),
        "evo_mean": (evo.mean(axis=1), plate_evo.mean(axis=1)),
    }
    delta = (fit - evo).mean(axis=1)
    plate_delta = (plate_fit - plate_evo).mean(axis=1)
    dense_inputs["delta_pcs"] = (delta, plate_delta)
    dense_results = []
    dense_predictions: dict[str, dict[str, np.ndarray]] = {}
    ridge_configs = _configurations("ridge", config)
    for name, (features, plate_features) in dense_inputs.items():
        arm = f"dense_{name}"
        pca_variance = (
            float(config["delta_pca_variance_fraction"]) if name == "delta_pcs" else None
        )
        search, selected_id, selected, epochs, cv_predictions = _fit_cv(
            arm, "ridge", ridge_configs, features, target, censoring, train_pool, frame,
            config, step2["objective"], device=device, persist_search=True,
            pca_variance=pca_variance,
        )
        search.to_csv(report5 / f"{arm}_search.csv", index=False)
        write_prediction_csv(cv_predictions, report5 / f"{arm}_cv_predictions.csv")
        predictions, ensemble = _fit_final_members(
            arm, "ridge", selected, epochs, features, plate_features, target, censoring,
            frame, train_pool, test, round_train, round_test, plate, config, step2["objective"],
            device=device, pca_variance=pca_variance,
        )
        write_prediction_csv(predictions, report5 / f"{arm}_predictions.csv")
        dense_predictions[arm] = ensemble
        dense_results.append({
            "arm": arm, "family": "ridge", "features": (
                int(PCA(n_components=pca_variance, svd_solver="full").fit(features[train_pool]).n_components_)
                if pca_variance is not None else features.shape[1]
            ),
            "validation_spearman": float(search.groupby("config_id")["validation_spearman"].mean().loc[selected_id]),
            "selected_config_id": selected_id, "selected_configuration": json.dumps(selected),
            "selected_epochs": epochs,
            "pg_kfold_test_spearman": float(spearmanr(ensemble["pg_kfold_test"], target[test]).statistic),
            "round_held_out_spearman": float(spearmanr(ensemble["round_held_out"], target[round_test]).statistic),
            "plate_spearman": float(spearmanr(ensemble["plate"], plate["1/s"]).statistic),
        })
    dense_frame = pd.DataFrame(dense_results)
    dense_frame.to_csv(report5 / "ired_dense_probe_results.csv", index=False)

    reference_features, _, _ = _phi(config, frame, 1, "sum", "all_live")
    architecture_rows = []
    architecture_fits = {}
    for family in ("ridge", "mlp", "random_forest"):
        configurations = _configurations(family, config)
        arm = f"phi_architecture_{family}"
        search, selected_id, selected, epochs, cv_predictions = _fit_cv(
            arm, family, configurations, reference_features, target, censoring, train_pool,
            frame, config, step2["objective"], device=device, persist_search=True,
        )
        search.to_csv(report9 / f"{arm}_search.csv", index=False)
        write_prediction_csv(cv_predictions, report9 / f"{arm}_cv_predictions.csv")
        score = float(search.groupby("config_id")["validation_spearman"].mean().loc[selected_id])
        architecture_rows.append({
            "family": family, "validation_spearman": score, "selected_config_id": selected_id,
            "selected_configuration": json.dumps(selected), "selected_epochs": epochs,
        })
        architecture_fits[family] = (selected, epochs, score)
    architecture_frame = pd.DataFrame(architecture_rows).sort_values("validation_spearman", ascending=False)
    architecture_frame.to_csv(report9 / "phi_architecture_shortlist.csv", index=False)
    family = str(architecture_frame.iloc[0]["family"])
    selected_family_config, _, _ = architecture_fits[family]

    phi_rows = []
    phi_predictions: dict[str, dict[str, np.ndarray]] = {}
    phi_cv_predictions: dict[str, pd.DataFrame] = {}
    for width, operator, feature_set in itertools.product(
        map(int, config["width_multipliers"]), config["pooling_operators"], config["feature_sets"]
    ):
        features, plate_features, feature_count = _phi(
            config, frame, width, str(operator), str(feature_set)
        )
        if features is None:
            phi_rows.append({
                "arm": f"phi_w{width}_{operator}_{feature_set}", "width_multiplier": width,
                "pooling": operator, "feature_set": feature_set, "features": 0,
                "status": "empty_feature_set",
            })
            continue
        arm = f"phi_w{width}_{operator}_{feature_set}"
        search, selected_id, _, epochs, cv_predictions = _fit_cv(
            arm, family, [selected_family_config], features, target, censoring, train_pool,
            frame, config, step2["objective"], device=device, persist_search=True,
        )
        search.to_csv(report9 / f"{arm}_cv.csv", index=False)
        write_prediction_csv(cv_predictions, report9 / f"{arm}_cv_predictions.csv")
        predictions, ensemble = _fit_final_members(
            arm, family, selected_family_config, epochs, features, plate_features, target,
            censoring, frame, train_pool, test, round_train, round_test, plate, config,
            step2["objective"], device=device,
        )
        write_prediction_csv(predictions, report9 / f"{arm}_predictions.csv")
        phi_predictions[arm] = ensemble
        phi_cv_predictions[arm] = cv_predictions
        validation_score = float(search["validation_spearman"].mean())
        phi_rows.append({
            "arm": arm, "width_multiplier": width, "pooling": operator,
            "feature_set": feature_set, "features": feature_count, "status": "complete",
            "family": family, "validation_spearman": validation_score, "selected_epochs": epochs,
            "pg_kfold_test_spearman": float(spearmanr(ensemble["pg_kfold_test"], target[test]).statistic),
            "round_held_out_spearman": float(spearmanr(ensemble["round_held_out"], target[round_test]).statistic),
            "plate_spearman": float(spearmanr(ensemble["plate"], plate["1/s"]).statistic),
            "pg_kfold_test_metrics": json.dumps(_bootstrap(ensemble["pg_kfold_test"], target[test], config, False)),
            "round_held_out_metrics": json.dumps(_bootstrap(ensemble["round_held_out"], target[round_test], config, False)),
            "plate_metrics": json.dumps(_bootstrap(ensemble["plate"], plate["1/s"].to_numpy(float), config, True)),
        })
    phi_frame = pd.DataFrame(phi_rows)
    phi_frame.to_csv(report9 / "ired_phi_feature_set_results.csv", index=False)
    complete_phi = phi_frame.loc[phi_frame["status"] == "complete"]
    best_phi_arm = str(complete_phi.sort_values("validation_spearman", ascending=False).iloc[0]["arm"])
    best_dense_arm = str(dense_frame.sort_values("validation_spearman", ascending=False).iloc[0]["arm"])
    paired = _paired_spearman_interval(
        dense_predictions[best_dense_arm]["pg_kfold_test"],
        phi_predictions[best_phi_arm]["pg_kfold_test"], target[test], config,
    )
    verdict = "dense_dominates" if paired["lower"] > 0 else "sparse_competitive"
    gate2 = {
        "dataset": "IRED", "decision_metric": "pg_kfold_test_spearman",
        "selection_data": "five_fold_cv_within_pg_kfold_training_pool",
        "best_dense_arm_by_validation": best_dense_arm,
        "best_phi_arm_by_validation": best_phi_arm,
        "paired_bootstrap_dense_minus_sparse": paired, "verdict": verdict,
        "step9_run_despite_verdict": True,
        "override_scope": "IRED only; Gate 2 is a resource-allocation gate",
    }
    (report7 / "gate2_result.json").write_text(json.dumps(gate2, indent=2) + "\n")

    summary = {
        "status": "complete", "dataset": "IRED", "selected_phi_family": family,
        "best_phi_arm_by_validation": best_phi_arm, "gate2_verdict": verdict,
        "gpu_wall_hours": (time.monotonic() - started) / 3600.0,
        "plate_labels_used_for_selection": False, "test_labels_used_for_selection": False,
        "expressivity_ladder_pending": True, "sample_efficiency_pending": True,
    }
    (report9 / "part_b_prediction_manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
