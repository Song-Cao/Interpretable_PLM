#!/usr/bin/env python3
"""Search, refit, and evaluate one non-classical IRED Step 2 arm."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossfit.datasets import get_split, load_assay, load_plate_dataset, wild_type_fitness
from crossfit.evaluation import (
    MondrianConformal,
    bootstrap_metric_intervals,
    combine_deep_ensemble,
    coverage_by_stratum,
    score_metrics,
)
from crossfit.models import build_model
from crossfit.persistence import (
    prediction_frame,
    upsert_checkpoint_manifest,
    write_prediction_csv,
    write_prediction_parquet,
)
from crossfit.representations import esm_mean, esm_mutpool, esm_residue_delta, onehot_positional
from crossfit.search import (
    boundary_hyperparameters,
    load_search_state,
    sample_configurations,
    training_pool_folds,
    write_search_state,
)
from crossfit.training import (
    fit_torch_fixed_epochs,
    fit_torch_predictor,
    predict_torch,
    prepare_feature_split,
    seed_everything,
)
from crossfit.utils import REPO_ROOT, load_config, stable_hash


ARM_NAMES = (
    "esm_mean_ridge",
    "esm_mean_mlp",
    "esm_mean_random_forest",
    "esm_mutpool_ridge",
    "esm_residue_conv",
    "esm_residue_delta_conv",
    "onehot_positional_conv",
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=ARM_NAMES, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--search-count", type=int, default=None)
    parser.add_argument("--benchmark-only", action="store_true")
    return parser.parse_args()


def _cache(config: dict, name: str) -> np.ndarray:
    return np.load(REPO_ROOT / config["paths"]["representations"] / name, mmap_mode="r")


def _cached_residues(config: dict, frame: pd.DataFrame, plate: bool = False) -> np.ndarray:
    if plate:
        return np.asarray(_cache(config, "plate_residue.npy"), dtype=np.float32)
    rows = np.asarray(_cache(config, "variant_rows.npy"), dtype=np.int64)
    row_to_cache = {int(value): index for index, value in enumerate(rows)}
    indices = np.asarray([row_to_cache[int(value)] for value in frame["variant_row"]], dtype=int)
    return np.asarray(_cache(config, "variant_residue.npy")[indices], dtype=np.float32)


def _representation(
    arm: str, config: dict, frame: pd.DataFrame, *, plate: pd.DataFrame | None = None
) -> np.ndarray:
    if arm == "onehot_positional_conv":
        sequences = (
            plate["aa_seq"].astype(str).tolist()
            if plate is not None
            else frame["mutant_sequence"].astype(str).tolist()
        )
        return onehot_positional(sequences)
    residues = _cached_residues(config, frame, plate=plate is not None)
    if arm.startswith("esm_mean"):
        return esm_mean(residues)
    if arm == "esm_mutpool_ridge":
        mutations = plate["mut"].tolist() if plate is not None else frame["mutations"].tolist()
        return esm_mutpool(residues, mutations)
    if arm == "esm_residue_delta_conv":
        wild_type = np.asarray(_cache(config, "wild_type_residue.npy"), dtype=np.float32)
        return esm_residue_delta(residues, wild_type)
    return residues


def _architecture(arm: str) -> str:
    if arm.endswith("random_forest"):
        return "random_forest"
    if arm.endswith("mlp"):
        return "mlp"
    if arm.endswith("conv"):
        return "conv_residual"
    return "ridge"


def _model_parameters(architecture: str, parameters: dict, seed: int | None = None) -> dict:
    excluded = {"learning_rate", "weight_decay", "batch_size"}
    output = {key: value for key, value in parameters.items() if key not in excluded}
    if architecture == "random_forest":
        output["random_state"] = seed
        output["n_jobs"] = -1
    return output


def _sample(space: dict, requested: int, seed: int) -> list[dict]:
    for count in range(requested, 29, -1):
        try:
            return sample_configurations(space, count, seed)
        except ValueError as error:
            if "fewer than" not in str(error):
                raise
    raise ValueError("search space contains fewer than the required 30 configurations")


def _bootstrap(prediction: np.ndarray, target: np.ndarray, config: dict, rank_only: bool) -> dict:
    settings = config["evaluation"]
    return bootstrap_metric_intervals(
        prediction, target,
        samples=int(settings["bootstrap_samples"]),
        seed=int(settings["bootstrap_seed"]),
        confidence_level=float(settings["confidence_level"]),
        top_k=() if rank_only else settings["top_k"],
        wild_type_fitness=None if rank_only else wild_type_fitness("IRED"),
        rank_only=rank_only,
    )


def _fit_rf(
    architecture: str,
    shape: tuple[int, ...],
    parameters: dict,
    features: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    seed: int,
):
    model = build_model(architecture, shape, _model_parameters(architecture, parameters, seed))
    model.fit(features[train], target[train])
    return model


def _position_sets(values: pd.Series) -> list[set[int]]:
    return [set() if pd.isna(value) or str(value) == "" else set(map(int, str(value).split(";"))) for value in values]


def _save_torch_checkpoint(fit, path: Path, parameters: dict, input_shape: tuple[int, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": fit.model.state_dict(),
        "residual_scale": fit.loss.residual_scale.state_dict(),
        "normalizer_mean": fit.normalizer.mean,
        "normalizer_scale": fit.normalizer.scale,
        "configuration": parameters,
        "epochs": int(getattr(fit, "epochs", getattr(fit, "best_epoch", -1) + 1)),
        "input_shape": input_shape,
    }, path)


def main() -> int:
    args = _arguments()
    config = load_config(REPO_ROOT / "Code" / "config" / "step2.yaml")
    if config["optimization"]["scheduler_metric"] != "validation_spearman":
        raise ValueError("Step 2 scheduling must monitor validation Spearman")
    frame = load_assay("IRED", with_sequences=True)
    plate = load_plate_dataset("IRED")
    features = _representation(args.arm, config, frame)
    plate_features = _representation(args.arm, config, frame, plate=plate)
    target = frame["y"].to_numpy(np.float32)
    censoring = frame["censored"].astype(str).to_numpy()
    train_pool, test = get_split("IRED", "pg_kfold", fold=int(config["evaluation"]["primary_test_fold"]))
    folds = training_pool_folds(train_pool, int(config["search"]["cv_folds"]), int(config["search"]["cv_seed"]))
    architecture = _architecture(args.arm)
    space = config["search"]["spaces"][architecture]
    requested = args.search_count or int(config["search"]["maximum_configurations_per_arm"])
    search_seed = int(config["search"]["sampling_seed"]) + stable_hash(args.arm) % 1_000_000
    configurations = _sample(space, requested, search_seed)
    input_shape = tuple(features.shape[1:])
    report_dir = REPO_ROOT / config["paths"]["reports"]
    report_dir.mkdir(parents=True, exist_ok=True)
    state_dir = REPO_ROOT / config["paths"]["search_state"]
    partial_path = state_dir / f"{args.arm}_partial.csv"
    persistence = config["persistence"]
    search_prediction_dir = REPO_ROOT / config["paths"]["search_predictions"] / args.arm
    search_checkpoint_dir = REPO_ROOT / config["paths"]["search_checkpoints"] / args.arm
    checkpoint_manifest = report_dir / f"{args.arm}_checkpoint_manifest.json"

    preparation_started = time.perf_counter()
    prepared_folds = None
    if architecture != "random_forest":
        prepared_folds = [
            prepare_feature_split(features[train], features[validation])
            for train, validation in folds
        ]
    preparation_seconds = time.perf_counter() - preparation_started

    timing_path = report_dir / f"{args.arm}_timing.json"
    timing = None
    if timing_path.exists():
        candidate = json.loads(timing_path.read_text())
        if (
            candidate.get("arm") == args.arm
            and int(candidate.get("sampling_seed", -1)) == search_seed
            and int(candidate.get("representative_fits", -1))
            == int(config["search"]["timing_fits"])
        ):
            timing = dict(candidate)
            timing["configuration_count"] = len(configurations)
            timing["projected_search_gpu_hours"] = (
                0.0
                if architecture == "random_forest" or args.device == "cpu"
                else float(
                    timing["mean_fit_seconds"] * len(configurations) * len(folds) / 3600.0
                )
            )
            timing["one_time_fold_preparation_seconds"] = preparation_seconds
            timing["reused_completed_timing_audit"] = True
            timing_path.write_text(json.dumps(timing, indent=2) + "\n")
    if timing is None:
        timing_rows = []
        for index, parameters in enumerate(configurations[: int(config["search"]["timing_fits"])]):
            train, validation = folds[index % len(folds)]
            started = time.perf_counter()
            if architecture == "random_forest":
                model = _fit_rf(architecture, input_shape, parameters, features, target, train, search_seed + index)
                model.predict(features[validation])
            else:
                seed_everything(search_seed + index)
                model = build_model(architecture, input_shape, _model_parameters(architecture, parameters))
                fit_torch_predictor(
                    model, features[train], target[train], censoring[train], features[validation], target[validation], censoring[validation],
                    objective_config=config["objective"], optimization_config=config["optimization"],
                    parameters=parameters, seed=search_seed + index, device=args.device,
                    prepared_features=prepared_folds[index % len(folds)],
                )
            timing_rows.append(time.perf_counter() - started)
        timing = {
            "arm": args.arm,
            "representative_fits": len(timing_rows),
            "mean_fit_seconds": float(np.mean(timing_rows)),
            "configuration_count": len(configurations),
            "projected_search_gpu_hours": 0.0 if architecture == "random_forest" or args.device == "cpu" else float(np.mean(timing_rows) * len(configurations) * len(folds) / 3600.0),
            "sampling_seed": search_seed,
            "one_time_fold_preparation_seconds": preparation_seconds,
        }
        timing_path.write_text(json.dumps(timing, indent=2) + "\n")
    print(json.dumps(timing, indent=2), flush=True)
    if args.benchmark_only:
        return 0

    search_rows = load_search_state(partial_path, args.arm, len(configurations), len(folds))
    if bool(persistence["write_search_predictions"]):
        search_rows = [
            row for row in search_rows
            if (
                search_prediction_dir
                / f"config_{int(row['configuration']):03d}_fold_{int(row['fold']):02d}.parquet"
            ).is_file()
        ]
    if bool(persistence["save_search_checkpoints"]):
        extension = "joblib" if architecture == "random_forest" else "pt"
        search_rows = [
            row for row in search_rows
            if (
                search_checkpoint_dir
                / f"config_{int(row['configuration']):03d}_fold_{int(row['fold']):02d}.{extension}"
            ).is_file()
        ]
    complete = {(int(row["configuration"]), int(row["fold"])) for row in search_rows}
    for configuration_index, parameters in enumerate(configurations):
        for fold_index, (train, validation) in enumerate(folds):
            if (configuration_index, fold_index) in complete:
                continue
            seed = search_seed + configuration_index * len(folds) + fold_index
            if architecture == "random_forest":
                fit = _fit_rf(architecture, input_shape, parameters, features, target, train, seed)
                prediction = fit.predict(features[validation])
                best_epoch = 0
            else:
                seed_everything(seed)
                model = build_model(architecture, input_shape, _model_parameters(architecture, parameters))
                fit = fit_torch_predictor(
                    model, features[train], target[train], censoring[train], features[validation], target[validation], censoring[validation],
                    objective_config=config["objective"], optimization_config=config["optimization"],
                    parameters=parameters, seed=seed, device=args.device,
                    prepared_features=prepared_folds[fold_index],
                )
                prediction = predict_torch(fit.model, fit.normalizer, features[validation], device=args.device)
                best_epoch = fit.best_epoch + 1
            held_out = prediction_frame(
                dataset="IRED", step=2, arm=args.arm, config_id=configuration_index,
                fold=fold_index, seed=seed, condition="cv_validation",
                variant_rows=frame.iloc[validation]["variant_row"],
                targets=target[validation], predictions=prediction,
            )
            if bool(persistence["write_search_predictions"]):
                write_prediction_parquet(
                    held_out,
                    search_prediction_dir
                    / f"config_{configuration_index:03d}_fold_{fold_index:02d}.parquet",
                )
            if bool(persistence["save_search_checkpoints"]):
                if architecture == "random_forest":
                    import joblib
                    checkpoint = search_checkpoint_dir / f"config_{configuration_index:03d}_fold_{fold_index:02d}.joblib"
                    checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    joblib.dump(fit, checkpoint)
                else:
                    checkpoint = search_checkpoint_dir / f"config_{configuration_index:03d}_fold_{fold_index:02d}.pt"
                    _save_torch_checkpoint(fit, checkpoint, parameters, input_shape)
                upsert_checkpoint_manifest(
                    checkpoint_manifest, dataset="IRED", step=2, arm=args.arm,
                    config_id=configuration_index, fold=fold_index, seed=seed,
                    condition="cv_validation", checkpoint=checkpoint,
                )
            validation_spearman = float(spearmanr(prediction, target[validation]).statistic)
            search_rows.append({
                "arm": args.arm,
                "configuration": configuration_index,
                "fold": fold_index,
                "validation_spearman": validation_spearman,
                "best_epoch": best_epoch,
                **parameters,
            })
            write_search_state(pd.DataFrame(search_rows), partial_path)
            complete.add((configuration_index, fold_index))
        fold_values = [row["validation_spearman"] for row in search_rows if row["configuration"] == configuration_index]
        if len(fold_values) == len(folds):
            print(f"{args.arm} config {configuration_index + 1}/{len(configurations)} mean={np.mean(fold_values):.5f}", flush=True)
    search_frame = pd.DataFrame(search_rows)
    if len(search_frame) != len(configurations) * len(folds):
        raise RuntimeError("search ended without every configuration-fold fit")
    search_frame = search_frame.sort_values(["configuration", "fold"]).reset_index(drop=True)
    write_search_state(search_frame, report_dir / f"{args.arm}_search.csv")
    mean_scores = search_frame.groupby("configuration")["validation_spearman"].mean()
    selected_index = int(mean_scores.idxmax())
    selected = configurations[selected_index]
    selected_epochs = max(1, int(round(search_frame.loc[search_frame["configuration"] == selected_index, "best_epoch"].median())))
    if bool(persistence["commit_selected_cv_predictions"]):
        write_prediction_csv(
            pd.concat([
                pd.read_parquet(
                    search_prediction_dir
                    / f"config_{selected_index:03d}_fold_{fold_index:02d}.parquet"
                )
                for fold_index in range(len(folds))
            ], ignore_index=True),
            report_dir / f"{args.arm}_cv_predictions.csv",
        )

    rng = np.random.default_rng(int(config["evaluation"]["calibration_seed"]))
    shuffled = rng.permutation(train_pool)
    n_calibration = max(1, int(round(len(shuffled) * float(config["evaluation"]["calibration_fraction"]))))
    calibration, fit_indices = np.sort(shuffled[:n_calibration]), np.sort(shuffled[n_calibration:])
    round_train, round_test = get_split("IRED", "round_held_out")
    test_members, round_members, plate_members, calibration_members = [], [], [], []
    member_metrics = []
    model_dir = REPO_ROOT / config["paths"]["models"] / "IRED" / args.arm
    model_dir.mkdir(parents=True, exist_ok=True)
    condition_rows = []
    for seed in map(int, config["optimization"]["ensemble_seeds"]):
        if architecture == "random_forest":
            fitted = _fit_rf(architecture, input_shape, selected, features, target, fit_indices, seed)
            round_fitted = _fit_rf(architecture, input_shape, selected, features, target, round_train, seed)
            test_prediction = fitted.predict(features[test])
            round_prediction = round_fitted.predict(features[round_test])
            plate_prediction = fitted.predict(plate_features)
            calibration_prediction = fitted.predict(features[calibration])
            import joblib
            pg_checkpoint = model_dir / f"seed_{seed}.joblib"
            round_checkpoint = model_dir / f"seed_{seed}_round_held_out.joblib"
            if bool(persistence["save_evaluation_checkpoints"]):
                joblib.dump(fitted, pg_checkpoint)
                joblib.dump(round_fitted, round_checkpoint)
        else:
            seed_everything(seed)
            pg_model = build_model(
                architecture, input_shape, _model_parameters(architecture, selected)
            )
            fitted = fit_torch_fixed_epochs(
                pg_model,
                features[fit_indices], target[fit_indices], censoring[fit_indices],
                objective_config=config["objective"], optimization_config=config["optimization"],
                parameters=selected, epochs=selected_epochs, seed=seed, device=args.device,
            )
            seed_everything(seed)
            round_model = build_model(
                architecture, input_shape, _model_parameters(architecture, selected)
            )
            round_fitted = fit_torch_fixed_epochs(
                round_model,
                features[round_train], target[round_train], censoring[round_train],
                objective_config=config["objective"], optimization_config=config["optimization"],
                parameters=selected, epochs=selected_epochs, seed=seed, device=args.device,
            )
            test_prediction = predict_torch(fitted.model, fitted.normalizer, features[test], device=args.device)
            round_prediction = predict_torch(round_fitted.model, round_fitted.normalizer, features[round_test], device=args.device)
            plate_prediction = predict_torch(fitted.model, fitted.normalizer, plate_features, device=args.device)
            calibration_prediction = predict_torch(
                fitted.model, fitted.normalizer, features[calibration], device=args.device
            )
            pg_checkpoint = model_dir / f"seed_{seed}.pt"
            round_checkpoint = model_dir / f"seed_{seed}_round_held_out.pt"
            if bool(persistence["save_evaluation_checkpoints"]):
                _save_torch_checkpoint(fitted, pg_checkpoint, selected, input_shape)
                _save_torch_checkpoint(round_fitted, round_checkpoint, selected, input_shape)
        if bool(persistence["save_evaluation_checkpoints"]):
            for condition, checkpoint in (
                ("pg_kfold_test", pg_checkpoint), ("round_held_out", round_checkpoint)
            ):
                upsert_checkpoint_manifest(
                    checkpoint_manifest, dataset="IRED", step=2, arm=args.arm,
                    config_id=selected_index, fold="final", seed=seed,
                    condition=condition, checkpoint=checkpoint,
                )
        condition_rows.extend([
            prediction_frame(
                dataset="IRED", step=2, arm=args.arm, config_id=selected_index,
                fold="final", seed=seed, condition="pg_kfold_test",
                variant_rows=frame.iloc[test]["variant_row"], targets=target[test],
                predictions=test_prediction,
            ),
            prediction_frame(
                dataset="IRED", step=2, arm=args.arm, config_id=selected_index,
                fold="final", seed=seed, condition="round_held_out",
                variant_rows=frame.iloc[round_test]["variant_row"], targets=target[round_test],
                predictions=round_prediction,
            ),
            prediction_frame(
                dataset="IRED", step=2, arm=args.arm, config_id=selected_index,
                fold="final", seed=seed, condition="plate",
                variant_rows=np.arange(len(plate)), targets=plate["1/s"],
                predictions=plate_prediction,
            ),
            prediction_frame(
                dataset="IRED", step=2, arm=args.arm, config_id=selected_index,
                fold="final", seed=seed, condition="pg_kfold_calibration",
                variant_rows=frame.iloc[calibration]["variant_row"], targets=target[calibration],
                predictions=calibration_prediction,
            ),
        ])
        if bool(persistence["commit_member_condition_predictions"]):
            write_prediction_csv(
                pd.concat(condition_rows, ignore_index=True),
                report_dir / f"{args.arm}_predictions.csv",
            )
        test_members.append(test_prediction)
        round_members.append(round_prediction)
        plate_members.append(plate_prediction)
        calibration_members.append(calibration_prediction)
        member_metrics.append(score_metrics(test_prediction, target[test], top_k=config["evaluation"]["top_k"], wild_type_fitness=wild_type_fitness("IRED")))

    test_ensemble = combine_deep_ensemble(np.stack(test_members))
    round_ensemble = combine_deep_ensemble(np.stack(round_members))
    plate_ensemble = combine_deep_ensemble(np.stack(plate_members))
    calibration_ensemble = combine_deep_ensemble(np.stack(calibration_members))
    ensemble_rows = [
        prediction_frame(
            dataset="IRED", step=2, arm=args.arm, config_id=selected_index,
            fold="final", seed="ensemble", condition="pg_kfold_test",
            variant_rows=frame.iloc[test]["variant_row"], targets=target[test],
            predictions=test_ensemble.mean, uncertainty=test_ensemble.standard_deviation,
        ),
        prediction_frame(
            dataset="IRED", step=2, arm=args.arm, config_id=selected_index,
            fold="final", seed="ensemble", condition="round_held_out",
            variant_rows=frame.iloc[round_test]["variant_row"], targets=target[round_test],
            predictions=round_ensemble.mean, uncertainty=round_ensemble.standard_deviation,
        ),
        prediction_frame(
            dataset="IRED", step=2, arm=args.arm, config_id=selected_index,
            fold="final", seed="ensemble", condition="plate",
            variant_rows=np.arange(len(plate)), targets=plate["1/s"],
            predictions=plate_ensemble.mean, uncertainty=plate_ensemble.standard_deviation,
        ),
        prediction_frame(
            dataset="IRED", step=2, arm=args.arm, config_id=selected_index,
            fold="final", seed="ensemble", condition="pg_kfold_calibration",
            variant_rows=frame.iloc[calibration]["variant_row"], targets=target[calibration],
            predictions=calibration_ensemble.mean,
            uncertainty=calibration_ensemble.standard_deviation,
        ),
    ]
    if bool(persistence["commit_member_condition_predictions"]):
        write_prediction_csv(
            pd.concat(condition_rows + ensemble_rows, ignore_index=True),
            report_dir / f"{args.arm}_predictions.csv",
        )
    fit_positions = set().union(*_position_sets(frame.iloc[fit_indices]["mutated_positions"]))
    calibration_unseen = np.asarray([
        bool(value - fit_positions)
        for value in _position_sets(frame.iloc[calibration]["mutated_positions"])
    ])
    test_unseen = np.asarray([
        bool(value - fit_positions)
        for value in _position_sets(frame.iloc[test]["mutated_positions"])
    ])
    calibrator = MondrianConformal(float(config["evaluation"]["conformal_alpha"])).fit(
        calibration_ensemble.mean,
        target[calibration],
        frame.iloc[calibration]["n_mutations"].to_numpy(int),
        calibration_unseen,
    )
    test_counts = frame.iloc[test]["n_mutations"].to_numpy(int)
    test_strata = list(zip(test_counts.tolist(), test_unseen.tolist()))
    supported = np.asarray([key in calibrator.quantiles_ for key in test_strata])
    conformal = {
        "calibration_counts": {str(key): value for key, value in calibrator.counts_.items()},
        "unsupported_test_rows": int((~supported).sum()),
        "coverage_by_stratum": {},
    }
    if supported.any():
        intervals = calibrator.predict_interval(
            test_ensemble.mean[supported], test_counts[supported], test_unseen[supported]
        )
        conformal["coverage_by_stratum"] = {
            str(key): value
            for key, value in coverage_by_stratum(
                intervals, target[test][supported], test_counts[supported], test_unseen[supported]
            ).items()
        }
    result = {
        "dataset": "IRED",
        "arm": args.arm,
        "sampling_seed": search_seed,
        "configuration_count": len(configurations),
        "selected_configuration": selected,
        "selected_epochs": selected_epochs,
        "boundary_hyperparameters": boundary_hyperparameters(selected, space),
        "validation_spearman_mean": float(mean_scores.loc[selected_index]),
        "pg_kfold_test": _bootstrap(test_ensemble.mean, target[test], config, False),
        "round_held_out": _bootstrap(round_ensemble.mean, target[round_test], config, False),
        "plate": _bootstrap(plate_ensemble.mean, plate["1/s"].to_numpy(float), config, True),
        "member_metric_mean": {key: float(np.mean([value[key] for value in member_metrics])) for key in member_metrics[0]},
        "member_metric_sd": {key: float(np.std([value[key] for value in member_metrics], ddof=1)) for key in member_metrics[0]},
        "mean_epistemic_sd": float(test_ensemble.standard_deviation.mean()),
        "calibration_rows_reserved": len(calibration),
        "mondrian_conformal": conformal,
        "prediction_persistence": {
            "contract_version": int(persistence["contract_version"]),
            "selected_cv": f"Reports/step2/{args.arm}_cv_predictions.csv",
            "conditions": f"Reports/step2/{args.arm}_predictions.csv",
            "search_partitions": f"Cache/step2/IRED/search_predictions/{args.arm}",
            "checkpoint_manifest": f"Reports/step2/{args.arm}_checkpoint_manifest.json",
        },
    }
    (report_dir / f"{args.arm}_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
