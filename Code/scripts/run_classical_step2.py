#!/usr/bin/env python3
"""Search and fit one exact classical IRED baseline arm."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossfit.datasets import (
    get_split,
    load_assay,
    load_plate_dataset,
    wild_type_fitness,
)
from crossfit.evaluation import MondrianConformal, bootstrap_metric_intervals, coverage_by_stratum
from crossfit.models import DiagonalRidgeRegressor
from crossfit.persistence import (
    prediction_frame,
    upsert_checkpoint_manifest,
    write_prediction_csv,
    write_prediction_parquet,
)
from crossfit.representations import aaindex_flat, load_aaindex_table, onehot
from crossfit.search import boundary_hyperparameters, sample_configurations, training_pool_folds
from crossfit.utils import REPO_ROOT, load_config, stable_hash


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--representation", choices=("onehot", "aaindex_flat"), required=True)
    parser.add_argument("--augmentation", choices=("unaugmented", "augmented"), required=True)
    parser.add_argument("--search-count", type=int, default=None)
    parser.add_argument("--benchmark-only", action="store_true")
    return parser.parse_args()


def _features(sequences: list[str], representation: str, config: dict) -> np.ndarray:
    if representation == "onehot":
        return onehot(sequences)
    table = load_aaindex_table(REPO_ROOT / config["paths"]["aaindex_table"])
    return aaindex_flat(sequences, table)


def _density_lookup(config: dict) -> dict[str, float]:
    path = REPO_ROOT / config["paths"]["density_cache"]
    if not path.exists():
        raise FileNotFoundError(f"augmented arm requires the pod-generated density cache: {path}")
    frame = pd.read_csv(path)
    if frame.columns.tolist() != ["aa_seq", "dpll"] or frame["aa_seq"].duplicated().any():
        raise ValueError("density cache must contain unique aa_seq,dpll rows")
    return dict(zip(frame["aa_seq"], frame["dpll"].astype(float)))


def _density(sequences: list[str], lookup: dict[str, float]) -> np.ndarray:
    missing = [sequence for sequence in sequences if sequence not in lookup]
    if missing:
        raise KeyError(f"density cache is missing {len(missing)} sequences")
    return np.asarray([lookup[sequence] for sequence in sequences], dtype=np.float64)


def _fit_predict(
    features: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    evaluation: np.ndarray,
    penalty: float,
    density: np.ndarray | None,
    density_penalty: float,
) -> tuple[DiagonalRidgeRegressor, np.ndarray]:
    model = DiagonalRidgeRegressor(penalty, density_penalty).fit(
        features[train], target[train], density=None if density is None else density[train]
    )
    prediction = model.predict(
        features[evaluation], density=None if density is None else density[evaluation]
    )
    return model, prediction


def _save_checkpoint(
    model: DiagonalRidgeRegressor,
    path: Path,
    *,
    site_penalty: float,
    density_penalty: float,
    fit_variant_rows: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        coefficient=model.coefficient_,
        intercept=model.intercept_,
        site_penalty=site_penalty,
        density_penalty=density_penalty,
        fit_variant_rows=np.asarray(fit_variant_rows, dtype=int),
    )


def _mutation_counts(mutations: pd.Series) -> np.ndarray:
    return mutations.map(lambda value: 0 if value == "WT" else len(str(value).split(":"))).to_numpy(int)


def _position_sets(values: pd.Series) -> list[set[int]]:
    return [set() if pd.isna(value) or str(value) == "" else set(map(int, str(value).split(";"))) for value in values]


def _bootstrap(
    prediction: np.ndarray,
    target: np.ndarray,
    config: dict,
    *,
    rank_only: bool,
) -> dict:
    evaluation = config["evaluation"]
    return bootstrap_metric_intervals(
        prediction,
        target,
        samples=int(evaluation["bootstrap_samples"]),
        seed=int(evaluation["bootstrap_seed"]),
        confidence_level=float(evaluation["confidence_level"]),
        top_k=() if rank_only else evaluation["top_k"],
        wild_type_fitness=None if rank_only else wild_type_fitness("IRED"),
        rank_only=rank_only,
    )


def main() -> int:
    args = _arguments()
    config = load_config(REPO_ROOT / "Code" / "config" / "step2.yaml")
    frame = load_assay("IRED", with_sequences=True)
    sequences = frame["mutant_sequence"].tolist()
    features = _features(sequences, args.representation, config)
    target = frame["y"].to_numpy(float)
    train_pool, test = get_split("IRED", "pg_kfold", fold=int(config["evaluation"]["primary_test_fold"]))
    folds = training_pool_folds(
        train_pool, int(config["search"]["cv_folds"]), int(config["search"]["cv_seed"])
    )
    lookup = _density_lookup(config) if args.augmentation == "augmented" else {}
    density = _density(sequences, lookup) if lookup else None
    space = config["search"]["spaces"]["diagonal_ridge"]
    count = args.search_count or int(config["search"]["maximum_configurations_per_arm"])
    arm = f"{args.representation}_ridge_{args.augmentation}"
    search_seed = int(config["search"]["sampling_seed"]) + stable_hash(arm) % 1_000_000
    configurations = sample_configurations(space, count, search_seed)

    timings = []
    for configuration in configurations[: int(config["search"]["timing_fits"])]:
        started = time.perf_counter()
        train, validation = folds[len(timings) % len(folds)]
        _fit_predict(
            features, target, train, validation, float(configuration["site_penalty"]), density,
            float(configuration["density_penalty"]),
        )
        timings.append(time.perf_counter() - started)
    projected_seconds = float(np.mean(timings) * len(configurations) * len(folds))
    timing_report = {
        "arm": arm,
        "representative_fits": len(timings),
        "mean_fit_seconds": float(np.mean(timings)),
        "projected_search_seconds": projected_seconds,
        "configuration_count": len(configurations),
    }
    report_dir = REPO_ROOT / config["paths"]["reports"]
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / f"{arm}_timing.json").write_text(json.dumps(timing_report, indent=2) + "\n")
    print(json.dumps(timing_report, indent=2))
    if args.benchmark_only:
        return 0

    persistence = config["persistence"]
    search_prediction_dir = REPO_ROOT / config["paths"]["search_predictions"] / arm
    search_checkpoint_dir = REPO_ROOT / config["paths"]["search_checkpoints"] / arm
    checkpoint_manifest = report_dir / f"{arm}_checkpoint_manifest.json"
    rows = []
    for config_index, configuration in enumerate(configurations):
        fold_scores = []
        for fold_index, (train, validation) in enumerate(folds):
            model, prediction = _fit_predict(
                features, target, train, validation, float(configuration["site_penalty"]), density,
                float(configuration["density_penalty"]),
            )
            fold_seed = search_seed + config_index * len(folds) + fold_index
            held_out = prediction_frame(
                dataset="IRED", step=2, arm=arm, config_id=config_index, fold=fold_index,
                seed=fold_seed, condition="cv_validation",
                variant_rows=frame.iloc[validation]["variant_row"],
                targets=target[validation], predictions=prediction,
            )
            if bool(persistence["write_search_predictions"]):
                write_prediction_parquet(
                    held_out,
                    search_prediction_dir / f"config_{config_index:03d}_fold_{fold_index:02d}.parquet",
                )
            if bool(persistence["save_search_checkpoints"]):
                checkpoint = search_checkpoint_dir / f"config_{config_index:03d}_fold_{fold_index:02d}.npz"
                _save_checkpoint(
                    model, checkpoint,
                    site_penalty=float(configuration["site_penalty"]),
                    density_penalty=float(configuration["density_penalty"]),
                    fit_variant_rows=frame.iloc[train]["variant_row"].to_numpy(int),
                )
                upsert_checkpoint_manifest(
                    checkpoint_manifest, dataset="IRED", step=2, arm=arm,
                    config_id=config_index, fold=fold_index, seed=fold_seed,
                    condition="cv_validation", checkpoint=checkpoint,
                )
            score = pd.Series(prediction).corr(pd.Series(target[validation]), method="spearman")
            fold_scores.append(float(score))
            rows.append(
                {
                    "arm": arm,
                    "configuration": config_index,
                    "fold": fold_index,
                    "validation_spearman": float(score),
                    **configuration,
                }
            )
        print(f"{arm} config {config_index + 1}/{len(configurations)} mean={np.mean(fold_scores):.5f}")
    search_frame = pd.DataFrame(rows)
    search_frame.to_csv(report_dir / f"{arm}_search.csv", index=False)
    means = search_frame.groupby("configuration")["validation_spearman"].mean()
    selected_index = int(means.idxmax())
    selected = configurations[selected_index]
    if bool(persistence["commit_selected_cv_predictions"]):
        selected_folds = [
            pd.read_parquet(
                search_prediction_dir / f"config_{selected_index:03d}_fold_{fold_index:02d}.parquet"
            )
            for fold_index in range(len(folds))
        ]
        write_prediction_csv(
            pd.concat(selected_folds, ignore_index=True),
            report_dir / f"{arm}_cv_predictions.csv",
        )

    evaluation = config["evaluation"]
    calibration_rng = np.random.default_rng(int(evaluation["calibration_seed"]))
    shuffled = calibration_rng.permutation(train_pool)
    n_calibration = max(1, int(round(len(shuffled) * float(evaluation["calibration_fraction"]))))
    calibration = np.sort(shuffled[:n_calibration])
    fit_indices = np.sort(shuffled[n_calibration:])
    model, test_prediction = _fit_predict(
        features, target, fit_indices, test, float(selected["site_penalty"]), density,
        float(selected["density_penalty"]),
    )
    _, calibration_prediction = _fit_predict(
        features, target, fit_indices, calibration, float(selected["site_penalty"]), density,
        float(selected["density_penalty"]),
    )
    model_dir = REPO_ROOT / config["paths"]["models"] / "IRED" / arm
    model_dir.mkdir(parents=True, exist_ok=True)
    round_train, round_test = get_split("IRED", "round_held_out")
    round_model, round_prediction = _fit_predict(
        features, target, round_train, round_test, float(selected["site_penalty"]), density,
        float(selected["density_penalty"]),
    )
    plate = load_plate_dataset("IRED")
    plate_sequences = plate["aa_seq"].astype(str).tolist()
    plate_features = _features(plate_sequences, args.representation, config)
    plate_density = _density(plate_sequences, lookup) if lookup else None
    plate_prediction = model.predict(plate_features, density=plate_density)

    condition_rows = []
    for seed in map(int, config["optimization"]["ensemble_seeds"]):
        pg_checkpoint = model_dir / f"seed_{seed}.npz"
        round_checkpoint = model_dir / f"seed_{seed}_round_held_out.npz"
        if bool(persistence["save_evaluation_checkpoints"]):
            _save_checkpoint(
                model, pg_checkpoint,
                site_penalty=float(selected["site_penalty"]),
                density_penalty=float(selected["density_penalty"]),
                fit_variant_rows=frame.iloc[fit_indices]["variant_row"].to_numpy(int),
            )
            _save_checkpoint(
                round_model, round_checkpoint,
                site_penalty=float(selected["site_penalty"]),
                density_penalty=float(selected["density_penalty"]),
                fit_variant_rows=frame.iloc[round_train]["variant_row"].to_numpy(int),
            )
            for condition, checkpoint in (
                ("pg_kfold_test", pg_checkpoint), ("round_held_out", round_checkpoint)
            ):
                upsert_checkpoint_manifest(
                    checkpoint_manifest, dataset="IRED", step=2, arm=arm,
                    config_id=selected_index, fold="final", seed=seed,
                    condition=condition, checkpoint=checkpoint,
                )
        condition_rows.extend([
            prediction_frame(
                dataset="IRED", step=2, arm=arm, config_id=selected_index,
                fold="final", seed=seed, condition="pg_kfold_test",
                variant_rows=frame.iloc[test]["variant_row"], targets=target[test],
                predictions=test_prediction,
            ),
            prediction_frame(
                dataset="IRED", step=2, arm=arm, config_id=selected_index,
                fold="final", seed=seed, condition="round_held_out",
                variant_rows=frame.iloc[round_test]["variant_row"], targets=target[round_test],
                predictions=round_prediction,
            ),
            prediction_frame(
                dataset="IRED", step=2, arm=arm, config_id=selected_index,
                fold="final", seed=seed, condition="plate",
                variant_rows=np.arange(len(plate)), targets=plate["1/s"],
                predictions=plate_prediction,
            ),
            prediction_frame(
                dataset="IRED", step=2, arm=arm, config_id=selected_index,
                fold="final", seed=seed, condition="pg_kfold_calibration",
                variant_rows=frame.iloc[calibration]["variant_row"], targets=target[calibration],
                predictions=calibration_prediction,
            ),
        ])
        if bool(persistence["commit_member_condition_predictions"]):
            write_prediction_csv(
                pd.concat(condition_rows, ignore_index=True),
                report_dir / f"{arm}_predictions.csv",
            )

    fit_positions = set().union(*_position_sets(frame.iloc[fit_indices]["mutated_positions"]))
    calibration_unseen = np.asarray(
        [bool(value - fit_positions) for value in _position_sets(frame.iloc[calibration]["mutated_positions"])]
    )
    test_unseen = np.asarray(
        [bool(value - fit_positions) for value in _position_sets(frame.iloc[test]["mutated_positions"])]
    )
    calibrator = MondrianConformal(float(evaluation["conformal_alpha"])).fit(
        calibration_prediction,
        target[calibration],
        frame.iloc[calibration]["n_mutations"].to_numpy(int),
        calibration_unseen,
    )
    test_strata = list(
        zip(frame.iloc[test]["n_mutations"].astype(int).tolist(), test_unseen.tolist())
    )
    supported = np.asarray([key in calibrator.quantiles_ for key in test_strata])
    conformal = {
        "calibration_counts": {str(key): value for key, value in calibrator.counts_.items()},
        "unsupported_test_rows": int((~supported).sum()),
        "coverage_by_stratum": {},
    }
    if supported.any():
        intervals = calibrator.predict_interval(
            test_prediction[supported],
            frame.iloc[test]["n_mutations"].to_numpy(int)[supported],
            test_unseen[supported],
        )
        conformal["coverage_by_stratum"] = {
            str(key): value
            for key, value in coverage_by_stratum(
                intervals,
                target[test][supported],
                frame.iloc[test]["n_mutations"].to_numpy(int)[supported],
                test_unseen[supported],
            ).items()
        }

    boundaries = boundary_hyperparameters(selected, space)
    # The density penalty is fixed at 1e-8 by the published formulation; it is
    # not a searched range and therefore cannot be a selected search boundary.
    boundaries.pop("density_penalty", None)
    result = {
        "dataset": "IRED",
        "arm": arm,
        "selection_split": "five-fold CV within pg_kfold fold-0 training pool",
        "selected_configuration": selected,
        "boundary_hyperparameters": boundaries,
        "validation_spearman_mean": float(means.loc[selected_index]),
        "pg_kfold_test": _bootstrap(test_prediction, target[test], config, rank_only=False),
        "round_held_out": _bootstrap(round_prediction, target[round_test], config, rank_only=False),
        "plate": _bootstrap(plate_prediction, plate["1/s"].to_numpy(float), config, rank_only=True),
        "ensemble_members": len(config["optimization"]["ensemble_seeds"]),
        "ensemble_sd_note": "zero: the exact convex ridge solution is seed-invariant",
        "calibration_rows": len(calibration),
        "fit_rows": len(fit_indices),
        "mondrian_conformal": conformal,
        "prediction_persistence": {
            "contract_version": int(persistence["contract_version"]),
            "selected_cv": f"Reports/step2/{arm}_cv_predictions.csv",
            "conditions": f"Reports/step2/{arm}_predictions.csv",
            "search_partitions": f"Cache/step2/IRED/search_predictions/{arm}",
            "checkpoint_manifest": f"Reports/step2/{arm}_checkpoint_manifest.json",
        },
    }
    (report_dir / f"{arm}_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
