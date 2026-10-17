#!/usr/bin/env python3
"""Tune and evaluate the Step 2 one-hot and AAIndex additive baselines."""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossfit import datasets
from crossfit.metrics import (
    gated_test_pairs,
    pair_metrics,
    pair_metrics_by_hamming,
    variant_metrics,
)
from crossfit.models import deterministic_folds, fit_additive_model, predict_additive
from crossfit.representations import (
    MutationVocabulary,
    aaindex_flat,
    load_georgiev_table,
    mutation_indicator,
)
from crossfit.utils import stable_hash


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--assays",
        nargs="*",
        help="Optional assay ids to run; existing rows for those assays are replaced.",
    )
    return parser.parse_args()


def _features(
    assay_id: str,
    frame: pd.DataFrame,
    representation: str,
    config: dict,
) -> tuple[sparse.csr_matrix, str]:
    wild_type = datasets.wt_sequence(assay_id)
    mutations = frame["mutations"].astype(str).tolist()
    if representation == "mutation_indicator":
        vocabulary = MutationVocabulary.from_mutations(wild_type, mutations)
        matrix = mutation_indicator(wild_type, mutations, vocabulary)
        metadata = json.dumps({"features": vocabulary.features}, separators=(",", ":"))
    elif representation == "aaindex_flat":
        table = load_georgiev_table(config["paths"]["georgiev_table"])
        matrix = aaindex_flat(wild_type, mutations, table)
        metadata = json.dumps(
            {"wild_type_length": len(wild_type), "channels": table.shape[1]},
            separators=(",", ":"),
        )
    else:
        raise ValueError(f"unknown additive representation: {representation}")
    return sparse.csr_matrix(matrix, dtype=np.float32), metadata


def _fit_kwargs(config: dict, *, max_epochs: int) -> dict:
    optimization = config["additive_baselines"]["optimization"]
    return {
        "learning_rate": optimization["learning_rate"],
        "max_epochs": max_epochs,
        "patience": optimization["patience"],
        "early_stopping_min_delta": optimization["early_stopping_min_delta"],
        "scheduler_factor": optimization["scheduler_factor"],
        "scheduler_patience": optimization["scheduler_patience"],
        "gradient_clip_norm": optimization["gradient_clip_norm"],
        "training_seed": optimization["seed"],
        "loss_config": config["loss"],
    }


def _replace_rows(path: Path, new: pd.DataFrame, assays: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path)
        old = old.loc[~old["assay_id"].isin(assays)]
        new = pd.concat((old, new), ignore_index=True)
    new.sort_values([column for column in ("assay_id", "representation", "n_train", "seed",
                                           "candidate", "fold", "hamming_distance")
                     if column in new.columns]).to_csv(path, index=False)


def _tune(
    assay_id: str,
    representation: str,
    matrix: sparse.csr_matrix,
    frame: pd.DataFrame,
    config: dict,
) -> tuple[dict, list[dict]]:
    baseline = config["additive_baselines"]
    tuning = baseline["tuning"]
    tune_indices, _ = datasets.get_split(
        assay_id,
        "low_n",
        N=tuning["n_train"],
        seed=tuning["source_seed"],
    )
    folds = deterministic_folds(
        len(tune_indices),
        tuning["folds"],
        stable_hash(assay_id, representation, "additive_cv", baseline["optimization"]["seed"]),
    )
    rows = []
    candidates = list(
        itertools.product(
            tuning["margin_sd"],
            tuning["regularization_weight"],
            tuning["weight_decay"],
        )
    )
    y = frame["y"].to_numpy(dtype=np.float32)
    sigma = frame["sigma_v"].to_numpy(dtype=np.float32)
    censor = frame["censored"].astype(str).to_numpy()
    for candidate_index, (margin, regularization, weight_decay) in enumerate(candidates):
        for fold_index, validation_local in enumerate(folds):
            training_local = np.concatenate(
                [fold for index, fold in enumerate(folds) if index != fold_index]
            )
            training = tune_indices[training_local]
            validation = tune_indices[validation_local]
            fit = fit_additive_model(
                matrix[training],
                y[training],
                censor[training],
                sigma[training],
                margin_sd=margin,
                regularization_weight=regularization,
                weight_decay=weight_decay,
                validation=(
                    matrix[validation],
                    y[validation],
                    censor[validation],
                    sigma[validation],
                ),
                **_fit_kwargs(config, max_epochs=baseline["optimization"]["max_epochs"]),
            )
            rows.append(
                {
                    "assay_id": assay_id,
                    "representation": representation,
                    "candidate": candidate_index,
                    "fold": fold_index,
                    "margin_sd": margin,
                    "regularization_weight": regularization,
                    "weight_decay": weight_decay,
                    "validation_spearman": fit.best_validation_spearman,
                    "best_epoch": fit.epochs,
                    "n_train_pairs": fit.n_pairs,
                    "status": fit.status,
                }
            )
    grid = pd.DataFrame(rows)
    candidate_summary = (
        grid.groupby(["candidate", "margin_sd", "regularization_weight", "weight_decay"], as_index=False)
        .agg(
            mean_validation_spearman=("validation_spearman", "mean"),
            finite_folds=("validation_spearman", "count"),
            selected_epochs=("best_epoch", "median"),
        )
        .sort_values(
            ["mean_validation_spearman", "finite_folds", "candidate"],
            ascending=[False, False, True],
        )
    )
    if candidate_summary["mean_validation_spearman"].notna().any():
        best = candidate_summary.loc[candidate_summary["mean_validation_spearman"].notna()].iloc[0]
    else:
        raise RuntimeError(f"{assay_id} {representation}: no finite validation Spearman")
    selected = {
        "assay_id": assay_id,
        "representation": representation,
        "margin_sd": float(best["margin_sd"]),
        "regularization_weight": float(best["regularization_weight"]),
        "weight_decay": float(best["weight_decay"]),
        "epochs": max(1, int(round(best["selected_epochs"]))),
        "mean_validation_spearman": float(best["mean_validation_spearman"]),
        "finite_folds": int(best["finite_folds"]),
        "feature_dim": matrix.shape[1],
    }
    return selected, rows


def _evaluate_representation(
    assay_id: str,
    representation: str,
    matrix: sparse.csr_matrix,
    metadata: str,
    frame: pd.DataFrame,
    selected: dict,
    config: dict,
) -> tuple[list[dict], list[dict]]:
    baseline = config["additive_baselines"]
    evaluation = baseline["evaluation"]
    y = frame["y"].to_numpy(dtype=np.float32)
    sigma = frame["sigma_v"].to_numpy(dtype=np.float32)
    censor = frame["censored"].astype(str).to_numpy()
    mutations = frame["mutations"].astype(str).to_numpy()
    run_rows = []
    distance_rows = []
    weights = []
    scales = []
    artifact_n = []
    artifact_seed = []
    statuses = []
    for n_train in baseline["low_n_sizes"]:
        for seed in baseline["seeds"]:
            training, test = datasets.get_split(assay_id, "low_n", N=n_train, seed=seed)
            fit = fit_additive_model(
                matrix[training],
                y[training],
                censor[training],
                sigma[training],
                margin_sd=selected["margin_sd"],
                regularization_weight=selected["regularization_weight"],
                weight_decay=selected["weight_decay"],
                validation=None,
                **_fit_kwargs(config, max_epochs=selected["epochs"]),
            )
            score = predict_additive(matrix[test], fit.weight)
            variant = variant_metrics(score, y[test], evaluation["top_k"])
            pair_seed = stable_hash(assay_id, n_train, seed, "additive_test_pairs")
            u, v = gated_test_pairs(
                y[test],
                censor[test],
                margin_sd=selected["margin_sd"],
                max_pairs=evaluation["max_pairs_per_run"],
                seed=pair_seed,
            )
            pair = pair_metrics(score, y[test], u, v)
            run_rows.append(
                {
                    "assay_id": assay_id,
                    "representation": representation,
                    "n_train": n_train,
                    "seed": seed,
                    "n_test": len(test),
                    "feature_dim": matrix.shape[1],
                    "margin_sd": selected["margin_sd"],
                    "regularization_weight": selected["regularization_weight"],
                    "weight_decay": selected["weight_decay"],
                    "epochs": fit.epochs,
                    "n_train_pairs": fit.n_pairs,
                    "n_test_pairs": len(u),
                    "residual_scale": fit.residual_scale,
                    "status": fit.status,
                    **variant,
                    **pair,
                }
            )
            if evaluation["hamming_stratified"]:
                for row in pair_metrics_by_hamming(score, y[test], mutations[test], u, v):
                    distance_rows.append(
                        {
                            "assay_id": assay_id,
                            "representation": representation,
                            "n_train": n_train,
                            "seed": seed,
                            **row,
                        }
                    )
            weights.append(fit.weight)
            scales.append(fit.residual_scale)
            artifact_n.append(n_train)
            artifact_seed.append(seed)
            statuses.append(fit.status)

    artifact_dir = Path(baseline["artifacts"]["models"])
    artifact_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact_dir / f"{assay_id}__{representation}.npz",
        weight=np.stack(weights),
        residual_scale=np.asarray(scales, dtype=np.float32),
        n_train=np.asarray(artifact_n, dtype=np.int32),
        seed=np.asarray(artifact_seed, dtype=np.int32),
        status=np.asarray(statuses, dtype=str),
        feature_metadata=np.asarray(metadata),
        selected_hyperparameters=np.asarray(json.dumps(selected, sort_keys=True)),
    )
    return run_rows, distance_rows


def _summary(runs: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "variant_spearman",
        "variant_pearson",
        "top_10_precision",
        "top_50_precision",
        "ndcg_10",
        "pair_spearman",
        "pair_pearson",
        "pair_rmse",
        "pair_sign_accuracy",
    ]
    grouped = runs.groupby(["assay_id", "representation", "n_train"])[metric_columns]
    summary = grouped.agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()
    counts = grouped.size().rename("n_seeds").reset_index()
    return summary.merge(counts, on=["assay_id", "representation", "n_train"])


def _gb1_sanity(runs: pd.DataFrame, config: dict) -> pd.DataFrame:
    evaluation = config["additive_baselines"]["evaluation"]
    gb1 = runs.loc[
        (runs["assay_id"] == "SPG1_STRSG_Olson_2014")
        & runs["n_train"].isin(evaluation["selection_sizes"])
    ]
    means = gb1.groupby("representation")["variant_spearman"].mean()
    if not {"mutation_indicator", "aaindex_flat"}.issubset(means.index):
        return pd.DataFrame()
    gap = float(means["aaindex_flat"] - means["mutation_indicator"])
    threshold = float(evaluation["gb1_substantial_spearman_gap"])
    return pd.DataFrame(
        [
            {
                "assay_id": "SPG1_STRSG_Olson_2014",
                "selection_sizes": ",".join(map(str, evaluation["selection_sizes"])),
                "one_hot_mean_spearman": means["mutation_indicator"],
                "aaindex_mean_spearman": means["aaindex_flat"],
                "aaindex_minus_one_hot": gap,
                "bug_threshold": threshold,
                "passes": gap <= threshold,
            }
        ]
    )


def main() -> None:
    args = _arguments()
    config = yaml.safe_load(Path("Code/config/step2.yaml").read_text())
    baseline = config["additive_baselines"]
    torch.set_num_threads(baseline["optimization"]["cpu_threads"])
    assay_ids = datasets.list_assays() if not args.assays else args.assays
    unknown = sorted(set(assay_ids) - set(datasets.list_assays()))
    if unknown:
        raise KeyError(f"unknown assays: {unknown}")

    tuning_rows = []
    selected_rows = []
    run_rows = []
    distance_rows = []
    for assay_id in assay_ids:
        frame = datasets.load_assay(assay_id)
        for representation in baseline["representations"]:
            matrix, metadata = _features(assay_id, frame, representation, config)
            selected, grid = _tune(assay_id, representation, matrix, frame, config)
            runs, distances = _evaluate_representation(
                assay_id, representation, matrix, metadata, frame, selected, config
            )
            tuning_rows.extend(grid)
            selected_rows.append(selected)
            run_rows.extend(runs)
            distance_rows.extend(distances)
            print(
                f"{assay_id} {representation}: features={matrix.shape[1]} "
                f"margin={selected['margin_sd']} epochs={selected['epochs']}"
            )

    artifacts = baseline["artifacts"]
    _replace_rows(Path(artifacts["tuning_grid"]), pd.DataFrame(tuning_rows), assay_ids)
    _replace_rows(
        Path(artifacts["selected_hyperparameters"]), pd.DataFrame(selected_rows), assay_ids
    )
    _replace_rows(Path(artifacts["runs"]), pd.DataFrame(run_rows), assay_ids)
    _replace_rows(Path(artifacts["pair_hamming"]), pd.DataFrame(distance_rows), assay_ids)
    all_runs = pd.read_csv(artifacts["runs"])
    _summary(all_runs).to_csv(artifacts["summary"], index=False)
    sanity = _gb1_sanity(all_runs, config)
    if not sanity.empty:
        sanity.to_csv(artifacts["gb1_sanity"], index=False)
        if not bool(sanity["passes"].iloc[0]):
            raise RuntimeError("AAIndex substantially exceeds one-hot on GB1; stop as preregistered")


if __name__ == "__main__":
    main()
