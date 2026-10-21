#!/usr/bin/env python3
"""Curate the IRED campaign and its external plate evaluation set."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crossfit import labels, qc as qcmod
from crossfit.qc import apply_mutations
from crossfit.splits import carve_layer_freeze, kfold_fallback, mutated_positions
from crossfit.utils import REPO_ROOT, load_config


def _path(config: dict, key: str) -> Path:
    return REPO_ROOT / config["paths"][key]


def _canonical_mutation(value: object, delimiter: str) -> str:
    mutation = str(value).strip()
    return "WT" if mutation == "WT" else ":".join(mutation.split(delimiter))


def _validate_reconstruction(frame: pd.DataFrame, wild_type: str) -> None:
    rebuilt = [apply_mutations(wild_type, mutation) for mutation in frame["mutant"]]
    bad = sum(observed != expected for observed, expected in zip(frame["sequence"], rebuilt))
    if bad:
        raise ValueError(f"{bad} IRED sequences disagree with their mutation strings")


def main() -> int:
    step1 = load_config()
    step2 = load_config(REPO_ROOT / "Code" / "config" / "step2.yaml")
    settings = step2["ired_curation"]
    dataset_id = settings["dataset_id"]
    columns = settings["campaign_columns"]

    source = pd.read_csv(
        _path(step2, "ired_campaign"),
        usecols=[columns["sequence"], columns["mutations"], columns["fitness"], columns["round"]],
    ).rename(
        columns={
            columns["sequence"]: "sequence",
            columns["mutations"]: "mutant",
            columns["fitness"]: "DMS Score",
            columns["round"]: "start_round",
        }
    )
    source["mutant"] = source["mutant"].map(
        lambda value: _canonical_mutation(value, settings["mutation_delimiter"])
    )
    wild_type_rows = source.loc[source["mutant"] == "WT"]
    if len(wild_type_rows) != 1:
        raise ValueError(f"IRED must contain exactly one wild type; found {len(wild_type_rows)}")
    wild_type = str(wild_type_rows["sequence"].iloc[0])
    wild_type_fitness = float(wild_type_rows["DMS Score"].iloc[0])
    _validate_reconstruction(source, wild_type)

    qcfg = step1["qc"]
    curated, record = qcmod.run_assay_qc(
        dataset_id,
        source,
        wild_type,
        score_col="DMS Score",
        censor_eps=qcfg["censor_eps"],
        censor_mass_frac=qcfg["censor_mass_frac"],
        dup_disagreement_sigma=qcfg["duplicate_disagreement_sigma"],
    )
    curated, sigma_source, n_real_sigma = labels.attach_sigma(curated)
    curated = labels.standardize(curated)
    if int((curated["censored"] == "ceiling").sum()) != 0:
        raise AssertionError("IRED has ceiling-censored rows; validate that Tobit branch first")

    round_by_sequence = source.set_index("sequence")["start_round"]
    curated["start_round"] = curated["sequence"].map(round_by_sequence).astype(int)
    reserve = carve_layer_freeze(
        curated,
        dataset_id,
        float(settings["layer_freeze_fraction"]),
        int(settings["layer_freeze_seed"]),
    )
    curated["is_layer_freeze_reserve"] = reserve
    folds = kfold_fallback(
        curated,
        dataset_id,
        int(settings["pg_kfold_folds"]),
        int(settings["split_seed"]),
    )
    curated["variant_row"] = np.arange(len(curated), dtype=int)
    curated["assay_id"] = dataset_id
    curated["mutated_positions"] = curated["mutant"].map(
        lambda value: ";".join(map(str, mutated_positions(value)))
    )
    variants = curated.rename(columns={"mutant": "mutations"})[
        [
            "assay_id", "variant_row", "mutations", "n_mutations", "mutated_positions",
            "y_raw", "y", "sigma_v", "sigma_source", "n_measurements", "censored",
            "dup_disagrees", "is_layer_freeze_reserve", "start_round",
        ]
    ]
    variant_path = _path(step2, "ired_variants")
    variant_path.parent.mkdir(parents=True, exist_ok=True)
    variants.to_parquet(variant_path, index=False)

    split_dir = _path(step2, "ired_pg_kfold").parent
    split_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"assay_id": dataset_id, "scheme": "pg_kfold", "variant_row": variants["variant_row"], "fold": folds}
    ).to_csv(_path(step2, "ired_pg_kfold"), index=False)
    train_rounds = set(map(int, settings["round_train"]))
    test_rounds = set(map(int, settings["round_test"]))
    if train_rounds & test_rounds or set(curated["start_round"]) != train_rounds | test_rounds:
        raise ValueError("round-held-out config must partition every observed IRED round")
    round_fold = np.where(curated["start_round"].isin(test_rounds), 1, 0)
    pd.DataFrame(
        {
            "assay_id": dataset_id,
            "scheme": "round_held_out",
            "variant_row": variants["variant_row"],
            "fold": round_fold,
        }
    ).to_csv(_path(step2, "ired_round_held_out"), index=False)

    plate_columns = settings["plate_columns"]
    plate = pd.read_csv(_path(step2, "ired_plate"), usecols=plate_columns)[plate_columns]
    plate["mut"] = plate["mut"].map(
        lambda value: _canonical_mutation(value, settings["mutation_delimiter"])
    )
    plate_for_check = plate.rename(columns={"aa_seq": "sequence", "mut": "mutant"})
    _validate_reconstruction(plate_for_check, wild_type)
    plate_raw_rows = len(plate)
    plate = plate.groupby(["aa_seq", "mut"], as_index=False, sort=False)["1/s"].mean()
    plate.to_csv(_path(step2, "ired_plate_curated"), index=False)

    processed = REPO_ROOT / "Data" / "processed"
    manifest_path = processed / "assay_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    manifest = manifest.loc[manifest["assay_id"] != dataset_id].copy()
    manifest_row = {
        "assay_id": dataset_id,
        "slot": "IRED-first",
        "wt_sequence": wild_type,
        "protein_id": "IRED",
        "selection_assay": "directed evolution campaign fitness",
        "source": "in-house",
        "organism": "not recorded",
        "phenotype_class": "Activity",
        "per_variant_sigma_present": False,
        "wt_len": len(wild_type),
        "n_variants_raw": len(source),
        "n_variants_curated": len(variants),
        "n_single": int((variants["n_mutations"] == 1).sum()),
        "n_multi": int((variants["n_mutations"] > 1).sum()),
        "max_n_mutations": int(variants["n_mutations"].max()),
        "censor_floor": bool(record.censor_floor_flagged),
        "censor_ceil": bool(record.censor_ceil_flagged),
        "ties_frac": record.ties_frac,
        "pg_kfold_provenance": "re_derived_deterministic",
        "wt_fitness_raw": wild_type_fitness,
        "external_plate_rows": len(plate),
    }
    manifest = pd.concat([manifest, pd.DataFrame([manifest_row])], ignore_index=True)
    manifest.to_csv(manifest_path, index=False)

    qc_path = processed / "qc_report.csv"
    qc_report = pd.read_csv(qc_path)
    qc_report = qc_report.loc[qc_report["assay_id"] != dataset_id].copy()
    qc_row = record.as_row()
    qc_row.update(
        orientation="higher_is_fitter_by_campaign_definition",
        sigma_dominant_source=sigma_source,
        n_variants_with_replicate_sigma=n_real_sigma,
        pg_kfold_provenance="re_derived_deterministic",
        mce_defined=True,
    )
    for key in qc_row:
        if key not in qc_report.columns:
            qc_report[key] = pd.NA
    qc_report = qc_report.astype(object)
    qc_report.loc[len(qc_report), list(qc_row)] = list(qc_row.values())
    qc_report.to_csv(qc_path, index=False)

    summary = {
        "dataset_id": dataset_id,
        "source_rows": len(source),
        "curated_variants_excluding_wild_type": len(variants),
        "wild_type_length": len(wild_type),
        "wild_type_fitness_raw": wild_type_fitness,
        "layer_freeze_reserve": int(reserve.sum()),
        "usable_default": int((~reserve).sum()),
        "floor_censored": int((variants["censored"] == "floor").sum()),
        "ceiling_censored": int((variants["censored"] == "ceiling").sum()),
        "plate_source_rows": plate_raw_rows,
        "plate_unique_variants": len(plate),
        "plate_campaign_overlap": int(plate["aa_seq"].isin(source["sequence"]).sum()),
        "round_counts": {str(key): int(value) for key, value in curated["start_round"].value_counts().sort_index().items()},
        "mutation_count_distribution": {
            str(key): int(value) for key, value in variants["n_mutations"].value_counts().sort_index().items()
        },
    }
    report_path = _path(step2, "reports") / "ired_curation_summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
