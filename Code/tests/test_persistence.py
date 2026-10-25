from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from crossfit.persistence import (
    PREDICTION_COLUMNS,
    prediction_frame,
    sha256_file,
    upsert_checkpoint_manifest,
    validate_prediction_frame,
    write_prediction_csv,
    write_prediction_parquet,
)
from crossfit.utils import REPO_ROOT


def test_prediction_schema_round_trips_csv_and_parquet(tmp_path: Path):
    frame = prediction_frame(
        dataset="IRED", step=3, arm="adapter", config_id="rank16", fold=2,
        seed=17, condition="cv_validation", variant_rows=[8, 4],
        targets=[0.2, 0.5], predictions=[0.3, 0.4], uncertainty=[0.1, 0.2],
    )
    assert list(frame.columns[: len(PREDICTION_COLUMNS)]) == list(PREDICTION_COLUMNS)
    csv_path = tmp_path / "predictions.csv"
    parquet_path = tmp_path / "predictions.parquet"
    write_prediction_csv(frame, csv_path)
    write_prediction_parquet(frame, parquet_path)
    validate_prediction_frame(pd.read_csv(csv_path))
    validate_prediction_frame(pd.read_parquet(parquet_path))


def test_prediction_schema_rejects_missing_or_duplicate_held_out_rows():
    with pytest.raises(ValueError, match="equal length"):
        prediction_frame(
            dataset="IRED", step=2, arm="ridge", config_id=0, fold=0, seed=0,
            condition="cv_validation", variant_rows=[1], targets=[0.0, 1.0],
            predictions=[0.5],
        )
    with pytest.raises(ValueError, match="one prediction"):
        prediction_frame(
            dataset="IRED", step=2, arm="ridge", config_id=0, fold=0, seed=0,
            condition="cv_validation", variant_rows=[1, 1], targets=[0.0, 1.0],
            predictions=[0.4, 0.5],
        )


def test_checkpoint_manifest_records_and_updates_hash(tmp_path: Path):
    checkpoint = tmp_path / "model.bin"
    manifest = tmp_path / "manifest.json"
    checkpoint.write_bytes(b"first")
    arguments = dict(
        dataset="IRED", step=3, arm="adapter", config_id="rank16", fold=0,
        seed=1, condition="cv_validation", checkpoint=checkpoint,
    )
    upsert_checkpoint_manifest(manifest, **arguments)
    first = json.loads(manifest.read_text())
    assert len(first) == 1
    assert first[0]["sha256"] == sha256_file(checkpoint)
    checkpoint.write_bytes(b"second")
    upsert_checkpoint_manifest(manifest, **arguments)
    second = json.loads(manifest.read_text())
    assert len(second) == 1
    assert second[0]["sha256"] != first[0]["sha256"]


def test_every_historical_fitted_result_has_predictions_or_an_explicit_legacy_audit():
    audit = pd.read_csv(REPO_ROOT / "Reports/step3/prediction_persistence_audit.csv")
    audited = set(audit.loc[audit["step"] == 2, "arm"])
    result_paths = sorted((REPO_ROOT / "Reports/step2").glob("*_result.json"))
    result_arms = set()
    for result_path in result_paths:
        result = json.loads(result_path.read_text())
        arm = str(result["arm"])
        result_arms.add(arm)
        prediction_path = result_path.with_name(result_path.name.replace("_result.json", "_predictions.csv"))
        assert prediction_path.is_file(), f"missing predictions for {arm}"
        predictions = pd.read_csv(prediction_path)
        if "prediction_persistence" in result:
            validate_prediction_frame(predictions)
            artifacts = result["prediction_persistence"]
            for key in ("selected_cv", "conditions"):
                path = REPO_ROOT / artifacts[key]
                assert path.is_file(), f"missing {key} predictions for {arm}"
                validate_prediction_frame(pd.read_csv(path))
        else:
            assert {"condition", "variant_row", "target", "prediction"}.issubset(predictions.columns)
            assert not predictions[["condition", "variant_row"]].duplicated().any()
            row = audit.loc[(audit["step"] == 2) & (audit["arm"] == arm)]
            assert len(row) == 1
            assert str(row.iloc[0]["cv_heldout_predictions"]).startswith("missing_")
    assert result_arms == audited


def test_persistence_is_enabled_by_default_for_future_fits():
    step2 = yaml.safe_load((REPO_ROOT / "Code/config/step2.yaml").read_text())
    step3 = yaml.safe_load((REPO_ROOT / "Code/config/step3.yaml").read_text())
    for key, value in step2["persistence"].items():
        if key != "contract_version":
            assert value is True, key
    for key, value in step3["persistence"].items():
        if key not in {"contract_version", "checkpoint_manifest"}:
            assert value is True, key


def test_step3_historical_fold_results_are_explicitly_audited_as_unrecoverable():
    audit = pd.read_csv(REPO_ROOT / "Reports/step3/prediction_persistence_audit.csv")
    step3 = audit.loc[audit["step"] == 3]
    assert set(step3["arm"]) == {
        "rank16_lr1e-4_registered", "rank16_lr3e-4", "rank32_lr1e-4", "rank32_lr3e-4",
    }
    registered = step3.loc[step3["arm"] == "rank16_lr1e-4_registered"].iloc[0]
    assert registered["cv_heldout_predictions"].startswith("fold0_")
    unavailable = step3.loc[step3["arm"] != "rank16_lr1e-4_registered"]
    assert unavailable["cv_heldout_predictions"].str.startswith("missing_").all()
    backfill = pd.read_csv(
        REPO_ROOT / "Reports/step3/rank16_lr1e-4_fold0_checkpoint_backfill_predictions.csv"
    )
    assert len(backfill) == 358
    assert backfill.groupby("arm").size().to_dict() == {
        "real_adapter": 179,
        "shuffled_adapter": 179,
    }


def test_gate1_checkpoint_backfill_is_inference_only():
    source = (REPO_ROOT / "Code/scripts/backfill_ired_gate1_fold0_predictions.py").read_text()
    assert "load_trainable_state" in source
    assert "predict_adapter" in source
    assert "fit_adapter" not in source
    assert "SequenceFitnessAdapter(adapted, int(backbone.embed.embedding_dim)).to(device)" in source
