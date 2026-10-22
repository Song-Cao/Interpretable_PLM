from __future__ import annotations

import hashlib
from pathlib import Path

import yaml

from crossfit.utils import REPO_ROOT


def _production_text() -> str:
    roots = [REPO_ROOT / "Code" / "crossfit", REPO_ROOT / "Code" / "scripts"]
    return "\n".join(path.read_text() for root in roots for path in root.glob("*.py"))


def _keys(value):
    if isinstance(value, dict):
        for key, child in value.items():
            yield str(key)
            yield from _keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from _keys(child)


def test_superseded_modules_stay_deleted():
    package = REPO_ROOT / "Code" / "crossfit"
    scripts = REPO_ROOT / "Code" / "scripts"
    for relative in (
        package / "sampling.py",
        package / "optimization.py",
        package / "pair_analysis.py",
        scripts / "analyze_pair_distances.py",
    ):
        assert not relative.exists()


def test_superseded_pairwise_and_cross_dataset_symbols_stay_absent():
    text = _production_text()
    forbidden = (
        "step_across_assays",
        "grouped_assay_folds",
        "prediction_groups",
        "aggregate_pairwise_scores",
        "censor_constraint_sign",
        "construct_pairs",
        "PairBatch",
        "PairDataset",
        "pair_dataset",
        "mutation_" + "indicator",
    )
    for symbol in forbidden:
        assert symbol not in text


def test_superseded_config_keys_stay_absent():
    config = yaml.safe_load((REPO_ROOT / "Code" / "config" / "step2.yaml").read_text())
    keys = set(_keys(config))
    assert "max_" + "hamming_distance" not in keys
    assert not any("sigma_v" in key for key in keys)


def test_frozen_aaindex_table_matches_recorded_hash():
    config = yaml.safe_load((REPO_ROOT / "Code" / "config" / "step2.yaml").read_text())
    path = REPO_ROOT / config["paths"]["aaindex_table"]
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    assert observed == config["representations"]["aaindex_sha256"]


def test_v12_excluded_architectures_stay_absent():
    config = yaml.safe_load((REPO_ROOT / "Code" / "config" / "step2.yaml").read_text())
    text = _production_text() + "\n" + str(config)
    assert "gradient_boosted_trees" not in text
    assert "attention_pooling" not in text


def test_scheduler_metric_is_validation_spearman_everywhere():
    step2 = yaml.safe_load((REPO_ROOT / "Code" / "config" / "step2.yaml").read_text())
    step3 = yaml.safe_load((REPO_ROOT / "Code" / "config" / "step3.yaml").read_text())
    assert step2["optimization"]["scheduler_metric"] == "validation_spearman"
    assert step3["adapter"]["scheduler_metric"] == "validation_spearman"
