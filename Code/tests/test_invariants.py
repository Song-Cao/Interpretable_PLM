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


def test_registered_excluded_architectures_stay_absent():
    config = yaml.safe_load((REPO_ROOT / "Code" / "config" / "step2.yaml").read_text())
    text = _production_text() + "\n" + str(config)
    assert "gradient_boosted_trees" not in text
    assert "attention_pooling" not in text


def test_v13_is_the_only_plan_and_dictionary_uses_width_terminology():
    plans = sorted(path.name for path in REPO_ROOT.glob("CROSSFIT_Implementation_Plan_v*.docx"))
    assert plans == ["CROSSFIT_Implementation_Plan_v13.docx"]
    step4 = yaml.safe_load((REPO_ROOT / "Code/config/step4.yaml").read_text())
    assert step4["width_multipliers"] == [1, 2, 4, 8]
    assert not any("alpha" in key.lower() for key in _keys(step4))
    step6 = yaml.safe_load((REPO_ROOT / "Code/config/step6.yaml").read_text())
    step7 = yaml.safe_load((REPO_ROOT / "Code/config/step7.yaml").read_text())
    assert step6["batch_topk"]["k"] == step7["batch_topk"]["k"] == 32
    assert step6["decoder_bias"] is step7["decoder_bias"] is False
    assert step7["width_multipliers"] == [1, 2, 4, 8]
    assert not any("alpha" in key.lower() for config in (step6, step7) for key in _keys(config))
    source = (REPO_ROOT / "Code/crossfit/crosscoder.py").read_text().lower()
    assert "l1" not in source


def test_scheduler_metric_is_validation_spearman_everywhere():
    step2 = yaml.safe_load((REPO_ROOT / "Code" / "config" / "step2.yaml").read_text())
    step3 = yaml.safe_load((REPO_ROOT / "Code" / "config" / "step3.yaml").read_text())
    assert step2["optimization"]["scheduler_metric"] == "validation_spearman"
    assert step3["adapter"]["scheduler_metric"] == "validation_spearman"


def test_gate1_layer_rule_retains_collapse_and_midnetwork_guards():
    step3 = yaml.safe_load((REPO_ROOT / "Code" / "config" / "step3.yaml").read_text())
    gate = step3["gate1"]
    assert gate["real_cka_floor"] == 0.5
    assert gate["exclude_first_layers"] == 3
    assert gate["exclude_last_layers"] == 3
    assert gate["near_maximum_drift_gap_fraction"] == 0.9
    assert gate["task_gain_test"] == "paired_two_sided_t"
    assert gate["task_gain_alpha"] == 0.05
