from __future__ import annotations

import numpy as np

from crossfit.taxonomy import (
    FeatureStatistics,
    benjamini_hochberg,
    calibrate_pooled_beta_tail,
    calibrate_pooled_empirical_tail,
    provisional_classes,
)


def test_benjamini_hochberg_uses_largest_passing_rank():
    observed = benjamini_hochberg(np.array([0.001, 0.011, 0.03, 0.9]), 0.05)
    assert observed.tolist() == [True, True, True, False]


def test_pooled_beta_calibration_fits_one_null_and_reports_seed_uncertainty():
    rng = np.random.default_rng(12)
    nulls = [rng.beta(20, 20, 400), rng.beta(21, 19, 400), rng.beta(19, 21, 400)]
    real = np.r_[np.array([0.01, 0.02, 0.98, 0.99]), rng.beta(20, 20, 96)]
    rejected, cutoff, report = calibrate_pooled_beta_tail(
        real, nulls, tail="upper", q=0.05
    )
    assert report["calibration_rule"] == "pooled_null_samples"
    assert report["null_features"] == 1200
    assert report["null_features_by_run"] == [400, 400, 400]
    assert len(report["leave_one_null_run_out"]) == 3
    assert rejected[2:].any()
    assert np.isfinite(cutoff)


def test_pooled_empirical_calibration_is_not_a_conjunction_over_runs():
    nulls = [np.linspace(0.6, 1.0, 500), np.linspace(0.55, 0.95, 500)]
    real = np.r_[np.array([0.01, 0.02]), np.linspace(0.7, 0.9, 98)]
    rejected, cutoff, report = calibrate_pooled_empirical_tail(
        real, nulls, tail="lower", q=0.05
    )
    assert rejected[:2].all()
    assert cutoff == 0.02
    assert report["null_features"] == 1000
    assert len(report["leave_one_null_run_out"]) == 2


def test_taxonomy_applies_registered_precedence_and_joint_latent_gate():
    statistics = FeatureStatistics(
        delta=np.array([0.5, 0.95, 0.95, 0.05, 0.5, 0.5, 0.5]),
        evo_decoder_norm=np.array([0.0, 0.2, 0.2, 1.0, 1.0, 1.0, 1.0]),
        fit_decoder_norm=np.array([0.0, 1.0, 1.0, 0.2, 1.0, 1.0, 1.0]),
        decoder_cosine=np.ones(7), activation_count=np.ones(7),
        activation_frequency=np.ones(7), self_pair_activation_frequency=np.ones(7),
        activation_frequency_change=np.zeros(7), error_ratio=np.ones(7),
        reconstruction_ratio=np.ones(7),
    )
    labels = provisional_classes(
        statistics, lower_delta=np.array([0, 0, 0, 1, 0, 0, 0], dtype=bool),
        upper_delta=np.array([0, 1, 1, 0, 0, 0, 0], dtype=bool),
        latent_error_pass=np.array([0, 1, 1, 0, 0, 0, 0], dtype=bool),
        latent_reconstruction_pass=np.array([0, 1, 0, 0, 0, 0, 0], dtype=bool),
        rotated=np.array([0, 1, 1, 1, 1, 0, 0], dtype=bool),
        frequency_shifted=np.array([0, 1, 1, 1, 1, 1, 0], dtype=bool),
        floor_fraction=0.1, activation_floor_count=1,
    )
    assert labels.tolist() == [
        "dead", "emergent_candidate", "rotated", "evolution_only",
        "rotated", "re_weighted", "conserved",
    ]
