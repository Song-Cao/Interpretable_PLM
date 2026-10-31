from __future__ import annotations

import numpy as np

from crossfit.taxonomy import (
    FeatureStatistics,
    benjamini_hochberg,
    provisional_classes,
)


def test_benjamini_hochberg_uses_largest_passing_rank():
    observed = benjamini_hochberg(np.array([0.001, 0.011, 0.03, 0.9]), 0.05)
    assert observed.tolist() == [True, True, True, False]


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
