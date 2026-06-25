"""CPU unit tests for domain-shift severity metrics (pure math)."""

from __future__ import annotations

import numpy as np

from cma.metrics.severity import (
    appearance_severity,
    feature_stats,
    frechet_distance,
    scale_severity,
)


def test_frechet_zero_for_identical_gaussians():
    mu = np.array([1.0, -2.0, 0.5])
    cov = np.diag([1.0, 2.0, 0.5])
    assert abs(frechet_distance(mu, cov, mu, cov)) < 1e-6


def test_frechet_equals_mean_gap_when_cov_equal():
    cov = np.eye(3)
    mu1 = np.zeros(3)
    mu2 = np.array([1.0, 2.0, 2.0])  # ||.||^2 = 9
    assert abs(frechet_distance(mu1, cov, mu2, cov) - 9.0) < 1e-5


def test_frechet_nonneg_and_symmetric():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(200, 4))
    b = rng.normal(size=(200, 4)) + 0.7
    mu_a, cov_a = feature_stats(a)
    mu_b, cov_b = feature_stats(b)
    d_ab = frechet_distance(mu_a, cov_a, mu_b, cov_b)
    d_ba = frechet_distance(mu_b, cov_b, mu_a, cov_a)
    assert d_ab > 0
    assert abs(d_ab - d_ba) < 1e-6


def test_scale_severity():
    assert scale_severity(1.0) == 0.0
    assert abs(scale_severity(2.0) - 1.0) < 1e-9
    assert abs(scale_severity(0.5) - 1.0) < 1e-9  # symmetric


def test_feature_stats_shapes():
    feats = np.random.default_rng(1).normal(size=(50, 6))
    mu, cov = feature_stats(feats)
    assert mu.shape == (6,)
    assert cov.shape == (6, 6)


def test_appearance_severity_zero_for_same_distribution():
    rng = np.random.default_rng(2)
    feats = rng.normal(size=(300, 5))
    mu, cov = feature_stats(feats)
    # severity of a distribution against its own stats ~ 0
    assert appearance_severity(feats, mu, cov) < 1e-6
