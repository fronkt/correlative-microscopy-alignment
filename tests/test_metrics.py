import numpy as np
import pytest

from cma.metrics import (
    mean_error,
    median_error,
    p_match_at_k,
    registration_metrics,
    success_rate,
)


def test_p_match_at_k_thresholds():
    pred = np.array([[0, 0], [1, 0], [4, 0], [10, 0]], dtype=float)
    gt = np.zeros_like(pred)
    # errors = [0, 1, 4, 10]
    assert p_match_at_k(pred, gt, 1.0) == pytest.approx(0.25)   # only the zero
    assert p_match_at_k(pred, gt, 3.0) == pytest.approx(0.5)    # 0, 1
    assert p_match_at_k(pred, gt, 5.0) == pytest.approx(0.75)   # 0, 1, 4
    assert p_match_at_k(pred, gt, 100.0) == pytest.approx(1.0)


def test_mean_median():
    pred = np.array([[0, 0], [3, 4], [6, 8]], dtype=float)
    gt = np.zeros_like(pred)
    # errors = [0, 5, 10]
    assert mean_error(pred, gt) == pytest.approx(5.0)
    assert median_error(pred, gt) == pytest.approx(5.0)


def test_success_rate():
    per_pair = np.array([0.5, 2.0, 4.9, 5.0, 12.0])
    # threshold 5.0: 0.5, 2.0, 4.9 succeed -> 3/5
    assert success_rate(per_pair, 5.0) == pytest.approx(0.6)


def test_p_match_at_k_validates():
    with pytest.raises(ValueError):
        p_match_at_k(np.zeros((1, 2)), np.zeros((1, 2)), k_px=0.0)


def test_registration_metrics_returns_all_fields():
    pred = np.array([[0, 0], [1, 0]], dtype=float)
    gt = np.zeros_like(pred)
    m = registration_metrics(pred, gt).as_dict()
    assert set(m) == {
        "mu_err",
        "med_err",
        "p_match_at_1",
        "p_match_at_3",
        "p_match_at_5",
        "p_match_at_10",
    }
