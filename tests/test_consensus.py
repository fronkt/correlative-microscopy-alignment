import numpy as np
import pytest

from cma.estimators import fit_transform


def _apply_h(H, xy):
    ones = np.ones((xy.shape[0], 1))
    return ((np.hstack([xy, ones]) @ H.T)[:, :2]
            / (np.hstack([xy, ones]) @ H.T)[:, 2:3])


def test_homography_recovery_clean():
    rng = np.random.default_rng(0)
    H_true = np.array(
        [
            [1.05, -0.10, 12.0],
            [0.10, 1.05, -8.0],
            [0.0, 0.0, 1.0],
        ]
    )
    src = rng.uniform(0, 512, size=(200, 2))
    dst = _apply_h(H_true, src)
    out = fit_transform(src, dst, family="homography")
    assert out.family == "homography"
    pred = _apply_h(out.as_3x3(), src)
    err = np.linalg.norm(pred - dst, axis=1)
    assert err.mean() < 0.5


def test_affine_recovery_clean():
    rng = np.random.default_rng(1)
    theta = np.deg2rad(15.0)
    A3 = np.array(
        [
            [1.2 * np.cos(theta), -1.2 * np.sin(theta), 30.0],
            [1.2 * np.sin(theta), 1.2 * np.cos(theta), -20.0],
            [0.0, 0.0, 1.0],
        ]
    )
    src = rng.uniform(0, 512, size=(200, 2))
    dst = _apply_h(A3, src)
    out = fit_transform(src, dst, family="affine")
    assert out.family == "affine"
    pred = _apply_h(out.as_3x3(), src)
    err = np.linalg.norm(pred - dst, axis=1)
    assert err.mean() < 0.5


def test_ransac_rejects_outliers():
    rng = np.random.default_rng(2)
    H_true = np.eye(3)
    H_true[0, 2] = 5.0
    H_true[1, 2] = -3.0
    src = rng.uniform(0, 512, size=(300, 2))
    dst = _apply_h(H_true, src)
    # Replace 30% with random outliers
    idx = rng.choice(len(src), size=90, replace=False)
    dst[idx] = rng.uniform(0, 512, size=(90, 2))
    out = fit_transform(src, dst, family="homography", ransac_threshold_px=3.0)
    # Inlier count should be at least the 210 true inliers (minus a few stragglers)
    assert out.n_inliers >= 180
    pred = _apply_h(out.as_3x3(), src)
    inlier_err = np.linalg.norm(pred - dst, axis=1)[out.inliers]
    assert inlier_err.mean() < 1.0


def test_too_few_points_raises():
    with pytest.raises(ValueError):
        fit_transform(np.zeros((3, 2)), np.zeros((3, 2)))
