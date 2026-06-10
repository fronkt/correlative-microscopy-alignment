"""Tests for register_v2 (coarse-to-fine + verification gate) and the verifier."""

import numpy as np
import pytest

from cma.data import synthesize_pair
from cma.matchers import SIFTMatcher
from cma.pipeline import register_v2, verification_score
from cma.pipeline.verify import REJECT


def _apply_h(H, xy):
    ones = np.ones((xy.shape[0], 1))
    hom = np.concatenate([xy, ones], axis=1)
    proj = hom @ H.T
    return proj[:, :2] / proj[:, 2:3]


def _pair(fov_ratio=0.20, seed=0):
    return synthesize_pair(
        source_size=1024,
        fov_ratio=fov_ratio,
        target_size=256,
        seed=seed,
        rotation_deg=5.0,
        noise_sigma=0.005,
    )


def test_verifier_prefers_true_transform():
    pair, H_gt = _pair()
    score_true = verification_score(pair.source, pair.target, H_gt)
    # A grossly wrong transform: identity (target's footprint is ~20% of
    # source and rotated, so identity misaligns badly).
    score_wrong = verification_score(pair.source, pair.target, np.eye(3))
    assert score_true > score_wrong


def test_verifier_rejects_degenerate():
    pair, _ = _pair()
    assert verification_score(pair.source, pair.target, np.zeros((3, 3))) == REJECT
    H_offscreen = np.array([[1, 0, 1e6], [0, 1, 1e6], [0, 0, 1]], dtype=float)
    assert verification_score(pair.source, pair.target, H_offscreen) == REJECT


@pytest.mark.parametrize("fov_ratio,seed", [(0.20, 0), (0.10, 1)])
def test_v2_recovers_homography(fov_ratio, seed):
    pair, _ = _pair(fov_ratio, seed)
    result = register_v2(
        pair.source, pair.target, SIFTMatcher(), pair.scale_ratio, family="auto"
    )
    pred_src = _apply_h(result.H_target_to_source, pair.gt.tgt_xy)
    err = np.linalg.norm(pred_src - pair.gt.src_xy, axis=1)
    assert err.mean() < 2.0, f"mean error {err.mean():.3f}px exceeds 2px"
    assert result.stage in {"direct", "tile", "direct+zoom", "tile+zoom"}
    assert result.score_final >= result.score_direct or result.score_direct == REJECT


def test_v2_never_worse_than_direct_under_verifier():
    """The acceptance gate: the final score can never drop below stage A's."""
    pair, _ = _pair(0.10, seed=2)
    result = register_v2(
        pair.source, pair.target, SIFTMatcher(), pair.scale_ratio
    )
    assert result.score_final >= result.score_direct


def test_v2_certainty_gate_filters():
    """With an impossible threshold every correspondence is dropped."""
    pair, _ = _pair()
    with pytest.raises(RuntimeError):
        register_v2(
            pair.source, pair.target, SIFTMatcher(), pair.scale_ratio,
            certainty_threshold=2.0,  # confidences are <= 1
            tile_fallback=False, zoom=False,
        )


def test_v2_exposes_inlier_correspondences_for_tps():
    pair, _ = _pair()
    result = register_v2(pair.source, pair.target, SIFTMatcher(), pair.scale_ratio)
    mask = result.transform.inliers
    assert mask.shape[0] == result.src_xy.shape[0] == result.tgt_xy.shape[0]
    assert mask.sum() >= 4
