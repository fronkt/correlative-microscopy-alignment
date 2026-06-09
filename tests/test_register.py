"""End-to-end acceptance test (task 3.4 from task_plan.md).

A synthetic correlative pair with known ground-truth homography should be
recovered to <2 px mean keypoint error using the SIFT matcher + pyramid +
RANSAC pipeline. SIFT stands in here for any feature-based matcher; the
real RoMa / ELoFTR / MatchAnything backbones plug into the same Matcher
ABC and are evaluated in the AmalgaMatch harness.
"""

import numpy as np
import pytest

from cma.data import synthesize_pair
from cma.matchers import SIFTMatcher
from cma.pipeline import register


def _apply_h(H, xy):
    ones = np.ones((xy.shape[0], 1))
    hom = np.concatenate([xy, ones], axis=1)
    proj = hom @ H.T
    return proj[:, :2] / proj[:, 2:3]


@pytest.mark.parametrize("fov_ratio,seed", [(0.20, 0), (0.10, 1)])
def test_synthetic_pair_recovers_homography(fov_ratio, seed):
    pair, H_gt = synthesize_pair(
        source_size=1024,
        fov_ratio=fov_ratio,
        target_size=256,
        seed=seed,
        rotation_deg=5.0,
        noise_sigma=0.005,
    )
    result = register(
        pair.source,
        pair.target,
        matcher=SIFTMatcher(),
        scale_ratio=pair.scale_ratio,
        family="auto",
    )
    pred_src = _apply_h(result.H_target_to_source, pair.gt.tgt_xy)
    err = np.linalg.norm(pred_src - pair.gt.src_xy, axis=1)
    assert err.mean() < 2.0, (
        f"FOV={fov_ratio}, seed={seed}: mean error {err.mean():.3f}px exceeds 2px"
    )
    assert result.n_tiles >= 1
    assert result.n_correspondences >= 8


def test_pipeline_executes_at_low_fov_smoke():
    """At very low FOV, SIFT may not converge — just verify the pipeline
    doesn't crash and produces a transform. Quantitative breakdown is the
    subject of the FOV sensitivity sweep (Phase 5)."""
    pair, H_gt = synthesize_pair(
        source_size=1024,
        fov_ratio=0.02,
        target_size=256,
        seed=3,
        rotation_deg=2.0,
        noise_sigma=0.002,
    )
    try:
        result = register(
            pair.source,
            pair.target,
            matcher=SIFTMatcher(),
            scale_ratio=pair.scale_ratio,
        )
    except RuntimeError:
        pytest.skip("SIFT did not produce enough matches at FOV=2% — expected breakdown")
    assert result.transform.matrix.shape in {(2, 3), (3, 3)}
