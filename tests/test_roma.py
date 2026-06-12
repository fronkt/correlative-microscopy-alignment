"""RoMa wrapper smoke tests.

Marked `slow` because the model loads ~1.5 GB of weights (RoMa + DINOv2) and
CPU inference takes ~30-60 s per match. Run with `pytest -m slow`.

Two checks mirror the MatchAnything verification:
  1. self-match: identical inputs should produce dense correspondences
     with near-zero a-b pixel distance.
  2. warped pair: a known-homography target should be recovered to
     < 2 px mean keypoint error after `register(...)`.
"""

import numpy as np
import pytest

pytest.importorskip("romatch")

from cma.data import synthesize_pair
from cma.data.synthetic import natural_source_image
from cma.matchers import RoMaMatcher
from cma.pipeline import register

pytestmark = pytest.mark.slow


def _apply_h(H, xy):
    ones = np.ones((xy.shape[0], 1))
    hom = np.concatenate([xy, ones], axis=1)
    proj = hom @ H.T
    return proj[:, :2] / proj[:, 2:3]


@pytest.fixture(scope="module")
def matcher():
    return RoMaMatcher(device="cpu", max_long_side=256)


def test_self_match_recovers_identity(matcher):
    img = natural_source_image(256)
    corr = matcher.match(img, img)
    assert len(corr) > 1000, f"expected dense correspondences, got {len(corr)}"
    d = np.linalg.norm(corr.a_xy - corr.b_xy, axis=1)
    assert d.mean() < 1.0, f"self-match mean displacement {d.mean():.3f}px exceeds 1px"


def test_ma_roma_loads_and_self_matches():
    """MA-RoMa weights are key-compatible with roma_outdoor and produce
    sane dense matches. Inversion robustness is checked in
    scripts/smoke_ma_roma.py (too slow for the suite)."""
    m = RoMaMatcher(variant="ma_outdoor", device="cpu", max_long_side=256)
    assert m.name == "ma_roma"
    img = natural_source_image(256)
    corr = m.match(img, img)
    assert len(corr) > 1000, f"expected dense correspondences, got {len(corr)}"
    d = np.linalg.norm(corr.a_xy - corr.b_xy, axis=1)
    assert d.mean() < 1.0, f"self-match mean displacement {d.mean():.3f}px exceeds 1px"


def test_warped_pair_recovers_homography(matcher):
    pair, _ = synthesize_pair(
        source_size=512,
        fov_ratio=0.25,
        target_size=128,
        seed=0,
        rotation_deg=5.0,
        noise_sigma=0.005,
        source_image=natural_source_image(512),
    )
    result = register(
        pair.source,
        pair.target,
        matcher=matcher,
        scale_ratio=pair.scale_ratio,
        family="auto",
    )
    pred = _apply_h(result.H_target_to_source, pair.gt.tgt_xy)
    err = np.linalg.norm(pred - pair.gt.src_xy, axis=1)
    assert err.mean() < 2.0, f"RoMa registration mean err {err.mean():.3f}px exceeds 2px"
    assert result.n_correspondences >= 100
