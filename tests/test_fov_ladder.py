"""Crop bookkeeping tests for the FOV ladder (no dataset, no matcher)."""

import numpy as np
import pytest

from cma.data.fov_ladder import crop_target_to_area_ratio
from cma.data.types import ImagePair, KeypointSet


def _pair(h=400, w=600, n=50, seed=0):
    rng = np.random.default_rng(seed)
    gt = KeypointSet(
        src_xy=rng.uniform(0, 1000, (n, 2)),
        tgt_xy=np.column_stack([rng.uniform(0, w - 1, n), rng.uniform(0, h - 1, n)]),
    )
    return ImagePair(
        source=rng.random((800, 1200)),
        target=rng.random((h, w)),
        scale_ratio=0.5,
        gt=gt,
    )


def test_all_gt_kept_and_shifted_consistently():
    pair = _pair()
    rung = crop_target_to_area_ratio(pair, base_area_ratio=0.8, desired_area_ratio=0.2)
    assert rung is not None
    x0, y0 = rung.crop_origin_xy
    # every GT row survives; tgt coords are the originals shifted by the
    # crop origin (out-of-crop points included), src coords untouched
    assert len(rung.pair.gt) == len(pair.gt)
    np.testing.assert_allclose(rung.pair.gt.tgt_xy + [x0, y0], pair.gt.tgt_xy)
    np.testing.assert_allclose(rung.pair.gt.src_xy, pair.gt.src_xy)


def test_crop_geometry_ratio_and_inside_count():
    pair = _pair(n=2000)
    rung = crop_target_to_area_ratio(pair, base_area_ratio=0.8, desired_area_ratio=0.05)
    assert rung is not None
    ch, cw = rung.pair.target.shape[:2]
    assert rung.area_ratio == pytest.approx(0.8 * ch * cw / (400 * 600))
    assert rung.area_ratio == pytest.approx(0.05, rel=0.05)
    # inside count matches a direct recount in the crop frame
    t = rung.pair.gt.tgt_xy
    inside = ((t[:, 0] >= 0) & (t[:, 0] <= cw - 1)
              & (t[:, 1] >= 0) & (t[:, 1] <= ch - 1))
    assert rung.n_gt_inside == int(inside.sum())
    assert 0 < rung.n_gt_inside < len(pair.gt)
    # the crop image is the corresponding window of the original target
    x0, y0 = rung.crop_origin_xy
    np.testing.assert_array_equal(
        rung.pair.target, pair.target[y0 : y0 + ch, x0 : x0 + cw])
    # scale_ratio (pixel size ratio) untouched by cropping
    assert rung.pair.scale_ratio == pair.scale_ratio


def test_rung_not_below_base_returns_none():
    pair = _pair()
    assert crop_target_to_area_ratio(pair, 0.3, 0.5) is None
    assert crop_target_to_area_ratio(pair, 0.3, 0.3) is None
