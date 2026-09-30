"""Tests for the frozen Arm 1 overlay detector and crop coordinate handling (src/cma/overlay.py)."""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from cma.overlay import (
    SAFETY_MARGIN,
    SHAM_FRAC,
    CropRule,
    analyze,
    crop_overlay,
    detect_overlay,
    overlay_rule,
    sham_rule,
    uncrop_homography,
)


def banner_image(h=600, w=800, band=48, seed=0) -> np.ndarray:
    """Noisy 'micrograph' with a flat metadata banner (two text-like lines of uniform glyphs) at the bottom."""
    rng = np.random.default_rng(seed)
    im = rng.integers(60, 200, size=(h, w)).astype(np.uint8)
    im[h - band:] = 30
    for line_top in (10, 28):
        for k in range(24):
            x = 20 + k * 30
            im[h - band + line_top: h - band + line_top + 12, x: x + 7] = 240
    return im


def test_detector_finds_bottom_banner_and_is_deterministic():
    im = banner_image()
    d1, d2 = detect_overlay(im), detect_overlay(im.copy())
    assert d1 == d2
    assert d1 is not None and d1[0] == "bottom" and 40 <= d1[1] <= 52
    assert analyze(im)["band_edges"] == ["bottom"]


def test_detector_flipped_gives_top_and_plain_image_none():
    im = banner_image()
    d = detect_overlay(im[::-1].copy())
    assert d is not None and d[0] == "top"
    rng = np.random.default_rng(1)
    assert detect_overlay(rng.integers(0, 255, size=(600, 800)).astype(np.uint8)) is None


def test_overlay_rule_union_and_margin():
    a_s, a_t = analyze(banner_image(seed=1)), analyze(banner_image(band=40, seed=2))
    rule = overlay_rule(a_s, a_t)
    assert rule is not None and rule.side == "bottom" and rule.rows_src == rule.rows_tgt
    biggest = max(detect_overlay(banner_image(seed=1))[1], detect_overlay(banner_image(band=40, seed=2))[1])
    assert rule.rows_src == math.ceil((biggest / 600 + SAFETY_MARGIN) * 600)
    # no shared overlay -> None
    assert overlay_rule(a_s, analyze(np.random.default_rng(3).integers(0, 255, (600, 800)).astype(np.uint8))) is None
    # different sides -> None
    assert overlay_rule(a_s, analyze(banner_image(seed=4)[::-1].copy())) is None


def test_sham_rule_fraction_from_bottom():
    r = sham_rule((1000, 900), (500, 500))
    assert r.side == "bottom" and r.rows_src == math.ceil(SHAM_FRAC * 1000) and r.rows_tgt == math.ceil(SHAM_FRAC * 500)


def test_crop_pixels_top_and_bottom():
    src = np.arange(100 * 10, dtype=np.float32).reshape(100, 10)
    tgt = src[:60].copy()
    for side in ("top", "bottom"):
        rule = CropRule("overlay", side, 7, 5)
        s, t = crop_overlay(src, tgt, rule)
        assert s.shape == (93, 10) and t.shape == (55, 10)
        assert np.array_equal(s, src[7:] if side == "top" else src[:93])
    s, t = crop_overlay(src, tgt, None)
    assert s is src and t is tgt


@pytest.mark.parametrize("side", ["bottom", "top"])
def test_homography_on_cropped_images_maps_original_gt(side):
    """H fitted on cropped coordinates, un-cropped, must reproduce the ORIGINAL-coordinate GT (no GT dropped)."""
    rng = np.random.default_rng(7)
    H_true = np.array([[0.4, 0.02, 120.0], [-0.03, 0.42, 200.0], [1e-5, -2e-5, 1.0]])  # target -> source, orig coords
    tgt_xy = np.column_stack([rng.uniform(0, 500, 40), rng.uniform(0, 400, 40)])
    src_xy = cv2.perspectiveTransform(tgt_xy[None].astype(np.float64), H_true)[0]
    rule = CropRule("overlay", side, 37, 23)
    (_, sdy), (_, tdy) = rule.offset_src, rule.offset_tgt
    if side == "bottom":
        assert (sdy, tdy) == (0, 0)  # bottom crop leaves coordinates unchanged
    else:
        assert (sdy, tdy) == (37, 23)
    src_c = src_xy - np.array([0.0, sdy])
    tgt_c = tgt_xy - np.array([0.0, tdy])
    H_c, _ = cv2.findHomography(tgt_c, src_c, method=0)  # what a matcher-fitted H on the cropped pair would be
    H_o = uncrop_homography(H_c, rule)
    proj = cv2.perspectiveTransform(tgt_xy[None].astype(np.float64), H_o)[0]
    assert np.abs(proj - src_xy).max() < 1e-3
    assert np.allclose(H_o / H_o[2, 2], H_true / H_true[2, 2], atol=1e-5)


def test_top_crop_without_uncrop_is_wrong():
    """Guards the shift: skipping the un-crop leaves an error of the crop offset."""
    H_id = np.eye(3)
    rule = CropRule("overlay", "top", 50, 20)
    assert np.allclose(uncrop_homography(H_id, rule)[:2, 2], [0.0, 30.0])
    assert np.allclose(uncrop_homography(H_id, CropRule("overlay", "bottom", 50, 20)), np.eye(3))
