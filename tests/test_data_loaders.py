"""Loader parsing tests for ANHIR + 3MOS, on synthetic fixtures.

These verify the parsing/interface against the real on-disk layout without the
multi-GB downloads (which happen at box-prep). Image IO goes through the same
cv2 path the loaders use in production.
"""

from __future__ import annotations

import cv2
import numpy as np

from cma.data.anhir import AnhirLoader
from cma.data.threemos import ThreeMosLoader
from cma.data.types import ImagePair


def _write_png(path, h=32, w=32):
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = (np.random.default_rng(0).random((h, w, 3)) * 255).astype(np.uint8)
    cv2.imwrite(str(path), arr)


def _write_landmarks(path, pts):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",X,Y"] + [f"{i},{x},{y}" for i, (x, y) in enumerate(pts)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_anhir_loader_parses_pair_and_landmarks(tmp_path):
    root = tmp_path
    _write_png(root / "tissueA" / "scale-100pc" / "img_s.png")
    _write_png(root / "tissueA" / "scale-100pc" / "img_t.png")
    src_pts = [(1.0, 2.0), (3.0, 4.0), (5.0, 6.0)]
    tgt_pts = [(1.5, 2.5), (3.5, 4.5), (5.5, 6.5)]
    _write_landmarks(root / "tissueA" / "scale-100pc" / "lm_s.csv", src_pts)
    _write_landmarks(root / "tissueA" / "scale-100pc" / "lm_t.csv", tgt_pts)
    (root / "dataset_medium.csv").write_text(
        "Source image,Target image,Source landmarks,Target landmarks,status\n"
        "tissueA/scale-100pc/img_s.png,tissueA/scale-100pc/img_t.png,"
        "tissueA/scale-100pc/lm_s.csv,tissueA/scale-100pc/lm_t.csv,training\n",
        encoding="utf-8",
    )
    loader = AnhirLoader(root)
    assert len(loader) == 1
    rec = loader.records[0]
    assert rec.tissue == "tissueA"
    assert rec.status == "training"
    pair = loader.load_pair(rec)
    assert isinstance(pair, ImagePair)
    assert pair.scale_ratio == 1.0
    assert pair.metadata["axis"] == "appearance"
    assert len(pair.gt) == 3
    np.testing.assert_allclose(pair.gt.src_xy[1], [3.0, 4.0])
    np.testing.assert_allclose(pair.gt.tgt_xy[1], [3.5, 4.5])


def test_anhir_status_filter(tmp_path):
    root = tmp_path
    for name in ("a", "b"):
        _write_png(root / f"{name}_s.png")
        _write_png(root / f"{name}_t.png")
        _write_landmarks(root / f"{name}_sl.csv", [(0.0, 0.0)])
        _write_landmarks(root / f"{name}_tl.csv", [(1.0, 1.0)])
    (root / "dataset_medium.csv").write_text(
        "Source image,Target image,Source landmarks,Target landmarks,status\n"
        "a_s.png,a_t.png,a_sl.csv,a_tl.csv,training\n"
        "b_s.png,b_t.png,b_sl.csv,b_tl.csv,evaluation\n",
        encoding="utf-8",
    )
    assert len(AnhirLoader(root, status="evaluation")) == 1
    assert len(AnhirLoader(root, status="training")) == 1
    assert len(AnhirLoader(root)) == 2


def test_threemos_loader_identity_gt(tmp_path):
    root = tmp_path
    for stem in ("scene1", "scene2"):
        _write_png(root / "opt" / f"{stem}.png", 40, 48)
        _write_png(root / "sar" / f"{stem}.png", 40, 48)
    _write_png(root / "opt" / "unpaired.png")  # no sar match -> skipped
    loader = ThreeMosLoader(root, n_gt_side=8)
    assert len(loader) == 2
    pair = loader.load_pair(loader.records[0])
    assert pair.metadata["coregistered"] is True
    assert len(pair.gt) == 64  # 8x8 identity grid
    np.testing.assert_array_equal(pair.gt.src_xy, pair.gt.tgt_xy)  # identity


def test_threemos_limit(tmp_path):
    root = tmp_path
    for i in range(5):
        _write_png(root / "opt" / f"s{i}.png")
        _write_png(root / "sar" / f"s{i}.png")
    assert len(ThreeMosLoader(root, limit=3)) == 3
