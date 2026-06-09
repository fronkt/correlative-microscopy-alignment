"""Smoke test for the AmalgaMatch loader using a synthesized on-disk fixture."""

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from cma.data import AmalgaMatchLoader


def _make_fixture(tmp_root: Path) -> Path:
    rng = np.random.default_rng(0)
    pairs = [
        ("p01", "g1", "alpha"),
        ("p02", "g1", "beta"),
        ("p03", "g2", "alpha"),
    ]
    with (tmp_root / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "group", "subclass", "source_pixel_nm", "target_pixel_nm"])
        for pid, g, s in pairs:
            w.writerow([pid, g, s, 10.0, 2.5])
    for pid, g, s in pairs:
        pdir = tmp_root / "images" / g / s / pid
        pdir.mkdir(parents=True)
        kdir = tmp_root / "keypoints" / g / s / pid
        kdir.mkdir(parents=True)
        src = (rng.random((128, 128)) * 255).astype(np.uint8)
        tgt = (rng.random((64, 64)) * 255).astype(np.uint8)
        cv2.imwrite(str(pdir / "source.png"), src)
        cv2.imwrite(str(pdir / "target.png"), tgt)
        with (kdir / "gt.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["src_x", "src_y", "tgt_x", "tgt_y"])
            w.writerow([10, 12, 5, 6])
            w.writerow([20, 22, 10, 11])
    return tmp_root


def test_loader_yields_all_pairs(tmp_path: Path) -> None:
    root = _make_fixture(tmp_path)
    loader = AmalgaMatchLoader(root)
    assert len(loader) == 3
    pairs = list(iter(loader))
    assert len(pairs) == 3
    img_pair, rec = pairs[0]
    assert img_pair.source.shape == (128, 128)
    assert img_pair.target.shape == (64, 64)
    assert img_pair.scale_ratio == pytest.approx(2.5 / 10.0)
    assert img_pair.gt is not None and len(img_pair.gt) == 2
    assert rec.group == "g1"


def test_loader_group_filter(tmp_path: Path) -> None:
    root = _make_fixture(tmp_path)
    loader = AmalgaMatchLoader(root)
    g2 = list(loader.iter(groups=["g2"]))
    assert len(g2) == 1
    assert g2[0][1].pair_id == "p03"


def test_loader_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        AmalgaMatchLoader(tmp_path / "nope")
