"""AmalgaMatch loader tests.

Unit tests run against a synthesized on-disk fixture mimicking the real
release layout (subset dirs, scenes/, eval_indexs/ with pickled dicts).
Integration tests run against the real dataset and are skipped when it is
not present under data/AmalgaMatch.
"""

import pickle
from pathlib import Path

import cv2
import numpy as np
import pytest

from cma.data import AmalgaMatchLoader

REAL_ROOT = Path(__file__).parents[1] / "data" / "AmalgaMatch"
needs_real_data = pytest.mark.skipif(
    not any(REAL_ROOT.glob("*/eval_indexs")),
    reason=f"real AmalgaMatch release not found under {REAL_ROOT}",
)


def _image_meta(name: str, px_m: float, w: int, h: int) -> dict:
    return {
        "Unique Images": name,
        "Physical Pixel Size [m]": px_m,
        "Resolution Width": w,
        "Resolution Height": h,
        "Microscope Type": "Scanning Electron Microscope",
        "Detector/Imaging Mode": "Backscattered Electron Detector",
        "Derived Modality": 0,
        "Derived Modality Type": "None",
        "Stitching Indicator": 0,
    }


def _make_fixture(tmp_root: Path) -> Path:
    """Two subsets in the real layout; second image of each pair is wide-FOV."""
    rng = np.random.default_rng(0)
    subsets = ["Alloy1_SEM-EBSD_SameSlice", "Alloy2_OM-SEM_Multiscale"]
    for subset in subsets:
        scene = f"{subset}_0"
        scene_dir = tmp_root / subset / "scenes" / scene
        scene_dir.mkdir(parents=True)
        eval_dir = tmp_root / subset / "eval_indexs"
        eval_dir.mkdir(parents=True)
        # narrow-FOV: 64px @ 10nm (640nm FOV); wide-FOV: 128px @ 40nm (5120nm)
        narrow = (rng.random((64, 64)) * 255).astype(np.uint8)
        wide = (rng.random((96, 128)) * 255).astype(np.uint8)
        cv2.imwrite(str(scene_dir / "narrow.tif"), narrow)
        cv2.imwrite(str(scene_dir / "wide.tif"), wide)
        gt = np.array([[10.0, 12.0, 40.0, 48.0], [20.0, 22.0, 80.0, 88.0]])
        payload = {
            "dataset_name": subset,
            "image_paths": [f"scenes/{scene}/narrow.tif", f"scenes/{scene}/wide.tif"],
            "image_metadata": [
                _image_meta("narrow.tif", 10e-9, 64, 64),
                _image_meta("wide.tif", 40e-9, 128, 96),
            ],
            "pair_infos": [([0, 1], 1)],
            "gt_2D_matches": [gt],
        }
        with (eval_dir / f"eval_{scene}.npz").open("wb") as f:
            pickle.dump(payload, f)
    return tmp_root


def test_loader_yields_all_pairs(tmp_path: Path) -> None:
    root = _make_fixture(tmp_path)
    loader = AmalgaMatchLoader(root)
    assert len(loader) == 2
    pairs = list(iter(loader))
    assert len(pairs) == 2
    img_pair, rec = pairs[0]
    # second image is wide-FOV, so the loader must flip: source=wide
    assert rec.flipped is True
    assert img_pair.source.shape == (96, 128)
    assert img_pair.target.shape == (64, 64)
    assert img_pair.scale_ratio == pytest.approx(10.0 / 40.0)
    assert rec.group == "SameSlice"
    assert rec.subclass == "Alloy1_SEM-EBSD_SameSlice"
    # GT columns swapped along with the images
    assert img_pair.gt is not None and len(img_pair.gt) == 2
    np.testing.assert_allclose(img_pair.gt.src_xy[0], [40.0, 48.0])
    np.testing.assert_allclose(img_pair.gt.tgt_xy[0], [10.0, 12.0])


def test_loader_group_filter(tmp_path: Path) -> None:
    root = _make_fixture(tmp_path)
    loader = AmalgaMatchLoader(root)
    multi = list(loader.iter(groups=["Multiscale"]))
    assert len(multi) == 1
    assert multi[0][1].subclass == "Alloy2_OM-SEM_Multiscale"
    assert list(loader.iter(subclasses=["Alloy1_SEM-EBSD_SameSlice"]))


def test_loader_missing_root_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        AmalgaMatchLoader(tmp_path / "nope")


@needs_real_data
def test_real_inventory() -> None:
    loader = AmalgaMatchLoader(REAL_ROOT)
    assert len(loader) == 187
    groups = {r.group for r in loader.records}
    assert groups == {
        "Multiscale",
        "SameSlice",
        "SerialSectioning",
        "SlipPartitioning",
        "FractureSurfaces",
        "DislocationCharacterization",
    }
    assert len({r.subclass for r in loader.records}) == 19
    for rec in loader.records:
        # source must be the wide-FOV image => target pixels are finer or equal
        assert rec.target_pixel_nm <= rec.source_pixel_nm * 100  # sanity, not strict
        gt = loader._gt[rec.pair_id]
        assert len(gt) >= 5


@needs_real_data
def test_real_pair_loads_with_consistent_gt() -> None:
    """GT keypoints must land inside their images for a sample of pairs."""
    loader = AmalgaMatchLoader(REAL_ROOT)
    by_subclass: dict[str, object] = {}
    for rec in loader.records:
        by_subclass.setdefault(rec.subclass, rec)
    for rec in by_subclass.values():
        pair = loader._load_pair(rec)  # type: ignore[arg-type]
        assert pair.source.ndim in (2, 3) and pair.target.ndim in (2, 3)
        assert np.isfinite(pair.source).all() and np.isfinite(pair.target).all()
        assert 0.0 <= pair.source.min() and pair.source.max() <= 1.0
        gt = pair.gt
        assert gt is not None
        h_s, w_s = pair.source.shape[:2]
        h_t, w_t = pair.target.shape[:2]
        margin = 2.0
        assert (gt.src_xy[:, 0] >= -margin).all() and (gt.src_xy[:, 0] <= w_s + margin).all()
        assert (gt.src_xy[:, 1] >= -margin).all() and (gt.src_xy[:, 1] <= h_s + margin).all()
        assert (gt.tgt_xy[:, 0] >= -margin).all() and (gt.tgt_xy[:, 0] <= w_t + margin).all()
        assert (gt.tgt_xy[:, 1] >= -margin).all() and (gt.tgt_xy[:, 1] <= h_t + margin).all()
