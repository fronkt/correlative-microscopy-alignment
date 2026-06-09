import numpy as np
import pytest

from cma.pyramid import build


def test_build_covers_source_at_scale_one():
    img = np.zeros((512, 512), dtype=np.float32)
    tiles = build(img, scale_ratio=1.0, tile_size=256, overlap=0.5)
    assert all(t.level == 0 for t in tiles)
    assert all(t.image.shape == (256, 256) for t in tiles)
    # With 50% overlap and tile=256 on a 512 image: x in {0, 128, 256}, y same.
    coords = {(t.x0, t.y0) for t in tiles}
    assert (0, 0) in coords
    assert (256, 256) in coords


def test_build_uses_multiple_levels_when_scale_ratio_large():
    img = np.zeros((1024, 1024), dtype=np.float32)
    tiles = build(img, scale_ratio=4.0, tile_size=256, overlap=0.5)
    levels = {t.level for t in tiles}
    assert levels == {0, 1, 2}, f"expected pyramid levels 0..2, got {levels}"


def test_tile_to_source_round_trip():
    img = np.zeros((1024, 1024), dtype=np.float32)
    tiles = build(img, scale_ratio=2.0, tile_size=256, overlap=0.5)
    # Pick a level-1 tile
    tile = next(t for t in tiles if t.level == 1)
    local = np.array([[0.0, 0.0], [128.0, 64.0]])
    src = tile.tile_to_source(local)
    expected = (local + np.array([tile.x0, tile.y0])) * tile.level_scale
    np.testing.assert_allclose(src, expected)


def test_invalid_inputs():
    img = np.zeros((64, 64), dtype=np.float32)
    with pytest.raises(ValueError):
        build(img, scale_ratio=1.0, tile_size=0)
    with pytest.raises(ValueError):
        build(img, scale_ratio=1.0, tile_size=32, overlap=1.0)
    with pytest.raises(ValueError):
        build(img, scale_ratio=-1.0, tile_size=32)
