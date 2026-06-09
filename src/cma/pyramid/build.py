"""Scale-aware pyramidal tile extraction with back-projection metadata."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Tile:
    """A single source-image tile, ready to feed a dense matcher.

    Coordinates inside `image` are local pixel coordinates of the tile. To
    back-project a local point (x_tile, y_tile) into the original source
    image, use `tile_to_source(...)`.
    """

    image: np.ndarray         # (tile_size, tile_size) tile image
    level: int                # pyramid level k (0 = original I_s)
    x0: int                   # left edge in level-k image coords
    y0: int                   # top edge in level-k image coords
    tile_size: int            # side length of the tile in pixels
    level_scale: float        # multiplier from level-k coords to level-0 (= 2**k)
    source_shape: tuple[int, int]  # original I_s shape (H, W)

    def tile_to_source(self, xy_tile: np.ndarray) -> np.ndarray:
        """Map (N, 2) points in tile coords to original I_s coords."""
        if xy_tile.ndim != 2 or xy_tile.shape[1] != 2:
            raise ValueError(f"expected (N, 2), got {xy_tile.shape}")
        # tile -> level-k image
        xy_level = xy_tile + np.array([self.x0, self.y0], dtype=xy_tile.dtype)
        # level-k -> level-0 (source)
        return xy_level * self.level_scale


def build(
    source: np.ndarray,
    scale_ratio: float,
    tile_size: int,
    overlap: float = 0.5,
) -> list[Tile]:
    """Construct a pyramid of tiles from `source`.

    The pyramid is downsampled by powers of two until one further halving
    would make the source coarser than the target's physical resolution
    (where `scale_ratio = pix_size(target) / pix_size(source)`, so a value
    of 4.0 means the target sees 4x larger physical pixels than the source).

    At every level we extract overlapping `tile_size x tile_size` patches
    with `overlap` fractional stride (e.g. 0.5 = 50% overlap).
    """
    if source.ndim not in (2, 3):
        raise ValueError(f"source must be 2D or 3D, got shape {source.shape}")
    if tile_size <= 0:
        raise ValueError(f"tile_size must be > 0, got {tile_size}")
    if not 0.0 <= overlap < 1.0:
        raise ValueError(f"overlap must be in [0, 1), got {overlap}")
    if scale_ratio <= 0:
        raise ValueError(f"scale_ratio must be > 0, got {scale_ratio}")

    H0, W0 = source.shape[:2]
    source_shape = (H0, W0)

    # Determine the deepest level: we keep going while the level-k pixel size
    # (= 2**k) stays at or below the target physical pixel size (= scale_ratio).
    # Equivalent: 2**k <= scale_ratio, so k_max = floor(log2(scale_ratio)),
    # clipped to >= 0.
    if scale_ratio >= 1.0:
        k_max = int(np.floor(np.log2(scale_ratio)))
    else:
        # Target is finer than source pixels — only level 0 makes sense.
        k_max = 0

    tiles: list[Tile] = []
    stride = max(1, int(round(tile_size * (1.0 - overlap))))

    current = source
    for k in range(k_max + 1):
        Hk, Wk = current.shape[:2]
        if Hk < tile_size or Wk < tile_size:
            # Pad the level-k image so at least one tile fits
            pad_h = max(0, tile_size - Hk)
            pad_w = max(0, tile_size - Wk)
            current_padded = cv2.copyMakeBorder(
                current, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101
            )
            Hk, Wk = current_padded.shape[:2]
            level_img = current_padded
        else:
            level_img = current

        ys = list(range(0, max(1, Hk - tile_size + 1), stride))
        xs = list(range(0, max(1, Wk - tile_size + 1), stride))
        # Ensure last tile reaches the bottom / right edge
        if ys[-1] + tile_size < Hk:
            ys.append(Hk - tile_size)
        if xs[-1] + tile_size < Wk:
            xs.append(Wk - tile_size)

        level_scale = float(2**k)
        for y0 in ys:
            for x0 in xs:
                patch = level_img[y0 : y0 + tile_size, x0 : x0 + tile_size]
                tiles.append(
                    Tile(
                        image=patch,
                        level=k,
                        x0=x0,
                        y0=y0,
                        tile_size=tile_size,
                        level_scale=level_scale,
                        source_shape=source_shape,
                    )
                )

        # Downsample for next level
        if k < k_max:
            new_h = max(1, current.shape[0] // 2)
            new_w = max(1, current.shape[1] // 2)
            current = cv2.resize(current, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return tiles
