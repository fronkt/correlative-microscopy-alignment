"""End-to-end registration pipeline: pyramid -> matcher -> consensus."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cma.estimators import EstimatedTransform, fit_transform
from cma.matchers.base import Matcher
from cma.pyramid import Tile, build


@dataclass
class RegistrationResult:
    """Output of `register`: transform + diagnostics."""

    transform: EstimatedTransform
    n_tiles: int
    n_correspondences: int
    H_target_to_source: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.H_target_to_source = self.transform.as_3x3()


def register(
    source: np.ndarray,
    target: np.ndarray,
    matcher: Matcher,
    scale_ratio: float,
    *,
    overlap: float = 0.5,
    family: str = "auto",
    ransac_threshold_px: float = 3.0,
) -> RegistrationResult:
    """Register `target` (narrow-FOV) into `source` (wide-FOV) coords.

    Returns a transform `H` such that source_xy = H @ target_xy.

    Steps:
      1. Build a pyramid of `source` whose tile size matches `target`'s spatial
         extent.
      2. For each tile, call `matcher.match(tile, target)` to get
         correspondences in (tile, target) coords.
      3. Back-project tile-side points to the original source frame.
      4. Pool across tiles and fit a robust transform with RANSAC.
    """
    if target.ndim not in (2, 3):
        raise ValueError(f"target must be 2D or 3D, got shape {target.shape}")

    tile_size = int(min(target.shape[:2]))
    tiles: list[Tile] = build(source, scale_ratio, tile_size=tile_size, overlap=overlap)
    if not tiles:
        raise RuntimeError("pyramid produced zero tiles")

    src_pts: list[np.ndarray] = []
    tgt_pts: list[np.ndarray] = []
    for tile in tiles:
        corr = matcher.match(tile.image, target)
        if len(corr) == 0:
            continue
        src_pts.append(tile.tile_to_source(corr.a_xy))
        tgt_pts.append(corr.b_xy)

    if not src_pts:
        raise RuntimeError("matcher returned no correspondences across all tiles")

    src_xy = np.concatenate(src_pts, axis=0)
    tgt_xy = np.concatenate(tgt_pts, axis=0)

    # Fit transform mapping target -> source: src_pred = H @ tgt
    transform = fit_transform(
        src_xy=tgt_xy,
        dst_xy=src_xy,
        family=family,  # type: ignore[arg-type]
        ransac_threshold_px=ransac_threshold_px,
    )

    return RegistrationResult(
        transform=transform,
        n_tiles=len(tiles),
        n_correspondences=int(len(src_xy)),
    )
