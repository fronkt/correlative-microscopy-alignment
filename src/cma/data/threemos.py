"""3MOS optical-SAR loader (appearance-axis domain #2).

3MOS (Ye et al., arXiv 2404.00838; Visual Intelligence 2025) provides
co-registered optical/SAR scene pairs. SAR speckle + inverted intensity vs.
optical is the canonical hard cross-modal shift for RGB-pretrained matchers.

Because pairs are co-registered, the correct correspondence is the identity:
GT is a regular grid of points with ``src_xy == tgt_xy``. A matcher succeeds
iff it recovers near-identity across the modality gap.

Layout (configurable; confirm against the repo at prep time)::

    <root>/<opt_subdir>/<scene>.<ext>
    <root>/<sar_subdir>/<scene>.<ext>     # same stem == a co-registered pair

License: CC BY-NC-ND 4.0 (NonCommercial, NoDerivatives — evaluate only, do not
redistribute derivatives).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cma.data.amalgamatch import _read_image_float
from cma.data.types import ImagePair, KeypointSet

_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


@dataclass(frozen=True)
class ThreeMosRecord:
    pair_id: str
    scene: str
    opt_path: Path
    sar_path: Path


def _identity_grid_points(h: int, w: int, n_side: int, margin: float = 0.1) -> np.ndarray:
    """(n_side², 2) regular grid of points inside the image, x,y order."""
    x0, x1 = margin * w, (1 - margin) * w
    y0, y1 = margin * h, (1 - margin) * h
    xs = np.linspace(x0, x1, n_side)
    ys = np.linspace(y0, y1, n_side)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float64)


class ThreeMosLoader:
    """Iterate 3MOS optical-SAR pairs with identity GT, cma loader interface."""

    def __init__(self, root: str | Path, opt_subdir: str = "opt",
                 sar_subdir: str = "sar", n_gt_side: int = 8,
                 limit: int | None = None) -> None:
        self.root = Path(root)
        self.n_gt_side = int(n_gt_side)
        opt_dir = self.root / opt_subdir
        sar_dir = self.root / sar_subdir
        if not opt_dir.is_dir() or not sar_dir.is_dir():
            raise FileNotFoundError(
                f"3MOS expects {opt_subdir}/ and {sar_subdir}/ under {self.root}. "
                f"See cma/data/threemos.py."
            )
        sar_by_stem = {p.stem: p for p in sar_dir.iterdir()
                       if p.suffix.lower() in _EXTS}
        self._records: list[ThreeMosRecord] = []
        for opt_path in sorted(opt_dir.iterdir()):
            if opt_path.suffix.lower() not in _EXTS:
                continue
            sar_path = sar_by_stem.get(opt_path.stem)
            if sar_path is None:
                continue
            self._records.append(ThreeMosRecord(
                pair_id=f"3mos#{opt_path.stem}",
                scene=opt_path.stem,
                opt_path=opt_path,
                sar_path=sar_path,
            ))
            if limit is not None and len(self._records) >= limit:
                break
        if not self._records:
            raise ValueError(f"no matched opt/sar stems under {self.root}")

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> list[ThreeMosRecord]:
        return list(self._records)

    def load_pair(self, rec: ThreeMosRecord) -> ImagePair:
        src = _read_image_float(rec.opt_path)   # optical = source
        tgt = _read_image_float(rec.sar_path)   # SAR = target
        h = min(src.shape[0], tgt.shape[0])
        w = min(src.shape[1], tgt.shape[1])
        pts = _identity_grid_points(h, w, self.n_gt_side)
        gt = KeypointSet(src_xy=pts.copy(), tgt_xy=pts.copy())  # co-registered
        return ImagePair(
            source=src, target=tgt, scale_ratio=1.0, gt=gt,
            metadata={"pair_id": rec.pair_id, "scene": rec.scene,
                      "axis": "appearance", "coregistered": True},
        )

    def __iter__(self) -> Iterator[tuple[ImagePair, ThreeMosRecord]]:
        for rec in self._records:
            yield self.load_pair(rec), rec
