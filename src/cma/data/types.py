from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class KeypointSet:
    """Paired keypoints between source (I_s) and target (I_t)."""

    src_xy: np.ndarray  # (N, 2) float, in I_s pixel coords
    tgt_xy: np.ndarray  # (N, 2) float, in I_t pixel coords

    def __post_init__(self) -> None:
        if self.src_xy.shape != self.tgt_xy.shape:
            raise ValueError(
                f"src/tgt shape mismatch: {self.src_xy.shape} vs {self.tgt_xy.shape}"
            )
        if self.src_xy.ndim != 2 or self.src_xy.shape[1] != 2:
            raise ValueError(f"expected (N, 2), got {self.src_xy.shape}")

    def __len__(self) -> int:
        return self.src_xy.shape[0]


@dataclass
class ImagePair:
    """Single correlative-microscopy registration item."""

    source: np.ndarray  # I_s, wide-FOV, (H, W) or (H, W, C), float in [0, 1]
    target: np.ndarray  # I_t, narrow-FOV
    scale_ratio: float  # pix_size(I_t) / pix_size(I_s); <1 means I_t is finer
    gt: Optional[KeypointSet] = None
    metadata: dict = field(default_factory=dict)
