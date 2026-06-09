"""Matcher abstract base + Correspondences container."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np


@dataclass
class Correspondences:
    """Dense or sparse correspondences between image A and image B."""

    a_xy: np.ndarray         # (N, 2) points in A coords
    b_xy: np.ndarray         # (N, 2) points in B coords
    confidence: np.ndarray   # (N,) confidence in [0, 1]

    def __post_init__(self) -> None:
        if self.a_xy.shape != self.b_xy.shape:
            raise ValueError(f"a/b shape mismatch: {self.a_xy.shape} vs {self.b_xy.shape}")
        if self.a_xy.ndim != 2 or self.a_xy.shape[1] != 2:
            raise ValueError(f"expected (N, 2), got {self.a_xy.shape}")
        if self.confidence.shape[0] != self.a_xy.shape[0]:
            raise ValueError(
                f"confidence length {self.confidence.shape[0]} != N={self.a_xy.shape[0]}"
            )

    def __len__(self) -> int:
        return self.a_xy.shape[0]


class Matcher(ABC):
    """Match two images, returning correspondences in (a, b) pixel coords."""

    name: str = "abstract"

    @abstractmethod
    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> Correspondences:
        ...
