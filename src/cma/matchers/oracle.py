"""OracleMatcher: uses a known homography to fabricate correspondences.

Purpose: exercise the consensus estimator with controlled noise/outlier
mixes in isolation. Intended for direct use against the full source image
(NOT through the pyramid wrapper — the oracle has no knowledge of tile
offsets and would reject most projections at the tile boundary). For
end-to-end pipeline tests use SIFTMatcher or a real backbone.
"""

from __future__ import annotations

import numpy as np

from cma.matchers.base import Correspondences, Matcher


def _apply_h(H: np.ndarray, xy: np.ndarray) -> np.ndarray:
    ones = np.ones((xy.shape[0], 1), dtype=xy.dtype)
    hom = np.concatenate([xy, ones], axis=1)
    proj = hom @ H.T
    return proj[:, :2] / proj[:, 2:3]


class OracleMatcher(Matcher):
    """Generates correspondences using a ground-truth homography.

    The orientation convention is fixed by `direction`:
      - `direction="b_to_a"` means the supplied H maps points in image B
        into image A: x_a = H @ x_b. The match() call returns sampled b
        points and their H-projected a points (the typical I_t -> I_s
        setup in this project).

    Args:
        H_b_to_a: 3x3 homography from B-frame to A-frame coords.
        n_samples: number of correspondences to emit per call.
        noise_px: gaussian pixel noise added to the A-side points (sim
            sub-pixel matcher noise).
        outlier_frac: fraction of returned correspondences replaced with
            random outliers (to test RANSAC).
        seed: RNG seed.
    """

    name = "oracle"

    def __init__(
        self,
        H_b_to_a: np.ndarray,
        n_samples: int = 256,
        noise_px: float = 0.0,
        outlier_frac: float = 0.0,
        seed: int = 0,
    ) -> None:
        if H_b_to_a.shape != (3, 3):
            raise ValueError(f"H must be 3x3, got {H_b_to_a.shape}")
        if not 0.0 <= outlier_frac < 1.0:
            raise ValueError(f"outlier_frac must be in [0, 1), got {outlier_frac}")
        self.H_b_to_a = H_b_to_a.astype(np.float64)
        self.n_samples = int(n_samples)
        self.noise_px = float(noise_px)
        self.outlier_frac = float(outlier_frac)
        self._rng = np.random.default_rng(seed)

    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> Correspondences:
        Ha, Wa = image_a.shape[:2]
        Hb, Wb = image_b.shape[:2]

        # Sample uniformly inside B (target), project to A (source), keep
        # only those whose A-projection lands inside image A.
        b_xy = self._rng.uniform(
            low=[2, 2],
            high=[Wb - 3, Hb - 3],
            size=(self.n_samples * 2, 2),
        )
        a_xy = _apply_h(self.H_b_to_a, b_xy)
        in_a = (
            (a_xy[:, 0] >= 0)
            & (a_xy[:, 0] < Wa)
            & (a_xy[:, 1] >= 0)
            & (a_xy[:, 1] < Ha)
        )
        b_xy = b_xy[in_a][: self.n_samples]
        a_xy = a_xy[in_a][: self.n_samples]

        if self.noise_px > 0:
            a_xy = a_xy + self._rng.normal(0.0, self.noise_px, size=a_xy.shape)

        n_out = int(round(len(a_xy) * self.outlier_frac))
        if n_out > 0:
            idx = self._rng.choice(len(a_xy), size=n_out, replace=False)
            a_xy[idx] = self._rng.uniform(
                low=[0, 0], high=[Wa - 1, Ha - 1], size=(n_out, 2)
            )

        conf = np.ones(len(a_xy), dtype=np.float64)
        if n_out > 0:
            conf[idx] = 0.1

        return Correspondences(a_xy=a_xy, b_xy=b_xy, confidence=conf)
