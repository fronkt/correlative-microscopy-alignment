"""Robust transform estimation: MAGSAC++ wrapper for affine + homography."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

TransformFamily = Literal["affine", "homography", "auto"]


@dataclass
class EstimatedTransform:
    """Result of fitting a transform from b-frame to a-frame coords."""

    matrix: np.ndarray          # 3x3 (homography) or 2x3 (affine)
    family: Literal["affine", "homography"]
    inliers: np.ndarray         # (N,) bool mask over input correspondences
    residuals: np.ndarray       # (N,) per-point residual after fit (px)
    n_inliers: int
    mean_inlier_residual: float

    def as_3x3(self) -> np.ndarray:
        """Return the transform as a 3x3 matrix regardless of family."""
        if self.matrix.shape == (3, 3):
            return self.matrix.astype(np.float64)
        H = np.eye(3, dtype=np.float64)
        H[:2, :3] = self.matrix
        return H


def _apply_h(H: np.ndarray, xy: np.ndarray) -> np.ndarray:
    ones = np.ones((xy.shape[0], 1), dtype=xy.dtype)
    hom = np.concatenate([xy, ones], axis=1)
    proj = hom @ H.T
    return proj[:, :2] / proj[:, 2:3]


def _residuals(M3x3: np.ndarray, src_xy: np.ndarray, dst_xy: np.ndarray) -> np.ndarray:
    pred = _apply_h(M3x3, src_xy)
    return np.linalg.norm(pred - dst_xy, axis=1)


def fit_transform(
    src_xy: np.ndarray,
    dst_xy: np.ndarray,
    family: TransformFamily = "auto",
    ransac_threshold_px: float = 3.0,
    max_iters: int = 10_000,
    confidence: float = 0.999,
) -> EstimatedTransform:
    """Fit a robust transform mapping `src_xy` -> `dst_xy`.

    `src_xy` is the source side of the correspondence (e.g. target-frame
    points), `dst_xy` is the destination side (e.g. source-frame points);
    naming follows opencv convention.

    If `family="auto"`, both affine and homography are fit and the one with
    the lower BIC-like score on inliers wins (penalising the extra DoF).
    """
    if src_xy.shape != dst_xy.shape:
        raise ValueError(f"shape mismatch: {src_xy.shape} vs {dst_xy.shape}")
    if len(src_xy) < 4:
        raise ValueError(f"need >= 4 correspondences, got {len(src_xy)}")

    src = src_xy.astype(np.float32).reshape(-1, 1, 2)
    dst = dst_xy.astype(np.float32).reshape(-1, 1, 2)

    candidates: list[EstimatedTransform] = []

    if family in ("homography", "auto"):
        H, mask = cv2.findHomography(
            src,
            dst,
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=ransac_threshold_px,
            maxIters=max_iters,
            confidence=confidence,
        )
        if H is not None and mask is not None:
            mask_bool = mask.ravel().astype(bool)
            res = _residuals(H, src_xy, dst_xy)
            n_in = int(mask_bool.sum())
            inlier_mean = float(res[mask_bool].mean()) if n_in else float("inf")
            candidates.append(
                EstimatedTransform(
                    matrix=H.astype(np.float64),
                    family="homography",
                    inliers=mask_bool,
                    residuals=res,
                    n_inliers=n_in,
                    mean_inlier_residual=inlier_mean,
                )
            )

    if family in ("affine", "auto"):
        A, mask = cv2.estimateAffine2D(
            src,
            dst,
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=ransac_threshold_px,
            maxIters=max_iters,
            confidence=confidence,
        )
        if A is not None and mask is not None:
            mask_bool = mask.ravel().astype(bool)
            A3 = np.eye(3, dtype=np.float64)
            A3[:2, :3] = A
            res = _residuals(A3, src_xy, dst_xy)
            n_in = int(mask_bool.sum())
            inlier_mean = float(res[mask_bool].mean()) if n_in else float("inf")
            candidates.append(
                EstimatedTransform(
                    matrix=A.astype(np.float64),
                    family="affine",
                    inliers=mask_bool,
                    residuals=res,
                    n_inliers=n_in,
                    mean_inlier_residual=inlier_mean,
                )
            )

    if not candidates:
        raise RuntimeError("transform estimation failed (no candidate found)")

    if family == "auto" and len(candidates) == 2:
        # BIC-style selection: penalise extra DoF (affine 6, homography 8).
        # score = N * log(mean_residual^2) + dof * log(N)  (lower is better)
        def score(c: EstimatedTransform) -> float:
            n = max(1, c.n_inliers)
            r2 = max(1e-6, c.mean_inlier_residual**2)
            dof = 8 if c.family == "homography" else 6
            return n * np.log(r2) + dof * np.log(n)

        return min(candidates, key=score)

    return candidates[0]
