"""Classical Control B pipeline: SIFT directly on (I_s, I_t) + optional MMI refine.

This deliberately does NOT use the pyramidal wrapper — it is the baseline the
proposed approach is benchmarked against.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import minimize

from cma.estimators import EstimatedTransform, fit_transform
from cma.matchers import Matcher, SIFTMatcher
from cma.metrics import mutual_information


@dataclass
class ClassicalResult:
    transform: EstimatedTransform
    n_correspondences: int
    refined: bool
    mi_initial: float
    mi_refined: float
    H_target_to_source: np.ndarray

    @classmethod
    def from_transform(
        cls,
        transform: EstimatedTransform,
        n_correspondences: int,
        refined: bool,
        mi_initial: float,
        mi_refined: float,
    ) -> ClassicalResult:
        return cls(
            transform=transform,
            n_correspondences=n_correspondences,
            refined=refined,
            mi_initial=mi_initial,
            mi_refined=mi_refined,
            H_target_to_source=transform.as_3x3(),
        )


def classical_register(
    source: np.ndarray,
    target: np.ndarray,
    matcher: Matcher | None = None,
    *,
    family: str = "auto",
    ransac_threshold_px: float = 3.0,
    refine_with_mi: bool = False,
    mi_max_iter: int = 40,
) -> ClassicalResult:
    """Direct SIFT + RANSAC registration, with optional MMI refinement."""
    matcher = matcher or SIFTMatcher()
    corr = matcher.match(source, target)
    if len(corr) < 4:
        raise RuntimeError(f"matcher returned only {len(corr)} correspondences (<4)")

    # Fit transform mapping target -> source (we call match(source, target)
    # so a_xy are source coords, b_xy are target coords).
    transform = fit_transform(
        src_xy=corr.b_xy,
        dst_xy=corr.a_xy,
        family=family,  # type: ignore[arg-type]
        ransac_threshold_px=ransac_threshold_px,
    )

    H_initial = transform.as_3x3().copy()
    mi_initial = _mi_at_h(source, target, H_initial)
    mi_refined = mi_initial
    refined = False

    if refine_with_mi:
        H_opt = _refine_mi(source, target, H_initial, max_iter=mi_max_iter)
        mi_new = _mi_at_h(source, target, H_opt)
        if mi_new > mi_initial:
            transform = EstimatedTransform(
                matrix=H_opt,
                family="homography",
                inliers=transform.inliers,
                residuals=transform.residuals,
                n_inliers=transform.n_inliers,
                mean_inlier_residual=transform.mean_inlier_residual,
            )
            mi_refined = mi_new
            refined = True

    return ClassicalResult.from_transform(
        transform=transform,
        n_correspondences=len(corr),
        refined=refined,
        mi_initial=mi_initial,
        mi_refined=mi_refined,
    )


def _mi_at_h(
    source: np.ndarray, target: np.ndarray, H: np.ndarray, max_side: int = 1024
) -> float:
    """Warp source into target's frame using H and compute MI vs target.

    MI is a global intensity statistic, so both images are evaluated at a
    capped resolution (`max_side`) with H rescaled accordingly — full-res
    warps made MMI refinement ~10x slower for no metric benefit.
    """
    source, f_s = _gray_capped(source, max_side)
    target, f_t = _gray_capped(target, max_side)
    # H maps target -> source in full-res coords; rescale to capped coords.
    S_s = np.diag([f_s, f_s, 1.0])
    S_t_inv = np.diag([1.0 / f_t, 1.0 / f_t, 1.0])
    H_capped = S_s @ H @ S_t_inv
    h, w = target.shape[:2]
    H_inv = np.linalg.inv(H_capped)
    warped = cv2.warpPerspective(
        source.astype(np.float32),
        H_inv,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return mutual_information(warped, target)


def _to_gray(img: np.ndarray) -> np.ndarray:
    """MI operates on single-channel intensity; collapse RGB(A) if present."""
    if img.ndim == 3:
        return cv2.cvtColor(img[..., :3].astype(np.float32), cv2.COLOR_RGB2GRAY)
    return img


def _gray_capped(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Grayscale + cap long side at `max_side`; returns (image, scale factor)."""
    g = _to_gray(img)
    h, w = g.shape[:2]
    f = min(1.0, max_side / max(h, w))
    if f < 1.0:
        g = cv2.resize(g, (max(1, round(w * f)), max(1, round(h * f))),
                       interpolation=cv2.INTER_AREA)
    return g, f


def _refine_mi(
    source: np.ndarray,
    target: np.ndarray,
    H_init: np.ndarray,
    max_iter: int = 40,
) -> np.ndarray:
    """Local Nelder-Mead refinement of the 8 free homography params."""
    base = H_init / H_init[2, 2]
    h0 = base.ravel()[:8]

    def neg_mi(h_flat: np.ndarray) -> float:
        H = np.append(h_flat, 1.0).reshape(3, 3)
        try:
            return -_mi_at_h(source, target, H)
        except (np.linalg.LinAlgError, cv2.error):
            return 1e9

    # Tight bounds via simplex scale rather than constraints
    res = minimize(
        neg_mi,
        h0,
        method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-4, "maxiter": max_iter, "adaptive": True},
    )
    H_opt = np.append(res.x, 1.0).reshape(3, 3)
    return H_opt
