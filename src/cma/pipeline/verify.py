"""Appearance verification of a candidate transform.

Dense matchers hallucinate smooth, self-consistent warps on non-overlapping
content, so geometric support (RANSAC inliers) alone cannot reject a wrong
candidate. This module scores a candidate H by resampling the source onto
the target grid and measuring mutual information over the valid overlap —
an independent, modality-robust judge used to gate every stage of
`register_v2` (a candidate is only accepted if it scores at least as well
as the incumbent).
"""

from __future__ import annotations

import cv2
import numpy as np

from cma.metrics import mutual_information

REJECT = float("-inf")


def verification_score(
    source: np.ndarray,
    target: np.ndarray,
    H_target_to_source: np.ndarray,
    *,
    max_side: int = 512,
    min_overlap: float = 0.05,
    bins: int = 32,
) -> float:
    """MI between target and the H-resampled source over their overlap.

    Returns REJECT (-inf) for degenerate H or overlap below `min_overlap`
    of the target area. Scores are comparable between candidates on the
    same pair only — never across pairs.
    """
    H = np.asarray(H_target_to_source, dtype=np.float64)
    if H.shape != (3, 3) or not np.isfinite(H).all():
        return REJECT
    if abs(np.linalg.det(H)) < 1e-12:
        return REJECT

    src, f_s = _gray_capped(source, max_side)
    tgt, f_t = _gray_capped(target, max_side)
    # H maps target -> source in full-res coords; move it to capped coords.
    H_capped = np.diag([f_s, f_s, 1.0]) @ H @ np.diag([1.0 / f_t, 1.0 / f_t, 1.0])

    h_t, w_t = tgt.shape[:2]
    # WARP_INVERSE_MAP treats M as the dst->src mapping, which is exactly
    # what H (target -> source) is.
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    warped = cv2.warpPerspective(src.astype(np.float32), H_capped, (w_t, h_t), flags=flags)
    mask = cv2.warpPerspective(
        np.ones(src.shape[:2], np.float32), H_capped, (w_t, h_t), flags=flags
    )
    valid = mask > 0.99
    if valid.mean() < min_overlap:
        return REJECT
    return mutual_information(warped[valid], tgt[valid], bins=bins)


def _gray_capped(img: np.ndarray, max_side: int) -> tuple[np.ndarray, float]:
    """Grayscale + cap long side at `max_side`; returns (image, scale factor)."""
    g = img
    if g.ndim == 3:
        g = cv2.cvtColor(g[..., :3].astype(np.float32), cv2.COLOR_RGB2GRAY)
    h, w = g.shape[:2]
    f = min(1.0, max_side / max(h, w))
    if f < 1.0:
        g = cv2.resize(g, (max(1, round(w * f)), max(1, round(h * f))),
                       interpolation=cv2.INTER_AREA)
    return g, f
