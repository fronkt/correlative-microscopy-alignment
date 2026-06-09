"""Joint-histogram mutual information for image-pair similarity.

Used by the classical Control B pipeline to refine an initial homography
estimated from SIFT correspondences. Higher MI = better alignment.
"""

from __future__ import annotations

import numpy as np


def mutual_information(
    a: np.ndarray,
    b: np.ndarray,
    bins: int = 32,
    eps: float = 1e-12,
) -> float:
    """Mutual information between two same-shape images using a joint histogram."""
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    a_flat = np.asarray(a).ravel().astype(np.float64)
    b_flat = np.asarray(b).ravel().astype(np.float64)

    # Robust normalisation to [0, 1] for binning
    a_norm = _normalise(a_flat)
    b_norm = _normalise(b_flat)

    joint, _, _ = np.histogram2d(a_norm, b_norm, bins=bins, range=[[0, 1], [0, 1]])
    p_joint = joint / max(joint.sum(), eps)
    p_a = p_joint.sum(axis=1, keepdims=True)
    p_b = p_joint.sum(axis=0, keepdims=True)
    denom = p_a * p_b
    nz = (p_joint > 0) & (denom > 0)
    mi = float((p_joint[nz] * np.log(p_joint[nz] / denom[nz])).sum())
    return mi


def _normalise(x: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(x, [1, 99])
    if hi <= lo:
        return np.clip(x - x.min(), 0, 1)
    y = (x - lo) / (hi - lo)
    return np.clip(y, 0.0, 1.0)
