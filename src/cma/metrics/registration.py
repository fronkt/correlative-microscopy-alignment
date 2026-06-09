"""Registration metrics: P_match@k, mean / median error, success rate."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


def _euclidean(pred_xy: np.ndarray, gt_xy: np.ndarray) -> np.ndarray:
    if pred_xy.shape != gt_xy.shape:
        raise ValueError(f"shape mismatch: {pred_xy.shape} vs {gt_xy.shape}")
    return np.linalg.norm(pred_xy - gt_xy, axis=1)


def p_match_at_k(pred_xy: np.ndarray, gt_xy: np.ndarray, k_px: float) -> float:
    """Fraction of points whose Euclidean error is within k pixels."""
    if k_px <= 0:
        raise ValueError(f"k_px must be > 0, got {k_px}")
    err = _euclidean(pred_xy, gt_xy)
    return float((err < k_px).mean()) if err.size else 0.0


def mean_error(pred_xy: np.ndarray, gt_xy: np.ndarray) -> float:
    err = _euclidean(pred_xy, gt_xy)
    return float(err.mean()) if err.size else float("nan")


def median_error(pred_xy: np.ndarray, gt_xy: np.ndarray) -> float:
    err = _euclidean(pred_xy, gt_xy)
    return float(np.median(err)) if err.size else float("nan")


def success_rate(per_pair_mu_err: np.ndarray, threshold_px: float = 5.0) -> float:
    """Fraction of pairs whose mean error is below `threshold_px`."""
    arr = np.asarray(per_pair_mu_err, dtype=float)
    return float((arr < threshold_px).mean()) if arr.size else 0.0


@dataclass
class RegistrationMetrics:
    mu_err: float
    med_err: float
    p_match_at_1: float
    p_match_at_3: float
    p_match_at_5: float
    p_match_at_10: float

    def as_dict(self) -> dict:
        return asdict(self)


def registration_metrics(pred_xy: np.ndarray, gt_xy: np.ndarray) -> RegistrationMetrics:
    return RegistrationMetrics(
        mu_err=mean_error(pred_xy, gt_xy),
        med_err=median_error(pred_xy, gt_xy),
        p_match_at_1=p_match_at_k(pred_xy, gt_xy, 1.0),
        p_match_at_3=p_match_at_k(pred_xy, gt_xy, 3.0),
        p_match_at_5=p_match_at_k(pred_xy, gt_xy, 5.0),
        p_match_at_10=p_match_at_k(pred_xy, gt_xy, 10.0),
    )
