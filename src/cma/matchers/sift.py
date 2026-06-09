"""SIFT + Lowe-ratio matcher. Doubles as the classical Control B backbone."""

from __future__ import annotations

import cv2
import numpy as np

from cma.matchers.base import Correspondences, Matcher


def _to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    arr = np.clip(img, 0.0, 1.0) * 255.0
    return arr.astype(np.uint8)


class SIFTMatcher(Matcher):
    """OpenCV SIFT + brute-force ratio test."""

    name = "sift"

    def __init__(self, max_features: int = 4000, lowe_ratio: float = 0.8) -> None:
        self.sift = cv2.SIFT_create(nfeatures=max_features)
        self.lowe_ratio = float(lowe_ratio)
        self.bf = cv2.BFMatcher(cv2.NORM_L2)

    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> Correspondences:
        a = _to_uint8(image_a)
        b = _to_uint8(image_b)
        if a.ndim == 3:
            a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        if b.ndim == 3:
            b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)

        kp_a, des_a = self.sift.detectAndCompute(a, None)
        kp_b, des_b = self.sift.detectAndCompute(b, None)
        if des_a is None or des_b is None or len(kp_a) < 2 or len(kp_b) < 2:
            return _empty()

        knn = self.bf.knnMatch(des_a, des_b, k=2)
        good = []
        for pair in knn:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < self.lowe_ratio * n.distance:
                good.append(m)
        if not good:
            return _empty()

        a_xy = np.array([kp_a[m.queryIdx].pt for m in good], dtype=np.float64)
        b_xy = np.array([kp_b[m.trainIdx].pt for m in good], dtype=np.float64)
        # Confidence ~ inverse of Lowe ratio (higher = more distinctive)
        conf = np.array(
            [1.0 - (m.distance / max(1e-6, n.distance))
             for m, n in (p for p in knn if len(p) == 2)
             if m.distance < self.lowe_ratio * n.distance],
            dtype=np.float64,
        )
        # Defensive: align confidence length with matches in case of degenerate pairs
        if conf.shape[0] != a_xy.shape[0]:
            conf = np.ones(a_xy.shape[0], dtype=np.float64)
        return Correspondences(a_xy=a_xy, b_xy=b_xy, confidence=conf)


def _empty() -> Correspondences:
    return Correspondences(
        a_xy=np.zeros((0, 2), dtype=np.float64),
        b_xy=np.zeros((0, 2), dtype=np.float64),
        confidence=np.zeros((0,), dtype=np.float64),
    )
