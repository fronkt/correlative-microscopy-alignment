"""Coarse-to-fine registration with a verification gate (pyramid v2).

The v1 pyramid pooled correspondences from every tile, which fails for
dense matchers: they return ~10k confident matches for every tile, so
non-overlapping tiles flood RANSAC (results/README.md, Phase 4 final).

v2 replaces blind pooling with verified stages:

  Stage A (direct):  match source vs target at the matcher's native cap.
                     This is the zero-shot baseline — v2 starts from the
                     best known method, not from scratch.
  Stage T (tiles):   only when stage A support is weak — per-tile match +
                     per-tile RANSAC, each candidate judged by the
                     appearance verifier, best tile wins.
  Stage Z (zoom):    project the target footprint into the source via the
                     incumbent H, crop that ROI (+margin), re-match at the
                     higher effective resolution this buys, refit.

Every stage's candidate is accepted only if `verification_score` does not
decrease, so by construction v2 is never worse than direct matching under
the verifier's judgement. Optional certainty gating filters matcher
correspondences by confidence before any fit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from cma.estimators import EstimatedTransform, fit_transform
from cma.matchers.base import Correspondences, Matcher
from cma.pipeline.verify import REJECT, verification_score
from cma.pyramid import build


@dataclass
class RegistrationV2Result:
    """Accepted transform + per-stage diagnostics."""

    transform: EstimatedTransform
    stage: str  # "direct" | "tile" | "direct+zoom" | "tile+zoom"
    score_direct: float
    score_final: float
    n_correspondences: int  # of the accepted stage
    src_xy: np.ndarray  # accepted-stage correspondences (source frame)
    tgt_xy: np.ndarray  # accepted-stage correspondences (target frame)
    H_target_to_source: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.H_target_to_source = self.transform.as_3x3()


@dataclass
class _Candidate:
    transform: EstimatedTransform
    src_xy: np.ndarray
    tgt_xy: np.ndarray
    score: float
    stage: str


def register_v2(
    source: np.ndarray,
    target: np.ndarray,
    matcher: Matcher,
    scale_ratio: float,
    *,
    family: str = "auto",
    ransac_threshold_px: float = 5.5,
    certainty_threshold: float | None = None,
    zoom: bool = True,
    zoom_margin: float = 0.3,
    zoom_iters: int = 3,
    tile_fallback: bool = True,
    weak_support_inliers: int = 50,
    tile_overlap: float = 0.5,
    verify_max_side: int = 512,
) -> RegistrationV2Result:
    """Register `target` (narrow FOV) into `source` (wide FOV) coords.

    Returns H such that source_xy = H @ target_xy. Raises RuntimeError only
    if no stage produces a verifiable transform.
    """
    if target.ndim not in (2, 3):
        raise ValueError(f"target must be 2D or 3D, got shape {target.shape}")

    def fit(src_xy: np.ndarray, tgt_xy: np.ndarray, stage: str) -> _Candidate | None:
        if len(src_xy) < 4:
            return None
        try:
            est = fit_transform(
                src_xy=tgt_xy, dst_xy=src_xy,
                family=family,  # type: ignore[arg-type]
                ransac_threshold_px=ransac_threshold_px,
            )
        except (ValueError, RuntimeError):
            return None
        score = verification_score(
            source, target, est.as_3x3(), max_side=verify_max_side
        )
        if score == REJECT:
            return None
        return _Candidate(est, src_xy, tgt_xy, score, stage)

    # ---- Stage A: direct ----------------------------------------------
    corr = _gated(matcher.match(source, target), certainty_threshold)
    best = fit(corr.a_xy, corr.b_xy, "direct")
    score_direct = best.score if best else REJECT

    # ---- Stage T: tile search (weak-support fallback) ------------------
    if tile_fallback and (best is None or best.transform.n_inliers < weak_support_inliers):
        tile_size = int(min(target.shape[:2]))
        for tile in build(source, scale_ratio, tile_size=tile_size, overlap=tile_overlap):
            corr_t = _gated(matcher.match(tile.image, target), certainty_threshold)
            if len(corr_t) < 4:
                continue
            cand = fit(tile.tile_to_source(corr_t.a_xy), corr_t.b_xy, "tile")
            if cand is not None and (best is None or cand.score > best.score):
                best = cand

    if best is None:
        raise RuntimeError("no stage produced a verifiable transform")

    # ---- Stage Z: iterated zoom refinement ------------------------------
    if zoom:
        prev_roi: tuple[int, int, int, int] | None = None
        for _ in range(max(0, zoom_iters)):
            roi = _target_footprint_roi(
                best.transform.as_3x3(), source, target, zoom_margin
            )
            if roi is None or roi == prev_roi:
                break
            prev_roi = roi
            x0, y0, x1, y1 = roi
            crop = source[y0:y1, x0:x1]
            corr_z = _gated(matcher.match(crop, target), certainty_threshold)
            if len(corr_z) < 4:
                break
            src_xy = corr_z.a_xy + np.array([x0, y0], dtype=np.float64)
            stage = best.stage if best.stage.endswith("+zoom") else best.stage + "+zoom"
            cand = fit(src_xy, corr_z.b_xy, stage)
            if cand is None or cand.score < best.score:
                break  # verifier stopped improving; keep the incumbent
            best = cand

    return RegistrationV2Result(
        transform=best.transform,
        stage=best.stage,
        score_direct=score_direct,
        score_final=best.score,
        n_correspondences=int(len(best.src_xy)),
        src_xy=best.src_xy,
        tgt_xy=best.tgt_xy,
    )


def _gated(corr: Correspondences, threshold: float | None) -> Correspondences:
    """Optional certainty gating: drop correspondences below `threshold`."""
    if threshold is None or len(corr) == 0:
        return corr
    keep = corr.confidence >= threshold
    return Correspondences(
        a_xy=corr.a_xy[keep], b_xy=corr.b_xy[keep], confidence=corr.confidence[keep]
    )


def _target_footprint_roi(
    H: np.ndarray, source: np.ndarray, target: np.ndarray, margin: float
) -> tuple[int, int, int, int] | None:
    """Bounding box (+margin) of the target's corners projected into source.

    Returns None when the zoom buys nothing: degenerate projection, or the
    ROI covers nearly the whole source already.
    """
    h_t, w_t = target.shape[:2]
    h_s, w_s = source.shape[:2]
    corners = np.array(
        [[0, 0, 1], [w_t, 0, 1], [w_t, h_t, 1], [0, h_t, 1]], dtype=np.float64
    )
    proj = (H @ corners.T).T
    if not np.isfinite(proj).all() or (np.abs(proj[:, 2]) < 1e-9).any():
        return None
    proj = proj[:, :2] / proj[:, 2:3]
    x_lo, y_lo = proj.min(axis=0)
    x_hi, y_hi = proj.max(axis=0)
    pad_x = (x_hi - x_lo) * margin
    pad_y = (y_hi - y_lo) * margin
    x0 = int(max(0, np.floor(x_lo - pad_x)))
    y0 = int(max(0, np.floor(y_lo - pad_y)))
    x1 = int(min(w_s, np.ceil(x_hi + pad_x)))
    y1 = int(min(h_s, np.ceil(y_hi + pad_y)))
    if x1 - x0 < 32 or y1 - y0 < 32:
        return None
    if (x1 - x0) * (y1 - y0) > 0.8 * h_s * w_s:
        return None  # ROI ~ whole source; re-matching it gains no resolution
    return x0, y0, x1, y1
