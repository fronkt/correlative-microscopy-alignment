"""Synthesize severer-FOV versions of real pairs by cropping the target.

Aim 3 of the research plan asks for the failure FOV per backbone, but the
real dataset has almost no pairs below area ratio 0.25 (n=4 under 0.05).
Cropping the *target* of a real cross-modal pair shrinks its physical
extent while leaving appearance, modality gap and pixel sizes untouched,
so a ladder of crops sweeps FOV with appearance held fixed.

Evaluation design: GT is NOT filtered to the crop. All GT target coords
are shifted into the crop frame and kept, including points outside the
cropped image. The registration transform is a global model (affine /
homography), so evaluating it at out-of-crop GT points measures
extrapolation — exactly the practical severe-FOV task of localizing a
small view inside a wide map. This keeps n_gt constant across rungs and
avoids censoring deep rungs on the dataset's sparse GT (median ~37
points/pair). Consequence: TPS-refined errors (fit on in-crop inliers)
are unreliable at out-of-crop points — ladder analyses must use the
transform-based ``mu_ed``, not ``mu_ed_tps``.

Conventions: pixel sizes (and hence ``ImagePair.scale_ratio``) are
unchanged by a crop; the FOV *area ratio* scales with the cropped pixel
area. Source image and source-side GT coords are untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from cma.data.types import ImagePair, KeypointSet


@dataclass(frozen=True)
class LadderRung:
    pair: ImagePair
    area_ratio: float  # achieved target/source physical area ratio
    crop_origin_xy: tuple[int, int]  # (x0, y0) of crop in original target
    n_gt_inside: int  # GT points whose target coord lies inside the crop


def crop_target_to_area_ratio(
    pair: ImagePair,
    base_area_ratio: float,
    desired_area_ratio: float,
) -> LadderRung | None:
    """Crop ``pair.target`` so the pair's FOV area ratio becomes ~desired.

    The crop window is centered on the GT target-point centroid (clipped
    to the image) so it stays inside the annotated overlap region.
    Returns None when the rung is not below the pair's base ratio.
    """
    if pair.gt is None or desired_area_ratio >= base_area_ratio:
        return None
    h, w = pair.target.shape[:2]
    f = float(np.sqrt(desired_area_ratio / base_area_ratio))
    ch, cw = max(8, round(h * f)), max(8, round(w * f))

    cx, cy = pair.gt.tgt_xy.mean(axis=0)
    x0 = int(np.clip(round(cx - cw / 2), 0, w - cw))
    y0 = int(np.clip(round(cy - ch / 2), 0, h - ch))

    tgt = pair.gt.tgt_xy - np.array([x0, y0], dtype=pair.gt.tgt_xy.dtype)
    inside = (
        (tgt[:, 0] >= 0) & (tgt[:, 0] <= cw - 1)
        & (tgt[:, 1] >= 0) & (tgt[:, 1] <= ch - 1)
    )
    gt = KeypointSet(src_xy=pair.gt.src_xy.copy(), tgt_xy=tgt)
    cropped = ImagePair(
        source=pair.source,
        target=pair.target[y0 : y0 + ch, x0 : x0 + cw].copy(),
        scale_ratio=pair.scale_ratio,
        gt=gt,
        metadata={**pair.metadata, "fov_ladder_crop": (x0, y0, cw, ch)},
    )
    achieved = base_area_ratio * (ch * cw) / (h * w)
    return LadderRung(
        pair=cropped, area_ratio=float(achieved),
        crop_origin_xy=(x0, y0), n_gt_inside=int(inside.sum()),
    )
