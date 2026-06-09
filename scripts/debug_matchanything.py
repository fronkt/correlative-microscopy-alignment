"""Diagnose where MatchAnything error is coming from.

We compare three setups on the same synthetic pair (fov=0.10):
  (1) MatchAnything called directly on (full I_s, I_t)  -- no pyramid
  (2) MatchAnything called on the GT tile of I_s and I_t (oracle crop)
  (3) MatchAnything inside the pyramid pipeline
"""

import time

import cv2
import numpy as np

from cma.data import synthesize_pair
from cma.estimators import fit_transform
from cma.matchers import MatchAnythingMatcher


def _apply_h(H, xy):
    ones = np.ones((xy.shape[0], 1))
    hom = np.concatenate([xy, ones], axis=1)
    proj = hom @ H.T
    return proj[:, :2] / proj[:, 2:3]


def _check(label, src_xy_in_full, tgt_xy, gt_kp):
    print(f"\n--- {label} ---")
    print(f"  n_correspondences={len(src_xy_in_full)}")
    if len(src_xy_in_full) < 8:
        print("  too few to fit a transform")
        return
    t = fit_transform(src_xy=tgt_xy, dst_xy=src_xy_in_full)
    pred = _apply_h(t.as_3x3(), gt_kp.tgt_xy)
    err = np.linalg.norm(pred - gt_kp.src_xy, axis=1)
    print(f"  family={t.family} n_inliers={t.n_inliers} mean_inlier_resid={t.mean_inlier_residual:.3f}")
    print(f"  GT err mean={err.mean():.3f} median={np.median(err):.3f}")


def main() -> None:
    pair, H_gt = synthesize_pair(1024, 0.10, 256, seed=0, rotation_deg=5.0)
    print(f"scale_ratio={pair.scale_ratio:.3f}  source={pair.source.shape}  target={pair.target.shape}")

    print("\nloading MatchAnything ...")
    matcher = MatchAnythingMatcher()
    print(f"device={matcher.device}  max_long_side={matcher.max_long_side}")

    # (1) Direct full-image matching
    t = time.perf_counter()
    corr = matcher.match(pair.source, pair.target)
    print(f"\n(1) DIRECT full-image match: {len(corr)} pairs in {time.perf_counter() - t:.2f}s")
    if len(corr) >= 4:
        _check("direct full", corr.a_xy, corr.b_xy, pair.gt)

    # (2) Oracle crop: pull the exact GT tile out of source, match against target
    # Use H_gt to find target center in source, take a window matching the
    # target's physical extent.
    side_in_s = 1024 * np.sqrt(0.10)
    center_s = _apply_h(H_gt, np.array([[128.0, 128.0]]))[0]
    cx, cy = float(center_s[0]), float(center_s[1])
    half = int(side_in_s / 2)
    x0 = max(0, int(round(cx - half)))
    y0 = max(0, int(round(cy - half)))
    x1 = min(1024, x0 + 2 * half)
    y1 = min(1024, y0 + 2 * half)
    crop = pair.source[y0:y1, x0:x1]
    print(f"\noracle crop: ({x0},{y0})->({x1},{y1}) shape={crop.shape}")

    corr = matcher.match(crop, pair.target)
    print(f"(2) ORACLE CROP match: {len(corr)} pairs")
    if len(corr) >= 4:
        # Map a_xy from crop frame to full source frame
        a_in_full = corr.a_xy + np.array([x0, y0])
        _check("oracle crop", a_in_full, corr.b_xy, pair.gt)


if __name__ == "__main__":
    main()
