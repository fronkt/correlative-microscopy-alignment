"""Fit an affine to each pair's GT correspondences and report residuals.

Validates the loader's coordinate conventions independent of any matcher:
if GT (tgt -> src) fits a low-residual affine and its scale agrees with the
pixel-size ratio, the column ordering / flip logic is correct.

Usage: python scripts/check_gt_consistency.py
"""

from __future__ import annotations

import numpy as np

from cma.data import AmalgaMatchLoader

loader = AmalgaMatchLoader("data/AmalgaMatch")

print(f"{len(loader.records)} pairs\n")
print(f"{'pair_id':<60} {'n':>3} {'resid_px':>9} {'gt_scale':>9} {'px_ratio':>9}")

worst = []
for rec in loader.records:
    gt = loader._gt[rec.pair_id]
    tgt = np.hstack([gt.tgt_xy, np.ones((len(gt), 1))])
    A, res, *_ = np.linalg.lstsq(tgt, gt.src_xy, rcond=None)
    pred = tgt @ A
    resid = float(np.sqrt(((pred - gt.src_xy) ** 2).sum(axis=1)).mean())
    # linear part scale: average singular value of the 2x2 block
    sv = np.linalg.svd(A[:2, :2], compute_uv=False)
    gt_scale = float(sv.mean())
    px_ratio = rec.target_pixel_nm / rec.source_pixel_nm
    worst.append((resid, rec.pair_id, gt_scale, px_ratio, len(gt)))

worst.sort(reverse=True)
for resid, pid, gt_scale, px_ratio, n in worst[:10]:
    print(f"{pid:<60} {n:>3} {resid:>9.2f} {gt_scale:>9.3f} {px_ratio:>9.3f}")

resids = np.array([w[0] for w in worst])
scale_agree = np.array([abs(w[2] - w[3]) / w[3] for w in worst])
print(f"\nresidual px: median {np.median(resids):.2f}, p90 {np.percentile(resids, 90):.2f}, max {resids.max():.2f}")
print(f"|gt_scale - px_ratio|/px_ratio: median {np.median(scale_agree):.3f}, p90 {np.percentile(scale_agree, 90):.3f}")
print(f"pairs with scale disagreement > 10%: {(scale_agree > 0.10).sum()}")
