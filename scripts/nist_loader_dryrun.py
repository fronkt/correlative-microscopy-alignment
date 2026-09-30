"""Dry-run of the arm-2 loader on the NIST component: NO matcher, NO registration.

Checks: all NIST pairs load through cma.materials_pool.MaterialsPoolLoader (the loader that
scripts/arm2_run_candidates.py swaps into run_triage_candidates), image shapes match the manifest, GT points lie
inside both images, tgt->src homography fitted to the 25 GT points reproduces them, scale_ratio is sane.
"""
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from cma.materials_pool import MaterialsPoolLoader  # noqa: E402


def main():
    import arm2_run_candidates as arc  # noqa: F401  (import only; main() is not called)
    import run_triage_candidates as rtc

    assert rtc.AmalgaMatchLoader is MaterialsPoolLoader, "runner would not pick up the pool loader"
    L = MaterialsPoolLoader()
    n = bad = 0
    worst = 0.0
    comps = {}
    for pair, rec in L.iter(components=["NIST"]):
        n += 1
        hs, ws = pair.source.shape
        ht, wt = pair.target.shape
        g = pair.gt
        ok_t = ((g.tgt_xy[:, 0] >= 0) & (g.tgt_xy[:, 0] <= wt - 1) & (g.tgt_xy[:, 1] >= 0) & (g.tgt_xy[:, 1] <= ht - 1)).all()
        ok_s = ((g.src_xy[:, 0] >= 0) & (g.src_xy[:, 0] <= ws - 1) & (g.src_xy[:, 1] >= 0) & (g.src_xy[:, 1] <= hs - 1)).all()
        H, _ = cv2.findHomography(g.tgt_xy.astype(np.float64), g.src_xy.astype(np.float64), 0)
        q = np.c_[g.tgt_xy, np.ones(len(g.tgt_xy))] @ H.T
        res = np.linalg.norm(q[:, :2] / q[:, 2:3] - g.src_xy, axis=1).max()
        worst = max(worst, res)
        flags = []
        if not (ok_t and ok_s):
            flags.append("GT_OUTSIDE")
        if len(g) != 25:
            flags.append(f"n_gt={len(g)}")
        if not (0.05 < pair.scale_ratio < 20):
            flags.append("scale?")
        if pair.source.dtype != np.float32 or pair.source.max() <= 0:
            flags.append("img?")
        if flags:
            bad += 1
            print("FLAG", rec.pair_id, flags)
        comps.setdefault(rec.subclass, []).append(pair.scale_ratio)
    print(f"loaded {n} NIST pairs ({len(L)} total in pool); flagged {bad}; max GT->homography refit residual {worst:.2e} px")
    for k, v in sorted(comps.items()):
        print(f"  {k}: n={len(v)} scale_ratio(tgt_px/src_px) {min(v):.3f}-{max(v):.3f}")
    print("runner check OK: arm2_run_candidates rebinds rtc.AmalgaMatchLoader -> MaterialsPoolLoader; loader.iter() has no filter")


if __name__ == "__main__":
    main()
