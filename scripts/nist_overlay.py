"""Visual sanity overlay: warp the LEFT image into the RIGHT frame with the composed GT and blend.

Only APPLIES a transform (no registration).  Usage:
  python scripts/nist_overlay.py 718_s100_bse1-om [out.png]
Reads tools/handcheck/demo_pairs.json.
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "handcheck"))
from handcheck import load_gray  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def norm8(a):
    lo, hi = np.percentile(a[a > 0] if (a > 0).any() else a, [1, 99.5])
    return (np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def main():
    pid = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(r"C:\Users\frank\Documents\materials-bench\nist\fits", pid + "_gt_overlay.png")
    rows = json.load(open(os.path.join(HERE, "..", "tools", "handcheck", "demo_pairs.json")))
    r = next(x for x in rows if x["pair_id"] == pid)
    L, R = load_gray(r["left_path"]), load_gray(r["right_path"])
    H = np.array(r["gt_h"]).reshape(3, 3)
    f = min(1.0, 1400 / max(R.shape))
    D = np.diag([f, f, 1.0])
    Hd = D @ H  # left full px -> right display px (approx; half-pixel offsets negligible at this scale)
    w, h = round(R.shape[1] * f), round(R.shape[0] * f)
    Lw = cv2.warpPerspective(norm8(L), Hd, (w, h))
    Rs = norm8(cv2.resize(R, (w, h), interpolation=cv2.INTER_AREA))
    blend = np.dstack([Rs, Lw, Lw])  # BGR: right in red, warped left in cyan
    cv2.imwrite(out, np.hstack([np.dstack([Rs] * 3), np.dstack([Lw] * 3), blend]))
    print(out)


if __name__ == "__main__":
    main()
