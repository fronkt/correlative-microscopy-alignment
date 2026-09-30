"""Compose cross-modal GT homographies for NIST serial-section pairs from same-modality fits.

GT(left raw -> right raw) = inv(H_right) @ S(regL -> regR) @ H_left
  H_x   : raw_x -> reg_x  (results/arm2/nist_gt_fits.json, from nist_fit_homography.py)
  S     : registered-frame relation between modalities (ASSUMPTION, README-based):
          BSE1_reg, BSE2_reg share one 2022x1980 grid (identity);
          OM_reg covers the same physical extent as BSE1_reg on a finer grid -> pure scaling by the
          size ratio, pixel-centre convention.
          EBSD_reg -> BSE1_reg offset is NOT recoverable from per-section files (no pair emitted).
Emits tools/handcheck/demo_pairs.json (+ a pair-list CSV) and optional overlay PNGs for visual sanity
(overlays only APPLY the composed transform; no registration is run on cross-modal pairs).
"""
import argparse
import csv
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FITS = os.path.join(HERE, "..", "results", "arm2", "nist_gt_fits.json")
ROOT = r"C:\Users\frank\Documents\materials-bench\nist"
SHAPES = {}  # filled from fits (reg_shape)


def path_of(alloy, s, kind):
    g = glob.glob(os.path.join(ROOT, alloy, f"s{s:03d}", f"{kind}__*"))
    return g[0]


def reg_scale(shape_from, shape_to):
    """Pixel-centre scaling between two registered grids covering the same physical extent."""
    (hf, wf), (ht, wt) = shape_from, shape_to
    sx, sy = wt / wf, ht / hf
    return np.array([[sx, 0, 0.5 * sx - 0.5], [0, sy, 0.5 * sy - 0.5], [0, 0, 1.0]])


def chain(fits, alloy, s, left, right):
    fl, fr = fits[f"{alloy}/s{s:03d}/{left}"], fits[f"{alloy}/s{s:03d}/{right}"]
    Hl, Hr = np.array(fl["H_full_dense"]), np.array(fr["H_full_dense"])
    S = reg_scale(fl["reg_shape"], fr["reg_shape"])
    if {left, right} <= {"bse1", "bse2"}:
        assert fl["reg_shape"] == fr["reg_shape"]
    G = np.linalg.inv(Hr) @ S @ Hl
    return G / G[2, 2]


PAIR_DESIGN = [("bse1", "om"), ("bse2", "bse1"), ("bse2", "om")]  # (left, right)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alloy", default="718")
    ap.add_argument("--sections", type=int, nargs="+", required=True)
    ap.add_argument("--out-json", default=os.path.join(HERE, "..", "tools", "handcheck", "demo_pairs.json"))
    ap.add_argument("--out-csv", default=None)
    ap.add_argument("--pairs", nargs="+", default=[f"{a}-{b}" for a, b in PAIR_DESIGN])
    a = ap.parse_args()
    fits = json.load(open(FITS))
    rows = []
    for s in a.sections:
        for pr in a.pairs:
            l, r = pr.split("-")
            G = chain(fits, a.alloy, s, l, r)
            rows.append({"pair_id": f"{a.alloy}_s{s:03d}_{l}-{r}",
                         "left_path": path_of(a.alloy, s, l + "_raw"),
                         "right_path": path_of(a.alloy, s, r + "_raw"),
                         "gt_h": G.ravel().tolist()})
    json.dump(rows, open(a.out_json, "w"), indent=1)
    if a.out_csv:
        with open(a.out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["pair_id", "left_path", "right_path", "gt_h"])
            for r in rows:
                w.writerow([r["pair_id"], r["left_path"], r["right_path"], json.dumps(r["gt_h"])])
    print(len(rows), "pairs ->", a.out_json)


if __name__ == "__main__":
    main()
