"""Build the NIST AM Bench serial-section component (component "NIST") of the arm-2 materials pool.

81 pairs: IN718 sections 25,50,..,500 x {BSE1-vs-optical, BSE2-vs-BSE1, BSE2-vs-optical}; IN625 sections
100,125,..,600 x {BSE2-vs-BSE1}.  GT = composed same-modality homographies (see nist_gt_chain.py); GT points =
a FIXED 5x5 grid of TARGET-image points mapped through the GT homography into the SOURCE image.
No matcher/registration is run.  Uses scripts/arm2_common.finalize_pair, then run scripts/arm2_assemble.py to
regenerate manifest.csv (this also runs the frozen overlay detector).

Rules (logged in each record and in results/arm2/nist_feasibility.md):
  * optical raw images are downsampled by a FIXED factor of 4 (cv2.INTER_AREA); BSE images stay native.
  * grey values: percentile stretch (5, 99.5) computed (pores/precipitates clip; optical texture spans <2% of 16-bit range) inside the specimen mask (whole image for BSE2), to uint16.
  * specimen mask: BSE1/optical = Otsu on a blurred 600-px thumbnail, largest connected component, holes filled;
    BSE2 = whole image (its FOV, 607/750 um, lies inside the specimen by construction).
  * grid: 5x5 linspace over [lo+0.15*ext, hi-0.15*ext] of the TARGET mask bounding box; points outside the mask or
    outside the source image are counted and logged (points are NOT moved).
  * cluster = "NIST:<alloy>:blk<section//100>" (alloy x block of 100 sections).
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import tifffile
from scipy import ndimage as ndi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arm2_common as ac  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"C:\Users\frank\Documents\materials-bench\nist"
OM_DS = 4
FOV_UM = {("718", "bse1"): 1750.0, ("718", "bse2"): 607.0, ("625", "bse1"): 1400.0, ("625", "bse2"): 750.0}
MOD_NAME = {"bse1": "BSE FOV1 (backscatter, raw uint16)", "bse2": "BSE FOV2 (backscatter, raw uint16)",
            "om": "reflected-light optical, etched (raw uint16, downsampled x4)"}
DESIGN = {"718": (list(range(25, 501, 25)), [("bse1", "om"), ("bse2", "bse1"), ("bse2", "om")]),
          "625": (list(range(100, 601, 25)), [("bse2", "bse1")])}


def load_fits():
    d = {}
    for p in sorted(glob.glob(os.path.join(HERE, "..", "results", "arm2", "nist_gt_fits*.json"))):
        d.update(json.load(open(p)))
    return d


def path_of(alloy, s, kind):
    g = [p for p in glob.glob(os.path.join(ROOT, alloy, f"s{s:03d}", f"{kind}__*")) if not p.endswith(".part")]
    assert len(g) == 1, (alloy, s, kind, g)
    return g[0]


def scale_mat(fx, fy):
    return np.array([[fx, 0, 0.5 * fx - 0.5], [0, fy, 0.5 * fy - 0.5], [0, 0, 1.0]])


def reg_scale(shape_from, shape_to):
    (hf, wf), (ht, wt) = shape_from, shape_to
    return scale_mat(wt / wf, ht / hf)


def specimen_mask(img, mod):
    """Return (bbox x0,y0,x1,y1 in full px, boolean mask at thumbnail scale, thumbnail scale factor)."""
    h, w = img.shape
    f = 600.0 / max(h, w)
    th, tw = round(h * f), round(w * f)
    t = cv2.resize(img.astype(np.float32), (tw, th), interpolation=cv2.INTER_AREA)
    if mod == "bse2":
        return (0, 0, w - 1, h - 1), np.ones((th, tw), bool), f
    t = cv2.GaussianBlur(t, (0, 0), 3)
    lo, hi = np.percentile(t, [1, 99.5])
    t8 = (np.clip((t - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    _, m = cv2.threshold(t8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lab, n = ndi.label(m > 0)
    if n == 0:
        return (0, 0, w - 1, h - 1), np.ones((th, tw), bool), f
    big = lab == (1 + int(np.argmax(ndi.sum(m > 0, lab, range(1, n + 1)))))
    big = ndi.binary_fill_holes(big)
    ys, xs = np.where(big)
    return (xs.min() / f, ys.min() / f, xs.max() / f, ys.max() / f), big, f


def stretch_u16(img, mask_thumb, f):
    h, w = img.shape
    t = cv2.resize(img.astype(np.float32), (mask_thumb.shape[1], mask_thumb.shape[0]), interpolation=cv2.INTER_AREA)
    v = t[mask_thumb]
    lo, hi = np.percentile(v, [5, 99.5])
    out = np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
    return (out * 65535 + 0.5).astype(np.uint16), (float(lo), float(hi))


def load_raw(alloy, s, mod):
    a = np.asarray(tifffile.imread(path_of(alloy, s, mod + "_raw")))
    D = np.eye(3)
    if mod == "om":
        h, w = a.shape
        nh, nw = round(h / OM_DS), round(w / OM_DS)
        a = cv2.resize(a.astype(np.float32), (nw, nh), interpolation=cv2.INTER_AREA)
        D = scale_mat(nw / w, nh / h)
    return a, D


def chain(fits, alloy, s, left, right):
    fl, fr = fits[f"{alloy}/s{s:03d}/{left}"], fits[f"{alloy}/s{s:03d}/{right}"]
    Hl, Hr = np.array(fl["H_full_dense"]), np.array(fr["H_full_dense"])
    S = reg_scale(fl["reg_shape"], fr["reg_shape"])
    G = np.linalg.inv(Hr) @ S @ Hl
    return G / G[2, 2]


def apply_h(H, p):
    q = np.c_[p, np.ones(len(p))] @ H.T
    return q[:, :2] / q[:, 2:3]


def build_pair(fits, alloy, s, left, right):
    """left = target (smaller FOV) modality, right = source (larger FOV)."""
    tgt_full, Dt = load_raw(alloy, s, left)  # Dt = identity (left never optical)
    src_full, Ds = load_raw(alloy, s, right)
    G = chain(fits, alloy, s, left, right)  # raw-left full px -> raw-right FULL px
    Gd = Ds @ G @ np.linalg.inv(Dt)  # target px -> source (possibly downsampled) px
    (x0, y0, x1, y1), mt, ft = specimen_mask(tgt_full, left)
    gx = np.linspace(x0 + 0.15 * (x1 - x0), x1 - 0.15 * (x1 - x0), 5)
    gy = np.linspace(y0 + 0.15 * (y1 - y0), y1 - 0.15 * (y1 - y0), 5)
    tgt_xy = np.array([(x, y) for y in gy for x in gx])
    src_xy = apply_h(Gd, tgt_xy)
    hs, ws = src_full.shape
    inside_src = ((src_xy[:, 0] >= 0) & (src_xy[:, 0] <= ws - 1) & (src_xy[:, 1] >= 0) & (src_xy[:, 1] <= hs - 1))
    ti = np.clip((tgt_xy * ft).round().astype(int), 0, np.array([mt.shape[1] - 1, mt.shape[0] - 1]))
    inside_mask = mt[ti[:, 1], ti[:, 0]]
    ms, fs_ = specimen_mask(src_full, right)[1:]
    tgt_u16, tr = stretch_u16(tgt_full, mt, ft)
    src_u16, sr = stretch_u16(src_full, ms, fs_)
    fl, fr = fits[f"{alloy}/s{s:03d}/{left}"], fits[f"{alloy}/s{s:03d}/{right}"]
    pid = f"NIST_{alloy}_s{s:03d}_{left}-{right}#0"
    fov_l = FOV_UM[(alloy, left)]
    fov_r = (src_full.shape[1] * OM_DS * 0.345) if right == "om" else FOV_UM[(alloy, right)]
    pxnm = {"bse1": {"718": 1750000 / 2048, "625": 1400000 / 2048}, "bse2": {"718": 607000 / 1757, "625": 750000 / 2194}}

    def px(alloy_, mod):
        return 345.0 * OM_DS if mod == "om" else pxnm[mod][alloy_]

    rec = ac.finalize_pair(
        pair_id=pid, component="NIST", cluster=f"NIST:{alloy}:blk{s // 100}", group="SerialSection",
        subclass=f"IN{alloy}-{left.upper()}-vs-{right.upper()}", img_a=tgt_u16, img_b=src_u16,
        xy_a=tgt_xy, xy_b=src_xy, name_a=f"{left}_raw s{s}", name_b=f"{right}_raw s{s}",
        px_a_nm=px(alloy, left), px_b_nm=px(alloy, right), mod_a=MOD_NAME[left], mod_b=MOD_NAME[right],
        licence="NIST public data (https://www.nist.gov/open/license); attribution requested",
        source_url=f"https://doi.org/10.18434/mds2-{2767 if alloy == '718' else 2765}",
        files={"archive_members": f"{os.path.basename(path_of(alloy, s, left + '_raw'))}; "
                                  f"{os.path.basename(path_of(alloy, s, right + '_raw'))}"},
        notes=("GT = same-modality raw->registered homographies (SIFT+DIS-flow dense-grid fit) chained through the "
               "authors' registered frame; inherits the authors' MI registration errors; "
               f"GT pts = fixed 5x5 grid in target mask bbox (15% margin); optical x1/{OM_DS}"),
        fov_a=fov_l, fov_b=fov_r)
    rec.update(dict(
        alloy=alloy, section=s, modality_pair=f"{left}-{right}", block=s // 100,
        target_is=left, source_is=right, om_downsample=OM_DS if "om" in (left, right) else 1,
        grid_inside_target_mask=int(inside_mask.sum()), grid_inside_source_image=int(inside_src.sum()),
        target_mask_bbox=[float(x0), float(y0), float(x1), float(y1)],
        stretch_target=tr, stretch_source=sr,
        gt_fit_target_modality={k: fl[k] for k in ("resid_dense_homography_px_reg", "resid_quadratic_px_reg",
                                                   "reg_um_per_px", "ncc_after_dense_h", "sift_inliers")},
        gt_fit_source_modality={k: fr[k] for k in ("resid_dense_homography_px_reg", "resid_quadratic_px_reg",
                                                   "reg_um_per_px", "ncc_after_dense_h", "sift_inliers")},
        gt_homography_target_to_source=Gd.tolist()))
    ac.save_record(rec)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alloy", choices=["718", "625"], nargs="+", default=["718", "625"])
    ap.add_argument("--sections", type=int, nargs="+", default=None)
    a = ap.parse_args()
    fits = load_fits()
    n = 0
    for alloy in a.alloy:
        secs, pairs = DESIGN[alloy]
        for s in a.sections or secs:
            for l, r in pairs:
                rec = build_pair(fits, alloy, s, l, r)
                n += 1
                print(rec["pair_id"], rec["h_s"], rec["w_s"], rec["h_t"], rec["w_t"],
                      "in_mask", rec["grid_inside_target_mask"], "in_src", rec["grid_inside_source_image"])
    print(n, "pairs built")


if __name__ == "__main__":
    main()
