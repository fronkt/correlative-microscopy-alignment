"""GT construction for NIST AM Bench serial sections: same-modality raw -> registered transform.

This is NOT the cross-modal registration being benchmarked.  It fits, per (alloy, section,
modality), the transform that carries the authors' RAW image onto their "Warped and Registered"
copy of the SAME modality (SIFT+RANSAC seed -> dense DIS flow -> homography fit to a dense grid of
correspondences).  The residual of the dense-grid homography fit measures how non-homographic the
authors' warp is.  The result still inherits the authors' automatic MI-registration errors.

Usage: python scripts/nist_fit_homography.py --alloy 718 --sections 100 250 400 [--modalities bse1 bse2 om ebsd]
Writes results/arm2/nist_gt_fits.json (merged) and diagnostic PNGs under <data>/fits/.
No cross-modal image pair is ever registered here.
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
import pandas as pd
import tifffile

ROOT = r"C:\Users\frank\Documents\materials-bench\nist"
OUT_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "arm2", "nist_gt_fits.json")
UM_PER_PX_REG = {"bse1": 0.854, "bse2": 0.854, "om": 0.345, "ebsd": 1.0}  # registered-frame pixel size (IN718 README)
FIT_MAX = 1800  # longest side used for the fit


def find(alloy, s, kind):
    g = glob.glob(os.path.join(ROOT, alloy, f"s{s:03d}", f"{kind}__*"))
    if len(g) != 1:
        raise FileNotFoundError(f"{alloy} s{s} {kind}: {g}")
    return g[0]


def read_ctf_bc(path):
    """Band-contrast image of a .ctf (rows in scan order, x fastest); masked -> 0."""
    with open(path, errors="replace") as f:
        lines = [next(f) for _ in range(40)]
    hdr = next(i for i, l in enumerate(lines) if l.startswith("Phase\t"))
    nx = int([l for l in lines if l.startswith("XCells")][0].split()[1])
    ny = int([l for l in lines if l.startswith("YCells")][0].split()[1])
    df = pd.read_csv(path, sep="\t", skiprows=hdr, usecols=["BC"])
    return df["BC"].to_numpy(np.float32).reshape(ny, nx)


def load_img(path):
    if path.endswith(".ctf"):
        return read_ctf_bc(path)
    return np.asarray(tifffile.imread(path), np.float32)


def to_u8(a, mask=None):
    a = np.nan_to_num(a, nan=0.0)
    v = a[a > 0] if (a > 0).any() else a.ravel()
    lo, hi = np.percentile(v, [1, 99.5])
    return (np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def scale_mat(f):
    return np.array([[f, 0, 0.5 * f - 0.5], [0, f, 0.5 * f - 0.5], [0, 0, 1.0]])


def downsample(a, maxside):
    f = min(1.0, maxside / max(a.shape))
    if f < 1:
        a = cv2.resize(a, (round(a.shape[1] * f), round(a.shape[0] * f)), interpolation=cv2.INTER_AREA)
    return a, f


def sift_h(a8, b8):
    clahe = cv2.createCLAHE(3.0, (8, 8))
    sift = cv2.SIFT_create(nfeatures=12000)
    ka, da = sift.detectAndCompute(clahe.apply(a8), None)
    kb, db = sift.detectAndCompute(clahe.apply(b8), None)
    if da is None or db is None or len(ka) < 8 or len(kb) < 8:
        return None, 0, 0
    m = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.8 * x[1].distance]
    if len(good) < 8:
        return None, len(good), 0
    pa = np.float32([ka[g.queryIdx].pt for g in good])
    pb = np.float32([kb[g.trainIdx].pt for g in good])
    H, inl = cv2.findHomography(pa, pb, cv2.RANSAC, 4.0, maxIters=5000, confidence=0.999)
    return H, len(good), int(inl.sum()) if inl is not None else 0


def ncc(a, b, m):
    a, b = a[m].astype(np.float64), b[m].astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def fit_modality(alloy, s, mod, save_dir):
    raw_p, reg_p = find(alloy, s, mod + "_raw"), find(alloy, s, mod + "_reg")
    raw, reg = load_img(raw_p), load_img(reg_p)
    raw_ds, fr = downsample(raw, FIT_MAX)
    reg_ds, fg = downsample(reg, FIT_MAX)
    a8, b8 = to_u8(raw_ds), to_u8(reg_ds)
    # rough intensity polarity is identical (same modality), so no inversion needed
    H0, ngood, ninl = sift_h(a8, b8)
    res = {"raw_shape": list(raw.shape), "reg_shape": list(reg.shape), "sift_good": ngood, "sift_inliers": ninl}
    if H0 is None or ninl < 15:
        res["status"] = "FAIL_SIFT"
        return res
    h, w = b8.shape
    warped = cv2.warpPerspective(a8, H0, (w, h), flags=cv2.INTER_LINEAR)
    valid = (cv2.warpPerspective(np.full(a8.shape, 255, np.uint8), H0, (w, h)) > 0) & (b8 > 0)
    valid = cv2.erode(valid.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
    res["ncc_after_sift_h"] = ncc(warped, b8, valid)
    # dense flow: warped raw -> reg (small residual warp), smoothed a little
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    flow = dis.calc(cv2.GaussianBlur(warped, (0, 0), 1.5), cv2.GaussianBlur(b8, (0, 0), 1.5), None)
    # grid of raw-image points whose image lies inside the valid reg region
    ys, xs = np.mgrid[0:a8.shape[0]:12, 0:a8.shape[1]:12]
    p = np.c_[xs.ravel(), ys.ravel()].astype(np.float64)
    y = (np.c_[p, np.ones(len(p))] @ H0.T)
    y = y[:, :2] / y[:, 2:3]
    inb = (y[:, 0] >= 0) & (y[:, 0] < w - 1) & (y[:, 1] >= 0) & (y[:, 1] < h - 1)
    p, y = p[inb], y[inb]
    ok = valid[np.clip(y[:, 1].round().astype(int), 0, h - 1), np.clip(y[:, 0].round().astype(int), 0, w - 1)]
    # texture mask: local std of the registered image (flow is meaningless on flat areas)
    loc = cv2.blur(b8.astype(np.float32) ** 2, (31, 31)) - cv2.blur(b8.astype(np.float32), (31, 31)) ** 2
    tex = np.sqrt(np.maximum(loc, 0))
    ok &= tex[np.clip(y[:, 1].round().astype(int), 0, h - 1), np.clip(y[:, 0].round().astype(int), 0, w - 1)] > np.percentile(tex[valid], 40)
    p, y = p[ok], y[ok]
    fx = cv2.remap(flow[..., 0], y[:, 0].astype(np.float32).reshape(-1, 1), y[:, 1].astype(np.float32).reshape(-1, 1), cv2.INTER_LINEAR).ravel()
    fy = cv2.remap(flow[..., 1], y[:, 0].astype(np.float32).reshape(-1, 1), y[:, 1].astype(np.float32).reshape(-1, 1), cv2.INTER_LINEAR).ravel()
    T = y + np.c_[fx, fy]  # dense correspondences raw(ds px) -> reg(ds px)
    # forward-backward-free reliability: drop outlier flows (> 40 px beyond the median flow)
    mag = np.hypot(fx, fy)
    keep = mag < np.median(mag) + 40
    p, T, mag = p[keep], T[keep], mag[keep]
    Hd, _ = cv2.findHomography(p, T, 0)
    # robust refit (trimmed) to avoid a few bad flow vectors dominating
    r = np.linalg.norm(cv2.perspectiveTransform(p.reshape(-1, 1, 2), Hd).reshape(-1, 2) - T, axis=1)
    sel = r < np.percentile(r, 90)
    Hd, _ = cv2.findHomography(p[sel], T[sel], 0)
    r = np.linalg.norm(cv2.perspectiveTransform(p.reshape(-1, 1, 2), Hd).reshape(-1, 2) - T, axis=1)
    r0 = np.linalg.norm(cv2.perspectiveTransform(p.reshape(-1, 1, 2), H0).reshape(-1, 2) - T, axis=1)
    # quadratic-polynomial (nonlinear) fit for comparison: how much is left after a smoother model?
    def poly_design(q):
        x_, y_ = q[:, 0], q[:, 1]
        return np.c_[np.ones(len(q)), x_, y_, x_ * x_, x_ * y_, y_ * y_]
    A = poly_design(p)
    coef = np.linalg.lstsq(A[sel], T[sel], rcond=None)[0]
    rp = np.linalg.norm(A @ coef - T, axis=1)
    # convert to FULL-RES pixel coordinates: raw_full -> reg_full
    Sr, Sg = scale_mat(fr), scale_mat(fg)
    Hfull = np.linalg.inv(Sg) @ Hd @ Sr
    H0full = np.linalg.inv(Sg) @ H0 @ Sr
    Hfull /= Hfull[2, 2]
    H0full /= H0full[2, 2]
    um = UM_PER_PX_REG[mod]
    pxg = 1.0 / fg  # ds px -> reg full px
    warped_d = cv2.warpPerspective(a8, Hd, (w, h))
    res.update({
        "status": "OK",
        "n_grid": int(len(p)),
        "H_full_dense": Hfull.tolist(),
        "H_full_sift": H0full.tolist(),
        "ncc_after_dense_h": ncc(warped_d, b8, valid),
        "dense_flow_median_px_reg": float(np.median(mag) * pxg),
        "resid_dense_homography_px_reg": {
            "median": float(np.median(r) * pxg), "p90": float(np.percentile(r, 90) * pxg),
            "p99": float(np.percentile(r, 99) * pxg), "max": float(r.max() * pxg)},
        "resid_sift_homography_px_reg": {"median": float(np.median(r0) * pxg), "p90": float(np.percentile(r0, 90) * pxg)},
        "resid_quadratic_px_reg": {"median": float(np.median(rp) * pxg), "p90": float(np.percentile(rp, 90) * pxg)},
        "reg_um_per_px": um,
        "resid_dense_homography_um": {"median": float(np.median(r) * pxg * um), "p90": float(np.percentile(r, 90) * pxg * um)},
        "reg_px_per_full_ds": pxg,
    })
    os.makedirs(save_dir, exist_ok=True)
    ov = np.dstack([b8, warped_d, warped_d])  # red = reg, cyan = warped raw (BGR order)
    cv2.imwrite(os.path.join(save_dir, f"{alloy}_s{s:03d}_{mod}_overlay.png"), ov)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alloy", required=True)
    ap.add_argument("--sections", type=int, nargs="+", required=True)
    ap.add_argument("--modalities", nargs="+", default=["bse1", "bse2", "om", "ebsd"])
    ap.add_argument("--out", default=None, help="output JSON (default results/arm2/nist_gt_fits.json)")
    a = ap.parse_args()
    out = os.path.abspath(a.out or OUT_JSON)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    allres = json.load(open(out)) if os.path.exists(out) else {}
    for s in a.sections:
        for m in a.modalities:
            key = f"{a.alloy}/s{s:03d}/{m}"
            try:
                allres[key] = fit_modality(a.alloy, s, m, os.path.join(ROOT, "fits"))
            except FileNotFoundError as e:
                allres[key] = {"status": "MISSING", "err": str(e)}
            r = allres[key]
            print(key, r.get("status"), {k: r[k] for k in ("sift_inliers", "ncc_after_sift_h", "ncc_after_dense_h",
                                                        "resid_dense_homography_px_reg", "resid_quadratic_px_reg") if k in r})
            json.dump(allres, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
