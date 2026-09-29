"""Phase A / A6: burned-in overlay audit over all 187 AmalgaMatch eval pairs (exploratory).

Detectors (thresholds are module constants, logged in results/phaseA/a6_log.md):
  BAND  : strip of rows/cols at an image edge (bottom, top, left, right) whose lines are dominated by one grey level
          (>= FLAT_FRAC of pixels within +-1 quantisation bin), <= MAX_DEPTH of the image, containing >= MIN_GLYPHS
          small high-contrast components (text glyphs / scale-bar ticks).  This is the "data bar" / metadata banner.
  LABEL : near-saturated (>=250) rectangular box in an image corner holding >= 2 dark holes (text on a white box,
          i.e. the "100 um" scale-bar label of the composites).
Pair level: SHARED_OVERLAY = both images have a BAND on the same edge (or a LABEL in the same corner).
            SHARED_FIXED   = SHARED_OVERLAY and both images have identical pixel size (a fixed-position lock is
                             then geometrically possible) and the band rows agree in height (+-1 % of H).
Writes a6_overlay_audit.json, a6_per_image.csv, a6_per_pair.csv, a6_thumbs/.  Log is written by hand (a6_log.md).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cma.data.amalgamatch import AmalgaMatchLoader, _long_path  # noqa: E402
from cma.triage import grid_points, transform_distance, youden_cutoff  # noqa: E402

OUT = ROOT / "results" / "phaseA"
THUMBS = OUT / "a6_thumbs"
DATA = Path(r"C:\Users\frank\Documents\correlative-microscopy-alignment\data\AmalgaMatch")
DESIGN = ("SameSlice", "SerialSectioning")
TH = 20.0
NEAR_ID_PX = 20.0
GT_DISP_MIN = 40.0

WORK_MAX = 3072      # detection runs on the image with long side downsampled to this
MAX_DEPTH = 0.12     # band deeper than this fraction of the dimension is rejected
MIN_DEPTH_PX = 6     # ... or thinner than this (working px)
FLAT_FRAC = 0.50     # fraction of a line within the modal +-1 bin (32 bins) to call it band-like (text rows)
ANCHOR_FRAC = 0.90   # a pure-background line; the band must END on one
MIN_GLYPH_H = 5      # working px
UNIFORM_H = 0.40     # glyph height within +-40 % of the median counts as 'uniform font'
MIN_UNIFORM = 0.70
MIN_LINE_FRAC = 0.75 # fraction of glyph centres inside the best two text lines (+-0.6 median height)
GAP = 3              # tolerated non-flat lines inside a band
MIN_GLYPHS = 20       # small high-contrast components needed inside the band
GLYPH_CONTRAST = 50  # abs grey difference from band background
LABEL_WHITE = 250
LABEL_CORNER = 0.30  # corner window (fraction of each dimension) searched for label boxes
LABEL_MIN_HOLES = 2
FREE_MIN_GLYPHS = 3   # free-standing saturated text (no box): >= this many uniform aligned glyphs in a corner
LABEL_MARGIN = 0.08  # box must lie within this fraction of both nearest edges (a corner label, not a bright blob)


def read_gray_u8(path):
    buf = np.fromfile(_long_path(Path(path)), dtype=np.uint8)
    im = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if im.ndim == 3:
        im = cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2GRAY)
    if im.dtype != np.uint8:
        a = im.astype(np.float32)
        lo, hi = np.percentile(a, [0.5, 99.5])
        im = np.clip((a - lo) / (hi - lo + 1e-9) * 255, 0, 255).astype(np.uint8)
    return im


def work_image(im):
    f = min(1.0, WORK_MAX / max(im.shape))
    if f < 1.0:
        im = cv2.resize(im, (int(round(im.shape[1] * f)), int(round(im.shape[0] * f))), interpolation=cv2.INTER_AREA)
    return im, f


def _edge_view(g, edge):
    """Array whose row 0 is the image edge line and rows increase inwards."""
    return {"bottom": g[::-1], "top": g, "left": g.T, "right": g.T[::-1]}[edge]


def band_on_edge(g, edge):
    a = _edge_view(g, edge)
    H = a.shape[0]
    depth = int(MAX_DEPTH * H)
    q = (a[:depth] // 8).astype(np.int32)  # 32 bins
    flat = np.zeros(depth, bool)
    anchor = np.zeros(depth, bool)
    for r in range(depth):
        hst = np.bincount(q[r], minlength=32)
        # modal bin plus its better neighbour
        m = int(hst.argmax())
        s = hst[m] + max(hst[m - 1] if m > 0 else 0, hst[m + 1] if m < 31 else 0)
        flat[r] = s >= FLAT_FRAC * q.shape[1]
        anchor[r] = s >= ANCHOR_FRAC * q.shape[1]
    # contiguous run from the edge allowing GAP-line gaps (a 1-3 line border at the very edge is skipped)
    end, gap, r = -1, 0, 0
    while r < depth:
        if flat[r]:
            end, gap = r, 0
        else:
            gap += 1
            if gap > GAP:
                break
        r += 1
    if end >= 0:
        idx = np.flatnonzero(anchor[:end + 1])
        end = int(idx[-1]) if len(idx) else -1
    h_band = end + 1
    res = dict(edge=edge, h_band_px=int(h_band), yes=False, n_glyphs=0, reason="")
    if h_band < MIN_DEPTH_PX:
        res["reason"] = "too_thin"
        return res
    reg = a[:h_band].astype(np.int16)
    bg = np.median(reg)
    fg = (np.abs(reg - bg) > GLYPH_CONTRAST).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    area_max = 0.02 * reg.size
    gl = [(st[k, cv2.CC_STAT_HEIGHT], st[k, cv2.CC_STAT_TOP] + st[k, cv2.CC_STAT_HEIGHT] / 2.0) for k in range(1, n)
          if st[k, cv2.CC_STAT_AREA] >= 6 and st[k, cv2.CC_STAT_AREA] <= area_max and MIN_GLYPH_H <= st[k, cv2.CC_STAT_HEIGHT] <= h_band]
    res["n_glyphs"] = len(gl)
    res["fg_frac"] = float(fg.mean())
    if len(gl) < MIN_GLYPHS:
        res["reason"] = "few_glyphs"
        return res
    hh = np.array([g_[0] for g_ in gl], float)
    cy = np.array([g_[1] for g_ in gl], float)
    med = float(np.median(hh))
    uni = float((np.abs(hh - med) <= UNIFORM_H * med).mean())
    # best two text lines: greedy windows of +-0.6*med around glyph centres
    rem = np.ones(len(cy), bool)
    covered = 0
    for _ in range(2):
        best, bm = 0, None
        for c in cy[rem]:
            m_ = rem & (np.abs(cy - c) <= 0.6 * med)
            if m_.sum() > best:
                best, bm = int(m_.sum()), m_
        if bm is None:
            break
        covered += best
        rem &= ~bm
    lines = covered / len(cy)
    res["uniform_h"], res["line_frac"] = uni, lines
    if uni >= MIN_UNIFORM and lines >= MIN_LINE_FRAC:
        res["yes"] = True
    else:
        res["reason"] = "not_text_like"
    return res


def label_in_corners(g):
    """Near-white rectangular boxes with dark holes (text) in a corner window."""
    h, w = g.shape
    out = []
    wh, ww = int(LABEL_CORNER * h), int(LABEL_CORNER * w)
    wins = {"bottom_left": (h - wh, h, 0, ww), "bottom_right": (h - wh, h, w - ww, w),
            "top_left": (0, wh, 0, ww), "top_right": (0, wh, w - ww, w)}
    for name, (y0, y1, x0, x1) in wins.items():
        sub = (g[y0:y1, x0:x1] >= LABEL_WHITE).astype(np.uint8)
        n, lab, st, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
        for k in range(1, n):
            x, y, bw, bh, ar = st[k]
            if bw < 0.02 * w or bw > 0.20 * w or bh < 0.01 * h or bh > 0.15 * h:
                continue
            if not (1.5 <= bw / bh <= 8) or ar / (bw * bh) < 0.4:
                continue
            gx, gy = x0 + x, y0 + y
            if min(gx, w - gx - bw) > LABEL_MARGIN * w or min(gy, h - gy - bh) > LABEL_MARGIN * h:
                continue
            box = sub[y:y + bh, x:x + bw]
            inv = (box == 0).astype(np.uint8)
            m, _, sti, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
            # holes = dark components not touching the box border
            holes = sum(1 for j in range(1, m) if sti[j, 0] > 0 and sti[j, 1] > 0 and sti[j, 0] + sti[j, 2] < bw
                        and sti[j, 1] + sti[j, 3] < bh and sti[j, 4] >= 3)
            if holes >= LABEL_MIN_HOLES:
                out.append(dict(corner=name, bbox=[int(x0 + x), int(y0 + y), int(bw), int(bh)], holes=int(holes)))
        # free-standing white text (scale-bar caption drawn straight onto the image, no box)
        if any(o["corner"] == name for o in out):
            continue
        gl = []
        for k in range(1, n):
            x, y, bw, bh, ar = st[k]
            gx, gy = x0 + x, y0 + y
            if not (0.008 * h <= bh <= 0.06 * h) or bw > 0.06 * w or ar < 6:
                continue
            if min(gx, w - gx - bw) > LABEL_MARGIN * w or min(gy, h - gy - bh) > LABEL_MARGIN * h:
                continue
            gl.append((gx, gy, bw, bh))
        if len(gl) >= FREE_MIN_GLYPHS:
            hs = np.array([g_[3] for g_ in gl], float)
            cyv = np.array([g_[1] + g_[3] / 2 for g_ in gl], float)
            cxv = np.array([g_[0] for g_ in gl], float)
            med = np.median(hs)
            keep = (np.abs(hs - med) <= UNIFORM_H * med) & (np.abs(cyv - np.median(cyv)) <= 0.6 * med)
            if keep.sum() >= FREE_MIN_GLYPHS and np.ptp(cxv[keep]) <= 0.15 * w:
                gk = [g_ for g_, kk in zip(gl, keep) if kk]
                x_ = min(g_[0] for g_ in gk); y_ = min(g_[1] for g_ in gk)
                out.append(dict(corner=name, bbox=[int(x_), int(y_), int(max(g_[0] + g_[2] for g_ in gk) - x_),
                                                   int(max(g_[1] + g_[3] for g_ in gk) - y_)], holes=-int(keep.sum())))
    return out


def detect(path):
    im = read_gray_u8(path)
    g, f = work_image(im)
    bands = {e: band_on_edge(g, e) for e in ("bottom", "top", "left", "right")}
    labels = label_in_corners(g)
    H, W = g.shape
    yes_edges = [e for e, b in bands.items() if b["yes"]]
    dim = {"bottom": H, "top": H, "left": W, "right": W}
    rec = dict(h=int(im.shape[0]), w=int(im.shape[1]), overlay=bool(yes_edges or labels),
               band_edges=yes_edges, label_corners=[l["corner"] for l in labels],
               band_frac={e: round(bands[e]["h_band_px"] / dim[e], 4) for e in yes_edges},
               n_glyphs={e: bands[e]["n_glyphs"] for e in yes_edges},
               labels=labels, rejected={e: (b["h_band_px"], b["n_glyphs"]) for e, b in bands.items()
                                        if not b["yes"] and b["h_band_px"] >= MIN_DEPTH_PX})
    return rec, im


def band_rows_native(im, edge, frac):
    h, w = im.shape
    k = int(round(frac * (h if edge in ("bottom", "top") else w)))
    return {"bottom": im[h - k:], "top": im[:k], "left": im[:, :k], "right": im[:, w - k:]}[edge]


def main():
    t0 = time.time()
    THUMBS.mkdir(parents=True, exist_ok=True)
    L = AmalgaMatchLoader(DATA)
    recs = L.records
    # ---------------- 1. per-image detection
    paths = {}
    for r in recs:
        for p in (r.source_path, r.target_path):
            paths.setdefault(str(p), p)
    img_rec, cache = {}, {}
    for i, (k, p) in enumerate(paths.items()):
        rec, im = detect(p)
        rec["idx"] = i
        rec["file"] = Path(p).name
        rec["subclass"] = Path(p).parents[2].name
        img_rec[k] = rec
        if rec["overlay"]:
            # keep the native-resolution band strips for the identity check (small)
            for e in rec["band_edges"]:
                cache[(k, e)] = band_rows_native(im, e, rec["band_frac"][e])
    print(f"detected {len(img_rec)} images in {time.time() - t0:.0f}s; overlay yes = {sum(v['overlay'] for v in img_rec.values())}")

    rows = []
    for k, v in img_rec.items():
        rows.append(dict(idx=v["idx"], subclass=v["subclass"], file=v["file"], h=v["h"], w=v["w"], overlay=v["overlay"],
                         band_edges="|".join(v["band_edges"]), band_frac=json.dumps(v["band_frac"]),
                         label_corners="|".join(v["label_corners"]), n_glyphs=json.dumps(v["n_glyphs"]),
                         rejected_bands=json.dumps(v["rejected"]), path=k))
    PI = pd.DataFrame(rows)
    PI.to_csv(OUT / "a6_per_image.csv", index=False)

    # ---------------- per-pair
    prow = []
    for r in recs:
        s, t = img_rec[str(r.source_path)], img_rec[str(r.target_path)]
        common_edges = sorted(set(s["band_edges"]) & set(t["band_edges"]))
        common_labels = sorted(set(s["label_corners"]) & set(t["label_corners"]))
        shared = bool(common_edges or common_labels)
        same_size = (s["h"], s["w"]) == (t["h"], t["w"])
        same_pix = abs(r.source_pixel_nm - r.target_pixel_nm) < 1e-9
        frac_ok, ident = False, None
        if common_edges:
            e = common_edges[0]
            frac_ok = abs(s["band_frac"][e] - t["band_frac"][e]) <= 0.01
            if same_size:
                a, b = cache[(str(r.source_path), e)], cache[(str(r.target_path), e)]
                n = min(a.shape[0], b.shape[0])
                ident = float((np.abs(a[:n].astype(int) - b[:n].astype(int)) <= 8).mean())
        fixed = bool(shared and same_size and (frac_ok or bool(common_labels)))
        prow.append(dict(pair_id=r.pair_id, group=r.group, subclass=r.subclass, src=Path(r.source_path).name,
                         tgt=Path(r.target_path).name, src_overlay=s["overlay"], tgt_overlay=t["overlay"],
                         SHARED_OVERLAY=shared, shared_edges="|".join(common_edges), shared_labels="|".join(common_labels),
                         same_image_size=same_size, same_pixel_size=same_pix, band_height_agree=frac_ok,
                         band_pixel_identity_frac=ident, SHARED_FIXED=fixed,
                         src_hw=f"{s['h']}x{s['w']}", tgt_hw=f"{t['h']}x{t['w']}"))
    PP = pd.DataFrame(prow).set_index("pair_id")

    # ---------------- candidates: near-identity analysis
    d = pd.read_csv(ROOT / "results" / "triage" / "candidates.csv")
    d["cand"] = d.backbone + "|" + d["mode"] + "|" + d["transform"].astype(str) + "|s" + d.seed.astype(int).astype(str)
    ok = d.status.eq("ok")
    d["err"] = np.where(ok, pd.to_numeric(d.mu_ed, errors="coerce"), np.inf)
    d["err"] = d.err.fillna(np.inf)
    nm, ni = pd.to_numeric(d.n_matches, errors="coerce"), pd.to_numeric(d.n_inliers, errors="coerce")
    d["S1"] = np.where(ok & (nm > 0), ni / nm, -np.inf)
    gt = d[(d.pool == "gt") & (d["mode"] == "homography")].set_index("pair_id")
    I3 = np.eye(3)

    def Hof(s):
        return np.array(json.loads(s)).reshape(3, 3)

    gt_disp, pts_cache = {}, {}
    for pid, r in gt.iterrows():
        pts_cache[pid] = grid_points(int(r.h_t), int(r.w_t), 5)
        gt_disp[pid] = transform_distance(Hof(r.H), I3, pts_cache[pid])
    PP["gt_disp_px"] = pd.Series(gt_disp)

    V = d[d.pool.isin(["core", "transform"])].copy()
    dist_id = []
    for _, r in V.iterrows():
        if r.status == "ok" and isinstance(r.H, str):
            dist_id.append(transform_distance(Hof(r.H), I3, pts_cache[r.pair_id]))
        else:
            dist_id.append(np.nan)
    V["dist_id"] = dist_id
    V["gt_disp"] = V.pair_id.map(gt_disp)
    V["shared"] = V.pair_id.map(PP.SHARED_OVERLAY)
    V["fixed"] = V.pair_id.map(PP.SHARED_FIXED)
    V["same_size"] = V.pair_id.map(PP.same_image_size)
    V["near_id"] = V.dist_id < NEAR_ID_PX
    V["fail"] = V.err > TH  # includes failed runs (inf)
    V["NIF"] = V.near_id & V.fail & (V.gt_disp > GT_DISP_MIN)
    V["nif_shared"] = V.NIF & V.shared
    V["nif_noshared"] = V.NIF & ~V.shared

    def tab(df, by):
        g = df.groupby(by)
        return pd.DataFrame(dict(n=g.size(), n_fail=g.fail.sum(), near_id=g.near_id.sum(), NIF=g.NIF.sum(),
                                 NIF_shared=g.nif_shared.sum(), NIF_noshared=g.nif_noshared.sum())).astype(int)

    md = V[(V.backbone == "ma_roma") & (V["mode"] == "direct") & (V.pool == "core") & (V.seed == 0) & (V["transform"] == "none")]
    assert len(md) == 187
    out = {"n_images": len(img_rec), "n_images_overlay": int(PI.overlay.sum()),
           "params": dict(WORK_MAX=WORK_MAX, MAX_DEPTH=MAX_DEPTH, MIN_DEPTH_PX=MIN_DEPTH_PX, FLAT_FRAC=FLAT_FRAC, GAP=GAP,
                          MIN_GLYPHS=MIN_GLYPHS, GLYPH_CONTRAST=GLYPH_CONTRAST, LABEL_WHITE=LABEL_WHITE,
                          LABEL_CORNER=LABEL_CORNER, LABEL_MIN_HOLES=LABEL_MIN_HOLES, NEAR_ID_PX=NEAR_ID_PX, GT_DISP_MIN=GT_DISP_MIN)}
    out["images_by_subclass"] = PI.groupby("subclass").overlay.agg(["sum", "count"]).astype(int).to_dict("index")
    gp = PP.groupby("group")
    out["pairs_by_group"] = {k: dict(n_pairs=len(v), SHARED_OVERLAY=int(v.SHARED_OVERLAY.sum()), SHARED_FIXED=int(v.SHARED_FIXED.sum()),
                                     same_image_size=int(v.same_image_size.sum()),
                                     any_overlay_one_side=int((v.src_overlay | v.tgt_overlay).sum())) for k, v in gp}
    out["pairs_by_subclass"] = {k: dict(n_pairs=len(v), SHARED_OVERLAY=int(v.SHARED_OVERLAY.sum()), SHARED_FIXED=int(v.SHARED_FIXED.sum()),
                                        same_image_size=int(v.same_image_size.sum())) for k, v in PP.groupby("subclass")}
    out["totals"] = dict(n_pairs=len(PP), SHARED_OVERLAY=int(PP.SHARED_OVERLAY.sum()), SHARED_FIXED=int(PP.SHARED_FIXED.sum()),
                         same_image_size=int(PP.same_image_size.sum()))
    out["nif_all_candidates"] = {"overall": dict(n=len(V), n_fail=int(V.fail.sum()), near_id=int(V.near_id.sum()), NIF=int(V.NIF.sum()),
                                                 NIF_shared=int(V.nif_shared.sum()), NIF_noshared=int(V.nif_noshared.sum()),
                                                 NIF_shared_fixed=int((V.NIF & V.fixed).sum()),
                                                 NIF_same_size=int((V.NIF & V.same_size).sum()),
                                                 NIF_noshared_samesize=int((V.nif_noshared & V.same_size).sum()),
                                                 NIF_noshared_diffsize=int((V.nif_noshared & ~V.same_size).sum())),
                                 "by_shared": tab(V, "shared").to_dict("index"),
                                 "by_group": tab(V, "group").to_dict("index"),
                                 "by_backbone": tab(V, "backbone").to_dict("index"),
                                 "by_group_x_shared": {f"{a}|{b}": v for (a, b), v in tab(V, ["group", "shared"]).to_dict("index").items()},
                                 "by_backbone_x_shared": {f"{a}|{b}": v for (a, b), v in tab(V, ["backbone", "shared"]).to_dict("index").items()}}
    out["nif_ma_roma_direct_seed0"] = {"overall": dict(n=len(md), n_fail=int(md.fail.sum()), near_id=int(md.near_id.sum()), NIF=int(md.NIF.sum()),
                                                       NIF_shared=int(md.nif_shared.sum()), NIF_noshared=int(md.nif_noshared.sum())),
                                       "by_group": tab(md, "group").to_dict("index"),
                                       "by_shared": tab(md, "shared").to_dict("index")}
    # near-identity locks among ALL candidate rows in no-shared-overlay pairs: which subclasses/backbones
    nn = V[V.nif_noshared]
    out["nif_noshared_detail"] = {"by_subclass": nn.subclass.value_counts().to_dict(), "by_backbone": nn.backbone.value_counts().to_dict(),
                                  "pairs": int(nn.pair_id.nunique()), "same_size_rows": int(nn.same_size.sum())}

    # ---------------- 4. S1 accepts over all 187 pairs
    prim = md.set_index("pair_id")
    des = prim.group.isin(DESIGN)
    cut = youden_cutoff(prim.S1[des].values, (prim.err[des] <= TH).values)
    acc = prim[prim.S1 >= cut]
    fa = acc[acc.err > TH]
    out["s1"] = dict(cutoff=float(cut), n_accepted=int(len(acc)), n_false_accept=int(len(fa)),
                     n_false_accept_heldout=int((~fa.group.isin(DESIGN)).sum()),
                     fa_near_identity=int(fa.near_id.sum()), fa_NIF=int(fa.NIF.sum()),
                     fa_NIF_shared=int(fa.nif_shared.sum()), fa_NIF_shared_fixed=int((fa.NIF & fa.fixed).sum()),
                     fa_NIF_noshared=int(fa.nif_noshared.sum()),
                     fa_shared_but_not_NIF=int((fa.shared & ~fa.NIF).sum()),
                     fa_not_shared_not_NIF=int((~fa.shared & ~fa.NIF).sum()),
                     fa_by_group=fa.group.value_counts().to_dict(),
                     fa_not_NIF_by_group=fa[~fa.NIF].group.value_counts().to_dict(),
                     fa_NIF_shared_by_subclass=fa[fa.nif_shared].subclass.value_counts().to_dict())
    fa_heldout = fa[~fa.group.isin(DESIGN)]
    out["s1"]["heldout"] = dict(n=len(fa_heldout), NIF_shared=int(fa_heldout.nif_shared.sum()))
    PP["ma_roma_err"] = prim.err
    PP["ma_roma_S1"] = prim.S1
    PP["ma_roma_dist_identity"] = prim.dist_id
    PP["ma_roma_NIF"] = prim.NIF
    PP["ma_roma_s1_accept"] = prim.S1 >= cut
    PP["n_cand_rows"] = V.groupby("pair_id").size()
    PP["n_NIF_rows"] = V.groupby("pair_id").NIF.sum()
    PP.to_csv(OUT / "a6_per_pair.csv")
    V[["pair_id", "cand", "group", "backbone", "pool", "err", "dist_id", "gt_disp", "shared", "fixed", "NIF"]].to_csv(
        OUT / "a6_candidate_rows.csv", index=False)

    # ---------------- 5. GPU sizing
    sh = PP[PP.SHARED_OVERLAY]
    out["gpu_sizing"] = dict(n_pairs_shared_overlay=int(len(sh)), n_pairs_shared_fixed=int(PP.SHARED_FIXED.sum()),
                             subsets=sh.subclass.value_counts().to_dict(), n_distinct_images=int(len(set(sh.src) | set(sh.tgt))),
                             n_pairs_shared_nif_maroma=int(sh.ma_roma_NIF.sum()))
    (OUT / "a6_overlay_audit.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")

    # ---------------- thumbnails for the eyeball validation
    sample = [99, 119, 111, 23, 22, 29, 30, 72, 155, 132, 144, 128, 31, 3, 35, 146, 152, 90]  # mix of flagged and not (idx = a6_per_image idx)
    keys = list(paths)
    sample_rows = []
    for i in sample:
        k = keys[i]
        v = img_rec[k]
        im = read_gray_u8(paths[k])
        h, w = im.shape
        f = 900 / max(h, w)
        t = cv2.cvtColor(cv2.resize(im, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
        for e, bf in v["band_frac"].items():
            hh, ww = t.shape[:2]
            kk = int(round(bf * (hh if e in ("bottom", "top") else ww)))
            box = {"bottom": (0, hh - kk, ww - 1, hh - 1), "top": (0, 0, ww - 1, kk), "left": (0, 0, kk, hh - 1),
                   "right": (ww - kk, 0, ww - 1, hh - 1)}[e]
            cv2.rectangle(t, box[:2], box[2:], (0, 0, 255), 2)
        for lb in v["labels"]:
            x, y, bw, bh = lb["bbox"]
            # bbox is in working px; convert via working factor
            fw = min(1.0, WORK_MAX / max(h, w))
            cv2.rectangle(t, (int(x / fw * f), int(y / fw * f)), (int((x + bw) / fw * f), int((y + bh) / fw * f)), (0, 255, 0), 2)
        txt = f"#{i} {v['file'][:28]} overlay={v['overlay']} {v['band_edges']} {v['label_corners']}"
        cv2.putText(t, txt, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        cv2.imwrite(str(THUMBS / f"img{i:03d}.png"), t)
        sample_rows.append(dict(idx=i, file=v["file"], overlay=v["overlay"], edges=v["band_edges"], corners=v["label_corners"]))
    pd.DataFrame(sample_rows).to_csv(THUMBS / "sample.csv", index=False)
    # contact sheet with flagged images of every subclass, for the eyeball audit (no box drawn, 200 px tiles)
    print(json.dumps({k: out[k] for k in ("totals", "pairs_by_group", "s1", "gpu_sizing")}, indent=1, default=float))
    print(f"done in {time.time() - t0:.0f}s")


def pilot():
    """EXPLORATORY CPU pilot: SIFT with the bottom band cropped from both images, on SHARED_FIXED pairs outside the
    42 A1 pairs (bottom bands only).  Same pipeline as A1 (family auto, RANSAC 5.5 px, per-pair seed)."""
    from cma.estimators import fit_transform
    from cma.matchers.sift import SIFTMatcher
    from cma.metrics import registration_metrics
    from cma.triage import pair_seed, project, seed_everything
    PP = pd.read_csv(OUT / "a6_per_pair.csv").set_index("pair_id")
    PI = pd.read_csv(OUT / "a6_per_image.csv")
    a1 = pd.read_csv(OUT / "a1_per_pair.csv").set_index("pair_id")
    L = AmalgaMatchLoader(DATA)
    recs = {r.pair_id: r for r in L.records}
    d = pd.read_csv(ROOT / "results" / "triage" / "candidates.csv")
    sift = d[(d.backbone == "sift") & (d.pool == "core")].set_index("pair_id")
    sm = SIFTMatcher()
    I3 = np.eye(3)
    rows = []
    todo = PP[PP.SHARED_FIXED & PP.shared_edges.eq("bottom") & ~PP.index.isin(a1.index)]
    fr = {row.path: json.loads(row.band_frac).get("bottom") for row in PI.itertuples()}
    for pid in todo.index:
        r = recs[pid]
        pair = L.load_pair(r)
        row = dict(pair_id=pid, subclass=r.subclass, sift_stored_err=float(pd.to_numeric(sift.loc[pid, "mu_ed"], errors="coerce")))
        for tag, crop in (("crop0", 0.0), ("cropband", None)):
            fs, ft = fr[str(r.source_path)], fr[str(r.target_path)]
            hs = pair.source.shape[0] if crop == 0.0 else int(round(pair.source.shape[0] * (1 - fs - 0.01)))
            ht = pair.target.shape[0] if crop == 0.0 else int(round(pair.target.shape[0] * (1 - ft - 0.01)))
            seed_everything(pair_seed(0, pid))
            corr = sm.match(pair.source[:hs], pair.target[:ht])
            err, dist = np.inf, np.nan
            if len(corr) >= 4:
                try:
                    est = fit_transform(src_xy=corr.b_xy, dst_xy=corr.a_xy, family="auto", ransac_threshold_px=5.5)
                    Hm = est.as_3x3()
                    err = float(registration_metrics(project(Hm, pair.gt.tgt_xy), pair.gt.src_xy).mu_err)
                    pts = grid_points(pair.target.shape[0], pair.target.shape[1], 5)
                    dist = transform_distance(Hm, I3, pts)
                except Exception:
                    pass
            row[f"err_{tag}"], row[f"dist_id_{tag}"], row[f"n_matches_{tag}"] = err, dist, len(corr)
        rows.append(row)
        print(pid, {k: round(v, 1) if isinstance(v, float) else v for k, v in row.items() if k.startswith(("err", "dist"))}, flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "a6_sift_pilot.csv", index=False)
    summ = dict(n=len(R), stored_success=int((R.sift_stored_err <= TH).sum()), crop0_success=int((R.err_crop0 <= TH).sum()),
                cropband_success=int((R.err_cropband <= TH).sum()), crop0_near_identity=int((R.dist_id_crop0 < NEAR_ID_PX).sum()),
                cropband_near_identity=int((R.dist_id_cropband < NEAR_ID_PX).sum()),
                crop0_reproduces_stored_within_1px=int((np.abs(R.err_crop0 - R.sift_stored_err) < 1).sum()),
                median_err_crop0=float(R.err_crop0.median()), median_err_cropband=float(R.err_cropband.median()))
    (OUT / "a6_sift_pilot.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    if "--pilot" in sys.argv:
        pilot()
    else:
        main()
