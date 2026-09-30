"""Burned-in overlay (metadata banner / scale-bar caption) detection and cropping. FROZEN for Arm 1.

The detector is a faithful port of the A6 audit (scripts/phaseA_a6_overlay_audit.py, results/phaseA/a6_log.md,
variant v4). Do not retune: Arm 1 is pre-registered on these exact constants. Thresholds:

  Working image: long side downsampled to WORK_MAX = 3072 (INTER_AREA); 16-bit / colour input is reduced to uint8
  grey by ``read_gray_u8`` exactly as A6 did (0.5-99.5 percentile stretch).
  BAND (tested on all four edges; only top/bottom are cropped): a line is band-like when >= FLAT_FRAC = 0.50 of its
  pixels lie in the modal +-1 bin of 32 grey bins, an anchor line when >= ANCHOR_FRAC = 0.90. Band = contiguous
  band-like run from the edge (gap <= GAP = 3 lines), trimmed to end on an anchor line, depth MIN_DEPTH_PX = 6 px
  .. MAX_DEPTH = 12 % of the dimension. Glyphs = 8-connected components of |pixel - band median| > GLYPH_CONTRAST = 50,
  area >= 6 px and <= 2 % of the band, height MIN_GLYPH_H = 5 px .. band height. Accepted iff >= MIN_GLYPHS = 20
  glyphs, >= MIN_UNIFORM = 0.70 of glyph heights within +-UNIFORM_H = 40 % of the median, and >= MIN_LINE_FRAC = 0.75
  of glyph centres inside the best two text lines (+-0.6 median height).
  LABEL (scale-bar caption in a corner, 30 % corner windows, LABEL_WHITE = 250): (a) boxed: saturated rectangular
  component, width 2-20 % / height 1-15 % of the image, aspect 1.5-8, fill >= 0.4, >= LABEL_MIN_HOLES = 2 interior
  dark holes, within LABEL_MARGIN = 8 % of both nearest edges; (b) free text: >= FREE_MIN_GLYPHS = 3 saturated
  glyph components of uniform height, on one line, spread <= 15 % of image width, within 8 % of both edges.

``detect_overlay(gray) -> (side, rows)``: side in {"bottom", "top"}, rows = band height in NATIVE pixels, or None.
  * a text band on the top/bottom edge gives rows = ceil(h_band_px / work_scale);
  * a corner caption (label) gives the side of its corner and rows = ceil(distance from that edge to the caption's
    far side / work_scale) (JUDGEMENT: the crop rule for the C103 caption-only pair is 'crop the band of rows that
    contains the caption');
  * if several candidates exist the one with the most rows wins; left/right bands are detected by ``analyze`` (for
    A6 parity) but never cropped (none occur in AmalgaMatch).

Crop rule (fixed): for a pair where both images carry an overlay on the same side, the UNION height is
  frac = max_i(rows_i / H_i) + SAFETY_MARGIN (2 % of height); each image loses ceil(frac * H_i) rows from that side.
  For equal-sized pairs (all 66 TEM pairs) this is exactly max(rows) + 2 % of H, rounded up, on both images. For the
  one C103 pair the two images differ in size, so the union is taken as a height FRACTION (JUDGEMENT).
  Sham: the fixed fraction SHAM_FRAC is removed from the bottom of both images of an overlay-free pair.
Coordinates: a bottom crop leaves retained pixel coordinates unchanged; a top crop of k rows shifts them by -k.
  ``uncrop_homography`` maps a transform fitted on the cropped pair (target_crop -> source_crop) back to original
  coordinates so GT points are evaluated in original coordinates and none are dropped.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

WORK_MAX = 3072
MAX_DEPTH = 0.12
MIN_DEPTH_PX = 6
FLAT_FRAC = 0.50
ANCHOR_FRAC = 0.90
MIN_GLYPH_H = 5
UNIFORM_H = 0.40
MIN_UNIFORM = 0.70
MIN_LINE_FRAC = 0.75
GAP = 3
MIN_GLYPHS = 20
GLYPH_CONTRAST = 50
LABEL_WHITE = 250
LABEL_CORNER = 0.30
LABEL_MIN_HOLES = 2
FREE_MIN_GLYPHS = 3
LABEL_MARGIN = 0.08

SAFETY_MARGIN = 0.02
# Median over the 67 overlay pairs of (band fraction + margin) is ~0.0857 (STEM banners 0.0654-0.0657 + 0.02);
# frozen at 0.086 so the sham removes the same fractional band the real crop removes.
SHAM_FRAC = 0.086


def read_gray_u8(path) -> np.ndarray:
    """A6 image reader: cv2.imdecode of the raw bytes, colour -> grey, 16-bit -> 0.5-99.5 percentile stretch."""
    from cma.data.amalgamatch import _long_path  # long-path safe open, as in A6

    buf = np.fromfile(_long_path(Path(path)), dtype=np.uint8)
    im = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if im.ndim == 3:
        im = cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2GRAY)
    if im.dtype != np.uint8:
        a = im.astype(np.float32)
        lo, hi = np.percentile(a, [0.5, 99.5])
        im = np.clip((a - lo) / (hi - lo + 1e-9) * 255, 0, 255).astype(np.uint8)
    return im


def work_image(im: np.ndarray):
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
        m = int(hst.argmax())
        s = hst[m] + max(hst[m - 1] if m > 0 else 0, hst[m + 1] if m < 31 else 0)
        flat[r] = s >= FLAT_FRAC * q.shape[1]
        anchor[r] = s >= ANCHOR_FRAC * q.shape[1]
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
          if st[k, cv2.CC_STAT_AREA] >= 6 and st[k, cv2.CC_STAT_AREA] <= area_max
          and MIN_GLYPH_H <= st[k, cv2.CC_STAT_HEIGHT] <= h_band]
    res["n_glyphs"] = len(gl)
    res["fg_frac"] = float(fg.mean())
    if len(gl) < MIN_GLYPHS:
        res["reason"] = "few_glyphs"
        return res
    hh = np.array([g_[0] for g_ in gl], float)
    cy = np.array([g_[1] for g_ in gl], float)
    med = float(np.median(hh))
    uni = float((np.abs(hh - med) <= UNIFORM_H * med).mean())
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
    """Near-white rectangular boxes with dark holes (text), or free saturated text, in a corner window."""
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
            holes = sum(1 for j in range(1, m) if sti[j, 0] > 0 and sti[j, 1] > 0 and sti[j, 0] + sti[j, 2] < bw
                        and sti[j, 1] + sti[j, 3] < bh and sti[j, 4] >= 3)
            if holes >= LABEL_MIN_HOLES:
                out.append(dict(corner=name, bbox=[int(x0 + x), int(y0 + y), int(bw), int(bh)], holes=int(holes)))
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
                x_ = min(g_[0] for g_ in gk)
                y_ = min(g_[1] for g_ in gk)
                out.append(dict(corner=name, bbox=[int(x_), int(y_), int(max(g_[0] + g_[2] for g_ in gk) - x_),
                                                   int(max(g_[1] + g_[3] for g_ in gk) - y_)], holes=-int(keep.sum())))
    return out


def analyze(gray: np.ndarray) -> dict:
    """Full A6 record for a uint8 grey image: band_edges, band_frac, label_corners, labels (working px), work_scale."""
    g, f = work_image(gray)
    bands = {e: band_on_edge(g, e) for e in ("bottom", "top", "left", "right")}
    labels = label_in_corners(g)
    H, W = g.shape
    yes_edges = [e for e, b in bands.items() if b["yes"]]
    dim = {"bottom": H, "top": H, "left": W, "right": W}
    return dict(h=int(gray.shape[0]), w=int(gray.shape[1]), overlay=bool(yes_edges or labels), band_edges=yes_edges,
                label_corners=[lb["corner"] for lb in labels],
                band_frac={e: round(bands[e]["h_band_px"] / dim[e], 4) for e in yes_edges},
                band_px={e: bands[e]["h_band_px"] for e in yes_edges}, labels=labels, work_scale=f, work_hw=(H, W))


def _candidates(a: dict) -> list[tuple[int, str, str]]:
    """(native rows, side, tag) for every top/bottom band and corner caption of an analyze() record.
    tag is 'band:<edge>' or 'label:<corner>' (the A6 pair-level 'same edge / same corner' identity)."""
    f, (Hw, _) = a["work_scale"], a["work_hw"]
    cands = []
    for e in ("bottom", "top"):
        if e in a["band_px"]:
            cands.append((math.ceil(a["band_px"][e] / f), e, f"band:{e}"))
    for lb in a["labels"]:
        _, y, _, bh = lb["bbox"]
        if lb["corner"].startswith("bottom"):
            cands.append((math.ceil((Hw - y) / f), "bottom", f"label:{lb['corner']}"))
        else:
            cands.append((math.ceil((y + bh) / f), "top", f"label:{lb['corner']}"))
    return cands


def detect_overlay(gray: np.ndarray):
    """Return (side, rows) with side in {"bottom","top"} and rows the native-pixel band height, or None."""
    a = analyze(gray)
    cands = _candidates(a)
    if not cands:
        return None
    rows, side, _ = max(cands)
    return side, int(min(rows, gray.shape[0] - 1))


# ------------------------------------------------------------------------------------------------ crop rule


@dataclass(frozen=True)
class CropRule:
    """Rows removed from `side` of the source and of the target (native pixels); kind = 'overlay' | 'sham'."""
    kind: str
    side: str
    rows_src: int
    rows_tgt: int

    @property
    def offset_src(self) -> tuple[int, int]:  # (dx, dy) added to original coords to get cropped coords is -offset
        return (0, self.rows_src if self.side == "top" else 0)

    @property
    def offset_tgt(self) -> tuple[int, int]:
        return (0, self.rows_tgt if self.side == "top" else 0)


def overlay_rule(a_src: dict, a_tgt: dict) -> CropRule | None:
    """Overlay crop rule from the two images' analyze() records; None unless A6 flags a SHARED overlay
    (same top/bottom band edge, or the same label corner; left/right bands are never cropped).

    Per image, rows = the largest overlay candidate on the shared side (so a data bar that is not itself shared
    is still removed). frac = max_i(rows_i / H_i) + SAFETY_MARGIN; each image loses ceil(frac * H_i) rows."""
    ca, cb = _candidates(a_src), _candidates(a_tgt)
    shared_tags = {t for _, _, t in ca} & {t for _, _, t in cb}
    if not shared_tags:
        return None
    sides = {s for _, s, t in ca if t in shared_tags}
    best = None
    for side in sorted(sides):
        rs = max(r for r, s, _ in ca if s == side)
        rt = max(r for r, s, _ in cb if s == side)
        fr = max(rs / a_src["h"], rt / a_tgt["h"])
        if best is None or fr > best[0]:
            best = (fr, side)
    frac, side = best[0] + SAFETY_MARGIN, best[1]
    return CropRule("overlay", side, math.ceil(frac * a_src["h"]), math.ceil(frac * a_tgt["h"]))


def sham_rule(shape_src, shape_tgt, frac: float = SHAM_FRAC) -> CropRule:
    return CropRule("sham", "bottom", math.ceil(frac * shape_src[0]), math.ceil(frac * shape_tgt[0]))


def _crop(img: np.ndarray, side: str, k: int) -> np.ndarray:
    if k <= 0:
        return img
    return img[k:] if side == "top" else img[: img.shape[0] - k]


def crop_overlay(src: np.ndarray, tgt: np.ndarray, rule: CropRule | None):
    """Crop both images by the rule. Returns (src_c, tgt_c); rule None returns the inputs."""
    if rule is None:
        return src, tgt
    return _crop(src, rule.side, rule.rows_src), _crop(tgt, rule.side, rule.rows_tgt)


def _translate(dx: float, dy: float) -> np.ndarray:
    return np.array([[1.0, 0.0, dx], [0.0, 1.0, dy], [0.0, 0.0, 1.0]])


def uncrop_homography(H_crop: np.ndarray, rule: CropRule | None) -> np.ndarray:
    """H fitted on the cropped pair (target_crop -> source_crop) expressed in ORIGINAL coordinates.

    p_crop = p_orig - offset  =>  H_orig = T(+off_src) @ H_crop @ T(-off_tgt). Bottom crops have zero offset."""
    H = np.asarray(H_crop, dtype=np.float64)
    if rule is None:
        return H
    (sx, sy), (tx, ty) = rule.offset_src, rule.offset_tgt
    return _translate(sx, sy) @ H @ _translate(-tx, -ty)
