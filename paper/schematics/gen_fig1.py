# -*- coding: utf-8 -*-
"""Figure 1 of the Microscopy & Microanalysis manuscript: the method schematic.

    cd paper/schematics && python gen_fig1.py

Two panels, and the content is fixed by the manuscript legend -- this script is a
redraw, not a redesign:

  (A) the verified coarse-to-fine wrapper, pyramid v2 (Methods 2.8)
  (B) the controlled field-of-view ladder (Methods 2.9)

Three things drive the drawing decisions:

1. Panel B's rung sizes are not free. The ladder crops to ABSOLUTE AREA ratios, so a
   rung's on-page side must go as sqrt(ratio) of the source side, and the source
   frame -- the denominator -- has to be on the page or the ratios name nothing.
   Both are asserted below.

2. Field of view is encoded by FRAME SIZE, never by colour. Okabe-Ito blue and
   orange already carry backbone identity in Figs 3-4 and error threshold in Figs 2
   and 5; a schematic that also used them would hang a third meaning on the same
   two hues. Size is what actually distinguishes a wide field from a narrow one,
   so the geometry carries it and the two hues that remain (green, vermillion) are
   free to mark the only two things the wrapper adds over a direct match: the
   conditional branch, and the gate.

3. Nothing here asserts an outcome. This is a methods figure; success rates, effect
   sizes and p-values belong to Figs 2-5. That includes not highlighting rung 0.1,
   which is where the ladder result lands.

Nothing in the figure is drawn that the legend or an on-page label does not name.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle
from matplotlib.lines import Line2D

import palette

palette.apply()


def spread_2d(n, rng, half_in, min_sep_in, wall=0.0, cand=192, avoid=()):
    """Mitchell best-candidate blue noise, in INCHES on the page.

    scatter.spread_positions is the 3-D original and the reasoning is its
    docstring's: at the couple of dozen points a panel holds, rng.uniform
    visibly clumps and leaves whole regions bare, and a reader reads a clump as
    structure -- here, as a claim about where correspondences lie. This is the
    2-D case, and it works in inches rather than axes fractions because panel B
    is wider than it is tall: equal separation in axes units is unequal
    separation on the page, and the page is what gets reviewed.

    `avoid` is a list of (x0, y0, x1, y1) keep-out boxes in the same inches. The
    panel's labels sit inside the rungs they name, and a marker landing on one
    does not read as a marker on a label -- it reads as a different digit.
    """
    h = np.asarray(half_in, float)

    def blocked(c):
        m = np.zeros(len(c), bool)
        for x0, y0, x1, y1 in avoid:
            m |= ((c[:, 0] > x0) & (c[:, 0] < x1)
                  & (c[:, 1] > y0) & (c[:, 1] < y1))
        return m

    pts = np.empty((n, 2))
    for i in range(n):
        c = rng.uniform(-h, h, (cand, 2))
        free = ~blocked(c)
        if not free.any():
            raise RuntimeError("keep-out boxes leave nowhere to place a point")
        c = c[free]
        if i == 0:
            pts[0] = c[0]
            continue
        d = np.linalg.norm(c[:, None, :] - pts[None, :i, :], axis=2).min(axis=1)
        score = d / min_sep_in
        if wall:
            score = score + wall * np.abs(c / h).max(axis=1)
        pts[i] = c[int(np.argmax(score))]
    return pts

OUT = "../figs"
STEM = "method_schematic"

# --- page geometry ----------------------------------------------------------
# Sized for a full-width figure in Microscopy & Microanalysis. The axes widths
# below are derived from these numbers, not guessed, because panels A and B both
# contain squares (image frames, ladder rungs) and a square drawn in axes
# fractions is only square if the axes' inch dimensions are known.
FIG_W, FIG_H = 7.20, 3.05
WR_A, WR_B = 1.50, 1.00
WSPACE = 0.10
L, R, T, B = 0.005, 0.995, 0.985, 0.015

_avail_w = (R - L) * FIG_W
_w_avg = _avail_w / (2.0 + WSPACE)
AW = 2.0 * _w_avg * WR_A / (WR_A + WR_B)      # panel A width, inches
BW = 2.0 * _w_avg * WR_B / (WR_A + WR_B)      # panel B width, inches
PH = (T - B) * FIG_H                          # both panels' height, inches


def fx(inches, panel_w):
    """On-page length -> x fraction of a panel of width `panel_w` inches."""
    return inches / panel_w


def fy(inches):
    """On-page length -> y fraction. Both panels share a height."""
    return inches / PH


# --- the numbers the figure is about ----------------------------------------
# Methods 2.9: "crop the target to absolute area ratios of 0.5, 0.25, 0.1, 0.05
# and 0.02". Absolute means relative to the source field, so the source itself is
# rung 1.0 and must be drawn -- without it the ratios have no denominator on the
# page, which is the defect this redraw exists to fix.
FOV_RATIOS = (1.0, 0.5, 0.25, 0.1, 0.05, 0.02)
SRC_SIDE_IN = 1.52          # on-page side of the source frame in panel B

_sides = [SRC_SIDE_IN * np.sqrt(r) for r in FOV_RATIOS]
for _r, _s in zip(FOV_RATIOS, _sides):
    assert abs((_s / SRC_SIDE_IN) ** 2 - _r) < 1e-12, (
        "rung %s is drawn at the wrong area" % _r)
assert _sides == sorted(_sides, reverse=True)

N_GT = 24                   # ground-truth correspondences drawn in panel B
GT_SEED = 11


# --- drawing helpers --------------------------------------------------------
def rbox(ax, cx, cy, w, h, text, ec=palette.INK, fc="white", lw=0.9, fs=6.8,
         tc=palette.INK, weight="normal", z=2):
    """A rounded process box, placed by its centre."""
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0,rounding_size=0.018",
        fc=fc, ec=ec, lw=lw, zorder=z, mutation_aspect=AW / PH))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            color=tc, fontweight=weight, zorder=z + 1, linespacing=1.35)


def diamond(ax, cx, cy, hw, hh, text, ec=palette.INK, fc="white", lw=0.9, fs=6.4):
    ax.add_patch(Polygon(
        [(cx, cy + hh), (cx + hw, cy), (cx, cy - hh), (cx - hw, cy)],
        closed=True, fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            color=palette.INK, zorder=3, linespacing=1.3)


def arrow(ax, p0, p1, color=palette.INK, lw=0.9, ls="-", z=4, head=7.0):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=head, color=color, lw=lw,
        linestyle=ls, zorder=z, shrinkA=0, shrinkB=0,
        joinstyle="miter", capstyle="butt"))


def route(ax, pts, color=palette.INK, lw=0.9, ls="-", z=4):
    """An orthogonal polyline with the arrowhead on the final segment."""
    pts = list(pts)
    xs, ys = zip(*pts[:-1])
    ax.add_line(Line2D(xs, ys, color=color, lw=lw, linestyle=ls, zorder=z,
                       solid_capstyle="projecting"))
    arrow(ax, pts[-2], pts[-1], color=color, lw=lw, ls=ls, z=z)


def edge_label(ax, x, y, text, color=palette.GREY, fs=5.9, ha="center",
               va="center", pad=1.6):
    """A label ON a connector. The white bbox is what stops a connector or a box
    edge running through the type; without it these are the first things to
    collide when any box moves."""
    ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=color,
            zorder=palette.Z_TOP, linespacing=1.3,
            bbox=dict(boxstyle="round,pad=%.2f" % (pad / 10), fc="white",
                      ec="none"))


def frame(ax, cx, cy, w, h, ec=palette.FRAME, fc=palette.FRAME_FILL, lw=0.9, z=2):
    """An image boundary: a field of view."""
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h,
                           fc=fc, ec=ec, lw=lw, zorder=z))


# ============================================================== figure
fig, (axA, axB) = plt.subplots(
    1, 2, figsize=(FIG_W, FIG_H), gridspec_kw={"width_ratios": [WR_A, WR_B]})
fig.subplots_adjust(left=L, right=R, top=T, bottom=B, wspace=WSPACE)
for ax in (axA, axB):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

# ------------------------------------------------------------------ panel A
palette.panel_tag(axA, "A", x=0.0, y=1.0, fontsize=9.5)
axA.text(0.50, 0.985, "Verified coarse-to-fine wrapper (pyramid v2)",
         ha="center", va="top", fontsize=8.0, fontweight="bold",
         color=palette.INK)

SPINE_Y = 0.815

# inputs: two image frames whose SIZE is the wide/narrow distinction
src_w, src_h = fx(0.42, AW), fy(0.42)
tgt_w, tgt_h = fx(0.24, AW), fy(0.24)
in_bottom = 0.762
src_cx, tgt_cx = 0.052, 0.170
src_cy, tgt_cy = in_bottom + src_h / 2, in_bottom + tgt_h / 2
frame(axA, src_cx, src_cy, src_w, src_h)
frame(axA, tgt_cx, tgt_cy, tgt_w, tgt_h)
axA.text(src_cx, 0.742, "wide-FOV\nsource $I_s$", ha="center", va="top",
         fontsize=5.6, color=palette.INK, linespacing=1.3)
axA.text(tgt_cx, 0.742, "narrow-FOV\ntarget $I_t$", ha="center", va="top",
         fontsize=5.6, color=palette.INK, linespacing=1.3)

# spine: match -> fit -> incumbent
MATCH_C, MATCH_W = 0.335, 0.168
FIT_C, FIT_W = 0.612, 0.258
TSTAR_C, TSTAR_W = 0.888, 0.148

rbox(axA, MATCH_C, SPINE_Y, MATCH_W, 0.100, "frozen dense\nmatcher")
rbox(axA, FIT_C, SPINE_Y, FIT_W, 0.140,
     "MAGSAC++ fit\naffine or homography\n+ optional TPS")
rbox(axA, TSTAR_C, SPINE_Y, TSTAR_W, 0.098, "incumbent\n$T^{*}$",
     fc=palette.STATE)

# Both images enter the matcher, so both get a head. The source's arrow leaves
# above the target frame's top edge; routed through its centre it would cross the
# other input, and a connector crossing a box reads as passing through it.
assert src_cy + 0.039 > tgt_cy + tgt_h / 2, "source connector would cross the target frame"
arrow(axA, (src_cx + src_w / 2, src_cy + 0.039),
      (MATCH_C - MATCH_W / 2, SPINE_Y + 0.030))
arrow(axA, (tgt_cx + tgt_w / 2, tgt_cy), (MATCH_C - MATCH_W / 2, SPINE_Y - 0.015))
arrow(axA, (MATCH_C + MATCH_W / 2, SPINE_Y), (FIT_C - FIT_W / 2, SPINE_Y))
arrow(axA, (FIT_C + FIT_W / 2, SPINE_Y), (TSTAR_C - TSTAR_W / 2, SPINE_Y))

# the branch: candidate stages run only on weak direct support
DIA_C, DIA_Y, DIA_HW, DIA_HH = FIT_C, 0.578, 0.132, 0.076
diamond(axA, DIA_C, DIA_Y, DIA_HW, DIA_HH, "direct support\nweak?")
arrow(axA, (FIT_C, SPINE_Y - 0.070), (DIA_C, DIA_Y + DIA_HH))

# "no" is not a path, it is the absence of one: the incumbent simply stands.
arrow(axA, (DIA_C + DIA_HW, DIA_Y), (DIA_C + DIA_HW + 0.042, DIA_Y),
      color=palette.GREY, lw=0.8)
axA.text(DIA_C + DIA_HW + 0.052, DIA_Y, "no: keep $T^{*}$", ha="left",
         va="center", fontsize=5.9, color=palette.GREY, zorder=palette.Z_TOP)

CAND_Y = 0.245
TILE_C, TILE_W = 0.170, 0.290
ZOOM_C, ZOOM_W = 0.500, 0.250
VER_C, VER_W = 0.838, 0.290

route(axA, [(DIA_C, DIA_Y - DIA_HH), (DIA_C, 0.402),
            (TILE_C, 0.402), (TILE_C, CAND_Y + 0.073)],
      color=palette.BRANCH)
edge_label(axA, DIA_C + 0.018, 0.478, "yes", color=palette.BRANCH, ha="left")

rbox(axA, TILE_C, CAND_Y, TILE_W, 0.146, "tile search over\nscaled source grid",
     ec=palette.BRANCH)
rbox(axA, ZOOM_C, CAND_Y, ZOOM_W, 0.146, "zoom refinement\non best region",
     ec=palette.BRANCH)
rbox(axA, VER_C, CAND_Y, VER_W, 0.146,
     "verifier: mutual\ninformation on overlap", ec=palette.GATE, lw=1.2)

arrow(axA, (TILE_C + TILE_W / 2, CAND_Y), (ZOOM_C - ZOOM_W / 2, CAND_Y),
      color=palette.BRANCH)
arrow(axA, (ZOOM_C + ZOOM_W / 2, CAND_Y), (VER_C - VER_W / 2, CAND_Y),
      color=palette.BRANCH)

# the gate's only output: replace the incumbent, or do not
RET_X = 0.952
route(axA, [(VER_C + 0.060, CAND_Y + 0.073), (RET_X, CAND_Y + 0.073),
            (RET_X, SPINE_Y - 0.049)], color=palette.GATE)
edge_label(axA, RET_X - 0.018, 0.452, "accept only if\nit beats $T^{*}$",
           color=palette.GATE, ha="right")

# ------------------------------------------------------------------ panel B
palette.panel_tag(axB, "B", x=0.0, y=1.0, fontsize=9.5)
axB.text(0.50, 0.985, "Controlled field-of-view ladder", ha="center", va="top",
         fontsize=8.0, fontweight="bold", color=palette.INK)
axB.text(0.50, 0.930,
         "appearance, modality and pixel size held fixed,\nso scale is the only variable",
         ha="center", va="top", fontsize=5.9, color=palette.GREY,
         style="italic", linespacing=1.35)

BCX, BCY = 0.360, 0.495
half = [(fx(s / 2, BW), fy(s / 2)) for s in _sides]

# ground truth first, so the rung outlines sit on top of it
rng = np.random.default_rng(GT_SEED)
hx0, hy0 = half[0]
_inset = 0.90                        # keep points off the source frame's own edge
# No wall bias here. Ground truth is claimed to be retained across the whole
# source field, so pushing points toward the edges would leave the inner rungs
# empty and quietly say correspondences live only at the periphery.
_pts_in = spread_2d(N_GT, rng, [SRC_SIDE_IN / 2 * _inset] * 2,
                    min_sep_in=SRC_SIDE_IN * 0.10)
# The panel's claim is that GT is retained ACROSS the source field, so the points
# must read as spread rather than clumped, and no two may touch on the page.
_d = np.linalg.norm(_pts_in[:, None, :] - _pts_in[None, :, :], axis=2)
np.fill_diagonal(_d, np.inf)
MARKER_IN = 0.026                    # drawn diameter of a GT marker
assert _d.min() > MARKER_IN * 2.0, (
    "ground-truth markers overlap on the page: min separation %.4f in" % _d.min())

# The panel says ground truth is kept across the whole field, so the count inside
# each rung has to fall with the rung, and a small rung must not read as empty of
# it. Both are claims about the drawing, so both are checked rather than eyeballed.
_inside = [int(((np.abs(_pts_in[:, 0]) < s / 2)
                & (np.abs(_pts_in[:, 1]) < s / 2)).sum()) for s in _sides]
assert _inside == sorted(_inside, reverse=True), (
    "points inside the rungs do not fall monotonically: %s" % _inside)
assert _inside[4] >= 1, (
    "no point lands inside the 0.05 rung, so the panel would say the small crops "
    "carry no ground truth")
pts = np.column_stack([_pts_in[:, 0] / BW, _pts_in[:, 1] / PH])

frame(axB, BCX, BCY, 2 * hx0, 2 * hy0)
axB.scatter(BCX + pts[:, 0], BCY + pts[:, 1], s=3.4, marker="o",
            facecolors="none", edgecolors=palette.POINT, linewidths=0.55,
            zorder=3)

for i, (hx, hy) in enumerate(half):
    if i:                                    # rung 0 is the source frame itself
        axB.add_patch(Rectangle((BCX - hx, BCY - hy), 2 * hx, 2 * hy,
                                fill=False, ec=palette.LADDER[i], lw=1.15,
                                zorder=4))

# The rungs carry no type at all. Labelling them in place cannot be done evenly:
# the band between the 0.10 and 0.05 rungs, and between 0.05 and 0.02, is thinner
# than a line of 5.9 pt type, and that is forced by sqrt(0.10), sqrt(0.05) and
# sqrt(0.02) being close rather than by the layout. A mixed scheme -- some rungs
# labelled in place, the crowded ones on leaders -- reads as two systems, and
# every such leader has to cross every rung outside the one it names. So all six
# go in one key, drawn from the same LADDER constants as the rectangles, which is
# what stops a swatch drifting from the rung it stands for.
LABEL_IN = 0.087                             # type height + its offset, inches
_band = [(half[i][1] - half[i + 1][1]) * PH for i in range(len(half) - 1)]
assert min(_band) < LABEL_IN, "bands now fit a label; reconsider the key"

KEY_X, KEY_TOP, KEY_DY = 0.700, 0.740, 0.070
KEY_GAP = 0.026          # the source is the denominator, not a rung: set it apart
for i, r in enumerate(FOV_RATIOS):
    y = KEY_TOP - i * KEY_DY - (KEY_GAP if i else 0.0)
    axB.add_line(Line2D([KEY_X, KEY_X + 0.050], [y, y],
                        color=palette.FRAME if i == 0 else palette.LADDER[i],
                        lw=0.9 if i == 0 else 1.15, zorder=palette.Z_TOP,
                        solid_capstyle="butt"))
    axB.text(KEY_X + 0.064, y, "1.00, source" if i == 0 else "%.2f" % r,
             ha="left", va="center", fontsize=5.9, color=palette.INK,
             zorder=palette.Z_TOP)
axB.text(KEY_X, KEY_TOP + 0.075, "crop area /\nsource area", ha="left",
         va="center", fontsize=5.9, color=palette.GREY, linespacing=1.35,
         zorder=palette.Z_TOP)

# claim 3 of the legend, shown rather than asserted in prose
_key_y = KEY_TOP - len(FOV_RATIOS) * KEY_DY - KEY_GAP - 0.030
axB.scatter([KEY_X + 0.025], [_key_y], s=3.4, marker="o", facecolors="none",
            edgecolors=palette.POINT, linewidths=0.55, zorder=palette.Z_TOP)
axB.text(KEY_X + 0.064, _key_y, "ground truth,\nkept in full", ha="left",
         va="center", fontsize=5.9, color=palette.INK, linespacing=1.35,
         zorder=palette.Z_TOP)

axB.text(0.50, 0.090, "points outside a crop are retained, and test\n"
                      "extrapolation of the fitted transform",
         ha="center", va="center", fontsize=5.9, color=palette.INK,
         linespacing=1.35)
axB.text(0.50, 0.022, "every rung run on every backbone, direct and wrapped",
         ha="center", va="center", fontsize=5.9, color=palette.GREY)

# ------------------------------------------------------------------ export
paths = palette.save_all(fig, STEM, outdir=OUT, dpi=600)
for p in paths:
    print("wrote", p)
print("panel A %.2f x %.2f in, panel B %.2f x %.2f in"
      % (AW, PH, BW, PH))
print("ladder sides (in):", ", ".join("%.3f" % s for s in _sides))
