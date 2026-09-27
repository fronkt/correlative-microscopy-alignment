"""ARCHIVED 2026-08-30. Superseded by paper/schematics/gen_fig1.py.

Provenance: this is the original Fig. 1 generator, kept verbatim below for
reference. It is archived rather than deleted because it wrote to the same path
as its replacement (paper/figs/method_schematic.png), so running it would have
silently overwritten the redraw with the old design. The guard below stops that.

Why it was replaced: panel B drew the ladder rungs without the source frame, so
the "target area / source area" ratios had no denominator on the page; the rung
sizes were computed against the 0.5 rung rather than the source, so the drawn
areas did not equal the stated ratios; and several labels collided with the boxes
and connectors they belonged to. See tasks/todo.md, Phase N.
"""
import sys
sys.exit("archived: run paper/schematics/gen_fig1.py instead")

"""ORIGINAL SOURCE FOLLOWS
'''Generate the method schematic (Fig. 1) for the manuscript.

Two panels:
  (a) Verified coarse-to-fine wrapper (pyramid v2) around a frozen dense matcher.
  (b) The FOV-ladder protocol that decouples scale from appearance.

Vector-clean, colourblind-safe, no external assets. Saves PNG (300 dpi) + PDF.
Run: python paper/make_schematic.py
'''
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "figs"
OUT.mkdir(parents=True, exist_ok=True)

# Okabe-Ito colourblind-safe palette
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
GREY = "#999999"
RED = "#D55E00"
INK = "#222222"


def box(ax, x, y, w, h, text, fc="white", ec=INK, fs=8.5, lw=1.2, tc=INK):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            fc=fc, ec=ec, lw=lw, zorder=2,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, zorder=3, wrap=True)


def arrow(ax, p0, p1, color=INK, lw=1.4, style="-|>", ls="-"):
    ax.add_patch(
        FancyArrowPatch(
            p0, p1, arrowstyle=style, mutation_scale=12,
            color=color, lw=lw, linestyle=ls, zorder=1,
            shrinkA=2, shrinkB=2,
        )
    )


fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.0, 4.5), gridspec_kw={"width_ratios": [1.32, 1.0]})
for ax in (axA, axB):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

# ---------------------------------------------------------------- panel (a)
axA.text(0.02, 0.96, "A", fontsize=14, fontweight="bold", va="top")
axA.text(0.5, 0.965, "Verified coarse-to-fine wrapper (pyramid v2)",
         ha="center", va="top", fontsize=10, fontweight="bold")

# inputs
box(axA, 0.03, 0.74, 0.20, 0.12, "Wide-FOV\nsource  $I_s$", fc="#EAF3FA", ec=BLUE)
box(axA, 0.03, 0.55, 0.20, 0.12, "Narrow-FOV\ntarget  $I_t$", fc="#FDF1DF", ec=ORANGE)

# direct match
box(axA, 0.31, 0.645, 0.22, 0.16, "Frozen dense\nmatcher\n(RoMa / MA-RoMa)", fc="white", ec=INK)
arrow(axA, (0.23, 0.80), (0.31, 0.755), BLUE)
arrow(axA, (0.23, 0.61), (0.31, 0.715), ORANGE)

# incumbent transform
box(axA, 0.61, 0.645, 0.22, 0.16, "MAGSAC++ fit\n(affine vs. homog.\nby BIC) + TPS", fc="white", ec=INK)
arrow(axA, (0.53, 0.725), (0.61, 0.725))
axA.text(0.565, 0.755, "direct", fontsize=7.5, color=GREY, ha="center")

box(axA, 0.85, 0.66, 0.12, 0.13, "incumbent\n$T^\\star$", fc="#EDEDED", ec=INK, fs=8)
arrow(axA, (0.83, 0.725), (0.85, 0.725))

# candidate stages
box(axA, 0.31, 0.30, 0.52, 0.20,
    "Candidate stages (only on weak support)\n"
    "  • tile search over scaled $I_s$ grid\n"
    "  • zoom refinement on best region",
    fc="#F3FBF7", ec=GREEN, fs=8.5)
arrow(axA, (0.42, 0.645), (0.42, 0.50), GREEN, ls="--")
axA.text(0.30, 0.575, "weak?", fontsize=7.5, color=GREEN, ha="right")

# verifier gate
box(axA, 0.34, 0.07, 0.46, 0.13,
    "Verifier: mutual information on overlap\n"
    "accept candidate only if it beats incumbent",
    fc="#FCEEE7", ec=RED, fs=8.5)
arrow(axA, (0.57, 0.30), (0.57, 0.20), RED)
arrow(axA, (0.80, 0.135), (0.905, 0.135), RED)
arrow(axA, (0.905, 0.135), (0.905, 0.66), RED, ls="--")
axA.text(0.86, 0.16, "if better,\nreplace $T^\\star$", fontsize=7, color=RED, ha="center")

# ---------------------------------------------------------------- panel (b)
axB.text(0.02, 0.96, "B", fontsize=14, fontweight="bold", va="top")
axB.text(0.5, 0.965, "FOV-ladder protocol", ha="center", va="top",
         fontsize=10, fontweight="bold")
axB.text(0.30, 0.885, "appearance fixed, scale swept on real pairs",
         ha="center", va="top", fontsize=8.2, color=GREY, style="italic")

# nested crops illustrating shrinking FOV
ratios = [(0.5, "#0072B2"), (0.25, "#56A7DD"), (0.1, ORANGE), (0.05, RED), (0.02, "#7A2E00")]
cx, cy = 0.30, 0.45
base = 0.42
for r, col in ratios:
    s = base * (r / 0.5) ** 0.5
    axB.add_patch(plt.Rectangle((cx - s / 2, cy - s / 2), s, s, fill=False,
                                ec=col, lw=1.6, zorder=3))
axB.text(cx, cy + base / 2 + 0.03, "target field-of-view area / source area",
         ha="center",
         fontsize=7.5, color=GREY)
axB.text(cx, cy - base / 2 - 0.05, "0.5 → 0.25 → 0.1 → 0.05 → 0.02",
         ha="center", fontsize=8, color=INK)

# protocol annotation. This is a METHODS figure: it states what is done at
# each rung and must not assert an outcome, so no success rates, effect sizes
# or p-values appear here. The ladder results are Fig. 4.
box(axB, 0.60, 0.55, 0.37, 0.30,
    "All ground-truth points\nretained, including those\noutside the crop",
    fc="#EDEDED", ec=INK, fs=8.5)
box(axB, 0.60, 0.14, 0.37, 0.32,
    "Each backbone evaluated\nat every rung, direct\nand wrapped",
    fc="#F3FBF7", ec=GREEN, fs=8.5)
arrow(axB, (0.52, 0.55), (0.60, 0.62), INK)
arrow(axB, (0.45, 0.30), (0.60, 0.30), GREEN)

fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.02, wspace=0.06)
fig.savefig(OUT / "method_schematic.png", dpi=300, bbox_inches="tight")
fig.savefig(OUT / "method_schematic.pdf", bbox_inches="tight")
print("wrote", OUT / "method_schematic.png")

"""
