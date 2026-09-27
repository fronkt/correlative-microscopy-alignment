"""Colour, type and output -- everything about how a paper's figures LOOK, in one file.

Copy this into the project and edit the colour block. Every figure script imports it,
so a phase is never drawn in two different colours across the paper, and a swatch in
the legend is literally the same constant as the panel it stands for.

The rcParams here are not preferences. Type 3 fonts get a figure bounced, flattened
SVG text cannot be edited by a co-author, and an uncustomised mathtext fontset ships
every subscripted label in a second typeface. Do not override them per figure.
"""
import numpy as np

# --- roles in this paper's schematics ---------------------------------------
# Fig. 1 is a method schematic, so its constants name ROLES in the pipeline, not
# materials. Two rules constrain the choices:
#
#  * Okabe-Ito blue (#0072B2) and orange (#E69F00) are already spoken for in the
#    data figures -- backbone identity in Figs 3-4, error threshold in Figs 2 and
#    5. A schematic that also used them would put a third meaning on the same two
#    hues. So the wide/narrow field of view is encoded by FRAME SIZE here, which
#    is what actually distinguishes them, and blue and orange do not appear.
#  * The two hues that ARE used, green and vermillion, mark the two things the
#    wrapper adds over a direct match: the conditional branch and the gate.
INK = "#1A1A1A"           # text and the main flow spine
LEADER = "#5A5A5A"        # leader lines: visible, never as dark as the text
GREY = "#6E6E6E"          # edge labels, secondary annotation
FRAME = "#3A3A3A"         # an image boundary (a field of view)
FRAME_FILL = "#F1F1F1"    # the pixels inside one; pale, it is background
STATE = "#E8E8E8"         # a stored value the pipeline carries (the incumbent)
BRANCH = "#009E73"        # Okabe-Ito green: the conditional candidate path
GATE = "#D55E00"          # Okabe-Ito vermillion: the verifier and its decision
POINT = "#1A1A1A"         # a ground-truth correspondence

# Sequential ramp for the field-of-view ladder. The rungs are an ORDERED quantity,
# so they get one hue ramped by value; categorical colours would invite the reader
# to look for five kinds of thing instead of one thing at five sizes.
LADDER = ["#C9C9C9", "#A8A8A8", "#878787", "#666666", "#454545", "#1A1A1A"]

PANEL_EDGE = "#3A3A3A"
PANEL_BG = "#FFFFFF"
ACCENT = GATE             # the one colour reserved for "look here"

# --- typography -------------------------------------------------------------
FONT = "Arial"
FONT_FALLBACK = ["Arial", "Helvetica", "DejaVu Sans"]

# Scene.draw hands out zorder in paint order, which climbs into the hundreds. Anything
# annotated onto the axes afterwards needs a zorder above all of it or it paints under
# the scene.
Z_TOP = 10_000


def hex2rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def rcparams():
    """Publication rcParams: sans-serif, true vector text, no Type-3 fonts."""
    return {
        "font.family": "sans-serif",
        "font.sans-serif": FONT_FALLBACK,
        # Without these four, mathtext (subscripts, Greek) silently falls back to
        # DejaVu and the figure ships two typefaces mixed inside single labels.
        "mathtext.fontset": "custom",
        "mathtext.rm": FONT,
        "mathtext.it": f"{FONT}:italic",
        "mathtext.bf": f"{FONT}:bold",
        "mathtext.default": "regular",
        "pdf.fonttype": 42,      # TrueType. Type 3 is rejected by most publishers
        "ps.fonttype": 42,
        "svg.fonttype": "none",  # keep text as text so the SVG stays editable
        "figure.dpi": 200,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.linewidth": 0.8,
    }


def apply():
    """Install the rcParams. Call once, before any figure is created."""
    import matplotlib
    matplotlib.rcParams.update(rcparams())


def save_all(fig, stem, outdir=".", dpi=600, close=True):
    """Write <stem>.pdf, .svg and .png.

    PDF is the submission copy, SVG is what co-authors edit, PNG is for pasting into
    the manuscript and -- more usefully -- for looking at, which is the only way
    defects in a schematic are ever found.
    """
    import os
    import matplotlib.pyplot as plt
    paths = []
    for ext in ("pdf", "svg", "png"):
        p = os.path.join(outdir, f"{stem}.{ext}")
        fig.savefig(p, dpi=dpi if ext == "png" else None)
        paths.append(p)
    if close:
        plt.close(fig)
    return paths


def leader(ax, text, xy, xytext, ha="left", va="center", fontsize=7.2,
           color=None, arrow=LEADER, lw=0.6, style="-", **kw):
    """Label something in the scene with a line to it.

    `xy` is a point in the axes' DATA coordinates, i.e. camera.project() output --
    pick the anchor in SCREEN space, on a part of the target with nothing else near
    it. Aiming at the geometrically nearest point of a target routinely lands the
    leader on top of some other object, and the label then names the wrong thing.

    `xytext` is an axes fraction: park text in the figure's white margin rather than
    over the geometry it describes.
    """
    return ax.annotate(
        text, xy=xy, xytext=xytext, textcoords=ax.transAxes,
        ha=ha, va=va, fontsize=fontsize, color=INK if color is None else color,
        zorder=Z_TOP, annotation_clip=False,
        arrowprops=dict(arrowstyle=style, color=arrow, lw=lw), **kw)


def panel_tag(ax, s, x=0.005, y=0.995, fontsize=8.5, weight="bold"):
    """The (a)/(D1) label, in a consistent place across every panel."""
    return ax.text(x, y, s, transform=ax.transAxes, ha="left", va="top",
                   fontsize=fontsize, fontweight=weight, color=INK, zorder=Z_TOP)
