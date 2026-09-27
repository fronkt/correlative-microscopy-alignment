"""Figures for the Microscopy and Microanalysis rewrite (after MAM-26-246).

    python scripts/plot_mam_rewrite.py            # every figure whose inputs exist
    python scripts/plot_mam_rewrite.py fig1 fig5  # a subset

The editor asked that "examples of the failures are shown". Figures 1-4 therefore show
real AmalgaMatch images (CC-BY-4.0, Durmaz et al.), and Figures 5-7 are deliberately
simpler than the charts they replace: one question per panel, counts written as
"k of n", intervals drawn, nothing a reader has to decode from a second legend.

Inputs: result CSVs, results/mam_rewrite_numbers.json (scripts/mam_rewrite_numbers.py),
and for Figures 2-4 the display re-runs cached by scripts/mam_examples_run.py.
Outputs: one file per image, paper/mam/figures/panels/Figure<N><letter>.tif (the upload),
a Figure<N>_layout.jpg sheet per figure showing the intended arrangement, and
FigureS<N>.pdf/.png for the supplementary material.

Why one file per image: the MSA office unsubmitted MAM-26-277 (2026-09-27) because the
composite figures were ~1600 pixels wide, multi-panel, and carried caption text. It asks for
each image as its own file, >= 2550 pixels wide at >= 300 dpi, with no legend inside. So
each panel holds only its image, its overlays, its axes and its letter; everything the old
panel titles and captions said is in the manuscript's legends.
"""

from __future__ import annotations

import csv
import json
import sys
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "paper" / "schematics"))
import mam_example_lib as L  # noqa: E402
import palette  # noqa: E402

plt.rcParams.update(palette.rcparams())
plt.rcParams.update({"font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7})

OUT = ROOT / "paper" / "mam" / "figures"
RES = ROOT / "results"
NUM = json.loads((RES / "mam_rewrite_numbers.json").read_text())
MM = 1 / 25.4
FULL = 174 * MM          # full page width
PANELS = OUT / "panels"
COL_MM, FULL_MM = 84, 174  # every panel is drawn at column width, Figure 5's bar charts at page width
DPI = 900                # OUP's floor for combined half-tone and line art is 600 dpi
MIN_PX = 2550            # the office's pixels-per-line requirement (8.5 in at 300 dpi)
BIG = 3200               # micrographs keep their own pixels up to this long side
# Intended arrangement, rows of panel letters in reading order: the office lays the panels out,
# so the letters must already read correctly. Figure 1 has one pair per row.
LAYOUT = {"1": ["AB", "CD", "EF"], "2": ["AB", "CDE"], "3": ["AB", "CD", "EF"],
          "4": ["ABC", "DEF", "GHI"], "5": ["A", "B"], "6": ["AB"], "7": ["AB"]}
_SHEET: dict[str, dict[str, Image.Image]] = {}
# Okabe-Ito. One meaning per hue across the whole paper:
TRUE = "#F0E442"         # yellow: where the narrow image really is (ground truth)
EST = "#D55E00"          # vermillion: where the method put it
DIRECT = "#0072B2"       # blue: a matcher used directly
CHECKED = "#E69F00"      # orange: the same matcher inside the checked search
POOLED = "#CC79A7"       # reddish purple: pool-then-fit tiling
KEPT = "#009E73"         # bluish green: correspondences robust fitting keeps
ABOVE = "#56B4E9"        # sky blue: certainty above the sampler cut-off
BELOW = "#000000"        # black: certainty below it
IN_T, OUT_T = "#009E73", "#D55E00"  # the tile inside / outside the true outline
GREY = "#6E6E6E"


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"wrote {name}")


def panel(ax, letter: str) -> None:
    """Put the panel letter at the head of the axes' title, left-aligned, so the two
    can never collide (the rejected package's figures had letters floating over text)."""
    t = ax.get_title(loc="left") or ax.get_title()
    size = (ax._left_title if ax.get_title(loc="left") else ax.title).get_fontsize()
    ax.set_title("")
    ax.set_title("", loc="left")
    head = rf"$\mathbf{{{letter}}}$"
    ax.set_title(f"{head}   {t}" if t else head, loc="left", fontsize=size if t else 9)


def save_panel(fig, figno: str, letter: str) -> None:
    """Write one panel as an RGB, LZW-compressed TIFF at DPI, with its letter at the top left."""
    fig.text(0, 1, letter, ha="left", va="bottom", fontsize=10, fontweight="bold")
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=DPI, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    plt.close(fig)
    im = Image.open(buf).convert("RGB")
    name = f"Figure{figno}{letter}"
    assert im.size[0] >= MIN_PX, f"{name} is {im.size[0]} pixels wide, below the required {MIN_PX}"
    PANELS.mkdir(parents=True, exist_ok=True)
    im.save(PANELS / f"{name}.tif", compression="tiff_lzw", dpi=(DPI, DPI))
    print(f"wrote {name}  {im.size[0]} x {im.size[1]} pixels")
    im.thumbnail((1400, 1400), Image.LANCZOS)
    _SHEET.setdefault(figno, {})[letter] = im


def image_axes():
    """A figure whose only axes fill it; save_image resizes it to the drawn data's aspect."""
    fig = plt.figure(figsize=(COL_MM * MM, COL_MM * MM))
    return fig, fig.add_axes([0, 0, 1, 1])


def save_image(fig, ax, figno: str, letter: str) -> None:
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    fig.set_size_inches(COL_MM * MM, COL_MM * MM * abs(y1 - y0) / abs(x1 - x0))
    save_panel(fig, figno, letter)


def fit_to_image(ax) -> None:
    """Clamp the view to the micrograph. A dot on its last pixel lies past the image's sticky
    edge, and the autoscale margin then adds a white strip along that side."""
    x0, x1, y1, y0 = ax.images[0].get_extent()
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)


def plot_axes(w_mm: float = COL_MM, h_mm: float = 64):
    fig, ax = plt.subplots(figsize=(w_mm * MM, h_mm * MM), layout="constrained")
    ax.spines[["top", "right"]].set_visible(False)
    return fig, ax


def layout_sheet(figno: str, width: int = 2400, gap: int = 40) -> None:
    """Paste the panels in their intended rows: a QA view, and the arrangement the office will build."""
    ims = _SHEET.pop(figno)
    assert set("".join(LAYOUT[figno])) == set(ims), f"Figure {figno}: panels {sorted(ims)} vs layout {LAYOUT[figno]}"
    rows = []
    for row in LAYOUT[figno]:
        tiles = [ims[k] for k in row]
        h = round((width - gap * (len(tiles) - 1)) / sum(t.size[0] / t.size[1] for t in tiles))
        rows.append([t.resize((round(t.size[0] * h / t.size[1]), h), Image.LANCZOS) for t in tiles])
    sheet = Image.new("RGB", (width, sum(r[0].size[1] for r in rows) + gap * (len(rows) - 1)), "white")
    y = 0
    for r in rows:
        x = 0
        for t in r:
            sheet.paste(t, (x, y))
            x += t.size[0] + gap
        y += r[0].size[1] + gap
    sheet.save(OUT / f"Figure{figno}_layout.jpg", quality=90, dpi=(300, 300))  # a preview: JPEG keeps the repo small
    print(f"wrote Figure{figno}_layout.jpg")


def show(ax, img, max_side=900, clahe=False):
    d, f = L.display(img, max_side)
    if clahe:  # local contrast for display only; stated in the legend where used
        import cv2
        eq = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        d = eq.apply(d) if d.ndim == 2 else np.dstack([eq.apply(d[..., i]) for i in range(3)])
    ax.imshow(d, cmap="gray", interpolation="lanczos")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.6)
    return f


def outline_on(ax, H, target_shape, f, color, ls="-", lw=1.6, label=None):
    poly = L.outline(H, target_shape) * f
    ax.plot(poly[:, 0], poly[:, 1], color=color, ls=ls, lw=lw, label=label,
            path_effects=_halo(lw))
    return poly


def _halo(lw=1.6):
    import matplotlib.patheffects as pe
    return [pe.Stroke(linewidth=lw + 1.0, foreground="black", alpha=0.55), pe.Normal()]


def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def fov_label(pid: str) -> str:
    with (RES / "fov_ratios_gt.csv").open(newline="", encoding="utf-8") as f:
        r = {row["pair_id"]: float(row["fov_area_ratio_gt"]) for row in csv.DictReader(f)}[pid]
    return f"{100 * r:.0f} %" if r >= 0.095 else f"{100 * r:.1f} %"


# ----------------------------------------------------------------- Figure 1
FIG1 = [  # (wide letter, narrow letter, pair); the pair's description is in the legend
    ("A", "B", "eval_CoNi-AM90_SEM-EBSD_SameSlice_0#0"),
    ("C", "D", "eval_TRIP1-Bainitic_LOM_SEM_EBSD_Multiscale_0#4"),
    ("E", "F", "eval_5842WCu-Spalled_SEM-SE_SEM-BSE_Multiscale_0#0"),
]


def fig1() -> None:
    """The registration problem, in three real pairs: wide image, then narrow image."""
    for wide, narrow, pid in FIG1:
        pair, rec = L.load(pid)
        fig, ax = image_axes()
        f = show(ax, pair.source, BIG)
        outline_on(ax, L.gt_homography(pair), pair.target.shape, f, TRUE, lw=2.0)
        ax.plot(pair.gt.src_xy[:, 0] * f, pair.gt.src_xy[:, 1] * f, "o", ms=3.4, mfc=TRUE, mec="black", mew=0.5)
        save_image(fig, ax, "1", wide)
        fig, ax = image_axes()
        f2 = show(ax, pair.target, BIG)
        ax.plot(pair.gt.tgt_xy[:, 0] * f2, pair.gt.tgt_xy[:, 1] * f2, "o", ms=3.4, mfc=TRUE, mec="black", mew=0.5)
        fit_to_image(ax)
        save_image(fig, ax, "1", narrow)
    layout_sheet("1")


# ----------------------------------------------------------------- Figure 2
def fig2() -> None:
    """Registration step by step on one pair: match, fit, score."""
    c = np.load(L.cache_path("pipeline_5842"), allow_pickle=False)
    pair, rec = L.load(str(c["pair_id"]))
    src, tgt, inl = c["src_xy"], c["tgt_xy"], c["inliers"].astype(bool)
    rng = np.random.default_rng(1)
    pick_in = rng.choice(np.flatnonzero(inl), size=min(60, inl.sum()), replace=False)
    pick_out = rng.choice(np.flatnonzero(~inl), size=min(60, (~inl).sum()), replace=False)
    # A, B: steps 1 and 2, the same 120 correspondences in both images
    for letter, img, xy in (("A", pair.source, src), ("B", pair.target, tgt)):
        fig, ax = image_axes()
        f = show(ax, img, BIG)
        for idx, col in ((pick_out, EST), (pick_in, KEPT)):
            ax.plot(xy[idx, 0] * f, xy[idx, 1] * f, "o", ms=3.6, mfc=col, mec="black", mew=0.4)
        fit_to_image(ax)
        save_image(fig, ax, "2", letter)
    # C: step 3, fitted outline over the true one
    fig, ax = image_axes()
    fb = show(ax, pair.source, BIG)
    outline_on(ax, L.gt_homography(pair), pair.target.shape, fb, TRUE, lw=2.4)
    outline_on(ax, c["H"], pair.target.shape, fb, EST, ls="--", lw=1.8)
    save_image(fig, ax, "2", "C")
    # D: step 4, the miss at each annotated point
    fig, ax = image_axes()
    gt, proj = pair.gt.src_xy, c["gt_proj"]
    fc = show(ax, pair.source, BIG)
    lim = np.array([gt[:, 0].min(), gt[:, 0].max(), gt[:, 1].min(), gt[:, 1].max()]) * fc
    pad = 0.12 * max(lim[1] - lim[0], lim[3] - lim[2])
    ax.set_xlim(lim[0] - pad, lim[1] + pad); ax.set_ylim(lim[3] + pad, lim[2] - pad)
    gain = 5.0  # stated in the legend
    for g, p in zip(gt * fc, proj * fc):
        ax.annotate("", xy=g + gain * (p - g), xytext=g,
                    arrowprops=dict(arrowstyle="-|>", color=EST, lw=1.3, mutation_scale=9))
    ax.plot(gt[:, 0] * fc, gt[:, 1] * fc, "o", ms=4.0, mfc=TRUE, mec="black", mew=0.5)
    save_image(fig, ax, "2", "D")
    # E: step 5, the score
    ed = np.linalg.norm(proj - gt, axis=1)
    fig, ax = plot_axes(COL_MM, 60)
    ax.hist(ed, bins=np.arange(0, 13, 1), color=GREY, edgecolor="white", lw=0.4)
    ax.axvline(ed.mean(), color=EST, lw=1.2)
    ax.axvline(10, color="black", lw=0.8, ls=":")
    top = ax.get_ylim()[1]
    ax.text(ed.mean() + 0.2, top * 0.92, f"mean {ed.mean():.1f} pixels", color=EST, fontsize=7)
    ax.text(10.2, top * 0.6, "10 pixels", fontsize=7)
    ax.set_xlabel("miss at each annotated point (pixels)")
    ax.set_ylabel("annotated points")
    save_panel(fig, "2", "E")
    layout_sheet("2")


# ----------------------------------------------------------------- Figure 3
def _tile_panel(ax, cc, pair, edge):
    """A tile, with the positions of the 10,000 correspondences it returned, coloured by
    whether the matcher's certainty cleared the sampler's cut-off."""
    x0, y0, s = int(cc["tile_x0"]), int(cc["tile_y0"]), int(cc["tile_size"])
    tile = pair.source[y0:y0 + s, x0:x0 + s]
    f = show(ax, tile, BIG)
    xy = (cc["src_xy"] - np.array([x0, y0])) * f
    above = cc["conf"] >= 0.999  # romatch sets certainty above its cut-off to exactly 1
    ax.plot(xy[~above, 0], xy[~above, 1], "o", ms=1.6, color=BELOW, mew=0, alpha=0.6)
    ax.plot(xy[above, 0], xy[above, 1], "o", ms=1.6, color=ABOVE, mew=0, alpha=0.8)
    fit_to_image(ax)
    for sp in ax.spines.values():
        sp.set_edgecolor(edge); sp.set_linewidth(4.0)
    return float(above.mean())


def fig3() -> None:
    """Why pooling tiles fails: every tile returns 10,000 correspondences."""
    from cma.pyramid import build
    ci = np.load(L.cache_path("tile_inside"), allow_pickle=False)
    co = np.load(L.cache_path("tile_outside"), allow_pickle=False)
    pair, rec = L.load(str(ci["pair_id"]))
    tiles = build(pair.source, pair.scale_ratio, tile_size=int(min(pair.target.shape[:2])), overlap=0.5)
    # A: the wide image, its full-resolution tiles, the two tiles of C and D, the true outline
    lvl0 = [t for t in tiles if t.level == 0]
    print(f"Figure 3A: {len(tiles)} tiles, {len(lvl0)} at full resolution (both numbers are in the legend)")
    fig, ax = image_axes()
    f = show(ax, pair.source, BIG, clahe=True)
    for t in lvl0:
        ax.add_patch(Rectangle((t.x0 * f, t.y0 * f), t.tile_size * f, t.tile_size * f,
                               fill=False, ec="white", lw=0.5, alpha=0.7))
    for cc, col in ((ci, IN_T), (co, OUT_T)):
        s = int(cc["tile_size"]) * f
        ax.add_patch(Rectangle((int(cc["tile_x0"]) * f, int(cc["tile_y0"]) * f), s, s, fill=False, ec=col, lw=2.4,
                               clip_on=False))  # the outside tile ends on the image's lower edge
    outline_on(ax, L.gt_homography(pair), pair.target.shape, f, TRUE, lw=2.2)
    fit_to_image(ax)  # edge tiles overhang the image; drawing past it only added a blank strip
    save_image(fig, ax, "3", "A")
    # B: the narrow image
    fig, ax = image_axes()
    show(ax, pair.target, BIG)
    save_image(fig, ax, "3", "B")
    # C, D: one tile inside the true outline, one outside it
    for cc, col, letter in ((ci, IN_T, "C"), (co, OUT_T, "D")):
        fig, ax = image_axes()
        _tile_panel(ax, cc, pair, col)
        save_image(fig, ax, "3", letter)
    # E: share of correspondences robust fitting keeps, across the 33 tiled pairs
    tiles_csv = {r["pair_id"]: r for r in csv.DictReader((RES / "pyramid_v1_tiles.csv").open(encoding="utf-8"))}
    rows = [r for r in csv.DictReader((RES / "baselines_A.csv").open(encoding="utf-8")) if r["backbone"] == "roma"]
    d = {r["pair_id"]: r for r in rows if r["mode"] == "direct"}
    p = {r["pair_id"]: r for r in rows if r["mode"] == "pyramid"}
    ids = [i for i, t in tiles_csv.items() if t["v1_status"] == "evaluated" and int(t["n_tiles"]) > 1]
    assert len(ids) == NUM["v1"]["tiled"]["n"]
    fd = np.array([int(d[i]["n_inliers"]) / int(d[i]["n_matches"]) for i in ids])
    fp = np.array([int(p[i]["n_inliers"]) / int(p[i]["n_matches"]) for i in ids])
    fig, ax = plot_axes(COL_MM, 70)
    for a, b in zip(fd, fp):
        ax.plot([0, 1], [a, b], color=POOLED, lw=0.6, alpha=0.6)
    ax.plot(np.zeros_like(fd), fd, "o", ms=3, color=DIRECT)
    ax.plot(np.ones_like(fp), fp, "o", ms=3, color=POOLED)
    ax.set_yscale("log"); ax.set_xlim(-0.5, 1.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["whole image\nmatched once", "tiles\npooled"])
    ax.set_ylabel("share kept by robust fitting")
    save_panel(fig, "3", "E")
    # F: error, whole image vs pooled tiles
    ed_d = np.array([float(d[i]["mu_ed"]) for i in ids])
    ed_p = np.array([float(p[i]["mu_ed"]) for i in ids])
    fig, ax = plot_axes(COL_MM, 76)
    lim = [1, 2e4]
    ax.plot(lim, lim, color=GREY, lw=0.6, ls="--")
    ax.plot(ed_d, ed_p, "o", ms=3.4, mfc=POOLED, mec="black", mew=0.3)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("error, whole image matched once (pixels)")
    ax.set_ylabel("error, tiles pooled (pixels)")
    save_panel(fig, "3", "F")
    layout_sheet("3")


# ----------------------------------------------------------------- Figure 4
GALLERY = [  # (letters of the row, display re-run, second re-run or None); rows are described in the legend
    ("ABC", "gal_appearance", None),   # appearance
    ("DEF", "gal_fov", None),          # field of view
    ("GHI", "gal_c103_zs", "gal_c103_ft"),  # forgetting
]


def _outlines_panel(ax, pair, H, max_side=BIG):
    f = show(ax, pair.source, max_side)
    outline_on(ax, L.gt_homography(pair), pair.target.shape, f, TRUE, lw=2.6)
    outline_on(ax, H, pair.target.shape, f, EST, ls="--", lw=2.0)
    h, w = pair.source.shape[:2]
    ax.set_xlim(-0.02 * w * f, 1.02 * w * f); ax.set_ylim(1.02 * h * f, -0.02 * h * f)


def _same_area_panel(ax, pair):
    """The part of the wide image that the narrow image shows, at the wide image's own
    pixel size (nearest-neighbour, so its pixels stay visible)."""
    poly = L.outline(L.gt_homography(pair), pair.target.shape)[:4]
    x0, y0 = np.floor(poly.min(0)).astype(int)
    x1, y1 = np.ceil(poly.max(0)).astype(int)
    h, w = pair.source.shape[:2]
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    crop = pair.source[y0:y1, x0:x1]
    d, _ = L.display(crop, 10_000)
    ax.imshow(d, cmap="gray", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    return x1 - x0, y1 - y0


def fig4() -> None:
    """Examples of failure: wide image with the true and the estimated outline, a second view,
    and the narrow image. The numbers the legend quotes for each row are printed here."""
    for letters, job, job2 in GALLERY:
        c = np.load(L.cache_path(job), allow_pickle=False)
        pid = str(c["pair_id"])
        pair, rec = L.load(pid)
        fig, ax = image_axes()
        _outlines_panel(ax, pair, c["H"])
        save_image(fig, ax, "4", letters[0])
        fig, ax = image_axes()
        if job2:
            c2 = np.load(L.cache_path(job2), allow_pickle=False)
            _outlines_panel(ax, pair, c2["H"])
            second = f"after fine-tuning, error {float(c2['mu_ed']):.0f} pixels"
        else:
            cw, ch = _same_area_panel(ax, pair)
            second = f"the narrow image's area in the wide image: {cw} x {ch} pixels"
        save_image(fig, ax, "4", letters[1])
        fig, ax = image_axes()
        show(ax, pair.target, BIG)
        save_image(fig, ax, "4", letters[2])
        print(f"Figure 4{letters}: {c['matcher']} error {float(c['mu_ed']):.0f} pixels; {second}; narrow = "
              f"{fov_label(pid)} of the wide area, {pair.target.shape[1]} x {pair.target.shape[0]} pixels; "
              f"{L.modality_label(rec, 'source')} / {L.modality_label(rec, 'target')}")
    layout_sheet("4")


# ----------------------------------------------------------------- Figure 5
LABELS = {
    "sift": "SIFT", "sift_mi": "SIFT + mutual information", "loftr": "LoFTR",
    "ma_eloftr": "MatchAnything-ELoFTR", "roma": "RoMa", "roma_v2": "RoMa + checked search",
    "ma_roma": "MatchAnything-RoMa", "ma_roma_v2": "MatchAnything-RoMa + checked search",
}


def fig5() -> None:
    """How many of the 187 pairs each method registers, against the ground-truth ceiling."""
    tab, orc = NUM["table187"], NUM["oracle"]
    keys = list(LABELS)[::-1]
    for letter, t, ceil in (("A", 10, orc["either_k10"]), ("B", 20, orc["either_k20"])):
        # page width, stacked, each with its own method labels: the panels are separate files
        fig, ax = plot_axes(FULL_MM, 58)
        y = np.arange(len(keys))
        k = np.array([tab[key][f"k{t}"] for key in keys])
        lo, hi = zip(*(wilson(x, 187) for x in k))
        cols = [CHECKED if key.endswith("_v2") else DIRECT for key in keys]
        ax.barh(y, k, color=cols, height=0.66)
        ax.errorbar(k, y, xerr=[k - 187 * np.array(lo), 187 * np.array(hi) - k], fmt="none",
                    ecolor="black", elinewidth=0.6, capsize=1.5)
        for yi, ki, h in zip(y, k, hi):
            ax.text(187 * h + 2, yi, f"{ki}", fontsize=6.5, va="center")
        ax.axvline(ceil, color="black", lw=0.9, ls="--")
        ax.text(ceil + 1.5, len(keys) - 0.6, f"best any global\ntransform can do: {ceil}", fontsize=6.8, va="top")
        ax.set_xlim(0, 187)
        ax.set_xlabel(f"pairs registered within {t} pixels (of 187)")
        ax.set_yticks(y)
        ax.set_yticklabels([LABELS[k] for k in keys])
        save_panel(fig, "5", letter)
    layout_sheet("5")


# ----------------------------------------------------------------- Figure 6
LADDER_PAIR = "eval_CoNi_SEM-SE_SEM-BSE_Cracks-SameSliceSerialSectioning_0#0"
RUNGS = [0.5, 0.25, 0.1, 0.05, 0.02]


def fig6() -> None:
    """Field-of-view ladder: crop the narrow image, hold everything else fixed."""
    from cma.data.fov_ladder import crop_target_to_area_ratio
    pair, rec = L.load(LADDER_PAIR)
    meta = {r["pair_id"]: float(r["fov_area_ratio"]) for r in csv.DictReader((RES / "fov_ratios.csv").open(encoding="utf-8"))}
    fig, axa = image_axes()
    f = show(axa, pair.source, BIG)
    Hgt = L.gt_homography(pair)
    outline_on(axa, Hgt, pair.target.shape, f, TRUE, lw=2.0)
    for i, rung in enumerate(RUNGS):
        cut = crop_target_to_area_ratio(pair, meta[LADDER_PAIR], rung)
        x0, y0, cw, ch = cut.pair.metadata["fov_ladder_crop"]
        box = np.array([[x0, y0], [x0 + cw, y0], [x0 + cw, y0 + ch], [x0, y0 + ch], [x0, y0]], float)
        pb = L.apply_h(Hgt, box) * f
        axa.plot(pb[:, 0], pb[:, 1], color=TRUE, lw=1.3)
        # labels on a leader to the right, one row per rung, so small rungs stay legible
        tx, ty = pb[:, 0].max(), pb[:, 1].min()
        lx, ly = axa.get_xlim()[1] * 0.99, axa.get_ylim()[1] + (i + 0.6) * 0.085 * abs(axa.get_ylim()[0] - axa.get_ylim()[1])
        axa.annotate(f"{rung:g}", xy=(tx, ty), xytext=(lx, ly), fontsize=8, color="black", ha="right",
                     va="center", bbox=dict(boxstyle="round,pad=0.15", fc=TRUE, ec="none"),
                     arrowprops=dict(arrowstyle="-", color=TRUE, lw=0.9, shrinkA=0, shrinkB=0))
    save_image(fig, axa, "6", "A")
    # B: MatchAnything-RoMa, base-matchable pairs, direct vs checked search
    ladder = list(csv.DictReader((RES / "fov_ladder.csv").open(encoding="utf-8")))
    base = [r for r in csv.DictReader((RES / "baselines_A.csv").open(encoding="utf-8"))
            if r["backbone"] == "ma_roma" and r["mode"] == "direct"]
    testbed = {r["pair_id"] for r in ladder}
    mu = lambda r: float(r["mu_ed"]) if r["status"] == "ok" and r["mu_ed"] else np.inf  # noqa: E731
    match = {r["pair_id"] for r in base if mu(r) < 20 and r["pair_id"] in testbed}
    assert len(match) == NUM["ladder"]["base_matchable"]["ma_roma"]  # the legend's "41 pairs"
    fig, axb = plot_axes(COL_MM, 70)
    x = np.arange(len(RUNGS) + 1)
    for mode, col, lab, dx in (("direct", DIRECT, "matcher used directly", -0.06),
                               ("pyramid_v2", CHECKED, "with the checked search", 0.06)):
        ks, ns = [], []
        nat = [r for r in csv.DictReader((RES / "baselines_A.csv").open(encoding="utf-8"))
               if r["backbone"] == "ma_roma" and r["mode"] == mode and r["pair_id"] in match]
        ks.append(sum(mu(r) < 10 for r in nat)); ns.append(len(nat))
        for rung in RUNGS:
            rr = [r for r in ladder if r["backbone"] == "ma_roma" and r["mode"] == mode and float(r["rung"]) == rung
                  and r["status"] != "skipped" and r["pair_id"] in match]
            ks.append(sum(mu(r) < 10 for r in rr)); ns.append(len(rr))
        ks, ns = np.array(ks), np.array(ns)
        rate = ks / ns
        lo, hi = zip(*(wilson(k, n) for k, n in zip(ks, ns)))
        axb.errorbar(x + dx, rate, yerr=[rate - np.array(lo), np.array(hi) - rate], color=col, marker="o",
                     ms=3.5, lw=1.2, capsize=2, elinewidth=0.7, label=lab)
        if mode == "direct":
            axb.text(3 + dx - 0.12, rate[3] - 0.06, f"{ks[3]} of {ns[3]}", color=col, fontsize=7, ha="right")
        else:
            axb.text(3 + dx + 0.1, rate[3] + 0.02, f"{ks[3]} of {ns[3]}", color=col, fontsize=7)
    axb.set_xticks(x); axb.set_xticklabels(["uncropped"] + [f"{r:g}" for r in RUNGS])
    axb.set_xlabel("cropped narrow area / wide area")
    axb.set_ylabel("fraction registered within 10 pixels")
    axb.set_ylim(0, 1)
    axb.legend(frameon=False, loc="upper right")
    save_panel(fig, "6", "B")
    layout_sheet("6")


# ----------------------------------------------------------------- Figure 7
def fig7() -> None:
    """Small fields of view and appearance differences come together."""
    gt = {r["pair_id"]: float(r["fov_area_ratio_gt"]) for r in csv.DictReader((RES / "fov_ratios_gt.csv").open(encoding="utf-8"))}
    nmi = {r["pair_id"]: float(r["nmi"]) for r in csv.DictReader((RES / "appearance_nmi.csv").open(encoding="utf-8"))}
    fig, ax = plot_axes(COL_MM + 6, 76)
    labels = ["under 0.05", "0.05 to 0.25", "0.25 to 0.5", "0.5 and over"]
    x = np.arange(4)
    for key, col, mk, dx in (("roma", DIRECT, "o", -0.15), ("roma_v2", CHECKED, "o", -0.05),
                             ("ma_roma", DIRECT, "s", 0.05), ("ma_roma_v2", CHECKED, "s", 0.15)):
        kn = [tuple(map(int, s.split("/"))) for s in NUM["strata_sr10"][key]]
        rate = np.array([k / n for k, n in kn])
        lo, hi = zip(*(wilson(k, n) for k, n in kn))
        ax.errorbar(x + dx, rate, yerr=[rate - np.array(lo), np.array(hi) - rate], fmt=mk, color=col,
                    mfc=col if mk == "o" else "white", ms=3.5, capsize=1.5, elinewidth=0.7, label=LABELS[key])
    ns = NUM["fov"]["strata_gt"]
    ax.set_xticks(x); ax.set_xticklabels([f"{lab}\n(n = {n})" for lab, n in zip(labels, ns)])
    ax.set_xlabel("narrow image's area as a fraction of the wide image's")
    ax.set_ylabel("fraction registered within 10 pixels")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, fontsize=6.5, loc="upper right", bbox_to_anchor=(1.02, 1.02))
    save_panel(fig, "7", "A")
    fig, ax = plot_axes(COL_MM, 76)
    ids = sorted(gt)
    xs, ys = np.array([gt[i] for i in ids]), np.array([nmi[i] for i in ids])
    ax.scatter(xs, ys, s=7, c=GREY, edgecolors="black", linewidths=0.2)
    for b in (0.05, 0.25, 0.5):
        ax.axvline(b, color=GREY, lw=0.5, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("narrow area / wide area")
    ax.set_ylabel("normalised mutual information\n(higher = more alike)")
    a = NUM["appearance"]
    print(f"Figure 7B: Pearson r = +{a['pearson_r']:.2f}, p = {a['pearson_p']:.3f}, {len(ids)} pairs (legend)")
    save_panel(fig, "7", "B")
    layout_sheet("7")


# ----------------------------------------------------------------- Supplementary
def figS1() -> None:
    """Metadata vs ground-truth field-of-view ratio."""
    rows = list(csv.DictReader((RES / "fov_ratios_gt.csv").open(encoding="utf-8")))
    fig, ax = plt.subplots(figsize=(90 * MM, 80 * MM))
    m = np.array([float(r["meta_area_ratio"]) for r in rows])
    g = np.array([float(r["gt_target_over_source"]) for r in rows])
    ok = np.array([r["metadata_consistent"] == "1" for r in rows])
    ax.plot([1e-4, 10], [1e-4, 10], color=GREY, lw=0.6, ls="--")
    ax.scatter(m[ok], g[ok], s=8, c=DIRECT, label=f"consistent ({ok.sum()} pairs)")
    ax.scatter(m[~ok], g[~ok], s=14, c=EST, marker="D", label=f"inconsistent ({(~ok).sum()} pairs)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("area ratio from the pixel-size metadata")
    ax.set_ylabel("area ratio implied by the annotated points")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "FigureS1")


def figS2() -> None:
    """The mirror-padded single tile of the pool-then-fit implementation."""
    import cv2
    from cma.pyramid import build
    pid = "eval_AF9628-Martensitic_SEM-SE-Stitch_EBSD_SameSlice_0#2"
    pair, rec = L.load(pid)
    tile = build(pair.source, pair.scale_ratio, tile_size=int(min(pair.target.shape[:2])), overlap=0.5)
    assert len(tile) == 1
    t = tile[0].image
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 70 * MM), gridspec_kw={"wspace": 0.08})
    f = show(axes[0], pair.source, 900)
    axes[0].set_title(f"wide image as recorded ({pair.source.shape[1]} x {pair.source.shape[0]} pixels)", fontsize=7)
    ft = show(axes[1], t, 900)
    h, w = pair.source.shape[:2]
    axes[1].add_patch(Rectangle((0, 0), w * ft, h * ft, fill=False, ec=TRUE, lw=1.4))
    axes[1].set_title(f"the one tile the implementation built ({t.shape[1]} x {t.shape[0]} pixels):\n"
                      f"the recorded image (outlined) and mirror copies filling the rest", fontsize=7)
    panel(axes[0], "A"); panel(axes[1], "B")
    save(fig, "FigureS2")


def figS3() -> None:
    """The same eight configurations scored after thin-plate-spline refinement."""
    from metric_sensitivity import err
    rows = {}
    for src in ("baselines_A.csv", "baselines_B.csv"):
        for r in csv.DictReader((RES / src).open(encoding="utf-8")):
            rows.setdefault((r["backbone"], r["mode"]), []).append(r)
    spec = {"sift": ("sift", "direct"), "sift_mi": ("sift", "classical"), "loftr": ("loftr", "direct"),
            "ma_eloftr": ("matchanything", "direct"), "roma": ("roma", "direct"),
            "roma_v2": ("roma", "pyramid_v2"), "ma_roma": ("ma_roma", "direct"), "ma_roma_v2": ("ma_roma", "pyramid_v2")}
    keys = list(LABELS)[::-1]
    fig, ax = plt.subplots(figsize=(FULL * 0.62, 62 * MM))
    y = np.arange(len(keys))
    for dy, metric, col, lab in ((-0.18, "raw", DIRECT, "unrefined (primary)"), (0.18, "tps", "#56B4E9", "after refinement")):
        k = []
        for key in keys:
            rr = rows[spec[key]]
            if key == "sift_mi":
                rr = [r for r in rr if True]
            k.append(sum(err(r, metric) < 10 for r in rr))
        ax.barh(y + dy, k, height=0.34, color=col, label=lab)
    ax.set_yticks(y); ax.set_yticklabels([LABELS[k] for k in keys])
    ax.set_xlabel("pairs registered within 10 pixels (of 187)")
    ax.legend(frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "FigureS3")


FIGS = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5, "fig6": fig6, "fig7": fig7,
        "figS1": figS1, "figS2": figS2, "figS3": figS3}
NEEDS = {"fig2": ["pipeline_5842"], "fig3": ["tile_inside", "tile_outside"],
         "fig4": ["gal_appearance", "gal_fov", "gal_c103_zs", "gal_c103_ft"]}

if __name__ == "__main__":
    for name in (sys.argv[1:] or list(FIGS)):
        missing = [j for j in NEEDS.get(name, []) if not L.cache_path(j).exists()]
        if missing:
            print(f"skip {name}: display re-runs not cached yet ({', '.join(missing)})")
            continue
        FIGS[name]()
