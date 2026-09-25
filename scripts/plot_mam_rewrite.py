"""Figures for the Microscopy and Microanalysis rewrite (after MAM-26-246).

    python scripts/plot_mam_rewrite.py            # every figure whose inputs exist
    python scripts/plot_mam_rewrite.py fig1 fig5  # a subset

The editor asked that "examples of the failures are shown". Figures 1-4 therefore show
real AmalgaMatch images (CC-BY-4.0, Durmaz et al.), and Figures 5-7 are deliberately
simpler than the charts they replace: one question per panel, counts written as
"k of n", intervals drawn, nothing a reader has to decode from a second legend.

Inputs: result CSVs, results/mam_rewrite_numbers.json (scripts/mam_rewrite_numbers.py),
and for Figures 2-4 the display re-runs cached by scripts/mam_examples_run.py.
Outputs: paper/mam/figures/Figure<N>.pdf (submission copy) and .png (manuscript copy),
and FigureS<N> for the supplementary material.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

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
            path_effects=_halo())
    return poly


def _halo():
    import matplotlib.patheffects as pe
    return [pe.Stroke(linewidth=2.6, foreground="black", alpha=0.55), pe.Normal()]


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
FIG1 = [
    ("A", "eval_CoNi-AM90_SEM-EBSD_SameSlice_0#0", "Same area, different contrast"),
    ("B", "eval_TRIP1-Bainitic_LOM_SEM_EBSD_Multiscale_0#4", "Different instrument and magnification"),
    ("C", "eval_5842WCu-Spalled_SEM-SE_SEM-BSE_Multiscale_0#0", "A small field inside a large one"),
]


def fig1() -> None:
    """The registration problem, in three real pairs."""
    fig, axes = plt.subplots(2, 3, figsize=(FULL, 122 * MM),
                             gridspec_kw={"height_ratios": [1, 1], "hspace": 0.42, "wspace": 0.1})
    for col, (letter, pid, title) in enumerate(FIG1):
        pair, rec = L.load(pid)
        ax = axes[0, col]
        f = show(ax, pair.source)
        outline_on(ax, L.gt_homography(pair), pair.target.shape, f, TRUE)
        ax.plot(pair.gt.src_xy[:, 0] * f, pair.gt.src_xy[:, 1] * f, "o", ms=2.2, mfc=TRUE, mec="black", mew=0.4)
        ax.set_title(title, fontsize=7.5)
        panel(ax, letter)
        ax.set_xlabel(f"Wide: {L.modality_label(rec, 'source')}", fontsize=6.5, labelpad=2)
        ax = axes[1, col]
        f2 = show(ax, pair.target)
        ax.plot(pair.gt.tgt_xy[:, 0] * f2, pair.gt.tgt_xy[:, 1] * f2, "o", ms=2.2, mfc=TRUE, mec="black", mew=0.4)
        ax.set_xlabel(f"Narrow: {L.modality_label(rec, 'target')}\n{fov_label(pid)} of the wide image's area, "
                      f"{len(pair.gt)} annotated points", fontsize=6.5, labelpad=2)
    save(fig, "Figure1")


# ----------------------------------------------------------------- Figure 2
def fig2() -> None:
    """Registration step by step on one pair: match, fit, score."""
    c = np.load(L.cache_path("pipeline_5842"), allow_pickle=False)
    pair, rec = L.load(str(c["pair_id"]))
    fig = plt.figure(figsize=(FULL, 118 * MM))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], hspace=0.45, wspace=0.42)
    axw, axn = fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2])
    fw = show(axw, pair.source, 1000)
    fn = show(axn, pair.target, 700)
    src, tgt, inl = c["src_xy"], c["tgt_xy"], c["inliers"].astype(bool)
    rng = np.random.default_rng(1)
    pick_in = rng.choice(np.flatnonzero(inl), size=min(60, inl.sum()), replace=False)
    pick_out = rng.choice(np.flatnonzero(~inl), size=min(60, (~inl).sum()), replace=False)
    for idx, col in ((pick_out, EST), (pick_in, KEPT)):
        axw.plot(src[idx, 0] * fw, src[idx, 1] * fw, "o", ms=2.3, mfc=col, mec="black", mew=0.3)
        axn.plot(tgt[idx, 0] * fn, tgt[idx, 1] * fn, "o", ms=2.3, mfc=col, mec="black", mew=0.3)
    axw.set_title(f"Steps 1 and 2: {len(src):,} proposed correspondences (120 drawn); robust fitting\n"
                  f"keeps the {inl.sum():,} consistent with one transform (green) and rejects "
                  f"{(~inl).sum():,} (vermillion)", fontsize=7, loc="left")
    panel(axw, "A")
    axn.set_title(f"narrow image, same points", fontsize=7)
    axn.set_xlabel(L.modality_label(rec, "target"), fontsize=6.5)
    axw.set_xlabel(f"wide image: {L.modality_label(rec, 'source')}", fontsize=6.5)
    # B: fitted outline vs true outline
    axb = fig.add_subplot(gs[1, 0])
    fb = show(axb, pair.source, 700)
    outline_on(axb, L.gt_homography(pair), pair.target.shape, fb, TRUE)
    outline_on(axb, c["H"], pair.target.shape, fb, EST, ls="--", lw=1.2)
    axb.set_title("Step 3: fitted outline (dashed)\nover the true one", fontsize=7)
    panel(axb, "B")
    # C: error vectors at the annotated points
    axc = fig.add_subplot(gs[1, 1])
    gt, proj = pair.gt.src_xy, c["gt_proj"]
    fc = show(axc, pair.source, 1400)
    lim = np.array([gt[:, 0].min(), gt[:, 0].max(), gt[:, 1].min(), gt[:, 1].max()]) * fc
    pad = 0.12 * max(lim[1] - lim[0], lim[3] - lim[2])
    axc.set_xlim(lim[0] - pad, lim[1] + pad); axc.set_ylim(lim[3] + pad, lim[2] - pad)
    gain = 5.0
    for g, p in zip(gt * fc, proj * fc):
        axc.annotate("", xy=g + gain * (p - g), xytext=g,
                     arrowprops=dict(arrowstyle="-|>", color=EST, lw=0.8, mutation_scale=5))
    axc.plot(gt[:, 0] * fc, gt[:, 1] * fc, "o", ms=2.6, mfc=TRUE, mec="black", mew=0.4)
    axc.set_title(f"Step 4: miss at each annotated\npoint (arrows drawn {gain:.0f}x longer)", fontsize=7)
    panel(axc, "C")
    # D: the score
    ed = np.linalg.norm(proj - gt, axis=1)
    axd = fig.add_subplot(gs[1, 2])
    axd.hist(ed, bins=np.arange(0, 13, 1), color=GREY, edgecolor="white", lw=0.4)
    axd.axvline(ed.mean(), color=EST, lw=1.2)
    axd.axvline(10, color="black", lw=0.8, ls=":")
    top = axd.get_ylim()[1]
    axd.text(ed.mean() + 0.2, top * 0.92, f"mean {ed.mean():.1f} pixels", color=EST, fontsize=6.5)
    axd.text(10.2, top * 0.6, "10 pixels", fontsize=6.5)
    axd.set_xlabel("miss at each point (pixels)")
    axd.set_ylabel("annotated points")
    axd.set_title("Step 5: the pair's error is the\nmean; under 10 pixels = registered", fontsize=7)
    axd.spines[["top", "right"]].set_visible(False)
    panel(axd, "D")
    save(fig, "Figure2")


# ----------------------------------------------------------------- Figure 3
def _tile_panel(ax, cc, pair, edge):
    """A tile, with the positions of the 10,000 correspondences it returned, coloured by
    whether the matcher's certainty cleared the sampler's cut-off."""
    x0, y0, s = int(cc["tile_x0"]), int(cc["tile_y0"]), int(cc["tile_size"])
    tile = pair.source[y0:y0 + s, x0:x0 + s]
    f = show(ax, tile, 500)
    xy = (cc["src_xy"] - np.array([x0, y0])) * f
    above = cc["conf"] >= 0.999  # romatch sets certainty above its cut-off to exactly 1
    ax.plot(xy[~above, 0], xy[~above, 1], ",", color="black", alpha=0.5)
    ax.plot(xy[~above, 0], xy[~above, 1], "o", ms=0.8, color=BELOW, mew=0, alpha=0.6)
    ax.plot(xy[above, 0], xy[above, 1], "o", ms=0.8, color=ABOVE, mew=0, alpha=0.8)
    for sp in ax.spines.values():
        sp.set_edgecolor(edge); sp.set_linewidth(2.0)
    return float(above.mean())


def fig3() -> None:
    """Why pooling tiles fails: every tile returns 10,000 correspondences."""
    from cma.pyramid import build
    ci = np.load(L.cache_path("tile_inside"), allow_pickle=False)
    co = np.load(L.cache_path("tile_outside"), allow_pickle=False)
    pair, rec = L.load(str(ci["pair_id"]))
    tiles = build(pair.source, pair.scale_ratio, tile_size=int(min(pair.target.shape[:2])), overlap=0.5)
    fig = plt.figure(figsize=(FULL, 128 * MM))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], width_ratios=[1, 1, 0.9, 1.1], hspace=0.5, wspace=0.75)
    axa = fig.add_subplot(gs[0, 0:2])
    f = show(axa, pair.source, 1000, clahe=True)
    lvl0 = [t for t in tiles if t.level == 0]
    for t in lvl0:
        axa.add_patch(Rectangle((t.x0 * f, t.y0 * f), t.tile_size * f, t.tile_size * f,
                                fill=False, ec="white", lw=0.35, alpha=0.7))
    for cc, col in ((ci, IN_T), (co, OUT_T)):
        s = int(cc["tile_size"]) * f
        axa.add_patch(Rectangle((int(cc["tile_x0"]) * f, int(cc["tile_y0"]) * f), s, s, fill=False, ec=col, lw=1.8))
    outline_on(axa, L.gt_homography(pair), pair.target.shape, f, TRUE)
    axa.set_title(f"The wide image cut into {len(tiles)} tiles ({len(lvl0)} full-\nresolution tiles drawn); "
                  f"true outline of the narrow image", fontsize=7)
    panel(axa, "A")
    axa.set_xlabel(f"{L.modality_label(rec, 'source')}, contrast adjusted", fontsize=6.5)
    axn = fig.add_subplot(gs[0, 2])
    show(axn, pair.target, 600)
    axn.set_title("narrow image", fontsize=7)
    import textwrap
    axn.set_xlabel(textwrap.fill(L.modality_label(rec, "target"), 18), fontsize=6.5)
    # D: inlier fraction across the 33 tiled pairs
    tiles_csv = {r["pair_id"]: r for r in csv.DictReader((RES / "pyramid_v1_tiles.csv").open(encoding="utf-8"))}
    rows = [r for r in csv.DictReader((RES / "baselines_A.csv").open(encoding="utf-8")) if r["backbone"] == "roma"]
    d = {r["pair_id"]: r for r in rows if r["mode"] == "direct"}
    p = {r["pair_id"]: r for r in rows if r["mode"] == "pyramid"}
    ids = [i for i, t in tiles_csv.items() if t["v1_status"] == "evaluated" and int(t["n_tiles"]) > 1]
    fd = np.array([int(d[i]["n_inliers"]) / int(d[i]["n_matches"]) for i in ids])
    fp = np.array([int(p[i]["n_inliers"]) / int(p[i]["n_matches"]) for i in ids])
    axd = fig.add_subplot(gs[0, 3])
    for a, b in zip(fd, fp):
        axd.plot([0, 1], [a, b], color=POOLED, lw=0.5, alpha=0.6)
    axd.plot(np.zeros_like(fd), fd, "o", ms=2.5, color=DIRECT)
    axd.plot(np.ones_like(fp), fp, "o", ms=2.5, color=POOLED)
    axd.set_yscale("log"); axd.set_xlim(-0.4, 1.4)
    axd.set_xticks([0, 1]); axd.set_xticklabels(["whole\nimage", "pooled\ntiles"])
    axd.set_ylabel("share kept by robust fitting")
    t = NUM["v1"]["tiled"]
    axd.set_title(f"All {t['n']} tiled pairs", fontsize=7)
    axd.spines[["top", "right"]].set_visible(False)
    panel(axd, "D")
    # B, C: one tile inside, one outside
    for k, (cc, col, lab) in enumerate(((ci, IN_T, "inside"), (co, OUT_T, "outside"))):
        ax = fig.add_subplot(gs[1, k])
        share = _tile_panel(ax, cc, pair, col)
        ax.set_title(f"Tile {lab} the outline:\n10,000 returned, {100 * share:.0f} % above\nthe certainty cut-off",
                     fontsize=6.8)
        panel(ax, "BC"[k])
    fig.axes[-1].legend(handles=[plt.Line2D([], [], ls="", marker="o", ms=3, color=ABOVE, label="above cut-off"),
                                 plt.Line2D([], [], ls="", marker="o", ms=3, color=BELOW, label="below cut-off")],
                        loc="upper center", bbox_to_anchor=(-0.1, -0.02), ncol=2, frameon=False, fontsize=6.3)
    # E: error, whole image vs pooled tiles
    ed_d = np.array([float(d[i]["mu_ed"]) for i in ids])
    ed_p = np.array([float(p[i]["mu_ed"]) for i in ids])
    axe = fig.add_subplot(gs[1, 2:4])
    lim = [1, 2e4]
    axe.plot(lim, lim, color=GREY, lw=0.6, ls="--")
    axe.plot(ed_d, ed_p, "o", ms=3, mfc=POOLED, mec="black", mew=0.3)
    axe.set_xscale("log"); axe.set_yscale("log"); axe.set_xlim(lim); axe.set_ylim(lim)
    axe.set_xlabel("error, whole image matched once (pixels)")
    axe.set_ylabel("error, tiles pooled (pixels)")
    axe.set_title(f"Pooling raised the error on {t['n_worse']} of {t['n']}\n(points above the dashed line)",
                  fontsize=7)
    axe.spines[["top", "right"]].set_visible(False)
    panel(axe, "E")
    save(fig, "Figure3")


# ----------------------------------------------------------------- Figure 4
GALLERY = [
    ("A", "gal_appearance", None, "Appearance: same area, same microscope, different detector"),
    ("B", "gal_fov", None, "Field of view: the narrow image covers under 2 % of the wide one"),
    ("C", "gal_c103_zs", "gal_c103_ft", "Forgetting: a pair the released matcher registers, before and after fine-tuning"),
]


def _outlines_panel(ax, pair, H, max_side=900):
    f = show(ax, pair.source, max_side)
    outline_on(ax, L.gt_homography(pair), pair.target.shape, f, TRUE)
    outline_on(ax, H, pair.target.shape, f, EST, ls="--", lw=1.2)
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
    """Examples of failure, each with the true and the estimated outline."""
    fig, axes = plt.subplots(3, 3, figsize=(FULL, 160 * MM),
                             gridspec_kw={"hspace": 0.62, "wspace": 0.12})
    for row, (letter, job, job2, title) in enumerate(GALLERY):
        c = np.load(L.cache_path(job), allow_pickle=False)
        pid = str(c["pair_id"])
        pair, rec = L.load(pid)
        name = {"ma_roma": "MatchAnything-RoMa", "roma": "RoMa"}[str(c["matcher"])]
        ax = axes[row, 0]
        _outlines_panel(ax, pair, c["H"])
        ax.set_title(f"{name}: error {float(c['mu_ed']):.0f} pixels", fontsize=7)
        panel(ax, letter)
        ax.set_xlabel(f"wide: {L.modality_label(rec, 'source')}", fontsize=6.5)
        ax.text(0.0, 1.30, title, transform=ax.transAxes, fontsize=7.5, fontweight="bold")
        ax2 = axes[row, 1]
        if job2:
            c2 = np.load(L.cache_path(job2), allow_pickle=False)
            _outlines_panel(ax2, pair, c2["H"])
            ax2.set_title(f"after fine-tuning: error {float(c2['mu_ed']):.0f} pixels", fontsize=7)
            ax2.set_xlabel("same wide image", fontsize=6.5)
        else:
            cw, ch = _same_area_panel(ax2, pair)
            ax2.set_title("the narrow image's area, as the\nwide image records it", fontsize=7)
            ax2.set_xlabel(f"{cw} x {ch} wide-image pixels", fontsize=6.5)
        ax3 = axes[row, 2]
        show(ax3, pair.target, 700)
        ax3.set_title(f"narrow image ({fov_label(pid)} of the wide\nimage's area)", fontsize=7)
        ax3.set_xlabel(f"{L.modality_label(rec, 'target')}\n{pair.target.shape[1]} x {pair.target.shape[0]} pixels",
                       fontsize=6.5)
    fig.legend(handles=[plt.Line2D([], [], color=TRUE, lw=1.6, label="true outline (ground truth)"),
                        plt.Line2D([], [], color=EST, lw=1.2, ls="--", label="where the matcher put it")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.02))
    save(fig, "Figure4")


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
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 62 * MM), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, t, ceil in ((axes[0], 10, orc["either_k10"]), (axes[1], 20, orc["either_k20"])):
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
        ax.text(ceil + 1.5, len(keys) - 0.6, f"best any global\ntransform can do: {ceil}", fontsize=6.3, va="top")
        ax.set_xlim(0, 187)
        ax.set_xlabel(f"pairs registered within {t} pixels (of 187)")
        ax.spines[["top", "right"]].set_visible(False)
        panel(ax, "AB"[t == 20])
    axes[0].set_yticks(np.arange(len(keys)))
    axes[0].set_yticklabels([LABELS[k] for k in keys])
    save(fig, "Figure5")


# ----------------------------------------------------------------- Figure 6
LADDER_PAIR = "eval_CoNi_SEM-SE_SEM-BSE_Cracks-SameSliceSerialSectioning_0#0"
RUNGS = [0.5, 0.25, 0.1, 0.05, 0.02]


def fig6() -> None:
    """Field-of-view ladder: crop the narrow image, hold everything else fixed."""
    from cma.data.fov_ladder import crop_target_to_area_ratio
    pair, rec = L.load(LADDER_PAIR)
    meta = {r["pair_id"]: float(r["fov_area_ratio"]) for r in csv.DictReader((RES / "fov_ratios.csv").open(encoding="utf-8"))}
    fig = plt.figure(figsize=(FULL, 72 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.3], wspace=0.28)
    axa = fig.add_subplot(gs[0, 0])
    f = show(axa, pair.source, 900)
    Hgt = L.gt_homography(pair)
    outline_on(axa, Hgt, pair.target.shape, f, TRUE, lw=1.4)
    for i, rung in enumerate(RUNGS):
        cut = crop_target_to_area_ratio(pair, meta[LADDER_PAIR], rung)
        x0, y0, cw, ch = cut.pair.metadata["fov_ladder_crop"]
        box = np.array([[x0, y0], [x0 + cw, y0], [x0 + cw, y0 + ch], [x0, y0 + ch], [x0, y0]], float)
        pb = L.apply_h(Hgt, box) * f
        axa.plot(pb[:, 0], pb[:, 1], color=TRUE, lw=0.9)
        # labels on a leader to the right, one row per rung, so small rungs stay legible
        tx, ty = pb[:, 0].max(), pb[:, 1].min()
        lx, ly = axa.get_xlim()[1] * 0.99, axa.get_ylim()[1] + (i + 0.6) * 0.085 * abs(axa.get_ylim()[0] - axa.get_ylim()[1])
        axa.annotate(f"{rung:g}", xy=(tx, ty), xytext=(lx, ly), fontsize=6, color="black", ha="right",
                     va="center", bbox=dict(boxstyle="round,pad=0.15", fc=TRUE, ec="none"),
                     arrowprops=dict(arrowstyle="-", color=TRUE, lw=0.6, shrinkA=0, shrinkB=0))
    axa.set_title("Crops of the narrow image, drawn where\nthey lie in the wide image", fontsize=7)
    panel(axa, "A")
    # B: MatchAnything-RoMa, base-matchable pairs, direct vs checked search
    ladder = list(csv.DictReader((RES / "fov_ladder.csv").open(encoding="utf-8")))
    base = [r for r in csv.DictReader((RES / "baselines_A.csv").open(encoding="utf-8"))
            if r["backbone"] == "ma_roma" and r["mode"] == "direct"]
    testbed = {r["pair_id"] for r in ladder}
    mu = lambda r: float(r["mu_ed"]) if r["status"] == "ok" and r["mu_ed"] else np.inf  # noqa: E731
    match = {r["pair_id"] for r in base if mu(r) < 20 and r["pair_id"] in testbed}
    axb = fig.add_subplot(gs[0, 1])
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
            axb.text(3 + dx - 0.12, rate[3] - 0.06, f"{ks[3]} of {ns[3]}", color=col, fontsize=6.5, ha="right")
        else:
            axb.text(3 + dx + 0.1, rate[3] + 0.02, f"{ks[3]} of {ns[3]}", color=col, fontsize=6.5)
    axb.set_xticks(x); axb.set_xticklabels(["uncropped"] + [f"{r:g}" for r in RUNGS])
    axb.set_xlabel("cropped narrow area / wide area")
    axb.set_ylabel("fraction registered within 10 pixels")
    axb.set_ylim(0, 1)
    lad = NUM["ladder"]["ma_roma_r010"]
    axb.set_title(f"MatchAnything-RoMa, on the {NUM['ladder']['base_matchable']['ma_roma']} pairs it\n"
                  f"registers within 20 pixels before cropping", fontsize=7)
    axb.legend(frameon=False, loc="upper right")
    axb.spines[["top", "right"]].set_visible(False)
    panel(axb, "B")
    save(fig, "Figure6")


# ----------------------------------------------------------------- Figure 7
def fig7() -> None:
    """Small fields of view and appearance differences come together."""
    gt = {r["pair_id"]: float(r["fov_area_ratio_gt"]) for r in csv.DictReader((RES / "fov_ratios_gt.csv").open(encoding="utf-8"))}
    nmi = {r["pair_id"]: float(r["nmi"]) for r in csv.DictReader((RES / "appearance_nmi.csv").open(encoding="utf-8"))}
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 64 * MM), gridspec_kw={"width_ratios": [1.1, 1], "wspace": 0.3})
    ax = axes[0]
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
    ax.legend(frameon=False, fontsize=6.3, loc="upper right", bbox_to_anchor=(1.02, 1.02))
    ax.set_title("Success by field-of-view group", fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    panel(ax, "A")
    ax = axes[1]
    ids = sorted(gt)
    xs, ys = np.array([gt[i] for i in ids]), np.array([nmi[i] for i in ids])
    ax.scatter(xs, ys, s=7, c=GREY, edgecolors="black", linewidths=0.2)
    for b in (0.05, 0.25, 0.5):
        ax.axvline(b, color=GREY, lw=0.5, ls=":")
    ax.set_xscale("log")
    ax.set_xlabel("narrow area / wide area")
    ax.set_ylabel("normalised mutual information\n(higher = more alike)")
    a = NUM["appearance"]
    ax.set_title(f"A weak trend: less area in common, less alike\n"
                 f"(Pearson r = +{a['pearson_r']:.2f}, p = {a['pearson_p']:.3f}, {len(ids)} pairs)", fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    panel(ax, "B")
    save(fig, "Figure7")


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
