"""Fig. 1 for the CJSJ triage paper: the triage procedure and three real MatchAnything-RoMa registrations.

Panels, chosen by a fixed rule from MatchAnything-RoMa's own runs (candidate ma_roma|direct|none|s0):
  (b) the correct registration (error <= 20 px) with the highest retained fraction S1
  (c) the failed registration with the highest S1 (a confident failure)
  (d) the failed registration with the lowest S1 among runs that returned a transform (a flagged failure)
Each panel shows the wide image, the true outline of the narrow image (from the homography fitted to the
annotated points, yellow) and where the registration placed it (dashed). Images: AmalgaMatch, Durmaz et al.,
CC BY 4.0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from cma.data import AmalgaMatchLoader  # noqa: E402
from cma.triage import project, to_gray  # noqa: E402

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 7.5, "savefig.dpi": 300})
PRIMARY = "ma_roma|direct|none|s0"


def outline(H: np.ndarray, h: int, w: int) -> np.ndarray:
    c = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1], [0, 0]], dtype=float)
    return project(H, c)


def plain_title(subclass: str) -> str:
    """'CoNi-AM90_SEM-DIC_EBSD_SlipPartitioning' -> 'CoNi alloy: SEM-DIC and EBSD'."""
    parts = subclass.split("_")
    material = parts[0].split("-")[0]
    mods = [p.replace("-Stitch", "").replace("SEM-SE", "SEM").replace("SEM-BSE", "SEM")
            for p in parts[1:-1] if p not in ("largeFOV",)]
    mods = list(dict.fromkeys(mods))
    kind = ("steel" if material in ("AF9628", "X2CrNi12", "TRIP1", "5842WCu")
            else "MAX-phase ceramic" if material == "Ti3AlC2" else "alloy")
    return f"{material} {kind}: " + (" and ".join(mods) if len(mods) > 1 else f"two {mods[0]} images")


def schematic(ax) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.75)
    texts = ["Batch of image pairs", "Register each pair\n(e.g. MatchAnything-RoMa\n+ robust fitting)",
             "Retained fraction S1\n= inliers / proposed\ncorrespondences", "Sort the batch by S1",
             "Rerun or discard the bottom;\ncheck the top by hand on\nimage kinds not yet validated"]
    w, gap = 1.64, 0.4
    for i, t in enumerate(texts):
        x = 0.05 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0.2), w, 1.2, boxstyle="round,pad=0.03", fc="#EEF3F8", ec="0.3", lw=0.6))
        ax.text(x + w / 2, 0.8, t, ha="center", va="center", fontsize=6.4)
        if i:
            ax.annotate("", xy=(x - 0.04, 0.8), xytext=(x - gap + 0.06, 0.8),
                        arrowprops=dict(arrowstyle="->", lw=0.8, color="0.2"))
    ax.text(0.05, 1.74, "(a)", fontsize=7.5, va="top")


def panel(ax, pair, H, Hgt, s1, err, title, color, letter) -> None:
    g = to_gray(pair.source)
    ax.imshow(g, cmap="gray", vmin=np.percentile(g, 1), vmax=np.percentile(g, 99))
    ht, wt = pair.target.shape[:2]
    gt, pr = outline(Hgt, ht, wt), outline(H, ht, wt)
    ax.plot(gt[:, 0], gt[:, 1], color="#F0E442", lw=1.4)
    ax.plot(pr[:, 0], pr[:, 1], color=color, lw=1.4, ls="--")
    hs, ws = g.shape
    both = np.vstack([gt, pr])
    both = both[np.all(np.isfinite(both), axis=1)]
    (x0, y0), (x1, y1) = both.min(axis=0), both.max(axis=0)
    if (x1 - x0) * (y1 - y0) < 0.25 * ws * hs:
        m = 0.6 * max(x1 - x0, y1 - y0, 0.08 * max(ws, hs))
        ax.set_xlim(max(0, x0 - m), min(ws, x1 + m))
        ax.set_ylim(min(hs, y1 + m), max(0, y0 - m))
    else:
        m = 0.03 * max(ws, hs)
        ax.set_xlim(min(0, x0) - m, max(ws, x1) + m)
        ax.set_ylim(max(hs, y1) + m, min(0, y0) - m)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"({letter}) {title}", fontsize=6.8)
    ax.text(0.02, 0.02, f"S1 = {s1:.2f}, error = {err:.0f} px", transform=ax.transAxes, fontsize=6.3,
            color="w", va="bottom", bbox=dict(fc="k", alpha=0.6, lw=0, pad=1.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/AmalgaMatch")
    ap.add_argument("--indir", default="results/triage")
    ap.add_argument("--out", default="paper/cjsj/figures")
    args = ap.parse_args()
    ind, out = Path(args.indir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cand = pd.read_csv(ind / "candidates.csv")
    cand["cand"] = (cand.backbone + "|" + cand["mode"] + "|" + cand["transform"].astype(str) + "|s"
                    + cand.seed.astype(int).astype(str))
    Hof = {(r.pair_id, r.cand): np.array(json.loads(r.H)).reshape(3, 3)
           for r in cand.itertuples() if r.status == "ok" and isinstance(r.H, str)}
    pp = pd.read_csv(ind / "per_pair.csv").set_index("pair_id")
    ok = pp.err_primary <= 20
    finite = np.isfinite(pp.err_primary)
    picks = [(pp[ok].S1_primary.idxmax(), "Confident and correct", "#009E73", "b"),
             (pp[~ok & finite].S1_primary.idxmax(), "Confident but wrong", "#D55E00", "c"),
             (pp[~ok & finite].S1_primary.idxmin(), "Low score and wrong", "#D55E00", "d")]

    loader = AmalgaMatchLoader(args.root)
    recs = {r.pair_id: r for r in loader.records}
    fig = plt.figure(figsize=(6.8, 3.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.5, 1])
    schematic(fig.add_subplot(gs[0, :]))
    for k, (pid, label, col, letter) in enumerate(picks):
        pair = loader.load_pair(recs[pid])
        panel(fig.add_subplot(gs[1, k]), pair, Hof[(pid, PRIMARY)], Hof[(pid, "gt|homography|none|s0")],
              pp.loc[pid, "S1_primary"], pp.loc[pid, "err_primary"],
              f"{label}\n{plain_title(recs[pid].subclass)}", col, letter)
    fig.tight_layout()
    fig.savefig(out / "fig1_method_examples.png")
    fig.savefig(out / "fig1_method_examples.pdf")
    print("panels:", [p[0] for p in picks])


if __name__ == "__main__":
    main()
