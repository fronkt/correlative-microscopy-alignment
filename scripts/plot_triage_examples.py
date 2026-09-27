"""Fig. 1 for the CJSJ triage paper: the method, and two real pairs chosen by a fixed rule.

Rule (fixed before looking at the pictures): among pairs where the S3-selected registration succeeds and
the best single method fails, the one with the highest S3 ("accepted"); among pairs where the selected
registration fails, the one with the lowest S3 ("flagged"). Each panel shows the wide image, the true
outline of the narrow image (from the homography fitted to the annotated points, yellow) and where the
selected registration placed it (dashed; green if accepted, vermillion if flagged).
Images: AmalgaMatch, Durmaz et al., CC BY 4.0.
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
NICE = {"sift": "SIFT", "loftr": "LoFTR", "roma": "RoMa", "ma_roma": "MatchAnything-RoMa",
        "matchanything": "MatchAnything-ELoFTR"}
TNICE = {"none": "", "invert": ", inverted", "histmatch": ", histogram-matched", "clahe": ", CLAHE",
         "gradmag": ", gradient magnitude"}


def outline(H: np.ndarray, h: int, w: int) -> np.ndarray:
    c = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1], [0, 0]], dtype=float)
    return project(H, c)


def label(cand: str) -> str:
    b, m, t, _ = cand.split("|")
    return NICE.get(b, b) + (" + zoom search" if m == "pyramid_v2" else "") + TNICE.get(t, "")


def schematic(ax) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 1.6)
    texts = ["Image pair\n(wide + narrow)", "15 registrations\n(matchers, zoom search,\nimage transforms)",
             "Label-free scores\nS1 retained fraction\nS2 agreement, S3 both", "Pick the top-scoring\nregistration",
             "Accept, or flag\nfor a human"]
    w, gap = 1.64, 0.4
    for i, t in enumerate(texts):
        x = 0.05 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0.2), w, 1.2, boxstyle="round,pad=0.03", fc="#EEF3F8", ec="0.3", lw=0.6))
        ax.text(x + w / 2, 0.8, t, ha="center", va="center", fontsize=6.8)
        if i:
            ax.annotate("", xy=(x - 0.04, 0.8), xytext=(x - gap + 0.06, 0.8),
                        arrowprops=dict(arrowstyle="->", lw=0.8, color="0.2"))


def panel(ax, loader, recs, pid, cand, H, Hgt, s3, err, accepted) -> None:
    pair = loader.load_pair(recs[pid])
    g = to_gray(pair.source)
    ax.imshow(g, cmap="gray", vmin=np.percentile(g, 1), vmax=np.percentile(g, 99))
    ht, wt = pair.target.shape[:2]
    gt = outline(Hgt, ht, wt)
    pr = outline(H, ht, wt)
    ax.plot(gt[:, 0], gt[:, 1], color="#F0E442", lw=1.4)
    ax.plot(pr[:, 0], pr[:, 1], color="#009E73" if accepted else "#D55E00", lw=1.4, ls="--")
    hs, ws = g.shape
    both = np.vstack([gt, pr])
    both = both[np.all(np.isfinite(both), axis=1)]
    (x0, y0), (x1, y1) = both.min(axis=0), both.max(axis=0)
    if (x1 - x0) * (y1 - y0) < 0.25 * ws * hs:  # small footprint: zoom in around it
        m = 0.6 * max(x1 - x0, y1 - y0, 0.08 * max(ws, hs))
        ax.set_xlim(max(0, x0 - m), min(ws, x1 + m))
        ax.set_ylim(min(hs, y1 + m), max(0, y0 - m))
    else:  # large footprint: show the whole image and the whole outline
        m = 0.03 * max(ws, hs)
        ax.set_xlim(min(0, x0) - m, max(ws, x1) + m)
        ax.set_ylim(max(hs, y1) + m, min(0, y0) - m)
    ax.set_xticks([])
    ax.set_yticks([])
    sub = recs[pid].subclass.replace("_", " ")
    ax.set_title(f"{'Accepted' if accepted else 'Flagged'}: {sub[:48]}", fontsize=6.8)
    ax.text(0.02, 0.02, f"{label(cand)}\nS3 = {s3:.2f}, error = {err:.0f} px", transform=ax.transAxes,
            fontsize=6.3, color="w", va="bottom", bbox=dict(fc="k", alpha=0.55, lw=0, pad=1.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/AmalgaMatch")
    ap.add_argument("--indir", default="results/triage")
    ap.add_argument("--out", default="paper/cjsj/figures")
    args = ap.parse_args()
    ind, out = Path(args.indir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pp = pd.read_csv(ind / "per_pair.csv").set_index("pair_id")
    cand = pd.read_csv(ind / "candidates.csv")
    cand["cand"] = (cand.backbone + "|" + cand["mode"] + "|" + cand["transform"].astype(str) + "|s"
                    + cand.seed.astype(int).astype(str))
    Hof = {(r.pair_id, r.cand): np.array(json.loads(r.H)).reshape(3, 3)
           for r in cand.itertuples() if r.status == "ok" and isinstance(r.H, str)}
    summ = json.loads((ind / "summary.json").read_text(encoding="utf-8"))
    best = summ["H3"]["best_single"]
    # S3 of the selected candidate, from candidates rows is not stored; recompute from per_pair picks
    s3 = pd.read_csv(ind / "per_pair_scores.csv").set_index(["pair_id", "cand"]).S3

    pp["s3_sel"] = [s3.get((p, c), np.nan) for p, c in zip(pp.index, pp.pick_S3)]
    good = pp[(pp.err_pick_S3 <= 20) & (pp.err_best_single > 20)]
    if good.empty:
        good = pp[pp.err_pick_S3 <= 20]
    acc_pid = good.s3_sel.idxmax()
    bad = pp[pp.err_pick_S3 > 20]
    flag_pid = bad.s3_sel.idxmin()

    loader = AmalgaMatchLoader(args.root)
    recs = {r.pair_id: r for r in loader.records}
    fig = plt.figure(figsize=(6.8, 3.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.42, 1])
    schematic(fig.add_subplot(gs[0, :]))
    for k, (pid, accepted) in enumerate([(acc_pid, True), (flag_pid, False)]):
        c = pp.loc[pid, "pick_S3"]
        panel(fig.add_subplot(gs[1, k]), loader, recs, pid, c, Hof[(pid, c)],
              Hof[(pid, "gt|homography|none|s0")], pp.loc[pid, "s3_sel"], pp.loc[pid, "err_pick_S3"], accepted)
    fig.tight_layout()
    fig.savefig(out / "fig1_method_examples.png")
    fig.savefig(out / "fig1_method_examples.pdf")
    print("examples:", acc_pid, "|", flag_pid, "| best single was", best)


if __name__ == "__main__":
    main()
