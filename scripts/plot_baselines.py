"""Baseline figures for the manuscript: SR bars (Fig. 2), per-group heatmap
(Fig. 3), and success vs. native FOV stratum (Fig. 5).

Writes 300-dpi PNGs straight into paper/figs/ so the manuscript figures are
regenerable from the committed CSVs:

    paper/figs/sr_bars.png        Fig. 2, TPS-refined error (declared pipeline)
    paper/figs/sr_bars_raw.png    unrefined matcher error

Usage: python scripts/plot_baselines.py [results/baselines_A.csv] [raw|tps]
The second argument selects the primary metric for the group heatmap and the
FOV-stratum curves; "raw" (default) writes the _raw variants.
    paper/figs/group_heatmap.png  Fig. 3
    paper/figs/fov_curves.png     Fig. 5

Every panel reports its sample size and a 95 % Wilson score interval on each
plotted rate. Wilson (not Wald/normal) is used deliberately: many strata here
sit at or very near zero success, where the Wald interval collapses to zero
width and understates uncertainty.

Usage: python scripts/plot_baselines.py [results/baselines_A.csv]
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
# Publisher requirements, not preferences: Type 3 fonts get a figure bounced, and
# flattened SVG text cannot be edited by a co-author. Without these the PDF
# exports carried Type 3 -- caught by scripts/build_mam_figures.py, not by eye.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np

# --------------------------------------------------------------------------
# Configuration set: EXACTLY the nine rows of Table 1, in Table 1 order, as
# (backbone, mode, source-file key, display name).
#
# Reviewer 2 flagged that the figure and table configuration sets diverged.
# The set is now pinned here rather than derived by filtering the CSV, so the
# two cannot drift apart again.
#
# Deliberately EXCLUDED and why:
#   ma_roma_ft/*              fine-tuned on 131 of the 187 pairs
#                             (results/split.json "train"). Its score over all
#                             187 pairs is train-contaminated (SR@10 0.389 on
#                             its own training pairs vs. 0.250 held out), so it
#                             must never enter an all-pairs aggregate. The
#                             fine-tuning results are reported separately on
#                             the held-out split only (Table 2).
#   matchanything/pyramid,    pyramid wrappers on the weaker ELoFTR backbone;
#   matchanything/pyramid_v2  not Table 1 rows.
#   matchanything_stretch     aspect-ratio ablation; not a Table 1 row.
#   roma/pyramid_v2+z3,       the 4.1c iterated-zoom and certainty-gating
#   roma/pyramid_v2+c50       ablations, reported in the text only.
# --------------------------------------------------------------------------
CONFIGS = [
    ("sift", "direct", "A", "SIFT (Control A)"),
    ("sift", "classical", "B", "SIFT + MI (Control B)"),
    ("loftr", "direct", "A", "LoFTR"),
    ("matchanything", "direct", "A", "MatchAnything-ELoFTR"),
    ("roma", "direct", "A", "RoMa (zero-shot)"),
    ("roma", "pyramid", "A", "RoMa + pyramid v1"),
    ("roma", "pyramid_v2", "A", "RoMa + pyramid v2"),
    ("ma_roma", "direct", "A", "MatchAnything-RoMa"),
    ("ma_roma", "pyramid_v2", "A", "MatchAnything-RoMa + pyramid v2"),
]

# internal task-group tokens -> prose labels shown to the reader
GROUP_LABELS = {
    "DislocationCharacterization": "Dislocation\ncharacterisation",
    "FractureSurfaces": "Fracture\nsurfaces",
    "Multiscale": "Multiscale",
    "SameSlice": "Same slice",
    "SerialSectioning": "Serial\nsectioning",
    "SlipPartitioning": "Slip\npartitioning",
}

# Okabe-Ito colourblind-safe
BLUE, ORANGE, GREEN, INK, GREY = "#0072B2", "#E69F00", "#009E73", "#222222", "#777777"

FOV_BINS = [(0.0, 0.05), (0.05, 0.25), (0.25, 0.5), (0.5, 10.0)]
FOV_LABELS = ["< 0.05", "0.05-0.25", "0.25-0.5", "$\\geq$ 0.5"]

OUT_DIR = Path("paper/figs")


# --------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """95 % Wilson score interval for a binomial proportion.

    Chosen over the normal-approximation (Wald) interval because many strata
    in this study have k = 0 or k = n, where Wald collapses to a zero-width
    interval and understates uncertainty.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def rate_with_err(hits: np.ndarray) -> tuple[float, float, float, int]:
    """-> (point estimate, lower error-bar length, upper error-bar length, n)."""
    n = int(hits.size)
    if n == 0:
        return (float("nan"), 0.0, 0.0, 0)
    k = int(hits.sum())
    p = k / n
    lo, hi = wilson(k, n)
    return p, p - lo, hi - p, n


def ed_tps(r: dict) -> float:
    """Declared pipeline metric: TPS-refined mean endpoint error, falling back
    to the unrefined transform error when the TPS column is blank."""
    if r["status"] != "ok":
        return float("inf")
    v = r["mu_ed_tps"] or r["mu_ed"]
    return float(v) if v else float("inf")


def ed_raw(r: dict) -> float:
    """Unrefined transform mean endpoint error (metric-sensitivity check)."""
    if r["status"] != "ok":
        return float("inf")
    return float(r["mu_ed"]) if r["mu_ed"] else float("inf")


# --------------------------------------------------------------------------
# Which error metric is primary for the heatmap and the FOV-stratum curves.
# The manuscript headlines the unrefined parametric error, because TPS coverage
# is not uniform across configurations (0.000 for Control B, 1.000 for the dense
# RoMa family), so a TPS-scored table compares configurations under different
# metrics. Both variants are generated; pass "tps" to make the refined error
# primary. The success-rate bar figure always emits both.
_METRIC = (sys.argv[2] if len(sys.argv) > 2 else "raw").lower()
if _METRIC not in ("raw", "tps"):
    sys.exit('metric must be "raw" or "tps", got %r' % _METRIC)
PRIMARY = ed_raw if _METRIC == "raw" else ed_tps
SUFFIX = "_raw" if _METRIC == "raw" else ""
NOTE = ("unrefined matcher transform error" if _METRIC == "raw"
        else "TPS-refined error")

path_a = Path(sys.argv[1] if len(sys.argv) > 1 else "results/baselines_A.csv")
path_b = path_a.parent / "baselines_B.csv"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def read(p: Path) -> list[dict]:
    with p.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# Control B (SIFT + mutual-information refinement) lives in its own results
# file; it is pulled in here so the figures cover all nine Table 1 rows rather
# than silently dropping one.
rows = {"A": read(path_a), "B": read(path_b) if path_b.exists() else []}

with (path_a.parent / "fov_ratios.csv").open(newline="", encoding="utf-8") as f:
    fov = {r["pair_id"]: float(r["fov_area_ratio"]) for r in csv.DictReader(f)}

by_cfg: dict[str, list[dict]] = {}
labels: list[str] = []
missing: list[str] = []
for backbone, mode, src, label in CONFIGS:
    sel = [r for r in rows[src] if r["backbone"] == backbone and r["mode"] == mode]
    if not sel:
        missing.append(label)
        continue
    by_cfg[label] = sel
    labels.append(label)
if missing:
    print("WARNING: no data for " + ", ".join(missing) + " -- omitted from the "
          "figures; the caption must state the omission explicitly.")

n_pairs = len({r["pair_id"] for r in by_cfg[labels[0]]})
print(f"configurations plotted: {len(labels)} of {len(CONFIGS)}   pairs: {n_pairs}")


# --- Figure 2: success-rate bars at 5/10/20 px -----------------------------
def sr_bar_figure(metric, out_name: str, metric_note: str) -> None:
    fig, ax = plt.subplots(figsize=(7.8, 6.4))
    y = np.arange(len(labels))[::-1]  # Table 1 order, top to bottom
    h = 0.26
    for i, (t, colour) in enumerate(zip((5, 10, 20), (BLUE, ORANGE, GREEN))):
        pts, los, his = [], [], []
        for lab in labels:
            p, lo, hi, _ = rate_with_err(
                np.array([metric(r) < t for r in by_cfg[lab]]))
            pts.append(p)
            los.append(lo)
            his.append(hi)
        ax.barh(y + (1 - i) * h, pts, height=h * 0.92, color=colour,
                edgecolor=INK, linewidth=0.4,
                label=f"success at {t} px", zorder=2)
        ax.errorbar(pts, y + (1 - i) * h, xerr=[los, his], fmt="none",
                    ecolor=INK, elinewidth=0.9, capsize=2.0, zorder=3)

    ax.set_yticks(y, [f"{lab}\n(n = {len(by_cfg[lab])} pairs)" for lab in labels],
                  fontsize=8.5)
    ax.set_xlabel("success rate (fraction of pairs registered within threshold)\n"
                  + metric_note, fontsize=9.5)
    ax.set_xlim(0, max(0.34, ax.get_xlim()[1]))
    ax.set_title("Registration success on the AmalgaMatch benchmark\n"
                 f"all {n_pairs} pairs; error bars are 95 % Wilson score intervals",
                 fontsize=10.5)
    ax.grid(axis="x", color="#DDDDDD", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=8.5, loc="upper right", framealpha=0.95)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    # M&M asks for charts and diagrams as vector with embedded fonts, and
    # sets 600-900 dpi for line art if raster is used at all. The PDF is the
    # submission copy; the PNG stays for pasting into the manuscript.
    fig.savefig(OUT_DIR / out_name, dpi=300)
    fig.savefig((OUT_DIR / out_name).with_suffix('.pdf'))
    plt.close(fig)
    print(f"wrote {OUT_DIR / out_name}")


sr_bar_figure(ed_tps, "sr_bars.png",
              "error metric: TPS-refined mean endpoint error")
sr_bar_figure(ed_raw, "sr_bars_raw.png",
              "error metric: unrefined matcher transform error (no TPS refinement)")

# --- Figure 3: per-task-group success-rate heatmap -------------------------
group_n: dict[str, int] = {}
for r in by_cfg[labels[0]]:
    group_n[r["group"]] = group_n.get(r["group"], 0) + 1
groups = sorted(group_n, key=lambda g: -group_n[g])

mat = np.full((len(labels), len(groups)), np.nan)
ci: dict[tuple[int, int], tuple[float, float]] = {}
for li, lab in enumerate(labels):
    for gi, g in enumerate(groups):
        hits = np.array([PRIMARY(r) < 10 for r in by_cfg[lab] if r["group"] == g])
        if hits.size:
            mat[li, gi] = hits.mean()
            ci[(li, gi)] = wilson(int(hits.sum()), int(hits.size))

vmax = max(0.35, float(np.nanmax(mat)))
fig, ax = plt.subplots(figsize=(10.2, 6.6))
im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
ax.set_xticks(range(len(groups)),
              [f"{GROUP_LABELS.get(g, g)}\n(n = {group_n[g]} pairs)" for g in groups],
              fontsize=8.5)
ax.set_yticks(range(len(labels)), labels, fontsize=8.5)
for li in range(len(labels)):
    for gi in range(len(groups)):
        if np.isnan(mat[li, gi]):
            continue
        col = "w" if mat[li, gi] < 0.55 * vmax else "k"
        lo, hi = ci[(li, gi)]
        ax.text(gi, li - 0.13, f"{mat[li, gi]:.2f}", ha="center", va="center",
                color=col, fontsize=9)
        ax.text(gi, li + 0.21, f"[{lo:.2f}, {hi:.2f}]", ha="center", va="center",
                color=col, fontsize=6.2)
cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
cb.set_label("success rate at 10 px", fontsize=9)
ax.set_xlabel("task group", fontsize=9.5)
ax.set_title("Success rate at 10 px by task group\n"
             f"{NOTE}; all {n_pairs} pairs; brackets are 95 % Wilson "
             "score intervals", fontsize=10.5)
fig.tight_layout()
fig.savefig(OUT_DIR / f"group_heatmap{SUFFIX}.png", dpi=300)
fig.savefig(OUT_DIR / f"group_heatmap{SUFFIX}.pdf")
plt.close(fig)
print(f"wrote {OUT_DIR / ('group_heatmap%s.png' % SUFFIX)}")

# --- Figure 5: success rate vs. native FOV area-ratio stratum --------------
# Small multiples rather than nine overlaid curves: once every point carries a
# 95 % interval (and the < 0.05 stratum holds only four pairs, so its interval
# is very wide) overlaid curves are unreadable.
bin_n = [sum(1 for r in by_cfg[labels[0]]
             if lo <= fov.get(r["pair_id"], 1.0) < hi) for lo, hi in FOV_BINS]
tick = [f"{lab}\n(n = {n})" for lab, n in zip(FOV_LABELS, bin_n)]

ncol = 3
nrow = int(np.ceil(len(labels) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(9.8, 7.6), sharex=True, sharey=True)
axes = np.atleast_1d(axes).ravel()
x = np.arange(len(FOV_BINS))
for ai, lab in enumerate(labels):
    ax = axes[ai]
    pts, los, his = [], [], []
    for lo_b, hi_b in FOV_BINS:
        hits = np.array([PRIMARY(r) < 10 for r in by_cfg[lab]
                         if lo_b <= fov.get(r["pair_id"], 1.0) < hi_b])
        p, lo, hi, _ = rate_with_err(hits)
        pts.append(p)
        los.append(lo)
        his.append(hi)
    ax.errorbar(x, pts, yerr=[los, his], marker="o", markersize=4.5,
                color=BLUE, ecolor=GREY, elinewidth=1.0, capsize=2.5, lw=1.4)
    ax.set_title(lab, fontsize=8.5)
    ax.grid(color="#EEEEEE", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
for ax in axes[len(labels):]:
    ax.set_visible(False)
axes[0].set_ylim(-0.03, 0.56)
axes[0].set_xticks(x, tick, fontsize=7.5)
for ai in range(len(labels)):
    if ai % ncol == 0:
        axes[ai].set_ylabel("success rate at 10 px", fontsize=9)
    if ai >= len(labels) - ncol:
        axes[ai].set_xlabel("field-of-view area ratio (target / source)",
                            fontsize=8.5)
fig.suptitle("Success rate at 10 px versus native field-of-view stratum\n"
             f"{NOTE}; all {n_pairs} pairs; error bars are 95 % "
             "Wilson score intervals", fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(OUT_DIR / f"fov_curves{SUFFIX}.png", dpi=300)
fig.savefig(OUT_DIR / f"fov_curves{SUFFIX}.pdf")
plt.close(fig)
print(f"wrote {OUT_DIR / ('fov_curves%s.png' % SUFFIX)}")
