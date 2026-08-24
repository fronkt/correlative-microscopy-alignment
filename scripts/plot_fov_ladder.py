"""Controlled FOV-ladder figure (Fig. 4): failure field of view per backbone.

Writes paper/figs/fov_ladder.png at 300 dpi and prints the underlying table.

Error metric
------------
This figure uses the UNREFINED matcher transform error (`mu_ed`), never the
TPS-refined error used by Fig. 2/3/5. That is deliberate and is stated on the
axis labels: the ladder crops the target, so a large share of the evaluation
points fall outside the cropped field of view, and a thin-plate spline
extrapolates unreliably outside its control-point hull (see
src/cma/data/fov_ladder.py). The unrefined global transform is the only metric
that is meaningful at every rung, and it is the metric the manuscript's ladder
headline (0.07 -> 0.23 at rung 0.1) is quoted on.

Population
----------
Each backbone's curve is restricted to the pairs that backbone registers at
native FOV (direct mu_ed < 20 px in baselines_A.csv) and that the ladder sweep
actually covers, so the curves read as "given a matchable pair, at what field
of view does it break". Skipped rungs (a pair whose native ratio is already
below the rung) are excluded; error rows count as failures.

Intervals
---------
Success rates carry 95 % Wilson score intervals (correct near 0 and 1, which
matters because several rungs sit at exactly zero). Median errors carry 95 %
percentile bootstrap intervals using the study's protocol: resample pairs with
replacement, B = 10,000. The wrapper-vs-direct gain annotated at rung 0.1 is
the study's paired bootstrap over per-pair errors, B = 10,000.

Usage: python scripts/plot_fov_ladder.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

B = 10_000
RUNGS = (0.5, 0.25, 0.10, 0.05, 0.02)
X_LABELS = ["native", "0.5", "0.25", "0.1", "0.05", "0.02"]

# (backbone, mode, legend name, short name for the n table, colour, linestyle,
#  marker, filled)
#
# ma_roma_ft is the decoder fine-tune, trained on 131 of the 187 benchmark
# pairs (results/split.json "train"). Most of this ladder testbed is therefore
# in its own training set, so its curves are marked with a dagger and drawn in
# grey: they are a within-training reference, not a held-out result, and no
# claim in the manuscript rests on them.
CONFIGS = [
    ("roma", "direct", "RoMa, direct",
     "RoMa direct", "#0072B2", "--", "o", False),
    ("roma", "pyramid_v2", "RoMa + pyramid v2",
     "RoMa + pyr v2", "#0072B2", "-", "o", True),
    ("ma_roma", "direct", "MatchAnything-RoMa, direct",
     "MA-RoMa direct", "#E69F00", "--", "s", False),
    ("ma_roma", "pyramid_v2", "MatchAnything-RoMa + pyramid v2",
     "MA-RoMa + pyr v2", "#E69F00", "-", "s", True),
    ("ma_roma_ft", "direct",
     "MatchAnything-RoMa fine-tuned, direct $^\\dagger$",
     "MA-RoMa-FT direct $^\\dagger$", "#999999", "--", "^", False),
    ("ma_roma_ft", "pyramid_v2",
     "MatchAnything-RoMa fine-tuned + pyramid v2 $^\\dagger$",
     "MA-RoMa-FT + pyr v2 $^\\dagger$", "#999999", "-", "^", True),
]
BACKBONES = ("roma", "ma_roma", "ma_roma_ft")
INK = "#222222"
OUT = Path("paper/figs/fov_ladder.png")


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    """95 % Wilson score interval; used instead of the normal approximation
    because many rungs here have zero successes."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def median_ci(vals: np.ndarray) -> tuple[float, float, float]:
    """Median and its 95 % percentile bootstrap interval (B = 10,000 resamples
    of the pairs, matching the study's bootstrap protocol)."""
    if vals.size == 0:
        return (float("nan"),) * 3
    med = float(np.median(vals))
    rng = np.random.default_rng(0)
    idx = rng.integers(0, vals.size, size=(B, vals.size))
    boot = np.median(vals[idx], axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return med, float(lo), float(hi)


def mu(r: dict) -> float:
    """Unrefined transform mean endpoint error; non-ok rows count as failures."""
    return float(r["mu_ed"]) if (r["status"] == "ok" and r["mu_ed"]) else float("inf")


with open("results/baselines_A.csv", newline="", encoding="utf-8") as f:
    base_rows = list(csv.DictReader(f))
with open("results/fov_ladder.csv", newline="", encoding="utf-8") as f:
    ladder = list(csv.DictReader(f))

# the ladder testbed (pairs actually swept) keeps every backbone's matchable
# set on the same fixed population
testbed = {r["pair_id"] for r in ladder}
matchable: dict[str, set[str]] = {
    bb: {r["pair_id"] for r in base_rows
         if r["backbone"] == bb and r["mode"] == "direct" and mu(r) < 20
         and r["pair_id"] in testbed}
    for bb in BACKBONES
}


def errs(bb: str, mode: str, rung: float | None) -> np.ndarray:
    """Per-pair errors for one config at one rung (rung=None -> native FOV)."""
    if rung is None:
        sel = [r for r in base_rows if r["backbone"] == bb and r["mode"] == mode
               and r["pair_id"] in matchable[bb]]
    else:
        sel = [r for r in ladder if r["backbone"] == bb and r["mode"] == mode
               and float(r["rung"]) == rung and r["status"] != "skipped"
               and r["pair_id"] in matchable[bb]]
    return np.array([mu(r) for r in sel])


# --------------------------------------------------------------------------
fig = plt.figure(figsize=(12.0, 6.6))
gs = fig.add_gridspec(2, 2, height_ratios=[3.5, 1.0], hspace=0.06, wspace=0.20)
ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
tab1, tab2 = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

x = np.arange(len(X_LABELS))
dodge = np.linspace(-0.13, 0.13, len(CONFIGS))

print(f"{'configuration':>20} {'native':>15}",
      *[f"{r:>15g}" for r in RUNGS])
for ci_, (bb, mode, label, short, colour, ls, mk, filled) in enumerate(CONFIGS):
    srs, sr_lo, sr_hi, meds, md_lo, md_hi, ns, n_fin = [], [], [], [], [], [], [], []
    cells = []
    for rung in (None, *RUNGS):
        e = errs(bb, mode, rung)
        hits = e < 10
        p, (lo, hi) = (hits.mean() if e.size else np.nan), wilson(int(hits.sum()), e.size)
        srs.append(p)
        # clamp: Wilson bounds can land a float epsilon on the wrong side of p
        sr_lo.append(max(p - lo, 0.0))
        sr_hi.append(max(hi - p, 0.0))
        fin = e[np.isfinite(e)]
        m, mlo, mhi = median_ci(fin)
        meds.append(m)
        md_lo.append(max(m - mlo, 0.0))
        md_hi.append(max(mhi - m, 0.0))
        ns.append(int(e.size))
        n_fin.append(int(fin.size))
        cells.append(f"{p:.2f}/{m:6.1f} (n={e.size})")
    print(f"{short.replace(' $^\\dagger$', '*'):>20}", *[f"{c:>15}" for c in cells])

    style = dict(color=colour, linestyle=ls, marker=mk, markersize=5.5, lw=1.5,
                 markerfacecolor=colour if filled else "white",
                 markeredgecolor=colour, label=label)
    ax1.errorbar(x + dodge[ci_], srs, yerr=[sr_lo, sr_hi], ecolor=colour,
                 elinewidth=0.9, capsize=2.0, alpha=0.95, **style)
    ax2.errorbar(x + dodge[ci_], meds, yerr=[md_lo, md_hi], ecolor=colour,
                 elinewidth=0.9, capsize=2.0, alpha=0.95, **style)

    # "numbers analysed" rows underneath each panel
    tab1.text(-0.62, ci_, short, ha="right", va="center", fontsize=7.0,
              color=colour)
    for tax, counts in ((tab1, ns), (tab2, n_fin)):
        for xi, n in zip(x, counts):
            tax.text(xi, ci_, str(n), ha="center", va="center", fontsize=7.4,
                     color=colour)

# --- significance annotation at the rung the manuscript quotes -------------
d = {r["pair_id"]: mu(r) for r in ladder
     if r["backbone"] == "ma_roma" and r["mode"] == "direct"
     and float(r["rung"]) == 0.10 and r["status"] != "skipped"
     and r["pair_id"] in matchable["ma_roma"]}
p_ = {r["pair_id"]: mu(r) for r in ladder
      if r["backbone"] == "ma_roma" and r["mode"] == "pyramid_v2"
      and float(r["rung"]) == 0.10 and r["status"] != "skipped"
      and r["pair_id"] in matchable["ma_roma"]}
ids = sorted(set(d) & set(p_))
ea, eb = np.array([d[i] for i in ids]), np.array([p_[i] for i in ids])
rng = np.random.default_rng(0)
bidx = rng.integers(0, len(ids), size=(B, len(ids)))
delta = float((eb < 10).mean() - (ea < 10).mean())
boot = (eb[bidx] < 10).mean(axis=1) - (ea[bidx] < 10).mean(axis=1)
blo, bhi = np.percentile(boot, [2.5, 97.5])
pval = float((boot <= 0).mean())
print(f"\nrung 0.10, MatchAnything-RoMa, pyramid v2 vs direct: n={len(ids)} "
      f"delta {delta:+.3f} CI [{blo:+.3f}, {bhi:+.3f}] p={pval:.4f}")

xa = 3.0
ax1.annotate("", xy=(xa + dodge[3], (eb < 10).mean()),
             xytext=(xa + dodge[2], (ea < 10).mean()),
             arrowprops=dict(arrowstyle="<->", color=INK, lw=1.1))
ax1.annotate(f"MatchAnything-RoMa at FOV 0.1\n"
             f"$\\Delta$ success rate = {delta:+.3f}\n"
             f"95 % CI [{blo:+.3f}, {bhi:+.3f}], $p$ = {pval:.4f}\n"
             f"paired bootstrap, B = 10,000, $n$ = {len(ids)}",
             xy=(xa + 0.02, 0.5 * ((ea < 10).mean() + (eb < 10).mean())),
             xytext=(-0.35, 0.135), fontsize=7.0, color=INK, va="center",
             ha="left",
             bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#BBBBBB",
                       lw=0.7),
             arrowprops=dict(arrowstyle="-", color="#888888", lw=0.8))

# --------------------------------------------------------------------------
ax1.set_ylabel("success rate at 10 px\n(unrefined matcher transform error)",
               fontsize=9.5)
ax1.set_title("Registration success versus field of view", fontsize=10.5)
ax1.set_ylim(-0.03, 1.0)
ax2.set_ylabel("median registration error (px)\n"
               "(unrefined matcher transform error)", fontsize=9.5)
ax2.set_yscale("log")
ax2.set_title("Median registration error versus field of view", fontsize=10.5)
for ax in (ax1, ax2):
    ax.set_xlim(-0.55, len(X_LABELS) - 0.45)
    ax.set_xticks(x, ["" for _ in X_LABELS])
    ax.grid(color="#EEEEEE", linewidth=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
ax1.legend(fontsize=6.8, loc="upper right", framealpha=0.95)

for tax, cap, lab_ha in ((tab1, "n, pairs evaluated:", "right"),
                         (tab2, "n, pairs with a finite error (same rows):",
                          "left")):
    tax.set_xlim(-0.55, len(X_LABELS) - 0.45)
    tax.set_ylim(len(CONFIGS) - 0.4, -1.25)
    tax.set_xticks(x, X_LABELS, fontsize=9)
    tax.set_yticks([])
    for s in ("top", "right", "left", "bottom"):
        tax.spines[s].set_visible(False)
    tax.tick_params(axis="x", length=0)
    tax.text(-0.62 if lab_ha == "right" else -0.5, -0.85, cap, ha=lab_ha,
             va="center", fontsize=7.4, fontweight="bold", color=INK)
    tax.set_xlabel("target field-of-view area ratio after cropping\n"
                   "(\"native\" = the uncropped pair)", fontsize=9.5)

fig.text(0.5, 0.035,
         "$^\\dagger$ the fine-tuned backbone was trained on 131 of the 187 "
         "benchmark pairs, most of this testbed included; its curves are a "
         "within-training reference, not a held-out result.",
         ha="center", fontsize=7.2, color="#555555")
fig.suptitle("Controlled field-of-view ladder: appearance, modality and pixel "
             "size held fixed, scale swept\n"
             "each backbone restricted to the pairs it registers at native "
             "field of view; error bars are 95 % Wilson score intervals (left) "
             "and 95 % bootstrap percentile intervals (right)",
             fontsize=9.5)
fig.subplots_adjust(left=0.155, right=0.985, top=0.855, bottom=0.155)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=300)
plt.close(fig)
print(f"wrote {OUT}")
