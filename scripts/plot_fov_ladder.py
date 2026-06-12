"""FOV ladder analysis (Aim 3): failure-FOV curves per backbone/mode.

Uses transform-based mu_ed ONLY (TPS extrapolates unreliably at
out-of-crop GT points — see src/cma/data/fov_ladder.py). Each backbone's
curve is restricted to the pairs that backbone registers at base FOV
(direct mu_ed < 20 px in baselines_A.csv), so curves read as "given a
matchable pair, at what FOV does it break". Skipped rungs (pair's base
ratio below the rung) are excluded; error rows count as failures.

Usage: python scripts/plot_fov_ladder.py
Writes reports/figs/baselines/fov_ladder.png and prints the table.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RUNGS = (0.5, 0.25, 0.10, 0.05, 0.02)
CONFIGS = [("roma", "direct"), ("roma", "pyramid_v2"),
           ("ma_roma", "direct"), ("ma_roma", "pyramid_v2")]


def mu(r: dict) -> float:
    return float(r["mu_ed"]) if (r["status"] == "ok" and r["mu_ed"]) else float("inf")


with open("results/baselines_A.csv", newline="", encoding="utf-8") as f:
    base_rows = list(csv.DictReader(f))
with open("results/fov_ladder.csv", newline="", encoding="utf-8") as f:
    ladder = list(csv.DictReader(f))

# per-backbone matchable set + base-FOV stats over that set
matchable: dict[str, set[str]] = {}
base_stat: dict[tuple[str, str], tuple[float, float]] = {}
for bb in ("roma", "ma_roma"):
    matchable[bb] = {r["pair_id"] for r in base_rows
                     if r["backbone"] == bb and r["mode"] == "direct" and mu(r) < 20}
    for mode in ("direct", "pyramid_v2"):
        sel = [r for r in base_rows
               if r["backbone"] == bb and r["mode"] == mode
               and r["pair_id"] in matchable[bb]]
        eds = np.array([mu(r) for r in sel])
        fin = eds[np.isfinite(eds)]
        base_stat[(bb, mode)] = (float(np.mean(eds < 10)),
                                 float(np.median(fin)) if len(fin) else np.nan)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
print(f"{'config':>20} {'base':>11}", *[f"{r:>11g}" for r in RUNGS])
for bb, mode in CONFIGS:
    label = bb if mode == "direct" else f"{bb}/{mode}"
    srs, meds = [base_stat[(bb, mode)][0]], [base_stat[(bb, mode)][1]]
    cells = [f"{srs[0]:.2f}/{meds[0]:6.1f}"]
    for rung in RUNGS:
        sel = [r for r in ladder
               if r["backbone"] == bb and r["mode"] == mode
               and float(r["rung"]) == rung and r["status"] != "skipped"
               and r["pair_id"] in matchable[bb]]
        eds = np.array([mu(r) for r in sel])
        fin = eds[np.isfinite(eds)]
        sr = float(np.mean(eds < 10)) if len(eds) else np.nan
        med = float(np.median(fin)) if len(fin) else np.nan
        srs.append(sr)
        meds.append(med)
        cells.append(f"{sr:.2f}/{med:6.1f} (n={len(sel)})")
    print(f"{label:>20}", *[f"{c:>11}" for c in cells])
    x = ["base", *[f"{r:g}" for r in RUNGS]]
    ax1.plot(x, srs, marker="o", label=label)
    ax2.plot(x, meds, marker="o", label=label)

ax1.set_xlabel("target FOV area ratio (cropped)")
ax1.set_ylabel("SR@10px (transform mu_ed)")
ax1.set_title("Failure FOV on base-matchable pairs")
ax1.legend(fontsize=8)
ax2.set_xlabel("target FOV area ratio (cropped)")
ax2.set_ylabel("median mu_ed (px)")
ax2.set_yscale("log")
ax2.set_title("Median error vs FOV")
ax2.legend(fontsize=8)
fig.tight_layout()
out = Path("reports/figs/baselines/fov_ladder.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=150)
print(f"wrote {out}")
