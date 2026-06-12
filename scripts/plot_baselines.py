"""Baseline figures (todo 1.5): SR bars, per-group heatmap, FOV curves.

Writes PNGs to reports/figs/baselines/ (gitignored artifacts; regenerate
from the committed CSVs at any time).

Usage: python scripts/plot_baselines.py [results/baselines_A.csv]
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/baselines_A.csv")
out_dir = Path("reports/figs/baselines")
out_dir.mkdir(parents=True, exist_ok=True)

with path.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
fov_path = path.parent / "fov_ratios.csv"
with fov_path.open(newline="", encoding="utf-8") as f:
    fov = {r["pair_id"]: float(r["fov_area_ratio"]) for r in csv.DictReader(f)}


def method(r: dict) -> str:
    return r["backbone"] if r["mode"] == "direct" else f"{r['backbone']}/{r['mode']}"


def ed(r: dict) -> float:
    if r["status"] != "ok":
        return float("inf")
    v = r["mu_ed_tps"] or r["mu_ed"]
    return float(v) if v else float("inf")


by_method: dict[str, list[dict]] = defaultdict(list)
for r in rows:
    # the 4.1c tag variants (pyramid_v2+z3 / +c50) were rejected; keep
    # figures to the un-tagged configurations
    if "+" not in r["mode"]:
        by_method[method(r)].append(r)
methods = sorted(by_method, key=lambda m: np.median([ed(r) for r in by_method[m]]))

# --- Figure 1: SR bars at 5/10/20 px --------------------------------------
fig, ax = plt.subplots(figsize=(9, 4.5))
x = np.arange(len(methods))
for i, t in enumerate((5, 10, 20)):
    srs = [np.mean([ed(r) < t for r in by_method[m]]) for m in methods]
    ax.bar(x + (i - 1) * 0.27, srs, width=0.25, label=f"SR@{t}px")
ax.set_xticks(x, methods, rotation=20, ha="right")
ax.set_ylabel("success rate (mean ED per pair, TPS)")
ax.set_title("AmalgaMatch zero-shot + pyramid baselines (n=187 pairs)")
ax.legend()
fig.tight_layout()
fig.savefig(out_dir / "sr_bars.png", dpi=150)

# --- Figure 2: per-group SR@10 heatmap -------------------------------------
groups = sorted({r["group"] for r in rows})
mat = np.zeros((len(groups), len(methods)))
for gi, g in enumerate(groups):
    for mi, m in enumerate(methods):
        sel = [r for r in by_method[m] if r["group"] == g]
        mat[gi, mi] = np.mean([ed(r) < 10 for r in sel]) if sel else np.nan
fig, ax = plt.subplots(figsize=(9, 4.5))
im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=max(0.35, np.nanmax(mat)))
ax.set_xticks(range(len(methods)), methods, rotation=20, ha="right")
ax.set_yticks(range(len(groups)), groups)
for gi in range(len(groups)):
    for mi in range(len(methods)):
        ax.text(mi, gi, f"{mat[gi, mi]:.2f}", ha="center", va="center",
                color="w" if mat[gi, mi] < 0.2 else "k", fontsize=8)
fig.colorbar(im, label="SR@10px")
ax.set_title("SR@10px by task group")
fig.tight_layout()
fig.savefig(out_dir / "group_heatmap.png", dpi=150)

# --- Figure 3: SR@10 vs FOV area-ratio bin ---------------------------------
# curated to the narrative methods; the full set makes the panel unreadable
FOV_METHODS = [
    "sift", "matchanything", "roma", "roma/pyramid",
    "roma/pyramid_v2", "ma_roma", "ma_roma/pyramid_v2",
]
bins = [(0.0, 0.05), (0.05, 0.25), (0.25, 0.5), (0.5, 10.0)]
labels = ["<0.05", "0.05-0.25", "0.25-0.5", ">=0.5"]
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for m in [m for m in FOV_METHODS if m in by_method]:
    ys = []
    for lo, hi in bins:
        sel = [r for r in by_method[m] if lo <= fov.get(r["pair_id"], 1.0) < hi]
        ys.append(np.mean([ed(r) < 10 for r in sel]) if sel else np.nan)
    ax.plot(labels, ys, marker="o", label=m)
ax.set_xlabel("FOV area ratio (target/source)")
ax.set_ylabel("SR@10px")
ax.set_title("Success vs FOV mismatch severity")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(out_dir / "fov_curves.png", dpi=150)

print(f"wrote 3 figures to {out_dir}")
