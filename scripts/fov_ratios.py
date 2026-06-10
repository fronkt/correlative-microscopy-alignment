"""Compute per-pair FOV ratios under both definitions (width and area).

The research plan's "FOV <= 5%" gate and the paper's "ratios down to 2%" need
a shared definition. Width ratio = (w_t * px_t) / (w_s * px_s); area ratio
multiplies in the height term. Emits results/fov_ratios.csv for joining into
analysis, and prints the distribution under each definition.

Usage: python scripts/fov_ratios.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from cma.data import AmalgaMatchLoader
from cma.data.amalgamatch import _load_eval

loader = AmalgaMatchLoader("data/AmalgaMatch")

rows = []
for rec in loader.records:
    data = _load_eval(rec.eval_path)
    # image_metadata is parallel to image_paths (names in "Unique Images" do
    # not always match the path basename), so index by position.
    names = [p.rsplit("/", 1)[-1] for p in data["image_paths"]]
    ms = data["image_metadata"][names.index(rec.source_path.name)]
    mt = data["image_metadata"][names.index(rec.target_path.name)]
    w_ratio = (mt["Resolution Width"] * rec.target_pixel_nm) / (
        ms["Resolution Width"] * rec.source_pixel_nm
    )
    h_ratio = (mt["Resolution Height"] * rec.target_pixel_nm) / (
        ms["Resolution Height"] * rec.source_pixel_nm
    )
    rows.append({
        "pair_id": rec.pair_id,
        "group": rec.group,
        "subclass": rec.subclass,
        "fov_width_ratio": f"{w_ratio:.5f}",
        "fov_area_ratio": f"{w_ratio * h_ratio:.5f}",
    })

out = Path("results/fov_ratios.csv")
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {len(rows)} rows to {out}\n")

for key in ("fov_width_ratio", "fov_area_ratio"):
    vals = np.array([float(r[key]) for r in rows])
    print(f"{key}: min {vals.min():.4f}  p10 {np.percentile(vals, 10):.4f}  "
          f"median {np.median(vals):.4f}  max {vals.max():.4f}")
    for t in (0.02, 0.05, 0.10, 0.25):
        print(f"    pairs <= {t:.2f}: {(vals <= t).sum()}")
