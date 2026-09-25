"""Field-of-view area ratio implied by the ground-truth correspondences.

`scripts/fov_ratios.py` derives each pair's FOV ratio from the pixel sizes in the
benchmark's image metadata. The submitted manuscript described its strata as
"GT-implied" but used that metadata ratio. The two agree on 174 of 187 pairs and
disagree by factors of 2.9 to ~9,400 on 13 pairs in three subsets (AF9628 scene 3,
In718 precipitation-strengthened DIC/EBSD, In718 solution-strengthened SE/EBSD),
where the metadata pixel sizes cannot be reconciled with the annotated points
(AF9628 scene 3: a 10,147 px SEM mosaic at 9,258 nm/px would be 94 mm wide).

This script fits a least-squares affine map target -> source through each pair's
ground-truth points. The determinant of its linear part converts target pixel area
into source pixel area, so the ratio of the two image frames' areas is

    r = |det A| * (w_t * h_t) / (w_s * h_s).

r > 1 means the image the loader calls the target is in fact the larger field. The
FOV area ratio used for strata is the symmetric min(r, 1/r): the narrower field's
area as a fraction of the wider one's, which is what "field-of-view mismatch" means
regardless of which image a pipeline happens to treat as the source.

Writes results/fov_ratios_gt.csv. Usage: python scripts/fov_ratios_gt.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from cma.data import AmalgaMatchLoader
from cma.data.amalgamatch import _load_eval

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "fov_ratios_gt.csv"
# A metadata ratio within this factor of the GT-implied one is "consistent".
CONSISTENT_WITHIN = 1.25


def main() -> None:
    loader = AmalgaMatchLoader(ROOT / "data" / "AmalgaMatch")
    meta_ratio = {}
    with (ROOT / "results" / "fov_ratios.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            meta_ratio[r["pair_id"]] = float(r["fov_area_ratio"])

    rows = []
    for rec in loader.records:
        gt = loader._gt[rec.pair_id]
        data = _load_eval(rec.eval_path)
        names = [p.rsplit("/", 1)[-1] for p in data["image_paths"]]
        ms = data["image_metadata"][names.index(rec.source_path.name)]
        mt = data["image_metadata"][names.index(rec.target_path.name)]
        area_s = ms["Resolution Width"] * ms["Resolution Height"]
        area_t = mt["Resolution Width"] * mt["Resolution Height"]
        X = np.hstack([gt.tgt_xy, np.ones((len(gt), 1))])
        A, *_ = np.linalg.lstsq(X, gt.src_xy, rcond=None)
        r = abs(float(np.linalg.det(A[:2, :2]))) * area_t / area_s
        meta = meta_ratio[rec.pair_id]
        rows.append({
            "pair_id": rec.pair_id,
            "subclass": rec.subclass,
            "meta_area_ratio": f"{meta:.5f}",
            "gt_target_over_source": f"{r:.5f}",
            "fov_area_ratio_gt": f"{min(r, 1.0 / r):.5f}",
            "metadata_consistent": int(1 / CONSISTENT_WITHIN <= r / meta <= CONSISTENT_WITHIN),
        })

    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    bad = [r for r in rows if not r["metadata_consistent"]]
    print(f"wrote {len(rows)} rows to {OUT.relative_to(ROOT)}")
    print(f"metadata inconsistent with GT (beyond x{CONSISTENT_WITHIN}): {len(bad)} pairs")
    for sub in sorted({r["subclass"] for r in bad}):
        sel = [r for r in bad if r["subclass"] == sub]
        f_ = np.median([float(r["gt_target_over_source"]) / float(r["meta_area_ratio"]) for r in sel])
        print(f"  {sub}: {len(sel)} pairs, GT/metadata ratio x{f_:.3g}")


if __name__ == "__main__":
    main()
