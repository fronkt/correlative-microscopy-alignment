"""Aggregate baselines CSV into headline tables (paper-protocol metrics).

Per backbone (and per group): mean/median ED, SR@{5,10,20}px on mean ED per
pair, failure counts, FOV-stratified breakdown. Failed pairs count as
non-successes in SR (denominator = all attempted pairs), matching the paper's
treatment of registration failures.

Usage: python scripts/summarize_baselines.py [results/baselines_A.csv]
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/baselines_A.csv")
rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
print(f"{len(rows)} rows from {path}\n")


def fnum(row: dict, key: str) -> float:
    v = row.get(key, "")
    return float(v) if v not in ("", None) else float("nan")


def sr(eds: list[float], thresh: float) -> float:
    arr = np.asarray(eds)
    return float((arr < thresh).mean()) if arr.size else float("nan")


def summarize(rows: list[dict], ed_key: str) -> dict:
    eds = [fnum(r, ed_key) if r["status"] == "ok" else float("inf") for r in rows]
    eds = [e if np.isfinite(e) or e == float("inf") else float("inf") for e in eds]
    finite = [e for e in eds if np.isfinite(e)]
    return {
        "n": len(rows),
        "ok": sum(r["status"] == "ok" for r in rows),
        "med_ed": float(np.median(finite)) if finite else float("nan"),
        "sr5": sr(eds, 5), "sr10": sr(eds, 10), "sr20": sr(eds, 20),
    }


by_backbone: dict[str, list[dict]] = defaultdict(list)
for r in rows:
    key = r["backbone"] if r["mode"] == "direct" else f"{r['backbone']}/{r['mode']}"
    by_backbone[key].append(r)

for ed_key, label in (("mu_ed", "parametric fit"), ("mu_ed_tps", "TPS refined")):
    print(f"=== mean ED per pair, {label} ===")
    print(f"{'backbone':<15} {'n':>4} {'ok':>4} {'med_ED':>10} {'SR@5':>6} {'SR@10':>6} {'SR@20':>6}")
    for bb in sorted(by_backbone):
        # for TPS: pairs without a TPS value fall back to parametric ED
        rws = by_backbone[bb]
        if ed_key == "mu_ed_tps":
            rws = [dict(r, mu_ed_tps=r["mu_ed_tps"] or r["mu_ed"]) for r in rws]
        s = summarize(rws, ed_key)
        print(f"{bb:<15} {s['n']:>4} {s['ok']:>4} {s['med_ed']:>10.2f} "
              f"{s['sr5']:>6.2f} {s['sr10']:>6.2f} {s['sr20']:>6.2f}")
    print()

print("=== SR@10 (TPS, fallback parametric) by group ===")
groups = sorted({r["group"] for r in rows})
print(f"{'group':<30}" + "".join(f"{bb:>16}" for bb in sorted(by_backbone)))
for g in groups:
    cells = []
    for bb in sorted(by_backbone):
        rws = [dict(r, mu_ed_tps=r["mu_ed_tps"] or r["mu_ed"])
               for r in by_backbone[bb] if r["group"] == g]
        s = summarize(rws, "mu_ed_tps")
        cells.append(f"{s['sr10']:>13.2f}/{s['n']:<2}")
    print(f"{g:<30}" + "".join(cells))

fov_path = path.parent / "fov_ratios.csv"
if fov_path.exists():
    area = {r["pair_id"]: float(r["fov_area_ratio"])
            for r in csv.DictReader(fov_path.open(newline="", encoding="utf-8"))}
    label = "area FOV ratio (results/fov_ratios.csv)"
else:
    area = None
    label = "width FOV ratio (CSV fov_ratio column)"

print(f"\n=== SR@10 (TPS) by FOV bin — {label} ===")
bins = [(0.0, 0.05), (0.05, 0.25), (0.25, 0.50), (0.50, 10.0)]
print(f"{'fov bin':<14}" + "".join(f"{bb:>16}" for bb in sorted(by_backbone)))
for lo, hi in bins:
    cells = []
    for bb in sorted(by_backbone):
        rws = []
        for r in by_backbone[bb]:
            ratio = area.get(r["pair_id"]) if area else (
                float(r["fov_ratio"]) if r["fov_ratio"] else None)
            if ratio is not None and lo <= ratio < hi:
                rws.append(dict(r, mu_ed_tps=r["mu_ed_tps"] or r["mu_ed"]))
        s = summarize(rws, "mu_ed_tps")
        cells.append(f"{s['sr10']:>13.2f}/{s['n']:<2}" if rws else f"{'-':>16}")
    print(f"{lo:.2f}-{hi:.2f}    " + "".join(cells))
