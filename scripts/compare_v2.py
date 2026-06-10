"""Compare pyramid_v2 against direct per backbone: headline, strata, stages.

Usage: python scripts/compare_v2.py <csv> [backbone]
"""

from __future__ import annotations

import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/baselines_A.csv")
backbone = sys.argv[2] if len(sys.argv) > 2 else "roma"

with path.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
with (Path("results/fov_ratios.csv")).open(newline="", encoding="utf-8") as f:
    fov = {r["pair_id"]: float(r["fov_area_ratio"]) for r in csv.DictReader(f)}


def ed(r: dict) -> float:
    if r["status"] != "ok":
        return float("inf")
    v = r["mu_ed_tps"] or r["mu_ed"]
    return float(v) if v else float("inf")


def table(sel: list[dict], label: str) -> None:
    eds = np.array([ed(r) for r in sel])
    fin = eds[np.isfinite(eds)]
    med = np.median(fin) if len(fin) else float("nan")
    print(f"{label:>18}: n={len(sel)} ok={sum(r['status'] == 'ok' for r in sel)} "
          f"medED={med:8.1f} SR@5={np.mean(eds < 5):.2f} "
          f"SR@10={np.mean(eds < 10):.2f} SR@20={np.mean(eds < 20):.2f}")


direct = [r for r in rows if r["backbone"] == backbone and r["mode"] == "direct"]
v2 = [r for r in rows if r["backbone"] == backbone and r["mode"] == "pyramid_v2"]
print(f"=== {backbone}: direct vs pyramid_v2 ===")
table(direct, "direct")
table(v2, "pyramid_v2")

print("\n--- by FOV stratum (SR@10) ---")
for lo, hi, lab in [(0, 0.05, "<0.05"), (0.05, 0.25, "0.05-0.25"),
                    (0.25, 0.5, "0.25-0.5"), (0.5, 10, ">=0.5")]:
    d = [r for r in direct if lo <= fov[r["pair_id"]] < hi]
    v = [r for r in v2 if lo <= fov[r["pair_id"]] < hi]
    sr_d = np.mean([ed(r) < 10 for r in d]) if d else float("nan")
    sr_v = np.mean([ed(r) < 10 for r in v]) if v else float("nan")
    print(f"  {lab:>10} (n={len(d):>3}): direct {sr_d:.2f} -> v2 {sr_v:.2f}")

print("\n--- accepted stage (family@stage column) ---")
stages = Counter(r["family"].split("@")[-1] if "@" in r["family"] else "?"
                 for r in v2 if r["status"] == "ok")
for s, c in stages.most_common():
    print(f"  {s}: {c}")

# pairwise: where v2 helped or hurt vs direct
d_by_id = {r["pair_id"]: ed(r) for r in direct}
deltas = [(ed(r) - d_by_id[r["pair_id"]], r["pair_id"]) for r in v2
          if r["pair_id"] in d_by_id
          and np.isfinite(ed(r)) and np.isfinite(d_by_id[r["pair_id"]])]
better = sum(1 for d, _ in deltas if d < -1)
worse = sum(1 for d, _ in deltas if d > 1)
print(f"\npairwise (finite both): better {better}, worse {worse}, "
      f"~same {len(deltas) - better - worse}")
print("worst regressions:")
for d, pid in sorted(deltas, reverse=True)[:5]:
    print(f"  {pid}: +{d:.1f}px")
print("best improvements:")
for d, pid in sorted(deltas)[:5]:
    print(f"  {pid}: {d:.1f}px")
