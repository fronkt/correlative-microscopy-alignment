"""Pairs crossing the SR@10 boundary between direct and pyramid_v2.

Usage: python scripts/v2_gain_loss.py <csv> [backbone]
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "results/baselines_A.csv")
backbone = sys.argv[2] if len(sys.argv) > 2 else "roma"


def ed(r: dict) -> float:
    if r["status"] != "ok":
        return float("inf")
    v = r["mu_ed_tps"] or r["mu_ed"]
    return float(v) if v else float("inf")


with path.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
d = {r["pair_id"]: ed(r) for r in rows
     if r["backbone"] == backbone and r["mode"] == "direct"}
v = {r["pair_id"]: ed(r) for r in rows
     if r["backbone"] == backbone and r["mode"] == "pyramid_v2"}

gained = [(p, d[p], v[p]) for p in v if v[p] < 10 <= d[p]]
lost = [(p, d[p], v[p]) for p in v if d[p] < 10 <= v[p]]
print(f"{backbone}: gained {len(gained)} pairs at SR@10, lost {len(lost)}")
for p, de, ve in gained:
    print(f"  GAIN {p}: {de:.1f} -> {ve:.1f}px")
for p, de, ve in lost:
    print(f"  LOSS {p}: {de:.1f} -> {ve:.1f}px")
