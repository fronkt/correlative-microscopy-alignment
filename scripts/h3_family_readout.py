"""H3 readout: which transform family does the BIC-style selector pick?

H3 says affine should suffice (vs full homography) for AmalgaMatch pairs.
The `family` column of the baselines CSV records the per-pair BIC choice,
so this is a free analysis over direct-mode ok rows.

Usage: python scripts/h3_family_readout.py [results/baselines_A.csv]
"""

from __future__ import annotations

import collections
import csv
import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "results/baselines_A.csv"
with open(path, newline="", encoding="utf-8") as f:
    rows = [
        r for r in csv.DictReader(f)
        if r["mode"] == "direct" and r["status"] == "ok"
    ]

print("H3 readout: BIC-selected family on direct-mode ok rows\n")
print(f"{'backbone':>22}  {'affine':>20}  {'homography':>20}")
for bb in sorted({r["backbone"] for r in rows}):
    sel = [r for r in rows if r["backbone"] == bb]
    cells = []
    for fam in ("affine", "homography"):
        eds = [float(r["mu_ed"]) for r in sel if r["family"] == fam and r["mu_ed"]]
        med = np.median(eds) if eds else float("nan")
        cells.append(f"{len(eds):>4} (med {med:8.1f}px)")
    print(f"{bb:>22}  {cells[0]}  {cells[1]}")

# Among the WELL-REGISTERED pairs (mu_ed < 20 px) — the only regime where
# the family choice is meaningful — how often is affine enough?
good = [r for r in rows if r["mu_ed"] and float(r["mu_ed"]) < 20]
fams = collections.Counter(r["family"] for r in good)
n = sum(fams.values())
print(f"\nwell-registered pairs (mu_ed < 20 px, all backbones): {n}")
for fam, c in fams.most_common():
    print(f"  {fam}: {c} ({c / max(n, 1):.0%})")
