"""Test-split (28 pairs) readout for the MA-RoMa fine-tuning experiment.

Summary table per method plus the pairs that flipped at each SR threshold
between zero-shot ma_roma and ma_roma_ft (direct). Paired bootstrap CIs
come from scripts/bootstrap_ci.py --split results/split.json:test.
"""

from __future__ import annotations

import csv
import json

import numpy as np

TPS = True  # paper protocol: TPS-refined ED when available


def ed(r: dict) -> float:
    if r["status"] != "ok":
        return float("inf")
    v = (r["mu_ed_tps"] or r["mu_ed"]) if TPS else r["mu_ed"]
    return float(v) if v else float("inf")


def main() -> None:
    split = json.load(open("results/split.json"))
    test = set(split["test"])
    rows = list(csv.DictReader(open("results/baselines_A.csv", encoding="utf-8")))

    print(f"{'method':30s} {'SR@5':>6s} {'SR@10':>6s} {'SR@20':>6s} "
          f"{'medED':>8s}   (28 test pairs, tps-refined)")
    sel: dict[tuple[str, str], dict[str, float]] = {}
    for bb in ("roma", "ma_roma", "ma_roma_ft"):
        for mode in ("direct", "pyramid_v2"):
            e = {r["pair_id"]: ed(r) for r in rows
                 if r["backbone"] == bb and r["mode"] == mode
                 and r["pair_id"] in test}
            if not e:
                continue
            sel[(bb, mode)] = e
            a = np.array(list(e.values()))
            fin = a[np.isfinite(a)]
            print(f"{bb + ':' + mode:30s} {np.mean(a < 5):6.3f} "
                  f"{np.mean(a < 10):6.3f} {np.mean(a < 20):6.3f} "
                  f"{np.median(fin):8.1f}")

    za, fa = sel[("ma_roma", "direct")], sel[("ma_roma_ft", "direct")]
    print("\nflips (ma_roma_ft vs ma_roma, direct):")
    for t in (10.0, 20.0):
        for pid in sorted(test):
            a0, a1 = za[pid], fa[pid]
            if (a0 < t) != (a1 < t):
                tag = "GAIN" if a1 < t else "LOSS"
                print(f"  @{t:.0f}px {tag} {pid:55s} zs={a0:9.1f} ft={a1:9.1f}")


if __name__ == "__main__":
    main()
