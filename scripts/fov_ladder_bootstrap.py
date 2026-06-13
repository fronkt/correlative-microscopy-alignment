"""Paired bootstrap of wrapper lift at a single FOV rung, per backbone.

For a given backbone, compares pyramid_v2 vs direct SR@10 at one rung over
that backbone's base-matchable testbed pairs (transform mu_ed, never TPS).
Mirrors the original ma_roma finding (rung 0.1: +0.150, p=0.0014).

Usage: python scripts/fov_ladder_bootstrap.py [rung] [backbone ...]
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

B = 10_000


def mu(r: dict) -> float:
    return float(r["mu_ed"]) if (r["status"] == "ok" and r["mu_ed"]) else float("inf")


def main() -> None:
    rung = float(sys.argv[1]) if len(sys.argv) > 1 else 0.10
    backbones = sys.argv[2:] or ["ma_roma", "ma_roma_ft"]
    with Path("results/baselines_A.csv").open(encoding="utf-8") as f:
        base = list(csv.DictReader(f))
    with Path("results/fov_ladder.csv").open(encoding="utf-8") as f:
        ladder = list(csv.DictReader(f))
    testbed = {r["pair_id"] for r in ladder}

    for bb in backbones:
        match = {r["pair_id"] for r in base
                 if r["backbone"] == bb and r["mode"] == "direct"
                 and mu(r) < 20 and r["pair_id"] in testbed}

        def at(mode: str, bb=bb, match=match) -> dict[str, float]:
            return {r["pair_id"]: mu(r) for r in ladder
                    if r["backbone"] == bb and r["mode"] == mode
                    and float(r["rung"]) == rung and r["status"] != "skipped"
                    and r["pair_id"] in match}

        d, p = at("direct"), at("pyramid_v2")
        ids = sorted(set(d) & set(p))
        ed, ep = np.array([d[i] for i in ids]), np.array([p[i] for i in ids])
        n = len(ids)
        rng = np.random.default_rng(0)
        idx = rng.integers(0, n, size=(B, n))
        d_obs = float((ep < 10).mean() - (ed < 10).mean())
        boot = (ep[idx] < 10).mean(axis=1) - (ed[idx] < 10).mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        pv = float((boot <= 0).mean())
        print(f"{bb:12s} rung {rung:g}  n={n:2d}  "
              f"SR@10 direct {(ed < 10).mean():.3f} -> pyr {(ep < 10).mean():.3f}  "
              f"delta {d_obs:+.3f}  CI [{lo:+.3f},{hi:+.3f}]  p={pv:.4f}")


if __name__ == "__main__":
    main()
