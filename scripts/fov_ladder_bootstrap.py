"""Paired bootstrap of wrapper lift at a single FOV rung, per backbone.

For a given backbone, compares pyramid_v2 vs direct SR@10 at one rung over
that backbone's base-matchable testbed pairs (transform mu_ed, never TPS).
Mirrors the original ma_roma finding (rung 0.1: +0.150, two-sided p=0.0028).

Usage: python scripts/fov_ladder_bootstrap.py [rung] [backbone ...]
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

B = 10_000


def two_sided_p(d_boot: np.ndarray) -> float:
    """Two-sided bootstrap p-value: twice the smaller tail mass, capped at 1.

    Convention: p = min(1, 2 * min(P(d <= 0), P(d >= 0))).  This is the
    convention the manuscript's methods text declares, and it is symmetric --
    the value does not depend on which configuration is labelled A and which
    is labelled B, nor on the sign of the observed difference.
    """
    p_le = float((d_boot <= 0).mean())
    p_ge = float((d_boot >= 0).mean())
    return min(1.0, 2.0 * min(p_le, p_ge))


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
        pv = two_sided_p(boot)
        print(f"{bb:12s} rung {rung:g}  n={n:2d}  "
              f"SR@10 direct {(ed < 10).mean():.3f} -> pyr {(ep < 10).mean():.3f}  "
              f"delta {d_obs:+.3f}  CI [{lo:+.3f},{hi:+.3f}]  p(two-sided)={pv:.4f}")


if __name__ == "__main__":
    main()
