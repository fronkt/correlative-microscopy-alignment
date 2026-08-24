"""Paired bootstrap over the 187 pairs: method B vs method A (task 4.3).

Resamples pairs with replacement, computing the delta in SR@{5,10,20} and
median ED on each replicate. Reports 95% percentile CIs and the two-sided
bootstrap p-value against the null "A and B do not differ", which is the
convention the manuscript's methods text declares.

Usage:
  python scripts/bootstrap_ci.py results/baselines_A.csv roma:direct roma:pyramid_v2
  python scripts/bootstrap_ci.py results/baselines_A.csv ma_roma:direct \
      ma_roma_ft:direct --split results/split.json:test
"""

from __future__ import annotations

import csv
import json
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


def ed(r: dict) -> float:
    if r["status"] != "ok":
        return float("inf")
    v = r["mu_ed_tps"] or r["mu_ed"]
    return float(v) if v else float("inf")


def load(path: Path, spec: str) -> dict[str, float]:
    backbone, mode = spec.split(":")
    with path.open(newline="", encoding="utf-8") as f:
        return {
            r["pair_id"]: ed(r) for r in csv.DictReader(f)
            if r["backbone"] == backbone and r["mode"] == mode
        }


def main() -> None:
    args = list(sys.argv[1:])
    keep = None
    if "--split" in args:
        i = args.index("--split")
        split_path, split_name = args[i + 1].rsplit(":", 1)
        keep = set(json.loads(Path(split_path).read_text())[split_name])
        del args[i:i + 2]
    path = Path(args[0])
    spec_a, spec_b = args[1], args[2]
    a, b = load(path, spec_a), load(path, spec_b)
    ids = sorted(set(a) & set(b))
    if keep is not None:
        ids = [i for i in ids if i in keep]
    if not ids:
        raise SystemExit(f"no common pairs between {spec_a} and {spec_b}")
    ea = np.array([a[i] for i in ids])
    eb = np.array([b[i] for i in ids])
    n = len(ids)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n, size=(B, n))

    print(f"{spec_b} vs {spec_a}, n={n} paired, B={B}\n")
    for t in (5.0, 10.0, 20.0):
        d_obs = float((eb < t).mean() - (ea < t).mean())
        d_boot = (eb[idx] < t).mean(axis=1) - (ea[idx] < t).mean(axis=1)
        lo, hi = np.percentile(d_boot, [2.5, 97.5])
        p = two_sided_p(d_boot)
        print(f"delta SR@{t:>4.0f}px: {d_obs:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  p(two-sided)={p:.4f}")

    fa, fb = np.isfinite(ea), np.isfinite(eb)
    both = fa & fb
    d_obs = float(np.median(eb[both]) - np.median(ea[both]))
    d_boot = []
    for row in idx:
        m = both[row]
        if m.sum() < 10:
            continue
        d_boot.append(np.median(eb[row][m]) - np.median(ea[row][m]))
    d_boot = np.asarray(d_boot)
    lo, hi = np.percentile(d_boot, [2.5, 97.5])
    p = two_sided_p(d_boot)
    print(f"delta median ED (finite-both): {d_obs:+.1f}px  "
          f"95% CI [{lo:+.1f}, {hi:+.1f}]  p(two-sided)={p:.4f}")


if __name__ == "__main__":
    main()
