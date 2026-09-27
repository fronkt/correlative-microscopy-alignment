"""Verification gate for the CJSJ triage paper. Exit 0 only if every check passes.

1. Data integrity: every (candidate, pair) present exactly once; transforms parse; counts as expected.
2. Independent recomputation: the key numbers in summary.json are recomputed here with separate,
   deliberately naive code (brute-force pairwise AUROC, direct counting), and must agree.
3. Paper consistency: paper/cjsj/paper.md must equal a fresh render of the templates from summary.json,
   so no number in the paper was typed by hand.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "results/triage/candidates.csv"
SUMMARY = ROOT / "results/triage/summary.json"
PRIMARY = "ma_roma|direct|none|s0"
checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, bool(ok), detail))


def brute_auroc(s: np.ndarray, y: np.ndarray) -> float:
    s = np.where(np.isfinite(s), s, np.where(s > 0, 1e300, -1e300))
    pos, neg = s[y], s[~y]
    gt = (pos[:, None] > neg[None, :]).sum()
    eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def main(csv_path: Path = CSV, summary_path: Path = SUMMARY, check_paper: bool = True) -> int:
    df = pd.read_csv(csv_path)
    s = json.loads(summary_path.read_text(encoding="utf-8"))
    df["cand"] = (df.backbone + "|" + df["mode"] + "|" + df["transform"].astype(str) + "|s"
                  + df.seed.astype(int).astype(str))
    n_pairs = df.pair_id.nunique()
    check("n_pairs matches summary", n_pairs == s["n_pairs"], f"{n_pairs} vs {s['n_pairs']}")
    dup = df.duplicated(["pair_id", "cand"]).sum()
    check("no duplicate (pair, candidate) rows", dup == 0, f"{dup} duplicates")
    per = df.groupby("cand").pair_id.nunique()
    check("every candidate covers every pair", bool((per == n_pairs).all()),
          ", ".join(f"{c}:{k}" for c, k in per.items() if k != n_pairs))
    ok = df.status.eq("ok")
    bad_h = [i for i, r in df[ok].iterrows() if len(json.loads(r.H)) != 9]
    check("every successful row has a 9-element transform", not bad_h, f"{len(bad_h)} bad")
    check("voter count", len(s["voters"]) == 15, str(len(s["voters"])))
    check("rerun controls present", df.pool.eq("control").groupby(df.cand).any().sum() == 10,
          str(df[df.pool.eq("control")].cand.nunique()))

    err = np.where(ok, pd.to_numeric(df.mu_ed, errors="coerce"), np.inf)
    df["err"] = np.nan_to_num(err, nan=np.inf)
    nm = pd.to_numeric(df.n_matches, errors="coerce")
    ni = pd.to_numeric(df.n_inliers, errors="coerce")
    df["S1"] = np.where(ok & (nm > 0), ni / nm, -np.inf)

    prim = df[df.cand == PRIMARY].sort_values("pair_id")
    y = (prim.err <= 20).values
    a = brute_auroc(prim.S1.values, y)
    check("H1 AUROC(S1) recomputed", abs(a - s["H1"]["auroc_S1_at_20"]["auroc"]) < 1e-9,
          f"{a:.6f} vs {s['H1']['auroc_S1_at_20']['auroc']:.6f}")
    check("H1 success count", int(y.sum()) == s["H1"]["auroc_S1_at_20"]["n_success"])

    voters = s["voters"]
    E = df[df.cand.isin(voters)].pivot(index="pair_id", columns="cand", values="err")
    best = (E <= 20).sum().idxmax()
    check("best single candidate", best == s["H3"]["best_single"], f"{best} vs {s['H3']['best_single']}")
    check("best single SR@20", abs((E[best] <= 20).mean() - s["H3"]["best_single_sr20"]) < 1e-12)
    check("oracle SR@20", abs((E.min(axis=1) <= 20).mean() - s["H3"]["oracle_sr20"]) < 1e-12)
    S1 = df[df.cand.isin(voters)].pivot(index="pair_id", columns="cand", values="S1").fillna(-np.inf)
    pick = S1.idxmax(axis=1)
    sr = np.mean([E.loc[p, pick[p]] <= 20 for p in E.index])
    check("H3 select-by-S1 SR@20", abs(sr - s["H3"]["select_S1"]["sr20"]) < 1e-12, f"{sr} vs {s['H3']['select_S1']['sr20']}")

    gt = df[(df.backbone == "gt") & (df["mode"] == "homography")]
    check("ceiling (GT homography within 20 px)", int((gt.err <= 20).sum()) == s["ceiling"]["homography_le_20"])

    tr = s["H1"]["transfer"]
    meta = prim.set_index("pair_id")
    design = meta.group.isin(["SameSlice", "SerialSectioning"]).values
    acc = meta.S1.values >= tr["cutoff"]
    held = ~design
    check("transfer held-out accepted count", int(acc[held].sum()) == tr["heldout_accepted"])
    check("transfer held-out base rate", abs(y[held].mean() - tr["heldout_base_rate"]) < 1e-12)

    if check_paper:
        sys.path.insert(0, str(ROOT / "scripts"))
        import render_cjsj_paper as r
        paper = (ROOT / "paper/cjsj/paper.md")
        if paper.exists():
            before = paper.read_text(encoding="utf-8")
            r.main()
            check("paper.md equals a fresh render from summary.json", paper.read_text(encoding="utf-8") == before)

    width = max(len(c[0]) for c in checks)
    for name, good, detail in checks:
        print(f"{'PASS' if good else 'FAIL'}  {name.ljust(width)}  {detail}")
    n_fail = sum(not g for _, g, _ in checks)
    print(f"\n{len(checks) - n_fail}/{len(checks)} checks passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        sys.exit(main(Path(args[0]), Path(args[1]), check_paper=False))
    sys.exit(main())
