"""Phase A3: per-subclass calibration of the S1 cut-off from k hand-checked pairs (exploratory)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cma.triage import auroc, youden_cutoff  # noqa: E402

OUT = ROOT / "results" / "phaseA"
OUT.mkdir(parents=True, exist_ok=True)
KS = (1, 2, 3, 5)
NDRAW = 1000
SEED = 20260929
DESIGN = ("SameSlice", "SerialSectioning")

d = pd.read_csv(ROOT / "results" / "triage" / "candidates.csv")
m = d[(d.backbone == "ma_roma") & (d["mode"] == "direct") & (d.seed == 0) & (d.pool == "core")].copy()
assert len(m) == 187
ok = m.status.eq("ok")
nm = pd.to_numeric(m.n_matches, errors="coerce")
ni = pd.to_numeric(m.n_inliers, errors="coerce")
# analyze_triage uses -inf for failed runs; here 0 (same ranking below every real S1, finite for midpoints)
m["S1"] = np.where(ok & (nm > 0), ni / nm, 0.0)
err = np.where(ok, pd.to_numeric(m.mu_ed, errors="coerce"), np.inf)
m["y"] = np.nan_to_num(err, nan=np.inf) <= 20

des = m.group.isin(DESIGN).values
GCUT = youden_cutoff(m.S1.values[des], m.y.values[des])
assert abs(GCUT - 0.171) < 0.002, GCUT

FALLBACK = "global"


def cut_rule(s, y, rule):
    """Return cut-off (accept if S1 >= cut). rule 'A' or 'B'."""
    if y.all():
        return -np.inf if rule == "B" else GCUT
    if (~y).all():
        return np.inf if rule == "B" else GCUT
    lo_s, hi_f = s[y].min(), s[~y].max()
    if lo_s > hi_f:
        return (lo_s + hi_f) / 2
    return youden_cutoff(s, y)


def counts(s, y, cut):
    acc = s >= cut
    return np.array([(acc & ~y).sum(), (~y).sum(), (acc & y).sum(), y.sum(), acc.sum()], float)


def rates(c):
    fa = c[0] / c[1] if c[1] > 0 else np.nan
    rec = c[2] / c[3] if c[3] > 0 else np.nan
    prec = c[2] / c[4] if c[4] > 0 else np.nan
    return fa, rec, prec


def summ(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"mean": None, "lo": None, "hi": None, "n_draws": 0}
    return {"mean": float(x.mean()), "lo": float(np.percentile(x, 2.5)), "hi": float(np.percentile(x, 97.5)),
            "n_draws": int(len(x))}


subs = sorted(m.subclass.unique())
data = {sc: (m.S1.values[(m.subclass == sc).values], m.y.values[(m.subclass == sc).values]) for sc in subs}
is_tem = {sc: "TEM_" in sc or "_TEM" in sc for sc in subs}
assert sum(is_tem.values()) == 4

rng = np.random.default_rng(SEED)
# per draw counts: [subclass][k][variant] -> array (NDRAW,5)
variants = ("global", "A", "B")
store = {}
for sc in subs:
    s, y = data[sc]
    n = len(s)
    for k in KS:
        if n <= k:
            continue
        arr = {v: np.zeros((NDRAW, 5)) for v in variants}
        for i in range(NDRAW):
            idx = rng.choice(n, k, replace=False)
            rest = np.setdiff1d(np.arange(n), idx)
            sr, yr = s[rest], y[rest]
            arr["global"][i] = counts(sr, yr, GCUT)
            arr["A"][i] = counts(sr, yr, cut_rule(s[idx], y[idx], "A"))
            arr["B"][i] = counts(sr, yr, cut_rule(s[idx], y[idx], "B"))
        store[(sc, k)] = arr

rows, res = [], {"global_cutoff": GCUT, "n_draws": NDRAW, "rng_seed": SEED, "ks": list(KS), "subclass_table": [],
                 "per_subclass": {}, "pooled": {}, "tem": {}}
for sc in subs:
    s, y = data[sc]
    a = auroc(s, y) if 0 < y.sum() < len(y) else None
    res["subclass_table"].append({"subclass": sc, "group": m.group[m.subclass == sc].iloc[0], "n": len(s),
                                  "n_success": int(y.sum()), "n_fail": int((~y).sum()),
                                  "global_accepted_failures_all": int(((s >= GCUT) & ~y).sum()),
                                  "auroc_within": None if a is None or np.isnan(a) else float(a),
                                  "tem": is_tem[sc], "too_small": bool(len(s) < 6 or y.sum() < 2 or (~y).sum() < 2)})
    for k in KS:
        if (sc, k) not in store:
            continue
        for v in variants:
            c = store[(sc, k)][v]
            r = np.array([rates(x) for x in c])
            fa, rec, pr = summ(r[:, 0]), summ(r[:, 1]), summ(r[:, 2])
            res["per_subclass"].setdefault(sc, {}).setdefault(str(k), {})[v] = {"false_accept": fa, "recall": rec, "precision": pr}
            rows.append({"subclass": sc, "tem": is_tem[sc], "n": len(s), "n_success": int(y.sum()), "k": k, "variant": v,
                         "fa_mean": fa["mean"], "fa_lo": fa["lo"], "fa_hi": fa["hi"],
                         "recall_mean": rec["mean"], "recall_lo": rec["lo"], "recall_hi": rec["hi"],
                         "prec_mean": pr["mean"], "prec_lo": pr["lo"], "prec_hi": pr["hi"],
                         "within_auroc": None if a is None else a})


def pooled(scs, k):
    out = {}
    scs = [q for q in scs if (q, k) in store]
    if not scs:
        return None
    for v in variants:
        tot = sum(store[(q, k)][v] for q in scs)  # draw-aligned sums (pairs weighted equally)
        r = np.array([rates(x) for x in tot])
        out[v] = {"false_accept": summ(r[:, 0]), "recall": summ(r[:, 1]), "precision": summ(r[:, 2]),
                  "mean_failures_accepted": float(tot[:, 0].mean()), "mean_successes_accepted": float(tot[:, 2].mean()),
                  "mean_heldout_failures": float(tot[:, 1].mean()), "mean_heldout_successes": float(tot[:, 3].mean())}
    out["n_subclasses"] = len(scs)
    out["labelled_pairs_total"] = int(k * len(scs))
    for v in ("A", "B"):
        sav = (sum(store[(q, k)]["global"] for q in scs)[:, 0] - sum(store[(q, k)][v] for q in scs)[:, 0])
        lost = (sum(store[(q, k)]["global"] for q in scs)[:, 2] - sum(store[(q, k)][v] for q in scs)[:, 2])
        out[f"failures_saved_{v}_vs_global"] = summ(sav)
        out[f"successes_lost_{v}_vs_global"] = summ(lost)
    return out


for k in KS:
    res["pooled"][str(k)] = pooled(subs, k)
    res["tem"][str(k)] = pooled([q for q in subs if is_tem[q]], k)
    res["tem"][str(k)]["per_subclass_note"] = "see per_subclass"
    res["pooled"][str(k)]["non_tem"] = pooled([q for q in subs if not is_tem[q]], k)

# full-set reference (global cut-off, all pairs, no draws)
S, Y = m.S1.values, m.y.values
res["global_full"] = {"n": len(S), "accepted": int((S >= GCUT).sum()),
                      "failures_accepted": int(((S >= GCUT) & ~Y).sum()), "failures": int((~Y).sum()),
                      "confident_false_accepts_heldout": int(((S >= GCUT) & ~Y & ~des).sum()),
                      "tem_false_accepts": int(((S >= GCUT) & ~Y & m.subclass.map(is_tem).values).sum())}
json.dump(res, open(OUT / "a3_subclass_calibration.json", "w"), indent=1, default=float)
pd.DataFrame(rows).to_csv(OUT / "a3_per_subclass.csv", index=False)

f = lambda x: "nan" if x is None else f"{x:.3f}"
print("GCUT", GCUT, res["global_full"])
for t in res["subclass_table"]:
    print(t["n"], t["n_success"], f(t["auroc_within"]), t["tem"], t["too_small"], t["subclass"][:50])
for k in KS:
    for name, blk in (("ALL", res["pooled"][str(k)]), ("TEM", res["tem"][str(k)])):
        if blk is None:
            continue
        s = f"k={k} {name} nsub={blk['n_subclasses']} lab={blk['labelled_pairs_total']} "
        for v in variants:
            b = blk[v]
            s += f"| {v}: FA {f(b['false_accept']['mean'])} rec {f(b['recall']['mean'])} prec {f(b['precision']['mean'])} "
        s += f"| saved A {blk['failures_saved_A_vs_global']['mean']:.2f} B {blk['failures_saved_B_vs_global']['mean']:.2f}"
        s += f" lostS A {blk['successes_lost_A_vs_global']['mean']:.2f} B {blk['successes_lost_B_vs_global']['mean']:.2f}"
        print(s)
