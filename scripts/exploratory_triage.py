"""EXPLORATORY analyses for the CJSJ triage paper (not pre-registered; decided after seeing H3 fail).

Writes results/triage/exploratory.json. The paper labels every number from here as exploratory.
  E1  fragility of the oracle's headroom (how many candidates succeed on oracle-only pairs; rerun flips)
  E2  within-pair discrimination: does S1 prefer a successful candidate over a failed one in the same pair?
  E3  which candidates selection-by-S1 picked, and in the lost pairs
  E4  selection by S1 restricted to the dense-matcher candidates (RoMa / MatchAnything-RoMa family)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from cma.triage import auroc, ci95, cluster_bootstrap, mcnemar_exact

IND = Path("results/triage")


def main() -> None:
    s = json.loads((IND / "summary.json").read_text(encoding="utf-8"))
    sc = pd.read_csv(IND / "per_pair_scores.csv")
    pp = pd.read_csv(IND / "per_pair.csv").set_index("pair_id")
    pairs = sorted(sc.pair_id.unique())
    scenes = pp.loc[pairs, "scene"].values
    voters, best = s["voters"], s["H3"]["best_single"]
    E = sc.pivot(index="pair_id", columns="cand", values="err").reindex(pairs)
    S1 = sc.pivot(index="pair_id", columns="cand", values="S1").reindex(pairs).fillna(-np.inf)
    Ev = E[voters]
    out: dict = {}

    nsucc = (Ev <= 20).sum(axis=1)
    oracle_only = (Ev.min(axis=1) <= 20) & (Ev[best] > 20)
    out["E1"] = {
        "pairs_no_candidate_succeeds": int((nsucc == 0).sum()),
        "oracle_only_pairs": int(oracle_only.sum()),
        "oracle_only_with_at_most_2_successes": int((nsucc[oracle_only] <= 2).sum()),
        "oracle_only_best_error_over_10px": int((Ev[oracle_only].min(axis=1) > 10).sum()),
    }
    seeds = [f"ma_roma|direct|none|s{i}" for i in range(6)]
    k = (E[seeds] <= 20).sum(axis=1)
    out["E1"].update(ma_roma_rerun_always=int((k == 6).sum()), ma_roma_rerun_never=int((k == 0).sum()),
                     ma_roma_rerun_flip=int(((k > 0) & (k < 6)).sum()))

    V = sc[sc.cand.isin(voters)].assign(ok=lambda d: d.err <= 20)
    w = [auroc(g.S1.values, g.ok.values) for _, g in V.groupby("pair_id") if 0 < g.ok.sum() < len(g)]
    out["E2"] = {"mixed_pairs": len(w), "within_pair_auroc_median": float(np.median(w)),
                 "within_pair_auroc_mean": float(np.mean(w))}

    pick = S1[voters].idxmax(axis=1)
    ok = np.array([Ev.loc[p, pick[p]] <= 20 for p in pairs])
    base_ok = (Ev[best] <= 20).values
    lost = (~ok) & base_ok
    fam = pick.str.split("|").str[0]
    out["E3"] = {"sift_picks": int((fam == "sift").sum()), "lost": int(lost.sum()),
                 "lost_sift_picks": int(((fam == "sift").values & lost).sum()),
                 "sift_S1_median": float(V[V.cand.str.startswith("sift|")].S1.median()),
                 "dense_S1_median": float(V[V.cand.str.split("|").str[0].isin(["roma", "ma_roma"])].S1.median())}

    dense = [c for c in voters if c.split("|")[0] in ("roma", "ma_roma")]
    pk = S1[dense].idxmax(axis=1)
    okd = np.array([Ev.loc[p, pk[p]] <= 20 for p in pairs])
    b, c = int((okd & ~base_ok).sum()), int((~okd & base_ok).sum())
    lo, hi = ci95(cluster_bootstrap(lambda idx: okd[idx].mean() - base_ok[idx].mean(), scenes, B=10_000))
    out["E4"] = {"n_candidates": len(dense), "n_ok": int(okd.sum()), "best_single_n": int(base_ok.sum()),
                 "won": b, "lost": c, "mcnemar_p": mcnemar_exact(b, c), "gain_ci": [lo, hi],
                 "oracle_n": int((Ev[dense].min(axis=1) <= 20).sum())}
    # E5: per task group, for the primary candidate (AUROC only where both classes exist)
    pr = pp.loc[pairs]
    okp = pr.err_primary <= 20
    groups = {}
    for g, idx in pr.groupby("group").groups.items():
        yy, ss = okp.loc[idx].values, pr.loc[idx, "S1_primary"].values
        groups[g] = {"n": int(len(idx)), "n_success": int(yy.sum()),
                     "auroc": auroc(ss, yy) if 0 < yy.sum() < len(yy) else None}
    testable = [v["auroc"] for v in groups.values() if v["auroc"] is not None]
    out["E5"] = {"groups": groups, "n_testable": len(testable), "min_auroc": float(min(testable)),
                 "max_auroc": float(max(testable))}
    # E6: consensus on a wrong answer: pairs whose most-agreeing registration is within 20 px of at least
    # half of the other voters, and how many of those pairs have no correct registration at all
    top = V.loc[V.groupby("pair_id").S2.idxmax()].set_index("pair_id")
    consensus = (-top.S2) <= 20
    allfail = (Ev > 20).all(axis=1).reindex(top.index)
    out["E6"] = {"consensus_pairs": int(consensus.sum()), "consensus_all_fail": int((consensus & allfail).sum())}
    # E7: which task groups the transferred cut-off's false accepts come from (held-out groups only)
    cut = s["H1"]["transfer"]["cutoff"]
    held = pr[~pr.group.isin(["SameSlice", "SerialSectioning"])]
    fa = held[(held.S1_primary >= cut) & (held.err_primary > 20)]
    out["E7"] = {"false_accepts": int(len(fa)), "by_group": fa.group.value_counts().to_dict(),
                 "tem_failures_heldout": int(((held.group == "DislocationCharacterization") & (held.err_primary > 20)).sum())}
    cand = pd.read_csv(IND / "candidates.csv")
    out["E3"]["sift_matches_median"] = float(cand[cand.backbone == "sift"].n_matches.median())
    (IND / "exploratory.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
