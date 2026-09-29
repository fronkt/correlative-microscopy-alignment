"""Phase A / A2 (exploratory): seed disagreement D as a failure signal for RoMa and MA-RoMa.

D = median over the seed pairs of the mean grid displacement (5 x 5 grid over the TARGET image, source px)
between the homographies of the 6 seed runs (core seed 0 + control seeds 1-5), transform none, mode direct.
Failed runs: strict variant D = +inf if any seed failed; 'ignore' variant drops failed seeds (needs >= 2 ok).
Conventions follow scripts/analyze_triage.py: success = mu_ed <= 20 (failed = inf), S1 = n_inliers/n_matches,
scene-clustered bootstrap B = 10,000, seed 0, rank percentiles taken once on the full sample.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from cma.triage import auroc, ci95, cluster_bootstrap, grid_points, transform_distance, youden_cutoff

B = 10_000
SEEDS = range(6)
DESIGN = ("SameSlice", "SerialSectioning")
BACKBONES = ("roma", "ma_roma")
OUT = Path("results/phaseA")


def bci(stat, clusters):
    b = cluster_bootstrap(stat, clusters, B=B, seed=0)
    lo, hi = ci95(b)
    return {"lo": lo, "hi": hi, "n_boot": int(len(b))}


def au(score, y, scenes):
    return {"auroc": auroc(score, y), **bci(lambda i: auroc(score[i], y[i]), scenes)}


def delta(sa, sb, y, scenes):
    return {"delta": auroc(sa, y) - auroc(sb, y),
            **bci(lambda i: auroc(sa[i], y[i]) - auroc(sb[i], y[i]), scenes)}


def pct_rank(x):
    return rankdata(np.where(np.isfinite(x), x, -1e300)) / len(x)


def quant(d):
    d = np.asarray(d, float)
    if len(d) == 0:
        return {}
    f = d[np.isfinite(d)]
    q = {f"q{int(p)}": (float(np.percentile(d, p)) if np.isfinite(np.percentile(d, p)) else "inf")
         for p in (0, 10, 25, 50, 75, 90, 100)}
    q.update(n=int(len(d)), n_inf=int((~np.isfinite(d)).sum()),
             frac_lt1=float((d < 1).mean()), frac_lt5=float((d < 5).mean()), frac_lt20=float((d < 20).mean()))
    return q


def main():
    df = pd.read_csv("results/triage/candidates.csv")
    ok = df.status.eq("ok")
    df["err"] = np.where(ok, pd.to_numeric(df.mu_ed, errors="coerce"), np.inf)
    df["err"] = df["err"].fillna(np.inf)
    nm, ni = pd.to_numeric(df.n_matches, errors="coerce"), pd.to_numeric(df.n_inliers, errors="coerce")
    df["S1"] = np.where(ok & (nm > 0), ni / nm, -np.inf)
    sub = df[df["mode"].eq("direct") & df["transform"].astype(str).eq("none") & df.backbone.isin(BACKBONES)
             & df.pool.isin(["core", "control"])]
    pairs = sorted(df.pair_id.unique())
    meta = df.drop_duplicates("pair_id").set_index("pair_id").loc[pairs, ["scene", "group", "subclass", "h_t", "w_t"]]
    scenes, groups = meta.scene.values, meta.group.values
    out, log_rows, per_pair = {"B": B, "n_pairs": len(pairs), "n_scenes": int(meta.scene.nunique())}, [], {}

    for bb in BACKBONES:
        s = sub[sub.backbone.eq(bb)]
        cnt = s.groupby("pair_id").seed.nunique().reindex(pairs).fillna(0).astype(int)
        dup = int(s.duplicated(["pair_id", "seed"]).sum())
        assert (cnt == 6).all() and dup == 0, (bb, cnt.value_counts().to_dict(), dup)
        E = s.pivot(index="pair_id", columns="seed", values="err").reindex(pairs)
        S1 = s.pivot(index="pair_id", columns="seed", values="S1").reindex(pairs)
        stat = s.pivot(index="pair_id", columns="seed", values="status").reindex(pairs)
        Hd = {(r.pair_id, int(r.seed)): (np.array(json.loads(r.H)).reshape(3, 3)
                                         if r.status == "ok" and isinstance(r.H, str) and r.H else None)
              for r in s.itertuples()}
        failed = (stat != "ok")
        res = {"n_runs": int(len(s)), "missing_pairs": 0, "failed_runs_by_seed": failed.sum().to_dict(),
               "pairs_with_any_failed_seed": int(failed.any(axis=1).sum())}
        # displacement table per pair for all 15 seed pairs (nan = a run failed)
        seedpairs = list(itertools.combinations(SEEDS, 2))
        M = np.full((len(pairs), len(seedpairs)), np.nan)
        for i, p in enumerate(pairs):
            pts = grid_points(int(meta.h_t.iloc[i]), int(meta.w_t.iloc[i]), 5)
            for k, (a, c) in enumerate(seedpairs):
                Ha, Hc = Hd[(p, a)], Hd[(p, c)]
                if Ha is not None and Hc is not None:
                    M[i, k] = transform_distance(Ha, Hc, pts)  # may be inf if degenerate projection
        anyfail = failed.any(axis=1).values
        Mi = np.where(np.isnan(M), np.inf, M)
        D_raw = np.median(Mi, axis=1)                          # failed pair-entries = inf inside the median
        D_strict = np.where(anyfail, np.inf, D_raw)            # PRIMARY: +inf if any seed failed
        D_ign = np.array([np.median(r[np.isfinite(r)]) if np.isfinite(r).sum() >= 1 else np.inf for r in M])
        # ignore variant: nan entries dropped; degenerate inf kept
        D_ign = np.array([np.median(r[~np.isnan(r)]) if (~np.isnan(r)).any() else np.inf for r in M])
        # leak-free variant: seeds 1-5 only (seed 0 is the label), 10 pairs
        idx15 = [k for k, (a, c) in enumerate(seedpairs) if a >= 1]
        D_no0 = np.where(failed[[1, 2, 3, 4, 5]].any(axis=1).values, np.inf, np.median(Mi[:, idx15], axis=1))
        # 2-seed: seeds 0 and 1
        k01 = seedpairs.index((0, 1))
        D2_strict = Mi[:, k01]
        D2_ign = D2_strict  # a failed run has no H, so the 2-seed distance is undefined either way -> inf
        y = (E[0] <= 20).values
        s1 = S1[0].values
        seedsucc = (E <= 20).values
        always, never = int(seedsucc.all(axis=1).sum()), int((~seedsucc.any(axis=1)).sum())
        res["success_seed0"] = int(y.sum())
        res["flips"] = {"always": always, "never": never, "flip": int(len(pairs) - always - never)}
        res["n_success_by_seed"] = seedsucc.sum(axis=0).tolist()
        res["failed_seed0_pairs"] = int(failed[0].sum())
        per_pair[bb] = dict(D_strict=D_strict, D_ignore=D_ign, D_raw=D_raw, D_no_seed0=D_no0, D2_seeds01=D2_strict,
                            y=y, S1=s1, n_success_seeds=seedsucc.sum(axis=1), any_failed=anyfail)

        variants = {"D_strict": D_strict, "D_ignore": D_ign, "D_raw_median_with_inf": D_raw,
                    "D_no_seed0": D_no0, "D2_seeds01": D2_strict}
        # 1. distributions
        res["D_distribution"] = {}
        for name, D in variants.items():
            res["D_distribution"][name] = {"all": quant(D), "seed0_success": quant(D[y]), "seed0_fail": quant(D[~y])}
        # 2-3. AUROC
        res["auroc_S1"] = au(s1, y, scenes)
        rS1 = pct_rank(s1)
        res["auroc_minusD"], res["auroc_combined"], res["delta_combined_minus_S1"] = {}, {}, {}
        for name, D in variants.items():
            nD = -D
            res["auroc_minusD"][name] = au(nD, y, scenes)
            comb = (rS1 + pct_rank(nD)) / 2
            res["auroc_combined"][name] = au(comb, y, scenes)
            res["delta_combined_minus_S1"][name] = delta(comb, s1, y, scenes)
            per_pair[bb]["comb_" + name] = comb
            log_rows.append((bb, name, res["auroc_minusD"][name]["auroc"], res["auroc_combined"][name]["auroc"],
                             res["delta_combined_minus_S1"][name]))
        # 6 extra: every 2-seed pair (a,c) (dead-end style robustness, point estimates only)
        two = {}
        for k, (a, c) in enumerate(seedpairs):
            two[f"{a}-{c}"] = {"auroc_minusD": auroc(-Mi[:, k], y),
                               "auroc_combined": auroc((rS1 + pct_rank(-Mi[:, k])) / 2, y)}
        res["two_seed_all_pairs_point"] = two
        # 4. within group
        wg = {}
        for g in sorted(set(groups)):
            m = groups == g
            if y[m].any() and (~y[m]).any():
                ss = scenes[m]
                wg[g] = {"n": int(m.sum()), "n_success": int(y[m].sum()),
                         "n_scenes": int(len(set(ss))), "auroc_minusD_strict": au(-D_strict[m], y[m], ss),
                         "auroc_S1": au(s1[m], y[m], ss)}
            else:
                wg[g] = {"n": int(m.sum()), "n_success": int(y[m].sum()), "skipped": "one class only"}
        res["within_group"] = wg
        # 5. the 42 (ma_roma only; also report for roma with the same recipe as a side note)
        design = np.isin(groups, DESIGN)
        cut = youden_cutoff(s1[design], y[design])
        held = ~design
        fa = held & (s1 >= cut) & ~y
        res["youden_cut"] = float(cut)
        res["n_confident_false_accepts"] = int(fa.sum())
        if bb == "ma_roma":
            assert int(fa.sum()) == 42, int(fa.sum())
        acc_s = held & (s1 >= cut) & y
        f = {"n_false_accepts": int(fa.sum()), "n_accepted_successes_heldout": int(acc_s.sum())}
        for name in ("D_strict", "D_ignore", "D2_seeds01", "D_no_seed0"):
            D = variants[name]
            f[name] = {"D_of_false_accepts": quant(D[fa]), "D_of_accepted_successes": quant(D[acc_s]),
                       "cutoffs": {}}
            for t in (1, 2, 5, 10, 20):
                f[name]["cutoffs"][str(t)] = {
                    "false_accepts_caught": int((D[fa] > t).sum()), "false_accepts_remaining": int((D[fa] <= t).sum()),
                    "successes_rejected": int((D[acc_s] > t).sum()),
                    "success_reject_frac": float((D[acc_s] > t).mean())}
            # best possible cut: smallest D cut-off that catches the most FA for each budget of rejected successes
        res["confident_false_accepts"] = f
        out[bb] = res
        per_pair[bb]["fa42"] = fa

    # CSV
    idx = pd.DataFrame({"pair_id": pairs, "scene": scenes, "group": groups})
    for bb in BACKBONES:
        for k, v in per_pair[bb].items():
            idx[f"{bb}__{k}"] = v
    OUT.mkdir(parents=True, exist_ok=True)
    idx.to_csv(OUT / "a2_per_pair.csv", index=False)
    (OUT / "a2_seed_disagreement.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")

    for bb in BACKBONES:
        r = out[bb]
        print(bb, "flips", r["flips"], "failed", r["pairs_with_any_failed_seed"], "S1", r["auroc_S1"])
        for k in r["auroc_minusD"]:
            print(" ", k, "AUROC(-D)", round(r["auroc_minusD"][k]["auroc"], 3),
                  "comb", round(r["auroc_combined"][k]["auroc"], 3),
                  "delta", {a: round(b, 3) for a, b in r["delta_combined_minus_S1"][k].items() if a != "n_boot"})


if __name__ == "__main__":
    main()
