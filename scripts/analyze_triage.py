"""Pre-registered analysis for the label-free triage study (tasks/todo.md, CJSJ section).

Reads results/triage/candidates.csv (from run_triage_candidates.py) and writes
results/triage/summary.json + results/triage/per_pair.csv. Every number the paper reports must come
from summary.json; scripts/verify_triage.py checks that.

Scores (label-free):
  S1 retained fraction = n_inliers / n_matches (Durmaz et al. 2026b)
  S2 agreement = -median over the other voting candidates of the mean displacement (source px) between
     the two transforms on a 5 x 5 grid over the target image
  S3 combined = mean of the rank percentiles of S1 and S2 over all voting (pair x candidate) rows
Success = unrefined mean error <= 20 px (primary); 10 px and 8 px secondary. Failed runs: error = inf.
Confidence intervals: bootstrap over scenes (pair_id before '#'), B = 10,000, seed 0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from cma.triage import (accepted_success, agreement_scores, auroc, aurc, ci95, cluster_bootstrap,
                        grid_points, mcnemar_exact, youden_cutoff)

THRESHOLDS = (20, 10, 8)
DESIGN_GROUPS = ("SameSlice", "SerialSectioning")  # the groups Durmaz et al. 2026b analysed
PRIMARY = "ma_roma|direct|none|s0"
COVERAGES = (0.25, 0.50, 0.75)


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["cand"] = (df.backbone + "|" + df["mode"] + "|" + df["transform"].astype(str) + "|s"
                  + df.seed.astype(int).astype(str))
    ok = df.status.eq("ok")
    df["err"] = np.where(ok, pd.to_numeric(df.mu_ed, errors="coerce"), np.inf)
    df["err"] = df["err"].fillna(np.inf)
    nm = pd.to_numeric(df.n_matches, errors="coerce")
    ni = pd.to_numeric(df.n_inliers, errors="coerce")
    df["S1"] = np.where(ok & (nm > 0), ni / nm, -np.inf)
    return df


def add_agreement(df: pd.DataFrame, voters: list[str]) -> pd.DataFrame:
    s2 = np.full(len(df), -np.inf)
    for _, g in df.groupby("pair_id"):
        r0 = g.iloc[0]
        pts = grid_points(int(r0.h_t), int(r0.w_t), n=5) if pd.notna(r0.h_t) else None
        Hs = {}
        for _, r in g.iterrows():
            Hs[r.cand] = (np.array(json.loads(r.H)).reshape(3, 3)
                          if r.status == "ok" and isinstance(r.H, str) and r.H else None)
        if pts is None:
            continue
        sc = agreement_scores(Hs, pts, voters)
        for i, c in zip(g.index, g.cand):
            s2[df.index.get_loc(i)] = sc[c]
    df["S2"] = s2
    return df


def add_combined(df: pd.DataFrame, voters: list[str]) -> pd.DataFrame:
    df["S3"] = np.nan
    v = df.cand.isin(voters)
    n = int(v.sum())
    r1 = rankdata(np.where(np.isfinite(df.loc[v, "S1"]), df.loc[v, "S1"], -1e300)) / n
    r2 = rankdata(np.where(np.isfinite(df.loc[v, "S2"]), df.loc[v, "S2"], -1e300)) / n
    df.loc[v, "S3"] = (r1 + r2) / 2.0
    return df


def boot_ci(stat, clusters, B, seed=0):
    b = cluster_bootstrap(stat, clusters, B=B, seed=seed)
    lo, hi = ci95(b)
    return {"lo": lo, "hi": hi, "n_boot": int(len(b))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/triage/candidates.csv")
    ap.add_argument("--outdir", default="results/triage")
    ap.add_argument("--B", type=int, default=10_000)
    args = ap.parse_args()
    B = args.B
    df = load(Path(args.csv))
    voters = sorted(df.loc[df.pool.isin(["core", "transform"]), "cand"].unique())
    df = add_agreement(df, voters)
    df = add_combined(df, voters)
    out: dict = {"n_rows": int(len(df)), "voters": voters, "B": B}

    pairs = sorted(df.pair_id.unique())
    out["n_pairs"] = len(pairs)
    meta = df.drop_duplicates("pair_id").set_index("pair_id").loc[pairs, ["scene", "group", "subclass"]]
    scenes = meta.scene.values
    out["n_scenes"] = int(meta.scene.nunique())
    out["n_subclasses"] = int(meta.subclass.nunique())

    # ---- feasibility ceiling
    gt = df[df.backbone.eq("gt")].pivot(index="pair_id", columns="mode", values="err").reindex(pairs)
    out["ceiling"] = {f"homography_le_{t}": int((gt["homography"] <= t).sum()) for t in THRESHOLDS}

    # ---- per-candidate success rates (descriptive; the candidates themselves are M&M-style baselines)
    E = df.pivot(index="pair_id", columns="cand", values="err").reindex(pairs)
    S = {k: df.pivot(index="pair_id", columns="cand", values=k).reindex(pairs) for k in ("S1", "S2", "S3")}
    out["sr20_by_candidate"] = {c: float((E[c] <= 20).mean()) for c in E.columns}

    # ---- H1: S1 predicts success for the primary candidate
    h1 = {}
    for t in THRESHOLDS:
        y = (E[PRIMARY] <= t).values
        s1 = S["S1"][PRIMARY].values
        a = auroc(s1, y)
        ci = boot_ci(lambda idx: auroc(s1[idx], y[idx]), scenes, B)
        h1[f"auroc_S1_at_{t}"] = {"auroc": a, **ci, "n_success": int(y.sum())}
        # roma direct for comparison
        yr = (E["roma|direct|none|s0"] <= t).values
        sr = S["S1"]["roma|direct|none|s0"].values
        h1[f"roma_auroc_S1_at_{t}"] = {"auroc": auroc(sr, yr), **boot_ci(lambda idx: auroc(sr[idx], yr[idx]), scenes, B),
                                       "n_success": int(yr.sum())}
    h1["supported"] = bool(h1["auroc_S1_at_20"]["auroc"] >= 0.80 and h1["auroc_S1_at_20"]["lo"] > 0.5)

    # transfer of the Youden cut-off from Durmaz's groups to the other four
    y20 = (E[PRIMARY] <= 20).values
    s1 = S["S1"][PRIMARY].values
    design = meta.group.isin(DESIGN_GROUPS).values
    cut = youden_cutoff(s1[design], y20[design])
    held = ~design
    acc = s1 >= cut
    hs, hy, ha = scenes[held], y20[held], acc[held]

    def diff(idx):
        a_ = ha[idx]
        if a_.sum() == 0:
            return np.nan
        return hy[idx][a_].mean() - hy[idx].mean()
    tr = {
        "cutoff": float(cut), "design_pairs": int(design.sum()), "heldout_pairs": int(held.sum()),
        "heldout_base_rate": float(hy.mean()), "heldout_accepted": int(ha.sum()),
        "heldout_accepted_success": float(hy[ha].mean()) if ha.any() else float("nan"),
        "heldout_recall": float(ha[hy].mean()) if hy.any() else float("nan"),
        "heldout_false_accept_rate": float(ha[~hy].mean()) if (~hy).any() else float("nan"),
        "design_accepted_success": float(y20[design][acc[design]].mean()) if acc[design].any() else float("nan"),
        "diff": float(hy[ha].mean() - hy.mean()) if ha.any() else float("nan"),
        "diff_ci": boot_ci(diff, hs, B),
    }
    tr["supported"] = bool(np.isfinite(tr["diff"]) and tr["diff_ci"]["lo"] > 0)
    h1["transfer"] = tr
    out["H1"] = h1

    # ---- H2: agreement / combined vs retained fraction
    h2 = {}
    y = y20
    for k in ("S2", "S3"):
        sk, s1v = S[k][PRIMARY].values, S["S1"][PRIMARY].values
        h2[f"primary_auroc_{k}"] = {"auroc": auroc(sk, y), **boot_ci(lambda idx: auroc(sk[idx], y[idx]), scenes, B)}
        h2[f"primary_delta_{k}_minus_S1"] = {
            "delta": auroc(sk, y) - auroc(s1v, y),
            **boot_ci(lambda idx: auroc(sk[idx], y[idx]) - auroc(s1v[idx], y[idx]), scenes, B)}
    # pooled over all voting candidate rows, clustered by scene
    V = df[df.cand.isin(voters)].reset_index(drop=True)
    yv, cv = (V.err <= 20).values, V.scene.values
    for k in ("S1", "S2", "S3"):
        sv = V[k].values
        h2[f"pooled_auroc_{k}"] = {"auroc": auroc(sv, yv), **boot_ci(lambda idx: auroc(sv[idx], yv[idx]), cv, B)}
    for k in ("S2", "S3"):
        sk, s1v = V[k].values, V["S1"].values
        h2[f"pooled_delta_{k}_minus_S1"] = {
            "delta": auroc(sk, yv) - auroc(s1v, yv),
            **boot_ci(lambda idx: auroc(sk[idx], yv[idx]) - auroc(s1v[idx], yv[idx]), cv, B)}
    h2["supported"] = bool(h2["pooled_delta_S3_minus_S1"]["lo"] > 0)
    out["H2"] = h2

    # ---- H3: per-pair selection among voters
    Ev = E[voters]
    sr = (Ev <= 20).mean()
    best_single = str(sr.idxmax())
    base_ok = (Ev[best_single] <= 20).values
    h3 = {"best_single": best_single, "best_single_sr20": float(sr.max()),
          "oracle_sr20": float((Ev.min(axis=1) <= 20).mean())}
    selected = {}
    for k in ("S1", "S2", "S3"):
        Sv = S[k][voters].fillna(-np.inf)
        pick = Sv.idxmax(axis=1)
        ok = np.array([Ev.loc[p, pick[p]] <= 20 for p in pairs])
        selected[k] = pick
        b = int((ok & ~base_ok).sum())
        c = int((~ok & base_ok).sum())
        h3[f"select_{k}"] = {
            "sr20": float(ok.mean()), "gain_vs_best_single": float(ok.mean() - base_ok.mean()),
            "won": b, "lost": c, "mcnemar_p": mcnemar_exact(b, c),
            "gain_ci": boot_ci(lambda idx: ok[idx].mean() - base_ok[idx].mean(), scenes, B)}
    # rerun control: best-of-6 seeds of one matcher, selected by S1, gain over its seed-0 run
    ctrl = {}
    for bb in ("roma", "ma_roma"):
        cs = [f"{bb}|direct|none|s{s}" for s in range(6) if f"{bb}|direct|none|s{s}" in E.columns]
        if len(cs) < 2:
            continue
        pick = S["S1"][cs].fillna(-np.inf).idxmax(axis=1)
        ok = np.array([E.loc[p, pick[p]] <= 20 for p in pairs])
        s0 = (E[cs[0]] <= 20).values
        ctrl[bb] = {"n_seeds": len(cs), "sr20_seed0": float(s0.mean()), "sr20_best_of_seeds_by_S1": float(ok.mean()),
                    "gain": float(ok.mean() - s0.mean()),
                    "oracle_best_of_seeds": float((E[cs].min(axis=1) <= 20).mean())}
    h3["rerun_control"] = ctrl
    bar = max([v["gain"] for v in ctrl.values()], default=0.0)
    h3["rerun_bar"] = bar
    for k in ("S1", "S2", "S3"):
        r = h3[f"select_{k}"]
        r["supported"] = bool(r["mcnemar_p"] < 0.05 and r["gain_vs_best_single"] > bar)
    out["H3"] = h3

    # ---- triage: risk-coverage for the primary candidate and for the S3 selection
    tri = {}
    for label, score, yy in [
        ("primary_S1", S["S1"][PRIMARY].values, y20),
        ("primary_S3", S["S3"][PRIMARY].values, y20),
        ("selected_S3", np.array([S["S3"].loc[p, selected["S3"][p]] for p in pairs]),
         np.array([Ev.loc[p, selected["S3"][p]] <= 20 for p in pairs])),
    ]:
        d = {"base_rate": float(np.mean(yy)), "aurc": aurc(score, yy),
             "aurc_ci": boot_ci(lambda idx: aurc(score[idx], yy[idx]), scenes, B)}
        for cov in COVERAGES:
            d[f"accepted_success_at_{int(cov * 100)}"] = {
                "value": accepted_success(score, yy, cov),
                **boot_ci(lambda idx: accepted_success(score[idx], yy[idx], cov), scenes, B)}
        tri[label] = d
    out["triage"] = tri

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "summary.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    pp = meta.copy()
    pp["err_primary"] = E[PRIMARY]
    for k in ("S1", "S2", "S3"):
        pp[f"{k}_primary"] = S[k][PRIMARY]
        pp[f"pick_{k}"] = selected[k]
        pp[f"err_pick_{k}"] = [Ev.loc[p, selected[k][p]] for p in pairs]
    pp["err_best_single"] = Ev[best_single]
    pp["err_oracle"] = Ev.min(axis=1)
    pp["err_gt_homography"] = gt["homography"]
    pp.to_csv(outdir / "per_pair.csv")
    print(json.dumps({k: out[k] for k in ("n_pairs", "n_scenes", "ceiling")}, indent=1))
    print("H1", json.dumps(out["H1"]["auroc_S1_at_20"]), "transfer", json.dumps(out["H1"]["transfer"], default=float))
    print("H2", json.dumps({k: v for k, v in out["H2"].items() if "delta" in k or k == "supported"}, default=float))
    print("H3", json.dumps({k: v for k, v in out["H3"].items() if k.startswith("select") or k in ("best_single_sr20", "oracle_sr20", "rerun_bar")}, default=float))


if __name__ == "__main__":
    main()
