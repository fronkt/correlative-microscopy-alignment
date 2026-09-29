"""Phase A4 (exploratory): matcher-comparable scores from stored columns only.

Scores (higher = better), per (pair x voting candidate) row:
  nfa4   = -logNFA, N_tests = C(n_matches, 4), null p = pi*5.5^2 / (w_s*h_s)
  nfa1   = -logNFA, N_tests = 1
  nfa4t  = as nfa4 but p uses the TARGET image area (sensitivity / dead-end check)
  wilson = Wilson 95% lower bound of n_inliers / n_matches
  ninl   = raw n_inliers
  S1     = n_inliers / n_matches (reference)

Which frame the 5.5 px threshold lives in (read from code, not guessed): fit_transform is called with
src_xy = target-frame points and dst_xy = source-frame points, residuals/mask are measured in dst =
SOURCE frame (cma/estimators/consensus.py; run_triage_candidates.py run_one). Estimator: cv2.USAC_MAGSAC
(MAGSAC++), maxIters 10000, confidence 0.999, family auto (homography vs affine by BIC). For pyramid_v2
tile/zoom stages the correspondences are mapped back to the full source frame before fitting, so the
full-source area is the right (if slightly generous) denominator there.
"""
from __future__ import annotations

import json
import sys
from math import comb, log, pi
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp
from scipy.stats import binom, pearsonr, spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from analyze_triage import load, boot_ci  # noqa: E402
from cma.triage import auroc, mcnemar_exact  # noqa: E402

RANSAC_PX = 5.5
B = 10_000
ZW = 1.959963984540054
SCORES = ["S1", "nfa4", "nfa1", "nfa4t", "wilson", "ninl"]
n_fallback = {"count": 0}


def log_binom_tail(k: int, n: int, p: float) -> float:
    """log P(X >= k), X ~ Binom(n, p), log space, never silently -inf."""
    if k <= 0:
        return 0.0
    if k > n:
        return -np.inf
    p = min(max(p, 0.0), 1.0)
    v = float(binom.logsf(k - 1, n, p))
    if np.isfinite(v):
        return v
    n_fallback["count"] += 1
    ks = np.arange(k, n + 1)
    return float(logsumexp(binom.logpmf(ks, n, p)))


def log_choose(n: int, r: int) -> float:
    if n < r:
        return -np.inf
    return float(gammaln(n + 1) - gammaln(r + 1) - gammaln(n - r + 1))


def log_nfa(k: int, n: int, p: float, tests: str) -> float:
    lt = log_choose(n, 4) if tests == "c4" else 0.0
    return lt + log_binom_tail(k, n, p)


def wilson_lb(k: float, n: float) -> float:
    ph = k / n
    z2 = ZW * ZW
    return float((ph + z2 / (2 * n) - ZW * np.sqrt(ph * (1 - ph) / n + z2 / (4 * n * n))) / (1 + z2 / n))


def self_check() -> list[dict]:
    """logNFA vs an independent brute-force computation on 3 toy cases (exact integer arithmetic)."""
    from fractions import Fraction
    cases = [(5, 20, 0.1), (12, 40, 0.05), (3, 8, 0.5)]
    res = []
    for k, n, p in cases:
        pf = Fraction(p).limit_denominator(10**6)
        tail = sum(Fraction(comb(n, j)) * pf**j * (1 - pf) ** (n - j) for j in range(k, n + 1))
        brute = log(comb(n, 4)) + log(float(tail))
        mine = log_nfa(k, n, p, "c4")
        assert abs(brute - mine) < 1e-9, (k, n, p, brute, mine)
        brute1 = log(float(tail))
        assert abs(brute1 - log_nfa(k, n, p, "one")) < 1e-9
        res.append({"k": k, "n": n, "p": p, "logNFA_c4_brute": brute, "logNFA_c4_mine": mine})
    return res


def add_scores(df: pd.DataFrame) -> pd.DataFrame:
    ok = df.status.eq("ok").values
    nm = pd.to_numeric(df.n_matches, errors="coerce").values
    ni = pd.to_numeric(df.n_inliers, errors="coerce").values
    ps = np.minimum(1.0, pi * RANSAC_PX**2 / (df.w_s.values * df.h_s.values))
    pt = np.minimum(1.0, pi * RANSAC_PX**2 / (df.w_t.values * df.h_t.values))
    out = {k: np.full(len(df), -np.inf) for k in ("nfa4", "nfa1", "nfa4t", "wilson", "ninl")}
    for i in range(len(df)):
        if not (ok[i] and np.isfinite(nm[i]) and nm[i] > 0 and np.isfinite(ni[i])):
            continue
        n, k = int(nm[i]), int(ni[i])
        out["nfa4"][i] = -log_nfa(k, n, ps[i], "c4")
        out["nfa1"][i] = -log_nfa(k, n, ps[i], "one")
        out["nfa4t"][i] = -log_nfa(k, n, pt[i], "c4")
        out["wilson"][i] = wilson_lb(k, n)
        out["ninl"][i] = float(k)
    for k, v in out.items():
        df[k] = v
    return df


def main() -> None:
    check = self_check()
    print("self-check OK", check)
    df = load(ROOT / "results/triage/candidates.csv")
    df = add_scores(df)
    voters = sorted(df.loc[df.pool.isin(["core", "transform"]), "cand"].unique())
    assert len(voters) == 15
    pairs = sorted(df.pair_id.unique())
    assert len(pairs) == 187
    meta = df.drop_duplicates("pair_id").set_index("pair_id").loc[pairs]
    scenes = meta.scene.values
    out: dict = {"self_check": check, "B": B, "voters": voters,
                 "ransac": {"estimator": "cv2.USAC_MAGSAC (MAGSAC++)", "max_iters": 10000, "confidence": 0.999,
                            "family": "auto (homography|affine by BIC)", "threshold_px": RANSAC_PX,
                            "frame": "SOURCE image (dst of fit_transform); p uses w_s*h_s"},
                 "n_tests_note": "N_tests = C(n_matches,4) (homography minimal sample; affine would be C(n,3)); "
                                 "variant N_tests=1 also reported"}

    V = df[df.cand.isin(voters)].reset_index(drop=True)
    yv, cv = (V.err <= 20).values, V.scene.values
    ok_v = V.status.eq("ok").values

    # ---------------- saturation check
    sat = {}
    for bb, g in V[ok_v].groupby("backbone"):
        d = {"n_rows": int(len(g))}
        for s in ("nfa4", "nfa1"):
            d[s] = {"min": float(g[s].min()), "median": float(g[s].median()), "max": float(g[s].max()),
                    "spearman_vs_n_inliers": float(spearmanr(g[s], g.ninl)[0]),
                    "pearson_vs_n_inliers": float(pearsonr(g[s], g.ninl)[0]),
                    "spearman_vs_S1": float(spearmanr(g[s], g.S1)[0]),
                    "pearson_vs_S1": float(pearsonr(g[s], g.S1)[0])}
        d["median_n_matches"] = float(pd.to_numeric(g.n_matches).median())
        d["median_n_inliers"] = float(g.ninl.median())
        sat[bb] = d
    g = V[ok_v]
    sat["ALL_voting"] = {s: {"spearman_vs_n_inliers": float(spearmanr(g[s], g.ninl)[0]),
                             "spearman_vs_S1": float(spearmanr(g[s], g.S1)[0]),
                             "frac_gt_700": float((g[s] > 700).mean())} for s in ("nfa4", "nfa1")}
    sat["binom_logsf_fallbacks_used"] = int(n_fallback["count"])
    out["saturation"] = sat

    # ---------------- per-candidate AUROC (187 pairs each)
    E = df.pivot(index="pair_id", columns="cand", values="err").reindex(pairs)
    Sc = {k: df.pivot(index="pair_id", columns="cand", values=k).reindex(pairs) for k in SCORES}
    per_cand = {}
    for c in voters:
        y = (E[c] <= 20).values
        per_cand[c] = {"n_success": int(y.sum())}
        for k in SCORES:
            per_cand[c][k] = auroc(Sc[k][c].values, y)
    out["auroc_per_candidate"] = per_cand
    out["auroc_ma_roma_direct"] = per_cand["ma_roma|direct|none|s0"]

    # ---------------- pooled AUROC + paired diff vs S1
    pooled = {}
    for k in SCORES:
        sv = V[k].values
        pooled[k] = {"auroc": auroc(sv, yv), **boot_ci(lambda idx, sv=sv: auroc(sv[idx], yv[idx]), cv, B)}
    s1v = V["S1"].values
    for k in SCORES[1:]:
        sv = V[k].values
        pooled[k]["delta_vs_S1"] = {
            "delta": auroc(sv, yv) - auroc(s1v, yv),
            **boot_ci(lambda idx, sv=sv: auroc(sv[idx], yv[idx]) - auroc(s1v[idx], yv[idx]), cv, B)}
    out["pooled_auroc"] = pooled
    out["pooled_n_rows"] = int(len(V))

    # ---------------- per-pair pick
    Ev = E[voters]
    sr = (Ev <= 20).sum()
    best_single = str(sr.idxmax())
    base_ok = (Ev[best_single] <= 20).values
    oracle = int((Ev.min(axis=1) <= 20).sum())
    h3 = {"best_single": best_single, "best_single_n": int(sr.max()), "oracle_n": oracle}
    bb_of = {c: c.split("|")[0] for c in voters}
    picks = {}
    oks = {}
    for k in SCORES:
        Sv = Sc[k][voters].fillna(-np.inf)
        pick = Sv.idxmax(axis=1)
        picks[k] = pick
        oks[k] = np.array([Ev.loc[p, pick[p]] <= 20 for p in pairs])
    s1ok = oks["S1"]
    h3["select_S1_n"] = int(s1ok.sum())
    for k in SCORES:
        ok = oks[k]
        b, c_ = int((ok & ~base_ok).sum()), int((~ok & base_ok).sum())
        d = {"n": int(ok.sum()), "vs_best_single": {
            "diff": int(ok.sum() - base_ok.sum()), "won": b, "lost": c_, "mcnemar_p": mcnemar_exact(b, c_),
            "diff_rate_ci": boot_ci(lambda idx: ok[idx].mean() - base_ok[idx].mean(), scenes, B)}}
        b2, c2 = int((ok & ~s1ok).sum()), int((~ok & s1ok).sum())
        d["vs_S1"] = {"diff": int(ok.sum() - s1ok.sum()), "won": b2, "lost": c2, "mcnemar_p": mcnemar_exact(b2, c2),
                      "diff_rate_ci": boot_ci(lambda idx: ok[idx].mean() - s1ok[idx].mean(), scenes, B)}
        pb = pd.Series([bb_of[picks[k][p]] for p in pairs], index=pairs)
        d["n_picks_sift"] = int((pb == "sift").sum())
        d["picks_by_backbone"] = {bb: int(v) for bb, v in pb.value_counts().items()}
        d["wins_losses_vs_best_single_by_backbone"] = {
            bb: {"won": int(((pb == bb).values & ok & ~base_ok).sum()),
                 "lost": int(((pb == bb).values & ~ok & base_ok).sum())} for bb in sorted(set(bb_of.values()))}
        d["sift_pick_success"] = int((ok & (pb == "sift").values).sum())
        d["losses_vs_oracle"] = int((~ok & (Ev.min(axis=1) <= 20).values).sum())
        h3[f"select_{k}"] = d
    out["H3"] = h3

    out["forking_paths_note"] = "Exploratory on data already seen; 6 scores x (per-cand, pooled, pick) all reported."
    outdir = ROOT / "results/phaseA"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "a4_matcher_score.json").write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    V[["pair_id", "scene", "cand", "backbone", "pool", "n_matches", "n_inliers", "err"] + SCORES].to_csv(
        outdir / "a4_scores.csv", index=False)

    print(json.dumps(sat, indent=1, default=float))
    print("pooled", {k: (round(v["auroc"], 4), v["lo"], v["hi"]) for k, v in pooled.items()})
    for k in SCORES[1:]:
        print("delta", k, pooled[k]["delta_vs_S1"])
    print("best", best_single, h3["best_single_n"], "oracle", oracle, "S1", h3["select_S1_n"])
    for k in SCORES:
        d = h3[f"select_{k}"]
        print(k, d["n"], "vsBest", d["vs_best_single"]["diff"], round(d["vs_best_single"]["mcnemar_p"], 4),
              d["vs_best_single"]["diff_rate_ci"], "sift", d["n_picks_sift"], d["picks_by_backbone"])


if __name__ == "__main__":
    main()
