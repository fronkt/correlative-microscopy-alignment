"""Arm 1 analysis, exactly as pre-registered in prereg/arm1_banner_crop.md.

Run:  PYTHONPATH=src python scripts/analyze_arm1.py [--csv results/arm1/arm1.csv]
Writes results/arm1/arm1_summary.json and results/arm1/arm1_report.md next to the CSV.

Implements H-A1, H-A2 (primary + 3 Holm-corrected secondaries), H-A3 (+ the untested S1 AUROC), the sham control,
the Ti3AlC2 control, the reproduction check, the per-scene / per-offset-configuration tables and the failed-run
audit. Nothing else is tested; the only extras are labelled DESCRIPTIVE.

Interpretive choices the prereg leaves open are collected in DEVIATIONS / AMBIGUITIES and printed in the report.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

from cma.triage import auroc, ci95, cluster_bootstrap, grid_points, mcnemar_exact, transform_distance

ROOT = Path(__file__).resolve().parents[1]
TH = 20.0            # success: mu_ed <= 20 px
LOCK_ID_PX = 20.0    # near-identity: H closer than this to identity (5x5 grid on target); strict, as in the A6 audit
LOCK_GT_PX = 40.0    # ... and GT homography farther than this from identity
S1_CUT = 0.1711      # fixed S1 cut-off (H-A3)
ALPHA = 0.05
SHAM_RULE = 0.10
REPRO_PX = 1.0
REPRO_FRAC = 0.95
EXPECTED_ROWS = 708
DESIGN_GROUPS = ("SameSlice", "SerialSectioning")   # the other four groups are the "4 held-out groups"
CONFIGS = [("roma", "direct"), ("ma_roma", "direct"), ("roma", "pyramid_v2"), ("ma_roma", "pyramid_v2")]
PRIMARY = ("ma_roma", "direct")
SECONDARY = [c for c in CONFIGS if c != PRIMARY]
INFRA_RE = re.compile(r"cuda|out of memory|\boom\b|timeout|timed out", re.I)
I3 = np.eye(3)

CONFOUND = ("In AmalgaMatch, overlay, identical image size and TEM modality coincide perfectly. Arm 1 can show that "
            "removing the band changes the outcome; it cannot separate the band from anything else about these TEM images.")
CLUSTER_NOTE = ("A pair-level p-value treats the pairs as independent, which they are not (67 pairs from 4 scenes and about 5 "
                "GT-offset configurations); the tests are pair-level as pre-registered, and the per-scene and per-offset "
                "tables must be read alongside them.")

AMBIGUITIES = [
    "One-sided vs two-sided: the prereg calls the hypotheses one-sided and directional (alpha 0.05) but its support rule "
    "says 'p < 0.05 and locks fall / more successes'. The primary p here is the ONE-SIDED exact McNemar (binomial tail in the "
    "predicted direction); the two-sided exact p (cma.triage.mcnemar_exact) is reported beside it. A hypothesis is supported "
    "only if the one-sided p < 0.05 AND the direction is right; where the two-sided p would give a different verdict this is "
    "flagged in the table.",
    "Lock thresholds: 'within 20 px of identity' is implemented strictly (< 20), and 'more than 40 px' as > 40, matching the "
    "frozen A6 audit (NEAR_ID_PX / GT_DISP_MIN). The 5x5 grid uses the ORIGINAL target size (from the stored CJSJ row), "
    "because for cropped runs the Arm-1 row reports H in original coordinates but its h_t/w_t may describe the cropped image.",
    "Failed / missing runs: no H means not a success and not a lock. A row the CSV lacks entirely is treated the same way "
    "in the paired tables, but the row-count check fails loudly (see 'Row counts').",
    "Duplicate keys (an infrastructure rerun): the LAST row per (pair, backbone, mode, crop) is used, matching 'a second "
    "failure counts as a failure'. Number of duplicates is reported.",
    "Sham '>10%' rule: computed as (runs whose success indicator differs between uncropped and sham-cropped, in either "
    "direction) / (paired sham runs) > 0.10, i.e. more than 8 of 80. The sham McNemar p is two-sided (no predicted direction).",
    "Reproduction: a fresh-vs-stored pair 'matches' if both mu_ed are finite and differ by <= 1 px, or both runs failed. "
    "Denominator = every uncropped Arm-1 run (overlay-none, Ti3AlC2, sham-none).",
    "Ti3AlC2 'succeed in the CJSJ run': judged from the stored CJSJ rows for the same configuration; the fresh rerun is "
    "compared to that.",
    "Offset configuration: GT translation = H_gt(target centre) - target centre (source px), each component rounded to "
    "the nearest 20 px, grouped per scene; the found configurations are listed (the prereg says 'about 5').",
    "H-A3 S1 = n_inliers / n_matches (as in the triage study; -inf for failed runs); false accept = S1 >= 0.1711 and "
    "mu_ed > 20. The AUROC bootstrap is a PAIR-level paired bootstrap (B = 10000, seed 0), because the prereg says "
    "'paired bootstrap' and there are too few scenes for a cluster bootstrap; its CI is therefore optimistic.",
    "Holm correction: applied to the ONE-SIDED p-values of the 3 secondary H-A2 configurations.",
]


# ----------------------------------------------------------------------------- statistics helpers


def mcnemar_directional(n_pos: int, n_neg: int) -> float:
    """One-sided exact McNemar: P(X >= n_pos), X ~ Binomial(n_pos + n_neg, 1/2).

    n_pos = discordant pairs in the PREDICTED direction, n_neg = in the opposite direction.
    """
    n = n_pos + n_neg
    if n == 0:
        return 1.0
    return float(sum(comb(n, i) for i in range(n_pos, n + 1)) / 2.0**n)


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * pvals[i])
        adj[i] = min(1.0, run)
    return [float(a) for a in adj]


def paired_test(base: np.ndarray, treat: np.ndarray, want: str) -> dict:
    """Paired binary comparison. ``want`` = 'fewer' (predict treat < base) or 'more' (predict treat > base).

    n_pos = discordant pairs in the predicted direction; n_neg = opposite.
    """
    base, treat = np.asarray(base, bool), np.asarray(treat, bool)
    only_base, only_treat = int((base & ~treat).sum()), int((~base & treat).sum())
    n_pos, n_neg = (only_base, only_treat) if want == "fewer" else (only_treat, only_base)
    p1 = mcnemar_directional(n_pos, n_neg)
    p2 = mcnemar_exact(only_base, only_treat)
    return dict(n=int(len(base)), base_count=int(base.sum()), treat_count=int(treat.sum()),
                only_base=only_base, only_treat=only_treat, n_pos=n_pos, n_neg=n_neg,
                direction_ok=bool(n_pos > n_neg), p_one_sided=p1, p_two_sided=p2,
                supported=bool(p1 < ALPHA and n_pos > n_neg),
                supported_two_sided=bool(p2 < ALPHA and n_pos > n_neg))


# ----------------------------------------------------------------------------- lock definition


def dist_to_identity(H, grid: np.ndarray) -> float:
    """Mean distance (px) between H and the identity map over the grid; inf if H is missing / non-finite."""
    if H is None:
        return float("inf")
    return transform_distance(np.asarray(H, float), I3, grid)


def is_lock(H, mu_ed: float, gt_dist_id: float, grid: np.ndarray) -> bool:
    """Near-identity lock: H within LOCK_ID_PX of identity AND mu_ed > 20 AND GT farther than LOCK_GT_PX from identity."""
    if H is None or not np.isfinite(mu_ed):
        return False
    return bool(dist_to_identity(H, grid) < LOCK_ID_PX and mu_ed > TH and gt_dist_id > LOCK_GT_PX)


def parse_H(s) -> np.ndarray | None:
    if s is None or (isinstance(s, float) and np.isnan(s)) or str(s).strip() == "":
        return None
    try:
        a = np.array(json.loads(s), dtype=float)
    except (ValueError, TypeError):
        return None
    return a.reshape(3, 3) if a.size == 9 and np.all(np.isfinite(a)) else None


def s1_score(n_inliers, n_matches, ok: bool) -> float:
    return float(n_inliers) / float(n_matches) if ok and n_matches and n_matches > 0 else float("-inf")


# ----------------------------------------------------------------------------- data


def read_csv(path) -> pd.DataFrame:
    """'none' is a real value in these files (transform, crop): never let pandas turn it into NaN."""
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def read_pairs(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]


def build_meta(cand: pd.DataFrame) -> pd.DataFrame:
    """One row per pair: original target size, group, scene, GT homography, GT distance to identity, GT offset."""
    gt = cand[(cand.pool == "gt") & (cand["mode"] == "homography")].drop_duplicates("pair_id").set_index("pair_id")
    rows = {}
    for pid, r in gt.iterrows():
        ht, wt = int(float(r.h_t)), int(float(r.w_t))
        grid = grid_points(ht, wt, 5)
        Hg = parse_H(r.H)
        d_id = dist_to_identity(Hg, grid)
        if Hg is not None:
            c = np.array([[(wt - 1) / 2.0, (ht - 1) / 2.0]])
            p = np.hstack([c, [[1.0]]]) @ Hg.T
            off = p[0, :2] / p[0, 2] - c[0]
        else:
            off = np.array([np.nan, np.nan])
        rows[pid] = dict(group=r.group, scene=r.scene, h_t=ht, w_t=wt, gt_dist_id=d_id, off_x=off[0], off_y=off[1],
                         cfg=(f"({int(np.round(off[0] / 20) * 20)},{int(np.round(off[1] / 20) * 20)})"
                              if np.all(np.isfinite(off)) else "n/a"))
    return pd.DataFrame.from_dict(rows, orient="index")


def prepare_runs(df: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Per-row derived indicators: ok, mu_ed, success, lock, S1."""
    out = df.copy()
    out["mu"] = pd.to_numeric(out["mu_ed"], errors="coerce")
    Hs = [parse_H(s) for s in out["H"]]
    out["ok"] = [(st == "ok") and (h is not None) and np.isfinite(m) for st, h, m in zip(out["status"], Hs, out["mu"])]
    out["success"] = out.ok & (out.mu <= TH)
    locks, s1s = [], []
    for (_, r), h, ok in zip(out.iterrows(), Hs, out["ok"]):
        m = meta.loc[r.pair_id] if r.pair_id in meta.index else None
        if m is None or not ok:
            locks.append(False)
        else:
            locks.append(is_lock(h, r.mu, m.gt_dist_id, grid_points(int(m.h_t), int(m.w_t), 5)))
        s1s.append(s1_score(pd.to_numeric(r.get("n_inliers", ""), errors="coerce"),
                            pd.to_numeric(r.get("n_matches", ""), errors="coerce"), ok))
    out["lock"] = locks
    out["S1"] = s1s
    return out


def dedupe(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    key = ["pair_id", "backbone", "mode", "crop"]
    n_dup = int(df.duplicated(key, keep="last").sum())
    return df.drop_duplicates(key, keep="last").reset_index(drop=True), n_dup


def paired_frame(runs: pd.DataFrame, pairs: list[str], configs, base_crop: str, treat_crop: str) -> pd.DataFrame:
    """One row per (pair, config) with the base and treated indicators (missing row = failure)."""
    idx = {(r.pair_id, r.backbone, r["mode"], r.crop): r for _, r in runs.iterrows()}
    recs = []
    for pid in pairs:
        for bb, mo in configs:
            b, t = idx.get((pid, bb, mo, base_crop)), idx.get((pid, bb, mo, treat_crop))
            recs.append(dict(pair_id=pid, backbone=bb, mode=mo, cfg=f"{bb}|{mo}",
                             have_base=b is not None, have_treat=t is not None,
                             succ_base=bool(b is not None and b.success), succ_treat=bool(t is not None and t.success),
                             lock_base=bool(b is not None and b.lock), lock_treat=bool(t is not None and t.lock),
                             S1_base=(b.S1 if b is not None else float("-inf")),
                             S1_treat=(t.S1 if t is not None else float("-inf")),
                             mu_base=(b.mu if b is not None else np.nan), mu_treat=(t.mu if t is not None else np.nan)))
    return pd.DataFrame(recs)


# ----------------------------------------------------------------------------- analyses


def analyse_ha1(P: pd.DataFrame) -> dict:
    t = paired_test(P.lock_base.values, P.lock_treat.values, "fewer")
    t["hypothesis"] = "H-A1"
    t["criterion"] = "one-sided exact McNemar p < 0.05 and near-identity locks fall when cropped"
    return t


def analyse_ha2(P: pd.DataFrame, sham_flag: bool) -> dict:
    prim = P[P.cfg == "%s|%s" % PRIMARY]
    res = {"primary": paired_test(prim.succ_base.values, prim.succ_treat.values, "more")}
    sec = {}
    for c in SECONDARY:
        s = P[P.cfg == "%s|%s" % c]
        sec[f"{c[0]}|{c[1]}"] = paired_test(s.succ_base.values, s.succ_treat.values, "more")
    adj = holm([v["p_one_sided"] for v in sec.values()])
    for (k, v), a in zip(sec.items(), adj):
        v["p_holm"] = a
        v["supported"] = bool(a < ALPHA and v["n_pos"] > v["n_neg"])
    res["secondary"] = sec
    res["sham_confounded"] = bool(sham_flag)
    res["primary"]["reported_as"] = ("cropping changes results generally; NOT attributed to the overlay (sham rule tripped)"
                                     if sham_flag else ("supported" if res["primary"]["supported"] else "not supported"))
    return res


def analyse_sham(P: pd.DataFrame) -> dict:
    disc = P.succ_base.values != P.succ_treat.values
    n = len(P)
    frac = float(disc.sum() / n) if n else float("nan")
    t = paired_test(P.succ_base.values, P.succ_treat.values, "more")
    return dict(n_runs=n, discordant=int(disc.sum()), fraction_changed=frac, successes_uncropped=t["base_count"],
                successes_sham=t["treat_count"], only_uncropped=t["only_base"], only_sham=t["only_treat"],
                p_two_sided=t["p_two_sided"], crop_itself_matters=bool(n > 0 and frac > SHAM_RULE),
                per_config={c: dict(n=int(len(g)), discordant=int((g.succ_base != g.succ_treat).sum()))
                            for c, g in P.groupby("cfg")})


def fa_indicator(S1: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """False accept: S1 >= fixed cut-off and mu_ed > 20 (failed run: -inf S1 -> never accepted)."""
    return (np.asarray(S1, float) >= S1_CUT) & (np.asarray(mu, float) > TH)


def analyse_ha3(runs: pd.DataFrame, cand: pd.DataFrame, meta: pd.DataFrame, overlay_pairs: list[str], B: int = 10_000) -> dict:
    held = sorted(p for p in meta.index if meta.loc[p, "group"] not in DESIGN_GROUPS)
    # stored CJSJ MA-RoMa direct rows (core pool, seed 0, transform none)
    st = cand[(cand.pool == "core") & (cand.backbone == PRIMARY[0]) & (cand["mode"] == PRIMARY[1]) &
              (cand["transform"] == "none") & (cand.seed == "0")].drop_duplicates("pair_id").set_index("pair_id")
    fresh_unc = runs[(runs.backbone == PRIMARY[0]) & (runs["mode"] == PRIMARY[1]) & (runs.crop == "none")].set_index("pair_id")
    fresh_ov = runs[(runs.backbone == PRIMARY[0]) & (runs["mode"] == PRIMARY[1]) & (runs.crop == "overlay")].set_index("pair_id")
    rows, source_counts = [], {"fresh_uncropped": 0, "stored_CJSJ": 0, "fresh_cropped": 0, "missing": 0}
    for p in held:
        def stored_row(p=p):
            if p not in st.index:
                return None
            r = st.loc[p]
            mu = pd.to_numeric(r.mu_ed, errors="coerce")
            ok = (r.status == "ok") and np.isfinite(mu) and parse_H(r.H) is not None
            return dict(S1=s1_score(pd.to_numeric(r.n_inliers, errors="coerce"),
                                    pd.to_numeric(r.n_matches, errors="coerce"), ok), mu=mu if ok else np.inf, ok=ok)
        if p in fresh_unc.index:
            r = fresh_unc.loc[p]
            unc, unc_src = dict(S1=r.S1, mu=r.mu if r.ok else np.inf, ok=bool(r.ok)), "fresh_uncropped"
        else:
            unc, unc_src = stored_row(), "stored_CJSJ"
            if unc is None:
                unc, unc_src = dict(S1=float("-inf"), mu=np.inf, ok=False), "missing"
        source_counts[unc_src] += 1
        if p in overlay_pairs:
            if p in fresh_ov.index:
                r = fresh_ov.loc[p]
                crp, crp_src = dict(S1=r.S1, mu=r.mu if r.ok else np.inf, ok=bool(r.ok)), "fresh_cropped"
            else:
                crp, crp_src = dict(S1=float("-inf"), mu=np.inf, ok=False), "missing"
        else:
            crp, crp_src = unc, unc_src   # same row in both arms
        if p in overlay_pairs:
            source_counts["fresh_cropped" if crp_src == "fresh_cropped" else "missing"] += 1
        rows.append(dict(pair_id=p, group=meta.loc[p, "group"], scene=meta.loc[p, "scene"], overlay=p in overlay_pairs,
                         unc_src=unc_src, S1_unc=unc["S1"], mu_unc=unc["mu"], S1_crp=crp["S1"], mu_crp=crp["mu"]))
    H = pd.DataFrame(rows)
    H["fa_unc"] = fa_indicator(H.S1_unc, H.mu_unc)
    H["fa_crp"] = fa_indicator(H.S1_crp, H.mu_crp)
    t = paired_test(H.fa_unc.values, H.fa_crp.values, "fewer")
    t.update(hypothesis="H-A3", criterion="one-sided exact McNemar p < 0.05 and fewer false accepts with cropped rows",
             cutoff=S1_CUT, n_heldout_pairs=len(H), rows_by_source=source_counts,
             n_not_rerun_stored=int((H.unc_src == "stored_CJSJ").sum()),
             n_not_rerun_stored_overlay_pairs=int(((H.unc_src == "stored_CJSJ") & H.overlay).sum()),
             fresh_unc_heldout=int((H.unc_src == "fresh_uncropped").sum()))
    # descriptive: false accepts by overlay membership
    t["false_accepts_overlay_pairs"] = dict(uncropped=int(H[H.overlay].fa_unc.sum()), cropped=int(H[H.overlay].fa_crp.sum()))
    t["false_accepts_other_pairs"] = int(H[~H.overlay].fa_unc.sum())
    # AUROC on DislocationCharacterization (reported, not tested)
    D = H[H.group == "DislocationCharacterization"].reset_index(drop=True)
    y_u, y_c = (D.mu_unc <= TH).values, (D.mu_crp <= TH).values
    a_u, a_c = auroc(D.S1_unc.values, y_u), auroc(D.S1_crp.values, y_c)

    def diff(idx):
        return auroc(D.S1_crp.values[idx], y_c[idx]) - auroc(D.S1_unc.values[idx], y_u[idx])
    if len(D) and np.isfinite(a_u) and np.isfinite(a_c):
        boot = cluster_bootstrap(diff, np.arange(len(D)), B=B, seed=0)
        lo, hi = ci95(boot) if len(boot) else (float("nan"), float("nan"))
    else:
        lo = hi = float("nan")
    t["auroc_dislocation"] = dict(n=int(len(D)), uncropped=a_u, cropped=a_c, diff=(a_c - a_u), diff_ci95=[lo, hi],
                                  note="reported, not tested; pair-level paired bootstrap")
    t["_table"] = H
    return t


def analyse_reproduction(runs: pd.DataFrame, cand: pd.DataFrame, pair_sets: dict[str, list[str]]) -> dict:
    st = cand[(cand.pool == "core") & (cand["transform"] == "none") & (cand.seed == "0")]
    st = st.set_index(["pair_id", "backbone", "mode"])
    st = st[~st.index.duplicated()]
    unc = runs[runs.crop == "none"]
    recs = []
    for _, r in unc.iterrows():
        k = (r.pair_id, r.backbone, r["mode"])
        if k not in st.index:
            recs.append(dict(pair_id=r.pair_id, cfg=f"{r.backbone}|{r['mode']}", match=False, note="no stored row"))
            continue
        s = st.loc[k]
        mu_s = pd.to_numeric(s.mu_ed, errors="coerce")
        ok_s = (s.status == "ok") and np.isfinite(mu_s)
        if r.ok and ok_s:
            m = bool(abs(r.mu - mu_s) <= REPRO_PX)
        else:
            m = bool((not r.ok) and (not ok_s))
        recs.append(dict(pair_id=r.pair_id, cfg=f"{r.backbone}|{r['mode']}", match=m, note="",
                         fresh=(r.mu if r.ok else np.nan), stored=(mu_s if ok_s else np.nan),
                         set=next((n for n, ps in pair_sets.items() if r.pair_id in ps), "?")))
    R = pd.DataFrame(recs)
    n = len(R)
    frac = float(R.match.mean()) if n else float("nan")
    return dict(n=n, matched=int(R.match.sum()), fraction=frac, threshold=REPRO_FRAC, reproduces=bool(n and frac >= REPRO_FRAC),
                per_config={c: dict(n=int(len(g)), matched=int(g.match.sum())) for c, g in R.groupby("cfg")} if n else {},
                per_set={c: dict(n=int(len(g)), matched=int(g.match.sum())) for c, g in R.groupby("set")} if "set" in R else {})


def analyse_ti3(runs: pd.DataFrame, cand: pd.DataFrame, pairs: list[str]) -> dict:
    st = cand[(cand.pool == "core") & (cand["transform"] == "none") & (cand.seed == "0")].set_index(["pair_id", "backbone", "mode"])
    out = []
    for pid in pairs:
        for bb, mo in CONFIGS:
            f = runs[(runs.pair_id == pid) & (runs.backbone == bb) & (runs["mode"] == mo) & (runs.crop == "none")]
            s = st.loc[(pid, bb, mo)] if (pid, bb, mo) in st.index else None
            mu_s = pd.to_numeric(s.mu_ed, errors="coerce") if s is not None else np.nan
            out.append(dict(pair_id=pid, cfg=f"{bb}|{mo}", have_fresh=len(f) > 0,
                            fresh_success=bool(len(f) and f.iloc[-1].success), fresh_mu=(float(f.iloc[-1].mu) if len(f) else np.nan),
                            stored_success=bool(s is not None and s.status == "ok" and np.isfinite(mu_s) and mu_s <= TH),
                            stored_mu=float(mu_s) if np.isfinite(mu_s) else np.nan))
    T = pd.DataFrame(out)
    return dict(n=int(len(T)), fresh_success=int(T.fresh_success.sum()), stored_success=int(T.stored_success.sum()),
                stored_success_and_fresh_success=int((T.stored_success & T.fresh_success).sum()),
                reproduces_environment=bool(len(T) and (T.fresh_success == T.stored_success).all()), _table=T)


def failed_run_audit(raw: pd.DataFrame) -> dict:
    bad = raw[raw.status != "ok"]
    groups: dict[str, dict] = {}
    for _, r in bad.iterrows():
        msg = re.sub(r"\s+", " ", str(r.get("error", ""))).strip() or "(empty error message)"
        g = groups.setdefault(msg, dict(count=0, infrastructure=bool(INFRA_RE.search(msg)), by_crop={}))
        g["count"] += 1
        g["by_crop"][r.crop] = g["by_crop"].get(r.crop, 0) + 1
    return dict(n_failed_rows=int(len(bad)), groups=[dict(error=k, **v) for k, v in sorted(groups.items(), key=lambda kv: -kv[1]["count"])],
                infrastructure_rows=int(sum(v["count"] for v in groups.values() if v["infrastructure"])))


def breakdown(P: pd.DataFrame, meta: pd.DataFrame, by: str) -> pd.DataFrame:
    """Overlay-pair breakdown by 'scene' or 'cfg' (offset configuration): locks pooled over 4 configs, MA-RoMa direct successes."""
    P = P.copy()
    P["grp"] = P.pair_id.map(meta[by] if by != "cfg" else (meta.scene + " " + meta.cfg))
    rows = []
    for g, d in P.groupby("grp"):
        m = d[d.cfg == "%s|%s" % PRIMARY]
        rows.append(dict(group=g, pairs=int(d.pair_id.nunique()), runs=len(d),
                         locks_uncropped=int(d.lock_base.sum()), locks_cropped=int(d.lock_treat.sum()),
                         maroma_direct_success_uncropped=int(m.succ_base.sum()), maroma_direct_success_cropped=int(m.succ_treat.sum()),
                         all_success_uncropped=int(d.succ_base.sum()), all_success_cropped=int(d.succ_treat.sum())))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- report


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join((f"{v:.3g}" if isinstance(v, float) else str(v)).replace("|", "\\|") for v in r.values) + " |")
    return "\n".join(lines)


def yn(b: bool) -> str:
    return "YES" if b else "NO"


def fmt_p(p: float) -> str:
    return f"{p:.4g}"


def render_report(S: dict, breakdowns: dict, csv_path: str) -> str:
    L: list[str] = []
    a = L.append
    a("# Arm 1 report: does cropping burned-in overlays remove the near-identity lock?")
    a("")
    a(f"Source CSV: `{csv_path}`. Analysis follows `prereg/arm1_banner_crop.md` exactly; the interpretive choices are listed at the end.")
    a("")
    rc = S["row_counts"]
    a("## Row counts")
    a(f"- Rows in file: {rc['n_rows']}; unique runs after dropping repeated attempts: {rc['n_unique']}; expected {EXPECTED_ROWS}: **{'OK' if rc['ok'] else 'MISMATCH'}**.")
    a(f"- Repeated attempts of the same run (last one used): {rc['n_duplicates']}.")
    a("- Unique runs by job (pair list x crop), observed / expected: " +
      "; ".join(f"{k} {v['observed']}/{v['expected']}" for k, v in rc["by_job"].items()))
    a(f"- Rows with a commit hash: {', '.join(rc['git_commits']) or 'none recorded'}.")
    a("")
    fr = S["failed_runs"]
    a("## Failed runs")
    a(f"{fr['n_failed_rows']} rows have status other than ok ({fr['infrastructure_rows']} of them with CUDA/OOM/timeout-type messages = infrastructure). By the prereg every failed run counts as a failure, and an infrastructure failure should have been rerun once.")
    if fr["groups"]:
        a("")
        a(md_table(pd.DataFrame([dict(count=g["count"], infrastructure=yn(g["infrastructure"]), by_crop=g["by_crop"], error=g["error"][:120]) for g in fr["groups"]])))
    a("")
    a("## H-A1: the lock goes away (67 overlay pairs x 4 configurations)")
    h = S["H-A1"]
    a(f"- Near-identity locks: uncropped {h['base_count']}, cropped {h['treat_count']} (of {h['n']} paired runs).")
    a(f"- Discordant pairs: lock only uncropped {h['only_base']}, lock only cropped {h['only_treat']}.")
    a(f"- Exact McNemar: one-sided p = {fmt_p(h['p_one_sided'])} (two-sided {fmt_p(h['p_two_sided'])}).")
    a(f"- Criterion: {h['criterion']}. **Supported: {yn(h['supported'])}.**"
      + ("" if h["supported"] == h["supported_two_sided"] else f" (Verdict would differ on the two-sided p: {yn(h['supported_two_sided'])}.)"))
    a("")
    a("## H-A2: success rises (primary: MA-RoMa direct, 67 overlay pairs)")
    h2 = S["H-A2"]
    p = h2["primary"]
    a(f"- SR@20 uncropped {p['base_count']}/{p['n']}, cropped {p['treat_count']}/{p['n']}; success only when cropped {p['only_treat']}, only when uncropped {p['only_base']}.")
    a(f"- Exact McNemar: one-sided p = {fmt_p(p['p_one_sided'])} (two-sided {fmt_p(p['p_two_sided'])}). Criterion: p < 0.05 and more successes.")
    a(f"- **Supported: {yn(p['supported'])}.** Reported as: {p['reported_as']}."
      + ("" if p["supported"] == p["supported_two_sided"] else f" (Verdict would differ on the two-sided p: {yn(p['supported_two_sided'])}.)"))
    a("")
    a("Secondary configurations (same test; Holm correction across the 3, on the one-sided p):")
    a("")
    a(md_table(pd.DataFrame([dict(config=k, successes_uncropped=v["base_count"], successes_cropped=v["treat_count"],
                                  only_cropped=v["only_treat"], only_uncropped=v["only_base"], p_one_sided=v["p_one_sided"],
                                  p_holm=v["p_holm"], supported=yn(v["supported"])) for k, v in h2["secondary"].items()])))
    a("")
    a("## H-A3: triage improves (S1 cut-off fixed at 0.1711, MA-RoMa direct, 4 held-out groups)")
    h3 = S["H-A3"]
    a(f"- Held-out pairs: {h3['n_heldout_pairs']}. False accepts, all-uncropped arm {h3['base_count']}; cropped-overlay arm {h3['treat_count']}.")
    a(f"- Discordant: false accept only uncropped {h3['only_base']}, only in cropped arm {h3['only_treat']}.")
    a(f"- Exact McNemar: one-sided p = {fmt_p(h3['p_one_sided'])} (two-sided {fmt_p(h3['p_two_sided'])}). Criterion: p < 0.05 and fewer false accepts. **Supported: {yn(h3['supported'])}.**"
      + ("" if h3["supported"] == h3["supported_two_sided"] else f" (Verdict would differ on the two-sided p: {yn(h3['supported_two_sided'])}.)"))
    a(f"- Overlay pairs alone: false accepts {h3['false_accepts_overlay_pairs']['uncropped']} uncropped vs {h3['false_accepts_overlay_pairs']['cropped']} cropped; other held-out pairs (identical rows in both arms): {h3['false_accepts_other_pairs']}.")
    au = h3["auroc_dislocation"]
    a(f"- Reported, not tested: S1 AUROC on DislocationCharacterization (n = {au['n']}): uncropped {au['uncropped']:.3f}, cropped {au['cropped']:.3f}, difference {au['diff']:+.3f}, paired-bootstrap 95% CI [{au['diff_ci95'][0]:.3f}, {au['diff_ci95'][1]:.3f}].")
    a("")
    a("> **DEVIATION NOTE for the human to adjudicate (not decided silently).** The prereg says H-A3 uses \"the fresh uncropped rows for all other held-out pairs\", but Arm 1 only reran the 67 overlay pairs, the 3 Ti3AlC2 pairs and 20 sham pairs.")
    a(f"> For the other held-out pairs I used the STORED CJSJ MA-RoMa direct rows (core pool, seed 0): {h3['n_not_rerun_stored']} of {h3['n_heldout_pairs']} held-out pairs (fresh uncropped rows were available for {h3['fresh_unc_heldout']}); of the stored ones {h3['n_not_rerun_stored_overlay_pairs']} are overlay pairs (should be 0).")
    a("> Because the cropped and uncropped arms use the SAME row for every non-overlay pair, these pairs are concordant and cannot affect the McNemar test; they do affect the absolute false-accept counts and the DislocationCharacterization AUROC. The reproduction check below shows how far stored and fresh rows agree.")
    a("")
    a("## Controls and validity checks")
    sh = S["sham"]
    a("### Sham crop (20 pairs x 4 configurations)")
    a(f"- Paired runs {sh['n_runs']}; successes uncropped {sh['successes_uncropped']}, sham-cropped {sh['successes_sham']}; success only uncropped {sh['only_uncropped']}, only sham {sh['only_sham']}; two-sided McNemar p = {fmt_p(sh['p_two_sided'])}.")
    a(f"- Runs whose success changed: {sh['discordant']} = {100 * sh['fraction_changed']:.1f}% (rule: more than 10% means the crop itself matters). **Crop itself matters: {yn(sh['crop_itself_matters'])}.**")
    if sh["crop_itself_matters"]:
        a("- Consequence: H-A2 is reported as \"cropping changes results generally\" and is not attributed to the overlay.")
    a("")
    t3 = S["ti3alc2"]
    a("### Ti3AlC2 (3 TEM pairs without a banner, uncropped)")
    a(f"- {t3['n']} runs; successes in fresh rerun {t3['fresh_success']}, in stored CJSJ rows {t3['stored_success']}; success in both {t3['stored_success_and_fresh_success']}. Environment reproduces on this control (fresh success = stored success on every run): **{yn(t3['reproduces_environment'])}**.")
    a("")
    a(md_table(S["_ti3_table"]))
    a("")
    rp = S["reproduction"]
    a("### Reproduction (fresh uncropped vs stored CJSJ mu_ed)")
    a(f"- Matched within 1 px on {rp['matched']}/{rp['n']} = {100 * rp['fraction']:.1f}% of runs (needs at least 95%). **Reproduces: {yn(rp['reproduces'])}.** Comparisons stay fresh-vs-fresh either way.")
    a("- Per configuration: " + "; ".join(f"{k} {v['matched']}/{v['n']}" for k, v in rp["per_config"].items()))
    a("- Per pair set: " + "; ".join(f"{k} {v['matched']}/{v['n']}" for k, v in rp["per_set"].items()))
    a("")
    a("### The confound, stated in advance (verbatim from the prereg)")
    a(f"> {CONFOUND}")
    a("")
    a("### Clustering")
    a(f"{CLUSTER_NOTE}")
    a("")
    a("## Per-scene breakdown (67 overlay pairs; locks pooled over the 4 configurations)")
    a("")
    a(md_table(breakdowns["scene"]))
    a("")
    a("## Per offset-configuration breakdown (scene + GT translation rounded to the nearest 20 px)")
    a("")
    a(md_table(breakdowns["cfg"]))
    a("")
    a("## Interpretation notes and choices the prereg left open")
    for x in AMBIGUITIES:
        a(f"- {x}")
    a("")
    a("(Everything above the interpretation notes is a pre-registered quantity; the breakdown tables are required descriptive reporting, not tests.)")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/arm1/arm1.csv")
    ap.add_argument("--candidates", default="results/triage/candidates.csv")
    ap.add_argument("--pairs-dir", default="results/arm1")
    ap.add_argument("--out-prefix", default="", help="default: <csv dir>/arm1")
    ap.add_argument("--allow-partial", action="store_true", help="do not abort when the row count differs from 708")
    ap.add_argument("--boot", type=int, default=10_000)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = ROOT / csv_path
    pdir = Path(args.pairs_dir) if Path(args.pairs_dir).is_absolute() else ROOT / args.pairs_dir
    cand = read_csv(ROOT / args.candidates if not Path(args.candidates).is_absolute() else args.candidates)
    raw = read_csv(csv_path)
    prefix = Path(args.out_prefix) if args.out_prefix else csv_path.parent / "arm1"

    overlay = read_pairs(pdir / "pairs_overlay.txt")
    ti3 = read_pairs(pdir / "pairs_control_ti3alc2.txt")
    sham = read_pairs(pdir / "pairs_sham.txt")
    assert (len(overlay), len(ti3), len(sham)) == (67, 3, 20), (len(overlay), len(ti3), len(sham))

    meta = build_meta(cand)
    print(f"offset-configuration grouping (GT translation / 20 px, per scene), overlay pairs:")
    cfgs = (meta.loc[overlay, "scene"] + " " + meta.loc[overlay, "cfg"]).value_counts()
    for k, v in cfgs.items():
        print(f"  {k}: {v} pairs")

    runs, n_dup = dedupe(raw)
    jobs = {"overlay|none": (overlay, "none"), "overlay|overlay": (overlay, "overlay"), "ti3alc2|none": (ti3, "none"),
            "sham|none": (sham, "none"), "sham|sham": (sham, "sham")}
    by_job = {}
    for name, (ps, crop) in jobs.items():
        obs = int(((runs.crop == crop) & runs.pair_id.isin(ps)).sum())
        by_job[name] = dict(observed=obs, expected=len(ps) * len(CONFIGS))
    stray = int(len(runs) - sum(v["observed"] for v in by_job.values()))
    row_ok = bool(len(runs) == EXPECTED_ROWS and all(v["observed"] == v["expected"] for v in by_job.values()) and stray == 0)
    row_counts = dict(n_rows=int(len(raw)), n_unique=int(len(runs)), n_duplicates=n_dup, expected=EXPECTED_ROWS, ok=row_ok,
                      by_job=by_job, stray_rows=stray,
                      git_commits=sorted(set(raw.get("git_commit", pd.Series(dtype=str))) - {""}))

    runs = prepare_runs(runs, meta)
    P_ov = paired_frame(runs, overlay, CONFIGS, "none", "overlay")
    P_sh = paired_frame(runs, sham, CONFIGS, "none", "sham")

    S: dict = {"csv": str(csv_path), "row_counts": row_counts, "failed_runs": failed_run_audit(raw)}
    S["sham"] = analyse_sham(P_sh)
    S["H-A1"] = analyse_ha1(P_ov)
    S["H-A2"] = analyse_ha2(P_ov, S["sham"]["crop_itself_matters"])
    h3 = analyse_ha3(runs, cand, meta, overlay, B=args.boot)
    S["H-A3"] = {k: v for k, v in h3.items() if k != "_table"}
    t3 = analyse_ti3(runs, cand, ti3)
    S["ti3alc2"] = {k: v for k, v in t3.items() if k != "_table"}
    S["_ti3_table"] = t3["_table"].round(2)
    S["reproduction"] = analyse_reproduction(runs, cand, {"overlay": overlay, "ti3alc2": ti3, "sham": sham})
    S["offset_configurations"] = cfgs.to_dict()
    S["ambiguities"] = AMBIGUITIES
    S["confound_caveat"] = CONFOUND

    breakdowns = {"scene": breakdown(P_ov, meta, "scene"), "cfg": breakdown(P_ov, meta, "cfg")}
    report = render_report(S, breakdowns, str(csv_path))
    if not row_ok:
        report = (f"> **WARNING: row-count check failed (expected {EXPECTED_ROWS} unique runs, see 'Row counts'). "
                  f"Results below are partial and NOT the pre-registered analysis.**\n\n") + report
    S.pop("_ti3_table")
    Path(f"{prefix}_summary.json").write_text(json.dumps(S, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                                              encoding="utf-8")
    Path(f"{prefix}_report.md").write_text(report, encoding="utf-8")
    print(f"wrote {prefix}_summary.json and {prefix}_report.md")
    if not row_ok and not args.allow_partial:
        sys.exit(f"ROW COUNT CHECK FAILED: {row_counts} (outputs were still written; rerun with --allow-partial to silence)")


if __name__ == "__main__":
    main()
