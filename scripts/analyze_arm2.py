"""Arm 2 analysis, exactly as pre-registered in prereg/arm2_materials_replication.md.

Run (after the GPU run AND after Frank's NIST hand-check):
  PYTHONPATH=src python scripts/analyze_arm2.py [--candidates results/arm2/candidates.csv]
Gate only (compute the NIST GT gate for the addendum, needs no candidate rows):
  PYTHONPATH=src python scripts/analyze_arm2.py --gate-only
Writes results/arm2/arm2_report.md and arm2_summary.json (and gate_result.json).

Implements H2-1..H2-4 (one-sided exact McNemar, cluster bootstrap B = 10,000 over the manifest `cluster` column),
both success thresholds (20 px primary; 1% of the source diagonal secondary), P alone / N alone (descriptive),
the oracle best-of-15, the GT ceiling rows, and the NIST GT gate. Without Frank's hand-check CSV the primary analysis
REFUSES to run; --no-gate-dry-run produces a clearly stamped DRY RUN (files named arm2_DRYRUN_*) that is not the
pre-registered analysis. Ambiguities are collected in AMBIGUITIES and printed in the report.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

from cma.materials_pool import DEFAULT_ROOT, ROOT_ENV, resolve_path
from cma.triage import auroc, ci95, cluster_bootstrap, mcnemar_exact

ROOT = Path(__file__).resolve().parents[1]

# ----------------------------------------------------------------------------- pre-registered constants
TH_PX = 20.0                    # primary success threshold: mu_ed <= 20 px
DIAG_FRAC = 0.01                # secondary: mu_ed <= 1% of the source-image diagonal
ALPHA = 0.05
B_DEFAULT = 10_000
MIN_CLASS = 8                   # H2-1 is untestable with < 8 successes or < 8 failures
AUROC_MIN = 0.80
# Carried over UNCHANGED from AmalgaMatch, never re-fitted (prereg "Parameters carried over unchanged"):
S1_CUT = 0.1711                 # Youden point on SameSlice + SerialSectioning in the CJSJ run
BEST_SINGLE = ("ma_roma", "pyramid_v2", "none")   # best single candidate, chosen on AmalgaMatch (prereg)
H21_CAND = ("ma_roma", "direct", "none")          # H2-1 / H2-2 score MA-RoMa direct
GATE_PX = 10.0                  # NIST GT gate: median hand-vs-GT error <= 10 px (half the 20 px threshold), target px
N_DEFAULT_PAIRS = 95            # 14 (P) + 81 (N), nominal
ROWS_PER_PAIR = 17              # 15 voting + 2 GT rows

# The 15 voting candidates in the prereg's listing order; R breaks ties toward the FIRST listed.
# Core: SIFT, LoFTR, RoMa, MA-RoMa, MatchAnything direct, then RoMa and MA-RoMa pyramid_v2;
# transform pool: RoMa then MA-RoMa direct on invert, histmatch, CLAHE, gradmag.
VOTERS: list[tuple[str, str, str]] = (
    [(b, "direct", "none") for b in ("sift", "loftr", "roma", "ma_roma", "matchanything")]
    + [(b, "pyramid_v2", "none") for b in ("roma", "ma_roma")]
    + [(b, "direct", t) for b in ("roma", "ma_roma") for t in ("invert", "histmatch", "clahe", "gradmag")]
)
assert len(VOTERS) == 15 and H21_CAND in VOTERS and BEST_SINGLE in VOTERS
GT_MODES = ("homography", "affine")
INFRA_RE = re.compile(r"cuda|out of memory|\boom\b|timeout|timed out", re.I)

AMBIGUITIES = [
    "Candidate order for R: the prereg lists 'Core: SIFT, LoFTR, RoMa, MA-RoMa and MatchAnything direct, plus RoMa and "
    "MA-RoMa pyramid_v2; Transform pool: RoMa and MA-RoMa direct on {invert, histmatch, CLAHE, gradmag}'. That listing order "
    "is used for every tie (R and pick-by-S1): sift, loftr, roma, ma_roma, matchanything, roma pyramid_v2, ma_roma pyramid_v2, "
    "roma x (invert, histmatch, clahe, gradmag), ma_roma x (same). R = argmax n_inliers over the 15 (a failed run has "
    "n_inliers = -inf; if all 15 fail, R fails). n_inliers of a pyramid_v2 row is the final-transform inlier count as written "
    "by run_triage_candidates.py.",
    "S1 = n_inliers / n_matches per row (as in the CJSJ / Arm 1 code); a failed run or one with zero matches has S1 = -inf, "
    "is never accepted, and counts as a failure. Pick-by-S1 (H2-3) = argmax S1 over the same 15, ties to the first listed.",
    "One-sided vs two-sided: the McNemar p in H2-3 / H2-4 is the ONE-SIDED exact binomial tail P(X >= n_pos), X ~ Bin(n_pos + "
    "n_neg, 1/2), in the predicted direction (R better). 'Supported' needs p < 0.05 AND R having more successes. The two-sided "
    "exact p (cma.triage.mcnemar_exact) is printed beside it. H2-4 has no stated support rule beyond the test; the same rule "
    "is applied and it is labelled secondary.",
    "Bootstrap CIs are the 95% PERCENTILE interval (2.5th and 97.5th percentiles, cma.triage.ci95) over B = 10,000 cluster "
    "resamples, seed 0, resampling whole manifest `cluster` values within the analysed set. H2-1 needs AUROC >= 0.80 and the "
    "2.5th percentile > 0.5; H2-2 needs the 2.5th percentile of (accepted success rate - base rate) > 0. The 5th percentile "
    "(the one-sided alpha = 0.05 bound) is reported beside it and would give the same verdict unless flagged. Replicates in "
    "which a statistic is undefined (one class missing, nobody accepted) are dropped and their number is reported.",
    "H2-1 untestable rule: counted on MA-RoMa direct success in the analysed set AFTER the GT gate (primary set), separately for "
    "each threshold. Untestable is reported as UNTESTABLE, neither supported nor failed, and no CI is computed.",
    "H2-2: accepted = S1 >= 0.1711 (inclusive). The statistic is the success rate among accepted MA-RoMa direct pairs minus "
    "the base success rate of MA-RoMa direct over all pairs in the set (the base rate includes the accepted pairs). If no pair "
    "is accepted the hypothesis is reported NOT SUPPORTED (no accepted pairs).",
    "Thresholds: success is mu_ed <= 20 px (inclusive), in SOURCE pixels, exactly as in the runner. The secondary threshold is "
    "mu_ed <= 0.01 * sqrt(h_s^2 + w_s^2), with the source size from the manifest (same frame as mu_ed). Every hypothesis is "
    "computed under both; the 20 px verdict is the primary one. Pair sets, S1 and R do not depend on the threshold; only the "
    "success labels do.",
    "Failed / missing runs: status != ok, a non-finite mu_ed, or a row absent from the CSV counts as a failure (mu_ed = inf, "
    "S1 = -inf, n_inliers = -inf). Row-count check: expected (manifest pairs) x 17; a mismatch stamps the report NOT the "
    "pre-registered analysis and exits non-zero unless --allow-partial. Repeated attempts of one key (an infrastructure rerun): "
    "the LAST row is used and the number of repeats is reported.",
    "Set definitions: P = every non-NIST manifest pair (components A and D, 14 pairs, 12 clusters nominally); N = NIST pairs of "
    "the pair types that pass the gate. Primary set = P union N(passing). 'P alone' / 'N alone' are descriptive point estimates "
    "(no bootstrap, no significance claims). Pair types are the manifest `subclass` (IN718-BSE1-vs-OM, IN718-BSE2-vs-BSE1, "
    "IN718-BSE2-vs-OM, IN625-BSE2-vs-BSE1). Pairs of a failing type are excluded from every analysis, including 'N alone'; only "
    "the gate table shows them.",
    "Missing NIST pairs (download failures) are simply absent from the manifest; the report gives the manifest count against the "
    "nominal 95 (14 + 81) and the number of NIST pairs.",
    "GT gate arithmetic: the prereg says the distance between the GT-mapped click and the click, 'in target px of the pool "
    "images'. The hand-check tool records left = target, right = source (pool PNG pixels). The GT-mapped position of the SOURCE "
    "click is taken through the inverse GT homography into the target frame and compared with the target click, so the "
    "distance is in target pixels (which are finer than source pixels, so this is the stricter reading). The GT homography is "
    "the least-squares fit (cv2.findHomography, method 0) to the pool's 25 GT points target -> source, which were generated from "
    "the chained GT homography, so it reproduces it to the CSV rounding. The median is taken over ALL clicked points of a pair "
    "type pooled (not a median of per-pair medians); pass = median <= 10 px inclusive. Skipped pairs contribute no points. A "
    "type with no clicked points cannot be verified and is EXCLUDED (reported as such). Source-px medians are shown too.",
    "The gate only reads Frank's click file; an absent file means the primary analysis is refused. --no-gate-dry-run skips the "
    "gate (all NIST types kept), writes arm2_DRYRUN_* files and stamps them; if a click file exists it is still ignored in a dry run.",
    "Overlay pairs: the prereg says any shared-overlay pair found later is 'analysed as it is and reported separately'. Manifest "
    "pairs whose overlay_src / overlay_tgt is not 'none' are listed in a separate descriptive table and stay in the primary set.",
    "H2-3 / H2-4 use the pair-level exact McNemar as pre-registered; pairs within a cluster are not independent, so a cluster "
    "bootstrap CI on the success-rate difference is printed beside each as a DESCRIPTIVE sensitivity check (not a decision rule).",
]


# ----------------------------------------------------------------------------- statistics
def mcnemar_one_sided(n_pos: int, n_neg: int) -> float:
    """One-sided exact McNemar: P(X >= n_pos), X ~ Binomial(n_pos + n_neg, 1/2)."""
    n = n_pos + n_neg
    if n == 0:
        return 1.0
    return float(sum(comb(n, i) for i in range(n_pos, n + 1)) / 2.0**n)


def paired_test(better: np.ndarray, other: np.ndarray) -> dict:
    """Paired binary test, predicted direction: ``better`` succeeds more than ``other``."""
    a, b = np.asarray(better, bool), np.asarray(other, bool)
    n_pos, n_neg = int((a & ~b).sum()), int((~a & b).sum())
    p1, p2 = mcnemar_one_sided(n_pos, n_neg), mcnemar_exact(n_pos, n_neg)
    return dict(n=int(len(a)), successes_better=int(a.sum()), successes_other=int(b.sum()), only_better=n_pos,
                only_other=n_neg, p_one_sided=p1, p_two_sided=p2,
                supported=bool(p1 < ALPHA and a.sum() > b.sum()))


def first_argmax(score: np.ndarray) -> np.ndarray:
    """Row-wise argmax with ties to the lowest column (= the candidate listed first). NaN counts as -inf."""
    s = np.where(np.isnan(score), -np.inf, score)
    return np.argmax(s, axis=1)


def pick_success(score: np.ndarray, succ: np.ndarray) -> np.ndarray:
    """Success of the candidate chosen by ``score`` (max, ties to first listed). A row whose scores are all -inf
    (every candidate failed) has no valid pick and counts as a failure."""
    s = np.where(np.isnan(score), -np.inf, score)
    idx = first_argmax(s)
    return succ[np.arange(len(succ)), idx] & np.isfinite(s.max(axis=1))


def _boot(stat, clusters, B, seed=0) -> tuple[np.ndarray, int]:
    boot = cluster_bootstrap(stat, clusters, B=B, seed=seed)
    return boot, int(B - len(boot))


def h21(s1: np.ndarray, y: np.ndarray, clusters: np.ndarray, B: int = B_DEFAULT) -> dict:
    """H2-1: AUROC of S1 for MA-RoMa direct success; untestable if < 8 successes or < 8 failures."""
    y = np.asarray(y, bool)
    n_s, n_f = int(y.sum()), int((~y).sum())
    point = auroc(s1, y)
    out = dict(n=int(len(y)), successes=n_s, failures=n_f, auroc=point, criterion="AUROC >= 0.80 and cluster-bootstrap 95% "
               "CI lower bound > 0.5; untestable if < 8 successes or < 8 failures")
    if n_s < MIN_CLASS or n_f < MIN_CLASS:
        out.update(status="UNTESTABLE", supported=None, ci95=None, ci_lower_5pct=None, dropped_replicates=None)
        return out
    boot, dropped = _boot(lambda idx: auroc(s1[idx], y[idx]), clusters, B)
    lo, hi = ci95(boot)
    lo5 = float(np.percentile(boot, 5))
    ok = bool(point >= AUROC_MIN and lo > 0.5)
    out.update(status="SUPPORTED" if ok else "NOT SUPPORTED", supported=ok, ci95=[lo, hi], ci_lower_5pct=lo5,
               dropped_replicates=dropped)
    return out


def h22(s1: np.ndarray, y: np.ndarray, clusters: np.ndarray, B: int = B_DEFAULT, cut: float = S1_CUT) -> dict:
    """H2-2: success rate among accepted (S1 >= cut) MA-RoMa direct pairs minus the base rate."""
    y = np.asarray(y, bool)
    acc = np.asarray(s1, float) >= cut

    def stat(idx):
        a = acc[idx]
        return float(y[idx][a].mean() - y[idx].mean()) if a.any() else float("nan")
    base = float(y.mean()) if len(y) else float("nan")
    out = dict(n=int(len(y)), cutoff=cut, n_accepted=int(acc.sum()), base_rate=base,
               criterion="cluster-bootstrap 95% CI of (accepted success rate - base rate) above 0")
    if not acc.any():
        out.update(accepted_rate=float("nan"), difference=float("nan"), status="NOT SUPPORTED (no accepted pairs)",
                   supported=False, ci95=None, ci_lower_5pct=None, dropped_replicates=None)
        return out
    acc_rate = float(y[acc].mean())
    boot, dropped = _boot(stat, clusters, B)
    lo, hi = ci95(boot) if len(boot) else (float("nan"), float("nan"))
    lo5 = float(np.percentile(boot, 5)) if len(boot) else float("nan")
    ok = bool(np.isfinite(lo) and lo > 0)
    out.update(accepted_rate=acc_rate, difference=acc_rate - base, status="SUPPORTED" if ok else "NOT SUPPORTED",
               supported=ok, ci95=[lo, hi], ci_lower_5pct=lo5, dropped_replicates=dropped)
    return out


def rate_diff_ci(a: np.ndarray, b: np.ndarray, clusters: np.ndarray, B: int) -> list[float]:
    d = a.astype(float) - b.astype(float)
    boot, _ = _boot(lambda idx: float(d[idx].mean()), clusters, B)
    lo, hi = ci95(boot) if len(boot) else (float("nan"), float("nan"))
    return [lo, hi]


def hypotheses(mu: np.ndarray, s1: np.ndarray, ninl: np.ndarray, thr: np.ndarray, clusters: np.ndarray,
               B: int = B_DEFAULT) -> dict:
    """H2-1..H2-4 + oracle on one set of pairs at one threshold vector (per-pair px). Arrays are (pairs, 15)."""
    succ = mu <= thr[:, None]
    ic, ib = VOTERS.index(H21_CAND), VOTERS.index(BEST_SINGLE)
    r_succ = pick_success(ninl, succ)
    s_succ = pick_success(s1, succ)
    best = succ[:, ib]
    h3 = paired_test(r_succ, s_succ)
    h3["rate_diff_ci95_cluster_descriptive"] = rate_diff_ci(r_succ, s_succ, clusters, B) if len(r_succ) else None
    h4 = paired_test(r_succ, best)
    h4["rate_diff_ci95_cluster_descriptive"] = rate_diff_ci(r_succ, best, clusters, B) if len(r_succ) else None
    return {
        "n_pairs": int(len(mu)),
        "H2-1": h21(s1[:, ic], succ[:, ic], clusters, B),
        "H2-2": h22(s1[:, ic], succ[:, ic], clusters, B),
        "H2-3": dict(h3, hypothesis="R vs pick-by-S1 (max S1 over the same 15)"),
        "H2-4": dict(h4, hypothesis="R vs best single (MA-RoMa pyramid_v2), secondary"),
        "SR": dict(R=int(r_succ.sum()), pick_by_S1=int(s_succ.sum()), best_single=int(best.sum()),
                   ma_roma_direct=int(succ[:, ic].sum()), oracle_best_of_15=int(succ.any(axis=1).sum()),
                   n=int(len(mu))),
        "per_candidate": {"|".join(v): int(succ[:, k].sum()) for k, v in enumerate(VOTERS)},
    }


def descriptive(mu, s1, ninl, thr) -> dict:
    """Point estimates only (P alone / N alone)."""
    if len(mu) == 0:
        return dict(n=0)
    succ = mu <= thr[:, None]
    ic, ib = VOTERS.index(H21_CAND), VOTERS.index(BEST_SINGLE)
    r, s = pick_success(ninl, succ), pick_success(s1, succ)
    return dict(n=int(len(mu)), R=int(r.sum()), pick_by_S1=int(s.sum()), best_single=int(succ[:, ib].sum()),
                ma_roma_direct=int(succ[:, ic].sum()), oracle_best_of_15=int(succ.any(axis=1).sum()),
                auroc_S1_ma_roma_direct=auroc(s1[:, ic], succ[:, ic]),
                n_accepted_at_cut=int((s1[:, ic] >= S1_CUT).sum()),
                accepted_success=int((succ[:, ic] & (s1[:, ic] >= S1_CUT)).sum()),
                p_R_vs_S1_one_sided=paired_test(r, s)["p_one_sided"],
                p_R_vs_best_one_sided=paired_test(r, succ[:, ib])["p_one_sided"])


# ----------------------------------------------------------------------------- NIST GT gate
def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.atleast_2d(np.asarray(pts, float))
    p = np.c_[pts, np.ones(len(pts))] @ np.asarray(H, float).T
    return p[:, :2] / p[:, 2:3]


def gate_errors(H_tgt2src: np.ndarray, left_tgt: np.ndarray, right_src: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-click error of the GT: (target px, source px). Left clicks are in the target, right clicks in the source.

    target-px error = |H^-1(source click) - target click|; source-px error = |H(target click) - source click|.
    """
    e_src = np.linalg.norm(apply_h(H_tgt2src, left_tgt) - np.asarray(right_src, float), axis=1)
    e_tgt = np.linalg.norm(apply_h(np.linalg.inv(H_tgt2src), right_src) - np.asarray(left_tgt, float), axis=1)
    return e_tgt, e_src


def evaluate_gate(errors: dict[str, np.ndarray], types: dict[str, str], all_types: list[str], limit: float = GATE_PX,
                  errors_src: dict[str, np.ndarray] | None = None) -> dict:
    """errors[pair_id] = per-click target-px errors; types[pair_id] = pair type. Returns per-type verdicts."""
    out = {}
    for t in sorted(set(all_types)):
        ids = [p for p in errors if types.get(p) == t and len(errors[p])]
        pts = np.concatenate([errors[p] for p in ids]) if ids else np.array([])
        med = float(np.median(pts)) if len(pts) else float("nan")
        row = dict(pairs_clicked=len(ids), points=int(len(pts)), median_target_px=med,
                   max_target_px=float(pts.max()) if len(pts) else float("nan"),
                   passes=bool(len(pts) and med <= limit),
                   reason=("median <= %g px" % limit) if len(pts) and med <= limit else
                   ("no clicked points: cannot be verified" if not len(pts) else "median > %g px" % limit))
        if errors_src is not None and ids:
            row["median_source_px"] = float(np.median(np.concatenate([errors_src[p] for p in ids])))
        out[t] = row
    return out


def norm_pair_id(pid: str) -> str:
    return pid.split("#")[0]


def read_handcheck(path: Path) -> dict[str, np.ndarray]:
    """{pair_id (no '#k'): array (n, 4) x_left, y_left, x_right, y_right} from tools/handcheck/handcheck.py output."""
    out: dict[str, list] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out.setdefault(norm_pair_id(r["pair_id"]), []).append(
                [float(r["x_left"]), float(r["y_left"]), float(r["x_right"]), float(r["y_right"])])
    return {k: np.array(v, float) for k, v in out.items()}


def fit_gt_h(gt_csv: Path) -> np.ndarray:
    import cv2
    a = np.loadtxt(gt_csv, delimiter=",", skiprows=1, ndmin=2)
    H, _ = cv2.findHomography(a[:, 2:4].astype(np.float64), a[:, :2].astype(np.float64), method=0)
    if H is None:
        raise RuntimeError(f"GT homography fit failed for {gt_csv}")
    return H


def run_gate(manifest: pd.DataFrame, root: Path, rebase: bool, click_csv: Path) -> dict:
    nist = manifest[manifest.component == "NIST"]
    types = {norm_pair_id(p): t for p, t in zip(nist.pair_id, nist.subclass)}
    gt_path = {norm_pair_id(p): resolve_path(root, g, rebase) for p, g in zip(nist.pair_id, nist.gt_path)}
    clicks = read_handcheck(click_csv)
    e_tgt, e_src, unknown = {}, {}, []
    for pid, arr in clicks.items():
        if pid not in types:
            unknown.append(pid)
            continue
        H = fit_gt_h(gt_path[pid])
        e_tgt[pid], e_src[pid] = gate_errors(H, arr[:, :2], arr[:, 2:])
    skips = []
    sk = Path(str(click_csv) + ".skips.csv")
    if sk.exists():
        with open(sk, newline="", encoding="utf-8") as f:
            skips = [dict(pair_id=r["pair_id"], reason=r.get("reason", "")) for r in csv.DictReader(f)]
    res = evaluate_gate(e_tgt, types, list(types.values()), errors_src=e_src)
    return dict(handcheck_csv=str(click_csv), limit_target_px=GATE_PX, types=res,
                pairs_clicked=len(e_tgt), pairs_clicked_not_in_manifest=unknown, skipped=skips,
                per_pair={p: dict(type=types[p], n=int(len(e)), median_target_px=float(np.median(e)),
                                  max_target_px=float(e.max())) for p, e in sorted(e_tgt.items())},
                passing_types=sorted(t for t, v in res.items() if v["passes"]),
                failing_types=sorted(t for t, v in res.items() if not v["passes"]))


# ----------------------------------------------------------------------------- data
def read_csv(path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def load_manifest(path: Path) -> pd.DataFrame:
    m = read_csv(path)
    m["h_s"] = m["h_s"].astype(float)
    m["w_s"] = m["w_s"].astype(float)
    m["diag_src"] = np.hypot(m.h_s, m.w_s)
    return m


def dedupe(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    key = ["pair_id", "backbone", "mode", "transform", "seed"]
    n_dup = int(df.duplicated(key, keep="last").sum())
    return df.drop_duplicates(key, keep="last").reset_index(drop=True), n_dup


def build_arrays(cand: pd.DataFrame, pair_ids: list[str]) -> dict:
    """mu / S1 / n_inliers as (pairs, 15) arrays in VOTERS order (failure: inf / -inf / -inf) + GT ceiling (pairs, 2)."""
    num = lambda s: pd.to_numeric(s, errors="coerce")  # noqa: E731
    d = cand[cand.seed == "0"].copy()
    d["mu"], d["ni"], d["nm"] = num(d.mu_ed), num(d.n_inliers), num(d.n_matches)
    ok = (d.status == "ok") & np.isfinite(d.mu)
    d["mu"] = np.where(ok, d.mu, np.inf)
    d["ni"] = np.where(ok & np.isfinite(d.ni), d.ni, -np.inf)
    d["s1"] = np.where(ok & (d.nm > 0) & np.isfinite(d.ni), d.ni / d.nm.where(d.nm > 0, 1), -np.inf)
    idx = {(r.pair_id, r.backbone, r["mode"], r["transform"]): r for _, r in d.iterrows()}
    n = len(pair_ids)
    mu, s1, ni = np.full((n, 15), np.inf), np.full((n, 15), -np.inf), np.full((n, 15), -np.inf)
    gt = np.full((n, 2), np.inf)
    for i, p in enumerate(pair_ids):
        for k, (b, m, t) in enumerate(VOTERS):
            r = idx.get((p, b, m, t))
            if r is not None:
                mu[i, k], s1[i, k], ni[i, k] = r.mu, r.s1, r.ni
        for k, m in enumerate(GT_MODES):
            r = idx.get((p, "gt", m, "none"))
            if r is not None:
                gt[i, k] = r.mu
    return dict(mu=mu, s1=s1, ninl=ni, gt=gt)


def failed_run_audit(raw: pd.DataFrame) -> dict:
    bad = raw[raw.status != "ok"]
    groups: dict[str, dict] = {}
    for _, r in bad.iterrows():
        msg = re.sub(r"\s+", " ", str(r.get("error", ""))).strip() or "(empty error message)"
        g = groups.setdefault(msg, dict(count=0, infrastructure=bool(INFRA_RE.search(msg))))
        g["count"] += 1
    return dict(n_failed_rows=int(len(bad)), infrastructure_rows=int(sum(v["count"] for v in groups.values() if v["infrastructure"])),
                groups=[dict(error=k, **v) for k, v in sorted(groups.items(), key=lambda kv: -kv[1]["count"])])


# ----------------------------------------------------------------------------- report
def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join((f"{v:.3g}" if isinstance(v, float) else str(v)).replace("|", "\\|") for v in r.values) + " |")
    return "\n".join(lines)


def fp(p: float) -> str:
    return f"{p:.4g}"


def fci(ci) -> str:
    return "n/a" if ci is None else f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def render_hyp(a, name: str, H: dict) -> None:
    h1, h2, h3, h4, sr = H["H2-1"], H["H2-2"], H["H2-3"], H["H2-4"], H["SR"]
    a(f"### {name} (n = {H['n_pairs']} pairs)")
    a(f"- **H2-1 S1 predicts MA-RoMa direct success (primary): {h1['status']}.** {h1['successes']} successes / {h1['failures']} failures; "
      f"AUROC {h1['auroc']:.3f}" + (f"; cluster-bootstrap 95% CI {fci(h1['ci95'])} (5th percentile {h1['ci_lower_5pct']:.3f}; dropped replicates {h1['dropped_replicates']})."
                                    if h1["ci95"] else "; no CI computed (untestable rule)."))
    a(f"- **H2-2 the cut-off transfers: {h2['status']}.** S1 >= {h2['cutoff']}: {h2['n_accepted']} accepted of {h2['n']}; base rate {h2['base_rate']:.3f}"
      + (f"; accepted success rate {h2['accepted_rate']:.3f}; difference {h2['difference']:+.3f}, CI {fci(h2['ci95'])} (5th percentile {h2['ci_lower_5pct']:.3f}; dropped replicates {h2['dropped_replicates']})."
         if h2["ci95"] else "."))
    for k, h in (("H2-3 R beats pick-by-S1", h3), ("H2-4 R beats best single MA-RoMa pyramid_v2 (secondary, low power stated in advance)", h4)):
        a(f"- **{k}: {'SUPPORTED' if h['supported'] else 'NOT SUPPORTED'}.** R {h['successes_better']} vs {h['successes_other']}; success only under R {h['only_better']}, "
          f"only under the comparator {h['only_other']}; exact McNemar one-sided p = {fp(h['p_one_sided'])} (two-sided {fp(h['p_two_sided'])}); "
          f"descriptive cluster-bootstrap CI of the SR difference {fci(h['rate_diff_ci95_cluster_descriptive'])}.")
    a(f"- SR counts: R {sr['R']}, pick-by-S1 {sr['pick_by_S1']}, best single {sr['best_single']}, MA-RoMa direct {sr['ma_roma_direct']}, "
      f"oracle best-of-15 {sr['oracle_best_of_15']} (of {sr['n']}).")
    a("")


def render_report(S: dict) -> str:
    L: list[str] = []
    a = L.append
    dry = S["dry_run"]
    a("# Arm 2 report: does label-free triage replicate on independent materials data?")
    a("")
    if dry:
        a("> **DRY RUN. The NIST GT gate was NOT applied (--no-gate-dry-run). This is NOT the pre-registered analysis.**")
        a("")
    a(f"Source CSV: `{S['csv']}`; manifest `{S['manifest']}`. Analysis follows `prereg/arm2_materials_replication.md`; constants carried over unchanged "
      f"from AmalgaMatch: S1 cut-off {S1_CUT}, best single = {'|'.join(BEST_SINGLE)}. Interpretive choices are listed at the end.")
    a("")
    rc = S["row_counts"]
    a("## Pool and row counts")
    a(f"- Manifest pairs: {rc['n_pairs']} (P {rc['n_P']}, NIST {rc['n_NIST']} of up to 81; nominal total {N_DEFAULT_PAIRS}; missing vs nominal {rc['missing_vs_nominal']}).")
    a(f"- Rows in file {rc['n_rows']}; unique keys {rc['n_unique']}; repeated attempts (last used) {rc['n_duplicates']}; stray rows (control pool or unknown pair) {rc['stray']}; "
      f"expected {rc['expected']} = {rc['n_pairs']} x {ROWS_PER_PAIR}: **{'OK' if rc['ok'] else 'MISMATCH'}**.")
    fr = S["failed_runs"]
    a(f"- Failed rows {fr['n_failed_rows']} ({fr['infrastructure_rows']} with CUDA/OOM/timeout-type messages = infrastructure, rerun once per the prereg). Every failed run counts as a failure.")
    if fr["groups"]:
        a("")
        a(md_table(pd.DataFrame([dict(count=g["count"], infrastructure="YES" if g["infrastructure"] else "no", error=g["error"][:120]) for g in fr["groups"]])))
    a("")
    a("## NIST GT gate")
    g = S["gate"]
    if g is None:
        a("Not applied (dry run). All NIST pair types are kept in this dry run.")
    else:
        a(f"Hand-check file `{g['handcheck_csv']}`: {g['pairs_clicked']} pairs clicked; skipped {len(g['skipped'])}"
          + (f" ({'; '.join(x['pair_id'] + ': ' + x['reason'] for x in g['skipped'])})" if g["skipped"] else "")
          + f". Rule: per pair type, the pooled median of the distance between the GT-mapped click and the click, in target px, must be <= {GATE_PX:g}.")
        a("")
        a(md_table(pd.DataFrame([dict(pair_type=t, pairs_clicked=v["pairs_clicked"], points=v["points"], median_target_px=v["median_target_px"],
                                      median_source_px=v.get("median_source_px", float("nan")), max_target_px=v["max_target_px"],
                                      verdict="PASS" if v["passes"] else "EXCLUDED", reason=v["reason"]) for t, v in g["types"].items()])))
    a(f"- Pair types in the primary set: {', '.join(S['n_types_included']) or 'none'}; excluded: {', '.join(S['n_types_excluded']) or 'none'}.")
    a(f"- Primary set: {S['primary_n']} pairs = P {S['primary_n_P']} + N {S['primary_n_N']}; clusters {S['primary_clusters']}.")
    a("")
    a("## Pre-registered hypotheses, primary set")
    a("")
    render_hyp(a, "Threshold 20 px (PRIMARY)", S["primary_20px"])
    render_hyp(a, "Threshold 1% of the source diagonal (secondary, reported for every hypothesis)", S["primary_1pct"])
    a("## Descriptive: P alone and N alone (point estimates only, no tests)")
    a("")
    rows = []
    for nm, key in (("P alone", "P"), ("N alone (gate-passing types)", "N")):
        for th, tk in (("20 px", "20px"), ("1% diag", "1pct")):
            d = S["descriptive"][key][tk]
            rows.append(dict(set=nm, threshold=th, **{k: v for k, v in d.items()}))
    a(md_table(pd.DataFrame(rows)))
    a("")
    a("## Per pair type / component (primary set, 20 px)")
    a("")
    a(md_table(pd.DataFrame(S["per_type"])))
    a("")
    a("## Ceiling and per-candidate success (primary set)")
    a(f"- GT-fit homography SR@20: {S['gt_ceiling']['homography']} / {S['primary_n']}; GT-fit affine SR@20: {S['gt_ceiling']['affine']} / {S['primary_n']}.")
    a("")
    a(md_table(pd.DataFrame([dict(candidate=k, sr20=v20, sr_1pct=v1) for (k, v20), v1 in
                             zip(S["primary_20px"]["per_candidate"].items(), S["primary_1pct"]["per_candidate"].values())])))
    a("")
    a("## Overlay pairs (manifest overlay_src / overlay_tgt not 'none'), analysed as-is and reported separately")
    a(f"{S['overlay_pairs'] or 'none'}")
    a("")
    a("## Interpretation notes and choices the prereg left open")
    for x in AMBIGUITIES:
        a(f"- {x}")
    a("")
    a("(Everything before the interpretation notes is a pre-registered quantity or required descriptive reporting; nothing here is a claim beyond the pre-registered tests.)")
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------- main
def _jd(o):
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def analyse(cand_raw: pd.DataFrame, manifest: pd.DataFrame, gate: dict | None, B: int, csv_path: str, man_path: str,
            dry_run: bool) -> dict:
    cand, n_dup = dedupe(cand_raw)
    pair_ids = list(manifest.pair_id)
    known = set(pair_ids)
    used = cand[cand.pair_id.isin(known) & (cand.pool != "control") & (cand.seed == "0")]
    stray = int(len(cand) - len(used))
    n_pairs = len(pair_ids)
    is_nist = (manifest.component == "NIST").values
    all_types = sorted(set(manifest.subclass[is_nist]))
    passing = set(all_types) if gate is None else set(gate["passing_types"])
    in_set = np.array([(not n) or (t in passing) for n, t in zip(is_nist, manifest.subclass)])
    arr = build_arrays(used, pair_ids)
    clusters = manifest.cluster.values
    diag = manifest.diag_src.values

    def sub(mask):
        return {k: v[mask] for k, v in arr.items()}, clusters[mask], diag[mask]

    A, cl, dg = sub(in_set)
    res20 = hypotheses(A["mu"], A["s1"], A["ninl"], np.full(len(cl), TH_PX), cl, B)
    res1 = hypotheses(A["mu"], A["s1"], A["ninl"], DIAG_FRAC * dg, cl, B)
    desc = {}
    for key, mask in (("P", in_set & ~is_nist), ("N", in_set & is_nist)):
        Ak, ck, dk = sub(mask)
        desc[key] = {"20px": descriptive(Ak["mu"], Ak["s1"], Ak["ninl"], np.full(len(ck), TH_PX)),
                     "1pct": descriptive(Ak["mu"], Ak["s1"], Ak["ninl"], DIAG_FRAC * dk)}
    per_type = []
    grp = np.where(is_nist, manifest.subclass.values, "P:" + manifest.component.values)
    for gname in sorted(set(grp[in_set])):
        m = in_set & (grp == gname)
        Ak, ck, dk = sub(m)
        d = descriptive(Ak["mu"], Ak["s1"], Ak["ninl"], np.full(len(ck), TH_PX))
        per_type.append(dict(group=gname, n=d["n"], clusters=int(len(set(ck))), ma_roma_direct=d["ma_roma_direct"], R=d["R"],
                             pick_by_S1=d["pick_by_S1"], best_single=d["best_single"], oracle=d["oracle_best_of_15"],
                             gt_homography=int((Ak["gt"][:, 0] <= TH_PX).sum())))
    ovl = manifest.get("overlay_src", pd.Series("none", index=manifest.index)).ne("none") | \
        manifest.get("overlay_tgt", pd.Series("none", index=manifest.index)).ne("none")
    ic = VOTERS.index(H21_CAND)
    ovl_rows = [dict(pair_id=p, ma_roma_direct_success=bool(arr["mu"][i, ic] <= TH_PX)) for i, p in enumerate(pair_ids)
                if ovl.iloc[i] and in_set[i]]
    nP, nN = int((in_set & ~is_nist).sum()), int((in_set & is_nist).sum())
    expected = n_pairs * ROWS_PER_PAIR
    n_used = len(used)
    ok_rows = bool(n_used == expected and stray == 0)
    return dict(
        csv=csv_path, manifest=man_path, dry_run=dry_run, gate=gate, B=B,
        row_counts=dict(n_rows=int(len(cand_raw)), n_unique=int(len(cand)), n_duplicates=n_dup, stray=stray, expected=expected,
                        n_used=n_used, ok=ok_rows, n_pairs=n_pairs, n_P=int((~is_nist).sum()), n_NIST=int(is_nist.sum()),
                        missing_vs_nominal=N_DEFAULT_PAIRS - n_pairs),
        failed_runs=failed_run_audit(cand_raw),
        n_types_included=sorted(t for t in all_types if t in passing), n_types_excluded=sorted(t for t in all_types if t not in passing),
        primary_n=int(in_set.sum()), primary_n_P=nP, primary_n_N=nN, primary_clusters=int(len(set(cl))),
        primary_20px=res20, primary_1pct=res1, descriptive=desc, per_type=per_type,
        gt_ceiling=dict(homography=int((A["gt"][:, 0] <= TH_PX).sum()), affine=int((A["gt"][:, 1] <= TH_PX).sum())),
        overlay_pairs=ovl_rows,
        constants=dict(S1_CUT=S1_CUT, BEST_SINGLE="|".join(BEST_SINGLE), TH_PX=TH_PX, DIAG_FRAC=DIAG_FRAC, B=B, voters=["|".join(v) for v in VOTERS]),
        ambiguities=AMBIGUITIES)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="results/arm2/candidates.csv")
    ap.add_argument("--manifest", default="", help=f"default: ${ROOT_ENV}/manifest.csv or the materials-bench folder")
    ap.add_argument("--root", default="", help="pool root for gt_points paths (default: ${ROOT_ENV} or the materials-bench folder)")
    ap.add_argument("--handcheck", default=str(DEFAULT_ROOT / "nist" / "handcheck_clicks.csv"))
    ap.add_argument("--gate-only", action="store_true", help="compute the NIST GT gate (results/arm2/gate_result.json) and stop")
    ap.add_argument("--no-gate-dry-run", action="store_true",
                    help="DRY RUN without the hand-check gate; writes arm2_DRYRUN_* and stamps them as not pre-registered")
    ap.add_argument("--allow-partial", action="store_true", help="do not exit non-zero when the row count differs")
    ap.add_argument("--out-dir", default="results/arm2")
    ap.add_argument("--boot", type=int, default=B_DEFAULT)
    a = ap.parse_args()

    root = Path(a.root or os.environ.get(ROOT_ENV) or DEFAULT_ROOT)
    rebase = bool(os.environ.get(ROOT_ENV) or a.root)
    man_path = Path(a.manifest) if a.manifest else root / "manifest.csv"
    manifest = load_manifest(man_path)
    out_dir = Path(a.out_dir) if Path(a.out_dir).is_absolute() else ROOT / a.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    gate = None
    click = Path(a.handcheck)
    if not a.no_gate_dry_run:
        if not click.is_file():
            sys.exit(f"REFUSING to produce the primary analysis: the NIST hand-check file {click} does not exist. The pre-registered "
                     f"GT gate (median hand-vs-GT error <= {GATE_PX:g} target px per pair type) must be decided by a human before the "
                     f"analysis. Run tools/handcheck/handcheck.py on tools/handcheck/nist_handcheck_pairs.json, or pass --no-gate-dry-run "
                     f"for a stamped DRY RUN that is not the pre-registered analysis.")
        gate = run_gate(manifest, root, rebase, click)
        (out_dir / "gate_result.json").write_text(json.dumps(gate, indent=2, default=_jd), encoding="utf-8")
        print("gate:", {t: ("PASS" if v["passes"] else "EXCLUDED") + f" median {v['median_target_px']:.1f}px" for t, v in gate["types"].items()})
        if a.gate_only:
            print(f"wrote {out_dir / 'gate_result.json'}")
            return
    elif a.gate_only:
        sys.exit("--gate-only needs the hand-check file; it cannot be combined with --no-gate-dry-run")

    cpath = Path(a.candidates) if Path(a.candidates).is_absolute() else ROOT / a.candidates
    S = analyse(read_csv(cpath), manifest, gate, a.boot, str(cpath), str(man_path), bool(a.no_gate_dry_run))
    prefix = out_dir / ("arm2_DRYRUN" if a.no_gate_dry_run else "arm2")
    report = render_report(S)
    if not S["row_counts"]["ok"]:
        rc = S["row_counts"]
        report = (f"> **WARNING: row-count check failed (expected {rc['expected']} rows for {rc['n_pairs']} pairs, found {rc['n_used']}, stray {rc['stray']}). "
                  f"Results below are partial and NOT the pre-registered analysis.**\n\n") + report
    Path(f"{prefix}_summary.json").write_text(json.dumps(S, indent=2, default=_jd), encoding="utf-8")
    Path(f"{prefix}_report.md").write_text(report, encoding="utf-8")
    print(f"wrote {prefix}_summary.json and {prefix}_report.md")
    if not S["row_counts"]["ok"] and not a.allow_partial:
        sys.exit(f"ROW COUNT CHECK FAILED: {S['row_counts']} (outputs were still written; rerun with --allow-partial to silence)")


if __name__ == "__main__":
    main()
