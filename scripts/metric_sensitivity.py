"""Metric-sensitivity analysis: raw parametric ED vs TPS-refined ED.

The manuscript scores registration accuracy on `mu_ed_tps`, falling back to the
parametric `mu_ed` whenever the TPS refinement produced no value.  This script
recomputes every headline number under BOTH metrics so that the dependence of
the paper's conclusions on that choice is explicit and reproducible.

It produces five blocks:

  A. Descriptive accuracy for each of the nine Table 1 configurations under
     both metrics: median ED plus SR@{5,10,20} with Wilson 95% CIs.
  B. TPS coverage per configuration -- how many of the 187 pairs actually carry
     a non-blank `mu_ed_tps`.  This is a comparability defect: the dense
     RoMa-family configurations are TPS-scored on all 187 pairs while the weak
     configurations silently fall back to the raw metric on the majority, so
     the reported column is not one metric but a mixture of two.
  C. Refinement regressions -- pairs where the raw fit succeeds (<10 px) and the
     TPS-refined fit fails (>=10 px), listed by pair_id.
  D. The two paired-bootstrap contrasts that carry the paper's native-pair
     claims (RoMa pyramid v2 vs RoMa direct; MatchAnything-RoMa vs RoMa), under
     both metrics.
  E. The FOV ladder wrapper contrast at rungs 0.5 / 0.25 / 0.10 for roma,
     ma_roma and ma_roma_ft, under both metrics, on the paper's base-matchable
     testbed and additionally restricted to pairs held out of fine-tuning.

Bootstrap protocol (unchanged from scripts/bootstrap_ci.py): paired bootstrap
over per-pair errors, B = 10,000 resamples, seed 0, percentile 95 % CI on the
difference.  One p-value is reported per contrast:

  * `p_two_sided`  = 2 * min(P(delta<=0), P(delta>=0)), clipped at 1.  This is
                     the two-sided convention the manuscript's methods text
                     declares.  It is symmetric: the value does not depend on
                     which configuration is labelled A and which is labelled B,
                     nor on the sign of the observed difference.

Outputs:
  results/metric_sensitivity.csv   -- tidy machine-readable form
  reports/metric_sensitivity.md    -- prose summary with the tables
  stdout                           -- the same report

Usage:  python scripts/metric_sensitivity.py      (no arguments, from repo root)
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORTS = ROOT / "reports"

B = 10_000
SEED = 0
THRESHOLDS = (5.0, 10.0, 20.0)
INF = float("inf")

# ---------------------------------------------------------------- Table 1 ----
# (label, source csv, backbone, mode).  Order follows Table 1 of the manuscript.
TABLE1 = [
    ("SIFT (Control A)", "baselines_A.csv", "sift", "direct"),
    ("SIFT + MI (Control B)", "baselines_B.csv", "sift", "classical"),
    ("LoFTR", "baselines_A.csv", "loftr", "direct"),
    ("MatchAnything-ELoFTR", "baselines_A.csv", "matchanything", "direct"),
    ("RoMa (zero-shot)", "baselines_A.csv", "roma", "direct"),
    ("RoMa + pyramid v1", "baselines_A.csv", "roma", "pyramid"),
    ("RoMa + pyramid v2", "baselines_A.csv", "roma", "pyramid_v2"),
    ("MatchAnything-RoMa", "baselines_A.csv", "ma_roma", "direct"),
    ("MatchAnything-RoMa + pyramid v2", "baselines_A.csv", "ma_roma", "pyramid_v2"),
]

# Configurations outside Table 1 that still live in the baseline CSVs; reported
# in the coverage/regression blocks only, flagged where fine-tuning taints them.
EXTRA = [
    ("MatchAnything-ELoFTR + pyramid v1", "baselines_A.csv", "matchanything", "pyramid"),
    ("MatchAnything-ELoFTR + pyramid v2", "baselines_A.csv", "matchanything", "pyramid_v2"),
    ("MatchAnything-ELoFTR (stretch)", "baselines_A.csv", "matchanything_stretch", "direct"),
    ("RoMa + pyramid v2 + certainty 0.5", "baselines_A.csv", "roma", "pyramid_v2+c50"),
    ("RoMa + pyramid v2 + zoom x3", "baselines_A.csv", "roma", "pyramid_v2+z3"),
    ("MatchAnything-RoMa fine-tuned", "baselines_A.csv", "ma_roma_ft", "direct"),
    ("MatchAnything-RoMa fine-tuned + pyramid v2", "baselines_A.csv", "ma_roma_ft", "pyramid_v2"),
]

FT_TAINTED = {"ma_roma_ft"}
FT_NOTE = ("fine-tuned on 131/187 pairs (results/split.json train); "
           "not comparable on the full 187-pair aggregate")


# ------------------------------------------------------------------ io ------
def load_csv(name: str) -> list[dict]:
    with (RESULTS / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def cell(row: dict, key: str) -> str:
    return (row.get(key) or "").strip()


def err(row: dict, metric: str) -> float:
    """Per-pair error under `metric`. Failed rows count as +inf (never a success).

    metric == "raw": the parametric fit error `mu_ed`.
    metric == "tps": `mu_ed_tps` when present, else `mu_ed` -- the paper's rule.
    """
    if row["status"] != "ok":
        return INF
    v = (cell(row, "mu_ed_tps") or cell(row, "mu_ed")) if metric == "tps" else cell(row, "mu_ed")
    return float(v) if v else INF


def has_tps(row: dict) -> bool:
    return bool(cell(row, "mu_ed_tps"))


# ------------------------------------------------------------ statistics ----
def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial rate (95 % by default)."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def median_finite(errs: np.ndarray) -> tuple[float, int]:
    fin = errs[np.isfinite(errs)]
    return (float(np.median(fin)) if fin.size else float("nan"), int(fin.size))


def boot_index(n: int) -> np.ndarray:
    return np.random.default_rng(SEED).integers(0, n, size=(B, n))


def two_sided_p(boot: np.ndarray) -> float:
    """Two-sided bootstrap p-value: twice the smaller tail mass, capped at 1."""
    p_le = float((boot <= 0).mean())
    p_ge = float((boot >= 0).mean())
    return min(1.0, 2.0 * min(p_le, p_ge))


def paired_rate_contrast(ea: np.ndarray, eb: np.ndarray, thresh: float) -> dict:
    """Paired bootstrap on SR@thresh, B minus A."""
    n = ea.size
    idx = boot_index(n)
    obs = float((eb < thresh).mean() - (ea < thresh).mean())
    boot = (eb[idx] < thresh).mean(axis=1) - (ea[idx] < thresh).mean(axis=1)
    lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
    return {
        "value": obs, "n": n, "ci_lo": lo, "ci_hi": hi,
        "sr_a": float((ea < thresh).mean()), "sr_b": float((eb < thresh).mean()),
        "p_two_sided": two_sided_p(boot),
    }


def paired_median_contrast(ea: np.ndarray, eb: np.ndarray) -> dict:
    """Paired bootstrap on median ED over pairs finite under BOTH configs."""
    n = ea.size
    idx = boot_index(n)
    both = np.isfinite(ea) & np.isfinite(eb)
    if both.sum() < 10:
        return {}
    obs = float(np.median(eb[both]) - np.median(ea[both]))
    boot = []
    for row in idx:
        m = both[row]
        if m.sum() < 10:
            continue
        boot.append(np.median(eb[row][m]) - np.median(ea[row][m]))
    boot = np.asarray(boot)
    lo, hi = (float(x) for x in np.percentile(boot, [2.5, 97.5]))
    return {
        "value": obs, "n": int(both.sum()), "ci_lo": lo, "ci_hi": hi,
        "p_two_sided": two_sided_p(boot),
    }


# ------------------------------------------------------------- collection ---
OUT: list[dict] = []
FIELDS = ["section", "metric", "config", "config_ref", "subset", "rung",
          "statistic", "value", "n", "ci_lo", "ci_hi",
          "p_two_sided", "pair_id", "note"]


def emit(**kw) -> None:
    OUT.append({k: kw.get(k, "") for k in FIELDS})


def fmt(v, nd: int = 4) -> str:
    if v == "" or v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf"
        return f"{v:.{nd}f}"
    return str(v)


def main() -> None:
    lines: list[str] = []

    def say(s: str = "") -> None:
        print(s)
        lines.append(s)

    csvs = {name: load_csv(name) for name in {c[1] for c in TABLE1 + EXTRA}}

    def rows_for(src: str, backbone: str, mode: str) -> list[dict]:
        return [r for r in csvs[src] if r["backbone"] == backbone and r["mode"] == mode]

    all_configs = TABLE1 + EXTRA

    say("# Metric sensitivity: raw parametric ED vs TPS-refined ED")
    say()
    say(f"Paired bootstrap B={B}, seed={SEED}, percentile 95 % CI. "
        "Rates carry Wilson 95 % intervals.")
    say()
    say("`raw` = the parametric fit error `mu_ed`. `tps` = `mu_ed_tps` with a "
        "fallback to `mu_ed` when blank, which is the rule the manuscript uses.")
    say()
    summary_anchor = len(lines)
    facts: dict[str, object] = {}

    # ---------------------------------------------------------- A. accuracy --
    say("## A. Table 1 accuracy under both metrics")
    say()
    say("| Configuration | metric | n | ok | med ED (px) | SR@5 [95% CI] | "
        "SR@10 [95% CI] | SR@20 [95% CI] |")
    say("|---|---|---:|---:|---:|---|---|---|")
    for label, src, bb, md in TABLE1:
        rws = rows_for(src, bb, md)
        n_ok = sum(r["status"] == "ok" for r in rws)
        for metric in ("raw", "tps"):
            e = np.array([err(r, metric) for r in rws])
            med, n_fin = median_finite(e)
            facts[f"med_{label}_{metric}"] = med
            cells = []
            for t in THRESHOLDS:
                k = int((e < t).sum())
                lo, hi = wilson(k, e.size)
                cells.append(f"{k / e.size:.3f} [{lo:.3f}, {hi:.3f}]")
                emit(section="A_accuracy", metric=metric, config=label,
                     subset="all_187", statistic=f"SR@{t:.0f}",
                     value=k / e.size, n=e.size, ci_lo=lo, ci_hi=hi,
                     note=f"k={k}")
            emit(section="A_accuracy", metric=metric, config=label,
                 subset="all_187", statistic="median_ED_px", value=med,
                 n=n_fin, note="median over pairs with a finite error")
            emit(section="A_accuracy", metric=metric, config=label,
                 subset="all_187", statistic="n_status_ok", value=float(n_ok),
                 n=e.size)
            say(f"| {label} | {metric} | {e.size} | {n_ok} | {med:.1f} | "
                + " | ".join(cells) + " |")
    say()
    say("SR counts pairs with error strictly below the threshold; rows whose "
        "`status` is not `ok` are scored as +inf and can never succeed. Median "
        "ED is taken over pairs with a finite error, matching "
        "`scripts/summarize_baselines.py`.")
    say()

    # ---------------------------------------------------------- B. coverage --
    say("## B. TPS coverage -- the comparability defect")
    say()
    say("| Configuration | pairs with non-blank `mu_ed_tps` | coverage | "
        "pairs scored on the raw fallback |")
    say("|---|---:|---:|---:|")
    cover: dict[str, float] = {}
    for label, src, bb, md in all_configs:
        rws = rows_for(src, bb, md)
        n = len(rws)
        k = sum(has_tps(r) for r in rws)
        if (label, src, bb, md) in TABLE1:
            cover[label] = k / n
        star = " *" if bb in FT_TAINTED else ""
        note = f"non_blank={k}; fallback_to_raw={n - k}"
        if bb in FT_TAINTED:
            note += f"; {FT_NOTE}"
        say(f"| {label}{star} | {k}/{n} | {k / n:.3f} | {n - k} |")
        emit(section="B_tps_coverage", metric="tps", config=label,
             subset="all_187", statistic="tps_coverage", value=k / n, n=n,
             note=note)
    say()
    say("`*` fine-tuned configuration; see the note in section E.")
    say()
    facts["cover_full"] = sorted(c for c, v in cover.items() if v == 1.0)
    facts["cover_worst"] = min(cover.items(), key=lambda kv: kv[1])

    # ------------------------------------------------------- C. regressions --
    say("## C. Pairs the refinement turns from success into failure")
    say()
    say("A raw error below 10 px paired with a TPS-refined error at or above "
        "10 px. These pairs are counted as successes by the raw metric and as "
        "failures by the metric the manuscript reports.")
    say()
    say("| Configuration | pair_id | raw ED (px) | TPS ED (px) |")
    say("|---|---|---:|---:|")
    total = 0
    per_config: dict[str, int] = defaultdict(int)
    for label, src, bb, md in all_configs:
        for r in rows_for(src, bb, md):
            if r["status"] != "ok":
                continue
            raw, tps = err(r, "raw"), err(r, "tps")
            if raw < 10.0 <= tps:
                total += 1
                per_config[label] += 1
                say(f"| {label} | `{r['pair_id']}` | {raw:.3f} | {tps:.3f} |")
                emit(section="C_regression", metric="raw_to_tps", config=label,
                     subset="all_187", statistic="raw_success_tps_failure",
                     value=tps, pair_id=r["pair_id"],
                     note=f"raw_ed={raw:.3f}; tps_ed={tps:.3f}")
    in_table1 = sum(per_config[c[0]] for c in TABLE1)
    say()
    say(f"**Total: {total} configuration-pair regressions** across the "
        f"{len(all_configs)} configurations in the baseline CSVs "
        f"({in_table1} of them inside Table 1).")
    emit(section="C_regression", metric="raw_to_tps", config="ALL",
         subset="all_187", statistic="regression_count", value=float(total),
         n=len(all_configs), note=f"in_table1={in_table1}")
    say()

    # The single pair that carries the severe-stratum claim.
    key_pair = "eval_5842WCu-Spalled_SEM-SE_SEM-BSE_Multiscale_0#0"
    area = {r["pair_id"]: float(r["fov_area_ratio"])
            for r in load_csv("fov_ratios.csv")}
    bins = [(0.0, 0.05), (0.05, 0.25), (0.25, 0.50), (0.50, 10.0)]
    say("### The pair that carries the severe-stratum claim")
    say()
    say(f"`{key_pair}` has FOV area ratio {area[key_pair]:.5f}, placing it in "
        "the 0.05-0.25 stratum. Under RoMa/direct it is 3.323 px raw and "
        "37.462 px after refinement.")
    say()
    say("| FOV area-ratio bin | n | metric | RoMa direct SR@10 | "
        "RoMa + pyramid v2 SR@10 |")
    say("|---|---:|---|---:|---:|")
    for lo, hi in bins:
        ids = {p for p, a in area.items() if lo <= a < hi}
        for metric in ("raw", "tps"):
            vals = []
            for bb, md in (("roma", "direct"), ("roma", "pyramid_v2")):
                e = np.array([err(r, metric)
                              for r in rows_for("baselines_A.csv", bb, md)
                              if r["pair_id"] in ids])
                k = int((e < 10).sum())
                vals.append((k, int(e.size)))
                if (lo, hi) == (0.05, 0.25):
                    facts[f"stratum_{metric}_{md}"] = (k, int(e.size))
                emit(section="C_fov_stratum", metric=metric,
                     config=f"roma/{md}", subset=f"fov_area_{lo:g}-{hi:g}",
                     statistic="SR@10",
                     value=k / e.size if e.size else float("nan"),
                     n=int(e.size), note=f"k={k}")
            say(f"| {lo:.2f}-{hi:.2f} | {len(ids)} | {metric} | "
                f"{vals[0][0] / vals[0][1]:.3f} ({vals[0][0]}/{vals[0][1]}) | "
                f"{vals[1][0] / vals[1][1]:.3f} ({vals[1][0]}/{vals[1][1]}) |")
    say()
    say("In the 0.05-0.25 stratum the TPS metric reads 0.000 -> 0.030, which is "
        "the manuscript's \"first non-zero severe-stratum result\". The raw "
        "metric reads 0.030 -> 0.061 in the same stratum: the direct baseline "
        "already had a success there, and it is the pair above. The wrapper "
        "gain in that stratum is +1 pair under both metrics; only its "
        "*from-zero* character is an artefact of the scoring choice.")
    say()

    # -------------------------------------------------------- D. contrasts --
    say("## D. Native-pair contrasts under both metrics")
    say()
    contrasts = [
        ("RoMa + pyramid v2 vs RoMa direct", ("roma", "pyramid_v2"), ("roma", "direct")),
        ("MatchAnything-RoMa vs RoMa direct", ("ma_roma", "direct"), ("roma", "direct")),
    ]
    say("| Contrast (B vs A) | metric | statistic | A | B | delta | 95% CI | "
        "p (two-sided) |")
    say("|---|---|---|---:|---:|---:|---|---:|")
    for label, (bbB, mdB), (bbA, mdA) in contrasts:
        rA = {r["pair_id"]: r for r in rows_for("baselines_A.csv", bbA, mdA)}
        rB = {r["pair_id"]: r for r in rows_for("baselines_A.csv", bbB, mdB)}
        ids = sorted(set(rA) & set(rB))
        for metric in ("raw", "tps"):
            ea = np.array([err(rA[i], metric) for i in ids])
            eb = np.array([err(rB[i], metric) for i in ids])
            for t in THRESHOLDS:
                s = paired_rate_contrast(ea, eb, t)
                say(f"| {label} | {metric} | SR@{t:.0f} | {s['sr_a']:.4f} | "
                    f"{s['sr_b']:.4f} | {s['value']:+.4f} | "
                    f"[{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}] | "
                    f"{s['p_two_sided']:.4f} |")
                if t == 10.0:
                    facts[f"contrast_{bbB}_{mdB}_{metric}"] = s
                emit(section="D_contrast", metric=metric, config=f"{bbB}/{mdB}",
                     config_ref=f"{bbA}/{mdA}", subset="all_187",
                     statistic=f"delta_SR@{t:.0f}", value=s["value"], n=s["n"],
                     ci_lo=s["ci_lo"], ci_hi=s["ci_hi"],
                     p_two_sided=s["p_two_sided"],
                     note=f"A={s['sr_a']:.4f}; B={s['sr_b']:.4f}")
            m = paired_median_contrast(ea, eb)
            if m:
                say(f"| {label} | {metric} | median ED (px) |  |  | "
                    f"{m['value']:+.1f} | [{m['ci_lo']:+.1f}, {m['ci_hi']:+.1f}] | "
                    f"{m['p_two_sided']:.4f} |")
                emit(section="D_contrast", metric=metric, config=f"{bbB}/{mdB}",
                     config_ref=f"{bbA}/{mdA}", subset="all_187",
                     statistic="delta_median_ED_px", value=m["value"], n=m["n"],
                     ci_lo=m["ci_lo"], ci_hi=m["ci_hi"],
                     p_two_sided=m["p_two_sided"],
                     note="pairs finite under both configurations")
    say()
    say("`p (two-sided)` is 2 x min(P(delta <= 0), P(delta >= 0)) over the "
        "bootstrap replicates, clipped at 1. That is the convention the "
        "manuscript's methods text declares and the values it prints; see the "
        "note at the foot of this report.")
    say()

    # ------------------------------------------------------- E. FOV ladder --
    say("## E. FOV ladder wrapper contrast (pyramid v2 vs direct)")
    say()
    ladder = load_csv("fov_ladder.csv")
    base = load_csv("baselines_A.csv")
    split = json.loads((RESULTS / "split.json").read_text(encoding="utf-8"))
    testbed = {r["pair_id"] for r in ladder}
    n_train = len(set(split["train"]) & testbed)
    n_val = len(set(split["val"]) & testbed)
    n_test = len(set(split["test"]) & testbed)
    heldout = (set(split["val"]) | set(split["test"])) & testbed

    say(f"Ladder testbed: {len(testbed)} pairs = {n_train} train / {n_val} val "
        f"/ {n_test} test with respect to `results/split.json`. `ma_roma_ft` "
        "was fine-tuned on the train split, so any ladder comparison that "
        "mixes train pairs into an ft-vs-zero-shot claim is training-enriched.")
    say()
    say("`base-matchable` is the manuscript's testbed filter: pairs whose "
        "full-FOV **direct** `mu_ed` is < 20 px for that backbone. It is "
        "computed per backbone, so the three backbones are evaluated on "
        "different denominators.")
    say()
    say("| Backbone | subset | rung | metric | n | direct SR@10 | pyr v2 SR@10 | "
        "delta | 95% CI | p (two-sided) |")
    say("|---|---|---:|---|---:|---:|---:|---:|---|---:|")

    def mu_raw(r: dict) -> float:
        return float(r["mu_ed"]) if (r["status"] == "ok" and cell(r, "mu_ed")) else INF

    pools: dict[str, tuple[int, int]] = {}
    for bb in ("roma", "ma_roma", "ma_roma_ft"):
        matchable = {r["pair_id"] for r in base
                     if r["backbone"] == bb and r["mode"] == "direct"
                     and mu_raw(r) < 20 and r["pair_id"] in testbed}
        pools[bb] = (len(matchable), len(matchable & heldout))
        subsets = [
            ("base-matchable", matchable),
            ("base-matchable AND held-out", matchable & heldout),
        ]
        for sub_label, keep in subsets:
            emit(section="E_fov_ladder", metric="", config=bb, subset=sub_label,
                 statistic="denominator_pool", value=float(len(keep)),
                 n=len(testbed),
                 note=("pairs passing the base-matchable filter"
                       + ("; split.json train excluded"
                          if "held-out" in sub_label else "")))
            for rung in (0.5, 0.25, 0.10):
                for metric in ("raw", "tps"):
                    d = {r["pair_id"]: err(r, metric) for r in ladder
                         if r["backbone"] == bb and r["mode"] == "direct"
                         and float(r["rung"]) == rung and r["status"] != "skipped"
                         and r["pair_id"] in keep}
                    p = {r["pair_id"]: err(r, metric) for r in ladder
                         if r["backbone"] == bb and r["mode"] == "pyramid_v2"
                         and float(r["rung"]) == rung and r["status"] != "skipped"
                         and r["pair_id"] in keep}
                    ids = sorted(set(d) & set(p))
                    if len(ids) < 2:
                        continue
                    ea = np.array([d[i] for i in ids])
                    eb = np.array([p[i] for i in ids])
                    s = paired_rate_contrast(ea, eb, 10.0)
                    facts[f"ladder_{bb}_{rung:g}_{metric}_"
                          f"{'held' if 'held-out' in sub_label else 'all'}"] = s
                    tainted = bb in FT_TAINTED and "held-out" not in sub_label
                    say(f"| {bb}{' *' if tainted else ''} | {sub_label} | "
                        f"{rung:g} | {metric} | {s['n']} | {s['sr_a']:.3f} | "
                        f"{s['sr_b']:.3f} | {s['value']:+.3f} | "
                        f"[{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}] | "
                        f"{s['p_two_sided']:.4f} |")
                    note = f"direct={s['sr_a']:.4f}; pyramid_v2={s['sr_b']:.4f}"
                    if tainted:
                        note += f"; {FT_NOTE}"
                    emit(section="E_fov_ladder", metric=metric,
                         config=f"{bb}/pyramid_v2", config_ref=f"{bb}/direct",
                         subset=sub_label, rung=f"{rung:g}", statistic="delta_SR@10",
                         value=s["value"], n=s["n"], ci_lo=s["ci_lo"],
                         ci_hi=s["ci_hi"],
                         p_two_sided=s["p_two_sided"], note=note)
    say()
    say("`*` denominator includes fine-tuning training pairs.")
    say()
    say("Per-backbone denominators on the ladder:")
    say()
    say("| Backbone | base-matchable | base-matchable AND held-out |")
    say("|---|---:|---:|")
    for bb in ("roma", "ma_roma", "ma_roma_ft"):
        say(f"| {bb} | {pools[bb][0]} | {pools[bb][1]} |")
    say()

    # ------------------------------------------------------------ summary --
    def srline(key: str) -> str:
        s = facts[key]
        return (f"{s['sr_a']:.4f} -> {s['sr_b']:.4f} "
                f"({s['value']:+.4f}, CI [{s['ci_lo']:+.4f}, {s['ci_hi']:+.4f}], "
                f"p = {s['p_two_sided']:.4f})")

    def ladline(key: str) -> str:
        s = facts[key]
        return (f"n = {s['n']}, {s['sr_a']:.3f} -> {s['sr_b']:.3f} "
                f"({s['value']:+.3f}, CI [{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}], "
                f"p = {s['p_two_sided']:.4f})")

    worst_label, worst_cov = facts["cover_worst"]
    summary = [
        "## Summary",
        "",
        "The manuscript scores accuracy on the TPS-refined error. Neither of "
        "its two significant native-pair results survives the switch to the raw "
        "parametric error that the matcher itself produces.",
        "",
        "1. **The wrapper gain is metric-dependent.** RoMa + pyramid v2 vs RoMa "
        "direct, SR@10 over all 187 pairs: "
        f"TPS {srline('contrast_roma_pyramid_v2_tps')}; "
        f"raw {srline('contrast_roma_pyramid_v2_raw')}. Under the raw metric "
        "the wrapper moves exactly zero pairs across the 10 px line.",
        "",
        "2. **The backbone gain is metric-dependent.** MatchAnything-RoMa vs "
        "RoMa direct, SR@10: "
        f"TPS {srline('contrast_ma_roma_direct_tps')}; "
        f"raw {srline('contrast_ma_roma_direct_raw')}. The raw effect points "
        "the same way at roughly half the size and its CI covers zero.",
        "",
        "3. **The TPS column is not one metric.** TPS coverage ranges from "
        f"{worst_cov:.3f} for {worst_label} up to 1.000 for "
        f"{', '.join(facts['cover_full'])}. The dense RoMa-family "
        "configurations are scored on refined error for all 187 pairs while "
        "the weak configurations are silently scored on raw error for the "
        "majority of theirs, so part of the gap between the two families in "
        "Table 1 is a difference in scoring, not in registration. The sharpest "
        "case is Control B (SIFT + mutual information), whose TPS coverage is "
        "zero: it is a control scored entirely on the raw metric against "
        "treatments scored entirely on the refined one.",
        "",
        f"4. **Refinement destroys {total} successes.** Across the "
        f"{len(all_configs)} configurations in the baseline CSVs, {total} "
        "configuration-pair combinations register below 10 px on the raw fit "
        "and above 10 px after refinement (section C). One of them, "
        f"`{key_pair}` under RoMa/direct (3.323 px raw, 37.462 px TPS), is the "
        "reason the 0.05-0.25 FOV stratum reads "
        f"{facts['stratum_tps_direct'][0]}/{facts['stratum_tps_direct'][1]} -> "
        f"{facts['stratum_tps_pyramid_v2'][0]}/"
        f"{facts['stratum_tps_pyramid_v2'][1]} under TPS and "
        f"{facts['stratum_raw_direct'][0]}/{facts['stratum_raw_direct'][1]} -> "
        f"{facts['stratum_raw_pyramid_v2'][0]}/"
        f"{facts['stratum_raw_pyramid_v2'][1]} under raw error. The wrapper "
        "still gains one pair in that stratum either way, but the "
        "\"first non-zero severe-stratum result\" framing is an artefact of "
        "the refinement failing on the one pair the direct baseline had "
        "already solved.",
        "",
        "5. **The FOV-ladder result is the strongest wrapper result, and it "
        "strengthens when the fine-tuning split is respected -- but it is a "
        "result on the raw metric only.** The ladder is computed and reported "
        "on raw (unrefined) matcher error, which is the only metric meaningful "
        "at every rung. MatchAnything-RoMa at rung 0.10 on raw error: "
        f"base-matchable {ladline('ladder_ma_roma_0.1_raw_all')}; restricted to "
        "pairs held out of fine-tuning "
        f"{ladline('ladder_ma_roma_0.1_raw_held')}. Rescored under the TPS "
        "metric the same rung weakens to non-significance "
        f"({ladline('ladder_ma_roma_0.1_tps_all')}): the effect points the same "
        "way at half the size, but two-sided it no longer clears 0.05, because "
        "refinement is close to useless at 10 % FOV. The ladder claim is "
        "therefore significant on the unrefined metric it is measured on, and "
        "must not be stated as holding under both metrics.",
        "",
        "6. **Descriptive medians move too.** The pyramid v1 collapse is "
        "quoted as median ED "
        f"{facts['med_RoMa (zero-shot)_tps']:.0f} -> "
        f"{facts['med_RoMa + pyramid v1_tps']:.0f} px; those are TPS-metric "
        "numbers and the corresponding raw pair is "
        f"{facts['med_RoMa (zero-shot)_raw']:.1f} -> "
        f"{facts['med_RoMa + pyramid v1_raw']:.1f} px (section A). The "
        "qualitative finding is unchanged and if anything larger on raw error, "
        "but the specific figures are metric-specific and should be labelled.",
        "",
        "Recommendation for the revision: report Table 1 under both metrics, "
        "state the TPS coverage per configuration, and demote the "
        "severe-stratum \"from zero\" sentence to a one-pair observation. The "
        "low-FOV ladder result should carry the wrapper claim.",
        "",
    ]
    lines[summary_anchor:summary_anchor] = summary

    # --------------------------------------------------------- caveats -----
    caveats = [
        "## Notes and caveats",
        "",
        "**p-value convention.** Every p-value in this report and in "
        "`results/metric_sensitivity.csv` is two-sided: "
        "2 x min(P(delta <= 0), P(delta >= 0)) over the paired-bootstrap "
        "replicates, clipped at 1. This matches the convention the "
        "manuscript's methods text declares, and it is symmetric under "
        "swapping the two configurations. An earlier version of this analysis "
        "quoted the one-sided tail mass P(delta <= 0), which is half these "
        "values wherever the observed difference is positive. The one "
        "substantive consequence of the correction is the FOV ladder at rung "
        "0.10 rescored under the TPS metric: 0.0434 one-sided becomes 0.0868 "
        "two-sided, which does not clear 0.05 (summary point 5). The two "
        "native-pair TPS results (0.0340 and 0.0350) and the ladder on raw "
        "error (0.0028) remain significant at 0.05 two-sided.",
        "",
        "**Fine-tuned configuration.** `ma_roma_ft` was fine-tuned on the 131 "
        "pairs in the `train` split of `results/split.json`. It must not enter "
        "any aggregate over all 187 pairs unless it is restricted to held-out "
        "pairs or explicitly marked; it is excluded from Table 1 and appears "
        "here only in the coverage, regression and ladder blocks, flagged.",
        "",
        "**Ladder denominators.** The base-matchable filter is computed per "
        "backbone, so the three backbones in section E are scored on different "
        "pair sets. Intersecting with the held-out split shrinks them further "
        "and unevenly. Any ft-versus-zero-shot statement read off the "
        "base-matchable rows is both training-enriched and computed on "
        "non-identical denominators.",
        "",
        "**Reproduce.** `python scripts/metric_sensitivity.py` from the repo "
        "root regenerates `results/metric_sensitivity.csv` and this report. "
        "Inputs: `results/baselines_A.csv`, `results/baselines_B.csv`, "
        "`results/fov_ladder.csv`, `results/fov_ratios.csv`, "
        "`results/split.json`.",
        "",
    ]
    for line in caveats:
        say(line)

    # --------------------------------------------------------------- write --
    RESULTS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    out_csv = RESULTS / "metric_sensitivity.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in OUT:
            w.writerow({k: fmt(v) for k, v in row.items()})

    out_md = REPORTS / "metric_sensitivity.md"
    header = [
        "<!-- Generated by scripts/metric_sensitivity.py. Do not edit by hand;",
        "     rerun `python scripts/metric_sensitivity.py` from the repo root. -->",
        "",
    ]
    out_md.write_text("\n".join(header + lines) + "\n", encoding="utf-8")
    print(f"\nwrote {out_csv} ({len(OUT)} rows)")
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()
