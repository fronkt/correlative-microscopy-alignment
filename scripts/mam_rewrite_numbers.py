"""Every number the M&M rewrite quotes, recomputed from the per-pair result files.

The gate for the rejected submission (verify_mam_draft.py) asserted hard-coded strings
copied from an earlier text. It passed 109 checks while the paper said "106 of 187
pairs become hard failures" about a GPU crash, and binned pairs by a metadata ratio it
called GT-implied. This script replaces the copied strings with recomputation: it
reads only result CSVs, writes results/mam_rewrite_numbers.json, and
scripts/verify_mam_rewrite.py checks the manuscript against that file.

Inputs: results/baselines_A.csv, baselines_B.csv, fov_ladder.csv, appearance_nmi.csv,
fov_ratios.csv (metadata), fov_ratios_gt.csv (scripts/fov_ratios_gt.py),
pyramid_v1_tiles.csv (scripts/pyramid_v1_audit.py), split.json.

Usage: python scripts/mam_rewrite_numbers.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from metric_sensitivity import (  # noqa: E402
    err, median_finite, paired_median_contrast, paired_rate_contrast, wilson,
)

RES = ROOT / "results"
OUT = RES / "mam_rewrite_numbers.json"
N = 187
BINS = [(0.0, 0.05), (0.05, 0.25), (0.25, 0.5), (0.5, np.inf)]
BIN_LABELS = ["<0.05", "0.05-0.25", "0.25-0.5", ">=0.5"]

# The 187-pair accuracy table. Pyramid v1 is NOT here: it was evaluated on 81 pairs
# (see pyramid_v1_audit.py) and gets its own same-pair table.
CONFIGS = [
    ("sift", "SIFT", "baselines_A.csv", "sift", "direct"),
    ("sift_mi", "SIFT + mutual information", "baselines_B.csv", "sift", "classical"),
    ("loftr", "LoFTR", "baselines_A.csv", "loftr", "direct"),
    ("ma_eloftr", "MatchAnything-ELoFTR", "baselines_A.csv", "matchanything", "direct"),
    ("roma", "RoMa", "baselines_A.csv", "roma", "direct"),
    ("roma_v2", "RoMa + checked tiling", "baselines_A.csv", "roma", "pyramid_v2"),
    ("ma_roma", "MatchAnything-RoMa", "baselines_A.csv", "ma_roma", "direct"),
    ("ma_roma_v2", "MatchAnything-RoMa + checked tiling", "baselines_A.csv", "ma_roma", "pyramid_v2"),
]
ZERO_SHOT_OR_WRAPPED = [c[0] for c in CONFIGS]


def load(name: str) -> list[dict]:
    with (RES / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def rows_by_pair(src: str, backbone: str, mode: str) -> dict[str, dict]:
    out = {r["pair_id"]: r for r in load(src) if r["backbone"] == backbone and r["mode"] == mode}
    return out


def r3(x: float) -> float:
    return float(f"{x:.3f}")


def main() -> None:
    num: dict = {}
    base = {c[0]: rows_by_pair(*c[2:]) for c in CONFIGS}
    pairs = sorted(base["roma"])
    assert len(pairs) == N and all(len(base[k]) == N for k in base), "every config must cover 187 pairs"

    def errs(key: str, metric: str = "raw", ids=None) -> np.ndarray:
        ids = pairs if ids is None else ids
        return np.array([err(base[key][p], metric) for p in ids])

    # ------------------------------------------------------------ table: 187 pairs
    table = {}
    for key, label, *_ in CONFIGS:
        e = errs(key)
        med, nfin = median_finite(e)
        table[key] = {"label": label, "median_ed": round(med, 1), "fits": int(nfin),
                      **{f"sr{t}": r3((e < t).mean()) for t in (5, 10, 20)},
                      **{f"k{t}": int((e < t).sum()) for t in (5, 10, 20)}}
        lo, hi = wilson(int((e < 10).sum()), N)
        table[key]["sr10_wilson"] = [r3(lo), r3(hi)]
    num["table187"] = table

    # Failure causes of every failed row in the table: must all be method failures.
    causes = {}
    for key in table:
        for p in pairs:
            r = base[key][p]
            if r["status"] != "ok":
                causes.setdefault(key, []).append(r["error"][:40])
    assert not any("CUDA" in c for v in causes.values() for c in v), "infrastructure failure in table"
    num["table187_failure_causes"] = {k: sorted(set(v)) for k, v in causes.items()}

    # ------------------------------------------------------------ pyramid v1 audit
    tiles = {r["pair_id"]: r for r in load("pyramid_v1_tiles.csv")}
    v1 = rows_by_pair("baselines_A.csv", "roma", "pyramid")
    evaluated = [p for p in pairs if tiles[p]["v1_status"] == "evaluated"]
    crashed = [p for p in pairs if tiles[p]["v1_status"] == "crashed"]
    assert len(evaluated) + len(crashed) == N
    assert all("CUDA error" in v1[p]["error"] for p in crashed)

    def v1_block(ids: list[str]) -> dict:
        d = np.array([err(base["roma"][p], "raw") for p in ids])
        v = np.array([err(v1[p], "raw") for p in ids])
        fd = np.array([int(base["roma"][p]["n_inliers"]) / int(base["roma"][p]["n_matches"]) for p in ids])
        fv = np.array([int(v1[p]["n_inliers"]) / int(v1[p]["n_matches"]) for p in ids])
        nt = np.array([int(tiles[p]["n_tiles"]) for p in ids])
        return {"n": len(ids),
                "median_ed_direct": round(float(np.median(d)), 1), "median_ed_v1": round(float(np.median(v)), 1),
                "k10_direct": int((d < 10).sum()), "k10_v1": int((v < 10).sum()),
                "k20_direct": int((d < 20).sum()), "k20_v1": int((v < 20).sum()),
                "inlier_frac_direct": round(float(np.median(fd)), 4),
                "inlier_frac_v1": float(f"{np.median(fv):.2g}"),
                "inlier_frac_ratio": round(float(np.median(fd) / np.median(fv))),
                "n_worse": int((v > d).sum()),
                "tiles_median": float(np.median(nt)), "tiles_min": int(nt.min()), "tiles_max": int(nt.max())}

    tiled = [p for p in evaluated if int(tiles[p]["n_tiles"]) > 1]
    padded = [p for p in evaluated if int(tiles[p]["n_tiles"]) == 1]
    assert all(tiles[p]["padded"] == "1" for p in padded)
    num["v1"] = {
        "evaluated": len(evaluated), "crashed": len(crashed),
        "crash_starts_at": pairs.index(crashed[0]) + 1 if crashed else None,
        "owed_tile_matches": sum(int(tiles[p]["n_tiles"]) for p in crashed),
        "owed_direct_successes": sum(err(base["roma"][p], "raw") < 10 for p in crashed),
        "padded_pairs_all187": sum(tiles[p]["padded"] == "1" and int(tiles[p]["n_tiles"]) == 1 for p in pairs),
        "max_pooled": max(int(v1[p]["n_matches"]) for p in evaluated),
        "all81": v1_block(evaluated), "tiled": v1_block(tiled), "padded": v1_block(padded),
    }

    # ------------------------------------------------------------ FOV ratios
    gt = {r["pair_id"]: r for r in load("fov_ratios_gt.csv")}
    fov = {p: float(gt[p]["fov_area_ratio_gt"]) for p in pairs}
    meta = {p: float(gt[p]["meta_area_ratio"]) for p in pairs}
    inconsistent = sorted(p for p in pairs if gt[p]["metadata_consistent"] == "0")

    def stratum(x: float) -> int:
        return next(i for i, (lo, hi) in enumerate(BINS) if lo <= x < hi)

    s_gt = {p: stratum(fov[p]) for p in pairs}
    s_meta = {p: stratum(meta[p]) for p in pairs}
    num["fov"] = {
        "metadata_inconsistent": len(inconsistent),
        "metadata_inconsistent_subsets": sorted({gt[p]["subclass"] for p in inconsistent}),
        "strata_gt": [sum(s_gt[p] == i for p in pairs) for i in range(4)],
        "strata_meta": [sum(s_meta[p] == i for p in pairs) for i in range(4)],
        "stratum_changes": sum(s_gt[p] != s_meta[p] for p in pairs),
        "min_ratio_gt": round(min(fov.values()), 4),
        "below_half_gt": sum(fov[p] < 0.5 for p in pairs),
        "median_ratio_gt": round(float(np.median(list(fov.values()))), 2),
    }

    strata = {}
    for key, label, *_ in CONFIGS:
        e = dict(zip(pairs, errs(key)))
        strata[key] = [f"{sum(e[p] < 10 for p in pairs if s_gt[p] == i)}/{sum(s_gt[p] == i for p in pairs)}"
                       for i in range(4)]
    num["strata_sr10"] = strata
    below = [p for p in pairs if fov[p] < 0.5]
    num["below_half_max_registered"] = max(
        int(sum(err(base[k][p], "raw") < 10 for p in below)) for k in ZERO_SHOT_OR_WRAPPED)

    # ------------------------------------------------------------ appearance axis
    nmi = {r["pair_id"]: float(r["nmi"]) for r in load("appearance_nmi.csv")}
    x = np.log10([fov[p] for p in pairs])
    y = np.array([nmi[p] for p in pairs])
    r, pv = stats.pearsonr(x, y)
    rs, ps = stats.spearmanr(x, y)
    xm = np.log10([meta[p] for p in pairs])
    rm, pm = stats.pearsonr(xm, y)
    num["appearance"] = {
        "pearson_r": round(float(r), 3), "pearson_p": round(float(pv), 3),
        "spearman_rho": round(float(rs), 3), "spearman_p": round(float(ps), 3),
        "pearson_r_metadata_ratio": round(float(rm), 3), "pearson_p_metadata_ratio": round(float(pm), 3),
        "median_nmi_by_stratum": [round(float(np.median([nmi[p] for p in pairs if s_gt[p] == i])), 4)
                                  for i in range(4)],
        "median_nmi_all": round(float(np.median(y)), 4),
    }

    # ------------------------------------------------------------ paired contrasts (187)
    def contrast(a: str, b: str, metric: str, t: float) -> dict:
        c = paired_rate_contrast(errs(a, metric), errs(b, metric), t)
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()}

    num["contrasts"] = {
        "wrapper_sr10_raw": contrast("roma", "roma_v2", "raw", 10),
        "wrapper_sr20_raw": contrast("roma", "roma_v2", "raw", 20),
        "wrapper_sr10_tps": contrast("roma", "roma_v2", "tps", 10),
        "backbone_sr10_raw": contrast("roma", "ma_roma", "raw", 10),
        "backbone_sr10_tps": contrast("roma", "ma_roma", "tps", 10),
    }
    for name, a, b in (("wrapper_median_raw", "roma", "roma_v2"), ("backbone_median_raw", "roma", "ma_roma")):
        c = paired_median_contrast(errs(a), errs(b))
        num["contrasts"][name] = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in c.items()}

    # Composition shift of the checked wrapper, by GT stratum.
    shift = []
    for i in range(4):
        ids = [p for p in pairs if s_gt[p] == i]
        d, v = errs("roma", ids=ids), errs("roma_v2", ids=ids)
        shift.append({"gained": int(((v < 10) & ~(d < 10)).sum()), "lost": int(((d < 10) & ~(v < 10)).sum())})
    num["wrapper_shift_by_stratum"] = shift

    # ------------------------------------------------------------ FOV ladder (rung 0.1)
    ladder = load("fov_ladder.csv")
    split = json.loads((RES / "split.json").read_text())
    testbed = {r["pair_id"] for r in ladder}
    heldout = (set(split["val"]) | set(split["test"])) & testbed
    bad = set(inconsistent)

    def ladder_contrast(bb: str, rung: float, subset: set[str] | None = None) -> dict:
        allrows = rows_by_pair("baselines_A.csv", bb, "direct")
        match = {p for p, r in allrows.items() if err(r, "raw") < 20 and p in testbed}
        if subset is not None:
            match &= subset

        def at(mode: str) -> dict[str, float]:
            return {r["pair_id"]: err(r, "raw") for r in ladder
                    if r["backbone"] == bb and r["mode"] == mode and float(r["rung"]) == rung
                    and r["status"] != "skipped" and r["pair_id"] in match}

        d, v = at("direct"), at("pyramid_v2")
        ids = sorted(set(d) & set(v))
        ed, ev = np.array([d[i] for i in ids]), np.array([v[i] for i in ids])
        n = len(ids)
        idx = np.random.default_rng(0).integers(0, n, size=(10_000, n))  # as fov_ladder_bootstrap.py
        boot = (ev[idx] < 10).mean(axis=1) - (ed[idx] < 10).mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        p = min(1.0, 2.0 * min(float((boot <= 0).mean()), float((boot >= 0).mean())))
        return {"n": n, "k_direct": int((ed < 10).sum()), "k_v2": int((ev < 10).sum()),
                "sr_direct": r3((ed < 10).mean()), "sr_v2": r3((ev < 10).mean()),
                "delta": r3((ev < 10).mean() - (ed < 10).mean()), "ci": [r3(lo), r3(hi)], "p": round(p, 4)}

    notbad = testbed - bad

    def ladder_stages(bb: str, rung: float) -> dict:
        allrows = rows_by_pair("baselines_A.csv", bb, "direct")
        match = {p for p, r in allrows.items() if err(r, "raw") < 20 and p in testbed}
        st: dict[str, int] = {}
        for r in ladder:
            if (r["backbone"] == bb and r["mode"] == "pyramid_v2" and float(r["rung"]) == rung
                    and r["status"] == "ok" and r["pair_id"] in match and err(r, "raw") < 10):
                s = r["family"].split("@")[1]
                st[s] = st.get(s, 0) + 1
        return st

    num["ladder_v2_success_stages_ma_roma_r010"] = ladder_stages("ma_roma", 0.1)
    num["ladder"] = {
        "testbed_pairs": len(testbed),
        "testbed_metadata_inconsistent": len(testbed & bad),
        "ma_roma_r010": ladder_contrast("ma_roma", 0.1),
        "ma_roma_r010_heldout": ladder_contrast("ma_roma", 0.1, heldout),
        "ma_roma_r010_consistent_only": ladder_contrast("ma_roma", 0.1, notbad),
        "roma_r010": ladder_contrast("roma", 0.1),
        "ma_roma_ft_r010_heldout": ladder_contrast("ma_roma_ft", 0.1, heldout),
        "base_matchable": {bb: len({p for p, r in rows_by_pair("baselines_A.csv", bb, "direct").items()
                                    if err(r, "raw") < 20 and p in testbed})
                           for bb in ("roma", "ma_roma", "ma_roma_ft")},
    }

    # ------------------------------------------------------------ transform family (H3)
    fam = {"affine": [], "homography": []}
    for bb in ("sift", "loftr", "matchanything", "matchanything_stretch", "roma", "ma_roma"):
        for p, r in rows_by_pair("baselines_A.csv", bb, "direct").items():
            if r["status"] == "ok" and r["mu_ed"] and float(r["mu_ed"]) < 20:
                fam[r["family"]].append(float(r["mu_ed"]))
    n_good = len(fam["affine"]) + len(fam["homography"])
    num["family"] = {"n_well_registered": n_good, "affine": len(fam["affine"]),
                     "homography": len(fam["homography"]),
                     "affine_pct": round(100 * len(fam["affine"]) / n_good),
                     "median_affine": round(float(np.median(fam["affine"])), 1),
                     "median_homography": round(float(np.median(fam["homography"])), 1)}

    # ------------------------------------------------------------ refinement coverage
    cov = {}
    for key, *_ in CONFIGS:
        cov[key] = sum(bool((base[key][p].get("mu_ed_tps") or "").strip()) for p in pairs)
    num["tps_coverage"] = cov

    # ------------------------------------------------------------ smaller quoted facts
    # Every RoMa-family direct call returns the sampler's full 10,000 correspondences.
    num["cap_hit"] = {k: sum(int(base[k][p]["n_matches"]) == 10_000 for p in pairs)
                      for k in ("roma", "ma_roma")}

    # Ground-truth floor: a global affine through each pair's own GT points.
    from cma.data import AmalgaMatchLoader
    loader = AmalgaMatchLoader(ROOT / "data" / "AmalgaMatch")
    resid = []
    for rec in loader.records:
        g = loader._gt[rec.pair_id]
        X = np.hstack([g.tgt_xy, np.ones((len(g), 1))])
        A, *_ = np.linalg.lstsq(X, g.src_xy, rcond=None)
        resid.append(float(np.linalg.norm(X @ A - g.src_xy, axis=1).mean()))
    npts = [len(loader._gt[r.pair_id]) for r in loader.records]
    # Ceiling: the best a single global transform can score, fitted to the GT itself.
    import cv2
    oracle_h = []
    for rec in loader.records:
        g = loader._gt[rec.pair_id]
        H, _ = cv2.findHomography(g.tgt_xy.astype(np.float64), g.src_xy.astype(np.float64), method=0)
        hom = np.hstack([g.tgt_xy, np.ones((len(g), 1))]) @ H.T
        oracle_h.append(float(np.linalg.norm(hom[:, :2] / hom[:, 2:3] - g.src_xy, axis=1).mean()))
    resid_a, resid_h = np.array(resid), np.array(oracle_h)
    num["oracle"] = {f"{fam}_k{t}": int((arr < t).sum())
                     for fam, arr in (("affine", resid_a), ("homography", resid_h)) for t in (5, 10, 20)}
    num["oracle"]["either_k10"] = int((np.minimum(resid_a, resid_h) < 10).sum())
    num["oracle"]["either_k20"] = int((np.minimum(resid_a, resid_h) < 20).sum())
    num["gt"] = {"affine_floor_median": round(float(np.median(resid)), 1),
                 "points_median": int(np.median(npts)), "points_min": int(min(npts)),
                 "points_max": int(max(npts)), "pairs_floor_over_10": int(sum(x >= 10 for x in resid))}

    # Ablations of the checked wrapper (all 187 pairs).
    c50 = rows_by_pair("baselines_A.csv", "roma", "pyramid_v2+c50")
    z3 = rows_by_pair("baselines_A.csv", "roma", "pyramid_v2+z3")
    e_v2 = {m: errs("roma_v2", m) for m in ("raw", "tps")}
    e_c50 = {m: np.array([err(c50[p], m) for p in pairs]) for m in ("raw", "tps")}
    e_z3 = {m: np.array([err(z3[p], m) for p in pairs]) for m in ("raw", "tps")}

    def rc(ea, eb, t):
        c = paired_rate_contrast(ea, eb, t)
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()}

    num["ablations"] = {
        "gate_sr20_raw": rc(e_v2["raw"], e_c50["raw"], 20), "gate_sr20_tps": rc(e_v2["tps"], e_c50["tps"], 20),
        "gate_sr10_raw": rc(e_v2["raw"], e_c50["raw"], 10),
        "zoom_sr10_raw": rc(e_v2["raw"], e_z3["raw"], 10), "zoom_sr20_raw": rc(e_v2["raw"], e_z3["raw"], 20),
    }

    # Per-pair behaviour of the checked wrapper on the primary (unrefined) metric.
    d_raw, v_raw = errs("roma"), errs("roma_v2")
    fin = np.isfinite(d_raw) & np.isfinite(v_raw)
    worse = fin & (v_raw > d_raw + 1e-9)
    material = fin & (v_raw > d_raw + 10) & (v_raw > 1.5 * d_raw)
    iw = int(np.argmax(np.where(fin, v_raw - d_raw, -np.inf)))
    fam_v2 = [base["roma_v2"][p]["family"] for p in pairs]
    num["v2_per_pair"] = {"worse": int(worse.sum()), "better": int((fin & (v_raw < d_raw - 1e-9)).sum()),
                          "unchanged": int((fin & (np.abs(v_raw - d_raw) <= 1e-9)).sum()),
                          "materially_worse": int(material.sum()),
                          "worst_from": round(float(d_raw[iw]), 1), "worst_to": round(float(v_raw[iw]), 1),
                          "stage_counts": {s: sum(f.split("@")[1] == s for f in fam_v2)
                                           for s in sorted({f.split("@")[1] for f in fam_v2})}}

    # Refinement turning successes into failures, over every configuration in the CSVs.
    regress = 0
    configs_seen = 0
    for src in ("baselines_A.csv", "baselines_B.csv"):
        rows = load(src)
        for key in sorted({(r["backbone"], r["mode"]) for r in rows}):
            configs_seen += 1
            for r in rows:
                if (r["backbone"], r["mode"]) == key and r["status"] == "ok" and r["mu_ed"] \
                        and (r.get("mu_ed_tps") or "").strip():
                    regress += float(r["mu_ed"]) < 10 <= float(r["mu_ed_tps"])
    num["tps_regressions"] = {"count": int(regress), "configs": configs_seen}

    # Fine-tuning: the one run whose per-pair rows survive (checkpoints/ma_roma_ft.pth),
    # scored on the held-out test split, both metrics.
    test = sorted(split["test"])
    ft = rows_by_pair("baselines_A.csv", "ma_roma_ft", "direct")
    zs = base["ma_roma"]
    ftb = {}
    for m in ("raw", "tps"):
        ez = np.array([err(zs[p], m) for p in test])
        ef = np.array([err(ft[p], m) for p in test])
        ftb[m] = {"n": len(test), "zs_k10": int((ez < 10).sum()), "ft_k10": int((ef < 10).sum()),
                  "zs_k20": int((ez < 20).sum()), "ft_k20": int((ef < 20).sum()),
                  "zs_sr20": r3((ez < 20).mean()), "ft_sr20": r3((ef < 20).mean()),
                  "zs_median": round(float(np.median(ez)), 1), "ft_median": round(float(np.median(ef)), 1)}
    tem = [p for p in test if "TEM" in p]
    c103 = [p for p in test if "C103" in p]
    ftb["tem_test"] = {"n": len(tem), "zs_median": round(float(np.median([err(zs[p], "raw") for p in tem])), 1),
                       "ft_median": round(float(np.median([err(ft[p], "raw") for p in tem])), 1)}
    ftb["c103_test"] = {p.replace("eval_", ""): [round(err(zs[p], "raw"), 1), round(err(ft[p], "raw"), 1)]
                        for p in c103}
    train_subclasses = {base["roma"][p]["subclass"] for p in split["train"]}
    ftb["c103_in_train"] = any("C103" in s for s in train_subclasses)
    ftb["train_pairs"] = len(split["train"])
    ftb["train_subclasses"] = len(train_subclasses)
    runs = load("seed_expand_summary.csv")

    def runs_of(cfg: str, mode: str, col: str) -> list[float]:
        return [float(r[col]) for r in runs if r["config"] == cfg and r["mode"] == mode]

    ftb["runs_4to8"] = {f"{cfg}_{mode}_{col}": [min(runs_of(cfg, mode, col)), max(runs_of(cfg, mode, col)),
                                               round(float(np.mean(runs_of(cfg, mode, col))), 3)]
                        for cfg in ("plain", "l2sp") for mode in ("direct", "pyramid_v2")
                        for col in ("sr10", "sr20", "med_ed")}
    num["finetune"] = ftb

    OUT.write_text(json.dumps(num, indent=2, default=float), encoding="utf-8")
    print(json.dumps(num, indent=1, default=float))


if __name__ == "__main__":
    main()
