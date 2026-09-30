"""Assemble the arm-2 materials pool: manifest.csv, GT-quality table, summary JSON.

Reads materials-bench/records/*.json (written by arm2_build_*.py), fits homography / affine to the GT points ONLY
(leave-one-out residuals) and writes:
  materials-bench/manifest.csv
  results/arm2/gt_quality.csv, results/arm2/manifest_summary.json
No matcher / registration algorithm is run on the images. The frozen overlay detector (src/cma/overlay.py) is run
only if it exists.
"""
from __future__ import annotations

import csv
import glob
import importlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from arm2_common import BENCH, RECORDS, WT  # noqa: E402

OUT = WT / "results" / "arm2"


def load_gt(path: str) -> tuple[np.ndarray, np.ndarray]:
    a = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return a[:, :2], a[:, 2:]


def fit_affine(tgt, src):
    A = np.hstack([tgt, np.ones((len(tgt), 1))])
    X, *_ = np.linalg.lstsq(A, src, rcond=None)
    return np.vstack([X.T, [0, 0, 1.0]])


def fit_h(tgt, src):
    H, _ = cv2.findHomography(tgt.astype(np.float64), src.astype(np.float64), method=0)
    return H


def proj(H, xy):
    p = np.hstack([xy, np.ones((len(xy), 1))]) @ H.T
    return p[:, :2] / p[:, 2:3]


def loo(fit, tgt, src, min_fit):
    n = len(tgt)
    if n - 1 < min_fit:
        return None
    r = []
    for i in range(n):
        m = np.ones(n, bool)
        m[i] = False
        try:
            H = fit(tgt[m], src[m])
            r.append(float(np.linalg.norm(proj(H, tgt[i:i + 1]) - src[i])))
        except Exception:  # noqa: BLE001
            r.append(float("nan"))
    return np.array(r)


def main() -> None:
    recs = [json.loads(Path(p).read_text(encoding="utf-8")) for p in sorted(glob.glob(str(RECORDS / "*.json")))]
    try:
        ov = importlib.import_module("cma.overlay")
        overlay_note = f"cma.overlay present: {[n for n in dir(ov) if not n.startswith('_')][:8]}"
    except Exception:  # noqa: BLE001
        ov = None
        overlay_note = "src/cma/overlay.py absent -> overlay detector NOT run (burned-in banners unchecked)"
    print(overlay_note)

    man_fields = ["pair_id", "component", "cluster", "group", "subclass", "source_path", "target_path", "gt_path",
                  "source_modality", "target_modality", "h_s", "w_s", "h_t", "w_t", "source_pixel_nm",
                  "target_pixel_nm", "flipped", "n_gt", "licence", "source_url", "sha256_source", "sha256_target",
                  "archive_members", "raw_sha256", "overlay_src", "overlay_tgt", "notes"]
    gq_fields = ["pair_id", "component", "cluster", "n_gt", "diag_src_px", "aff_fit_mean_px", "aff_loo_mean_px",
                 "aff_loo_max_px", "aff_loo_mean_pct_diag", "hom_fit_mean_px", "hom_loo_mean_px", "hom_loo_max_px",
                 "hom_loo_mean_pct_diag", "gt_scale_tgt_to_src", "flag_lt4pts", "flag_aff_loo_gt20px", "flags"]
    man_rows, gq_rows = [], []
    for r in recs:
        src, tgt = load_gt(r["gt_path"])[0], load_gt(r["gt_path"])[1]  # columns: src_x,src_y | tgt_x,tgt_y
        diag = float(np.hypot(r["h_s"], r["w_s"]))
        row = {k: r.get(k, "") for k in man_fields}
        for side, key in (("overlay_src", "source_path"), ("overlay_tgt", "target_path")):
            if ov is None:
                row[side] = "not_run"
            else:
                det = ov.detect_overlay(ov.read_gray_u8(r[key]))
                row[side] = "none" if det is None else f"{det[0]}:{det[1]}"
        row["archive_members"] = r["files"].get("archive_members", "")
        row["raw_sha256"] = json.dumps(r["files"].get("raw_sha256", {}))
        man_rows.append(row)
        g = dict(pair_id=r["pair_id"], component=r["component"], cluster=r["cluster"], n_gt=len(src),
                 diag_src_px=round(diag, 1))
        flags = []
        if len(src) >= 3:
            Ha = fit_affine(tgt, src)
            g["aff_fit_mean_px"] = round(float(np.linalg.norm(proj(Ha, tgt) - src, axis=1).mean()), 2)
            g["gt_scale_tgt_to_src"] = round(float(np.sqrt(abs(np.linalg.det(Ha[:2, :2])))), 4)
            la = loo(fit_affine, tgt, src, 3)
            if la is not None:
                g.update(aff_loo_mean_px=round(float(np.nanmean(la)), 2), aff_loo_max_px=round(float(np.nanmax(la)), 2),
                         aff_loo_mean_pct_diag=round(float(np.nanmean(la)) / diag * 100, 3))
        if len(src) >= 4:
            Hh = fit_h(tgt, src)
            g["hom_fit_mean_px"] = round(float(np.linalg.norm(proj(Hh, tgt) - src, axis=1).mean()), 2)
            lh = loo(fit_h, tgt, src, 4)
            if lh is not None:
                g.update(hom_loo_mean_px=round(float(np.nanmean(lh)), 2), hom_loo_max_px=round(float(np.nanmax(lh)), 2),
                         hom_loo_mean_pct_diag=round(float(np.nanmean(lh)) / diag * 100, 3))
        g["flag_lt4pts"] = int(len(src) < 4)
        if len(src) < 4:
            flags.append("lt4pts")
        a = g.get("aff_loo_mean_px")
        g["flag_aff_loo_gt20px"] = int(a is not None and a > 20)
        if a is None:
            flags.append("affine LOO undefined")
        elif a > 20:
            flags.append("aff_loo>20px")
        g["flags"] = ";".join(flags)
        gq_rows.append(g)

    OUT.mkdir(parents=True, exist_ok=True)
    with open(BENCH / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=man_fields)
        w.writeheader()
        w.writerows(man_rows)
    with open(OUT / "gt_quality.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=gq_fields)
        w.writeheader()
        w.writerows(gq_rows)
    by_comp = Counter(r["component"] for r in recs)
    clusters = defaultdict(set)
    for r in recs:
        clusters[r["component"]].add(r["cluster"])
    flagged = [g["pair_id"] for g in gq_rows if g["flags"]]
    overlay_hits = [dict(pair_id=r["pair_id"], src=r["overlay_src"], tgt=r["overlay_tgt"]) for r in man_rows
                    if r["overlay_src"] not in ("none", "not_run") or r["overlay_tgt"] not in ("none", "not_run")]
    summ = dict(n_pairs=len(recs), pairs_per_component=dict(by_comp),
                clusters_per_component={k: len(v) for k, v in clusters.items()}, n_clusters=len({r["cluster"] for r in recs}),
                licences=dict(Counter(r["licence"] for r in recs)), flagged_pairs=flagged,
                n_flag_aff_loo_gt20=sum(g["flag_aff_loo_gt20px"] for g in gq_rows),
                n_flag_lt4pts=sum(g["flag_lt4pts"] for g in gq_rows), overlay_detector=overlay_note, overlay_hits=overlay_hits,
                n_gt_points_total=int(sum(g["n_gt"] for g in gq_rows)))
    (OUT / "manifest_summary.json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print(json.dumps(summ, indent=1))
    for g in gq_rows:
        print(g["pair_id"], g["n_gt"], "aff_loo", g.get("aff_loo_mean_px"), "hom_loo", g.get("hom_loo_mean_px"),
              "pct", g.get("aff_loo_mean_pct_diag"), g["flags"])


if __name__ == "__main__":
    main()
