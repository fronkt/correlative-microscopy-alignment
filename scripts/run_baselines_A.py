"""Control A: zero-shot backbone baselines on the real AmalgaMatch release.

Per (pair, backbone): match source/target directly (no pyramid), robust-fit a
transform, project GT target points into source coords, report Euclidean
distances. Protocol follows the AmalgaMatch paper (Durmaz et al.): RANSAC
reprojection threshold 5.5 px, then thin-plate-spline refinement on inliers;
ED is reported for both the parametric fit and the TPS refinement. Success
rate at {5, 10, 20} px mean-ED is an aggregate over pairs (computed in
analysis, not here).

Writes one CSV row per (pair, backbone, mode), appending incrementally so an
interrupted run resumes by skipping completed rows.

Usage:
  python scripts/run_baselines_A.py --backbones sift --limit 2        # smoke
  python scripts/run_baselines_A.py --backbones sift,loftr,roma,matchanything
  python scripts/run_baselines_A.py --backbones roma --mode pyramid   # Phase 4
"""

from __future__ import annotations

import argparse
import csv
import time
import traceback
from pathlib import Path

import numpy as np

from cma.data import AmalgaMatchLoader
from cma.estimators import fit_transform
from cma.metrics import registration_metrics
from cma.pipeline import register

FIELDS = [
    "pair_id", "group", "subclass", "backbone", "mode", "fov_ratio", "scale_ratio",
    "n_gt", "n_matches", "n_inliers", "family", "runtime_s",
    "mu_ed", "med_ed", "p1", "p3", "p5", "p10",
    "mu_ed_tps", "med_ed_tps", "status", "error",
]
RANSAC_PX = 5.5  # paper protocol


def make_matcher(name: str, device: str):
    if name == "sift":
        from cma.matchers import SIFTMatcher
        return SIFTMatcher()
    if name == "loftr":
        from cma.matchers import LoFTRMatcher
        return LoFTRMatcher(device=device)
    if name == "roma":
        from cma.matchers import RoMaMatcher
        return RoMaMatcher(device=device)
    if name == "ma_roma":
        from cma.matchers import RoMaMatcher
        return RoMaMatcher(device=device, variant="ma_outdoor")
    if name == "matchanything":
        from cma.matchers import MatchAnythingMatcher
        return MatchAnythingMatcher(device=device)
    if name == "matchanything_stretch":
        from cma.matchers import MatchAnythingMatcher
        m = MatchAnythingMatcher(device=device, resize_mode="stretch")
        m.name = "matchanything_stretch"  # distinct rows in the results CSV
        return m
    raise ValueError(f"unknown backbone: {name}")


def project(H: np.ndarray, xy: np.ndarray) -> np.ndarray:
    h = np.hstack([xy, np.ones((len(xy), 1))])
    p = (H @ h.T).T
    return p[:, :2] / p[:, 2:3]


def tps_project(
    fit_tgt: np.ndarray, fit_src: np.ndarray, query_tgt: np.ndarray,
    max_pts: int = 2000, smoothing: float = 1.0,
) -> np.ndarray:
    from scipy.interpolate import RBFInterpolator
    if len(fit_tgt) > max_pts:
        idx = np.random.default_rng(0).choice(len(fit_tgt), max_pts, replace=False)
        fit_tgt, fit_src = fit_tgt[idx], fit_src[idx]
    rbf = RBFInterpolator(fit_tgt, fit_src, kernel="thin_plate_spline", smoothing=smoothing)
    return rbf(query_tgt)


def run_pair(pair, rec, matcher, mode: str, certainty: float | None = None) -> dict:
    row: dict = {
        "pair_id": rec.pair_id, "group": rec.group, "subclass": rec.subclass,
        "backbone": matcher.name, "mode": mode, "scale_ratio": f"{pair.scale_ratio:.5f}",
        "n_gt": len(pair.gt), "status": "ok", "error": "",
    }
    h_s, w_s = pair.source.shape[:2]
    h_t, w_t = pair.target.shape[:2]
    row["fov_ratio"] = f"{(w_t * rec.target_pixel_nm) / (w_s * rec.source_pixel_nm):.4f}"

    t0 = time.perf_counter()
    if mode == "classical":
        from cma.pipeline import classical_register
        result = classical_register(pair.source, pair.target, matcher,
                                    ransac_threshold_px=RANSAC_PX, refine_with_mi=True)
        H = result.H_target_to_source
        inl_src = inl_tgt = None  # correspondence arrays not exposed; MMI is the refiner
        row["n_matches"] = result.n_correspondences
        row["n_inliers"] = result.transform.n_inliers
        row["family"] = result.transform.family + ("+mmi" if result.refined else "")
    elif mode.startswith("pyramid_v2"):
        from cma.pipeline import register_v2
        result = register_v2(pair.source, pair.target, matcher, pair.scale_ratio,
                             ransac_threshold_px=RANSAC_PX,
                             certainty_threshold=certainty)
        H = result.H_target_to_source
        mask = result.transform.inliers
        inl_src = result.src_xy[mask]
        inl_tgt = result.tgt_xy[mask]
        row["n_matches"] = result.n_correspondences
        row["n_inliers"] = result.transform.n_inliers
        row["family"] = f"{result.transform.family}@{result.stage}"
    elif mode == "pyramid":
        result = register(pair.source, pair.target, matcher, pair.scale_ratio,
                          ransac_threshold_px=RANSAC_PX)
        H = result.H_target_to_source
        mask = result.transform.inliers
        inl_src = result.src_xy[mask]
        inl_tgt = result.tgt_xy[mask]
        row["n_matches"] = result.n_correspondences
        row["n_inliers"] = result.transform.n_inliers
        row["family"] = result.transform.family
    else:
        corr = matcher.match(pair.source, pair.target)
        row["n_matches"] = len(corr)
        if len(corr) < 4:
            raise RuntimeError(f"only {len(corr)} correspondences")
        est = fit_transform(src_xy=corr.b_xy, dst_xy=corr.a_xy,
                            family="auto", ransac_threshold_px=RANSAC_PX)
        H = est.as_3x3()
        row["n_inliers"] = est.n_inliers
        row["family"] = est.family
        inl_tgt = corr.b_xy[est.inliers]
        inl_src = corr.a_xy[est.inliers]
    row["runtime_s"] = f"{time.perf_counter() - t0:.2f}"

    proj = project(H, pair.gt.tgt_xy)
    m = registration_metrics(proj, pair.gt.src_xy)
    row.update(mu_ed=f"{m.mu_err:.3f}", med_ed=f"{m.med_err:.3f}",
               p1=f"{m.p_match_at_1:.3f}", p3=f"{m.p_match_at_3:.3f}",
               p5=f"{m.p_match_at_5:.3f}", p10=f"{m.p_match_at_10:.3f}")

    row["mu_ed_tps"] = row["med_ed_tps"] = ""
    if inl_tgt is not None and len(inl_tgt) >= 10:
        try:
            proj_tps = tps_project(inl_tgt, inl_src, pair.gt.tgt_xy)
            ed = np.linalg.norm(proj_tps - pair.gt.src_xy, axis=1)
            row["mu_ed_tps"] = f"{float(ed.mean()):.3f}"
            row["med_ed_tps"] = f"{float(np.median(ed)):.3f}"
        except Exception as e:  # noqa: BLE001 — TPS failure must not kill the row
            row["error"] = f"tps: {e}"
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/AmalgaMatch")
    ap.add_argument("--out", default="results/baselines_A.csv")
    ap.add_argument("--backbones", default="sift")
    ap.add_argument("--mode", default="direct",
                    choices=["direct", "pyramid", "pyramid_v2", "classical"])
    ap.add_argument("--tag", default="", help="suffix for the mode column, e.g. c50 -> pyramid_v2+c50 (new resume key)")
    ap.add_argument("--certainty", type=float, default=None,
                    help="certainty gate threshold for pyramid_v2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="stop after N pairs per backbone")
    ap.add_argument("--subclasses", default="", help="comma-separated filter")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, str, str]] = set()
    if out.exists():
        with out.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done.add((r["pair_id"], r["backbone"], r["mode"]))
    write_header = not out.exists()

    loader = AmalgaMatchLoader(args.root)
    subclasses = [s for s in args.subclasses.split(",") if s] or None
    mode = args.mode + (f"+{args.tag}" if args.tag else "")

    with out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        for backbone in args.backbones.split(","):
            matcher = make_matcher(backbone, args.device)
            n_done = 0
            for pair, rec in loader.iter(subclasses=subclasses):
                if (rec.pair_id, backbone, mode) in done:
                    continue
                try:
                    row = run_pair(pair, rec, matcher, mode, certainty=args.certainty)
                except Exception as e:  # noqa: BLE001 — log per-pair failure, keep sweeping
                    row = {k: "" for k in FIELDS}
                    row.update(pair_id=rec.pair_id, group=rec.group, subclass=rec.subclass,
                               backbone=backbone, mode=mode, n_gt=len(loader._gt[rec.pair_id]),
                               status="failed", error=str(e)[:200])
                    traceback.print_exc()
                writer.writerow(row)
                f.flush()
                print(f"[{backbone}] {rec.pair_id}: {row['status']} "
                      f"mu_ed={row.get('mu_ed', '')} tps={row.get('mu_ed_tps', '')} "
                      f"({row.get('runtime_s', '')}s)")
                n_done += 1
                if args.limit and n_done >= args.limit:
                    break
            del matcher
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass


if __name__ == "__main__":
    main()
