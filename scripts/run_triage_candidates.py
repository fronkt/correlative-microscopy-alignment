"""Generate candidate registrations for the label-free triage study (CJSJ, tasks/todo.md).

One row per (pair, backbone, mode, input transform, seed): the fitted 3x3 transform (target -> source),
correspondence and inlier counts, and the unrefined error against the ground-truth points. Rows are
appended as they finish, so an interrupted run resumes by skipping completed keys.

Candidate pools (see the pre-registered design):
  core      sift, loftr, roma, ma_roma, matchanything (direct); roma, ma_roma (pyramid_v2)
  transform roma, ma_roma direct x {invert, histmatch, clahe, gradmag}
  control   roma, ma_roma direct, seeds 1-5 (plain reruns; never vote)
  gt        homography / affine fitted to the ground-truth points themselves (feasibility ceiling)

Usage:
  python scripts/run_triage_candidates.py --pools gt,core --backbones sift --limit 2 --device cpu   # smoke
  python scripts/run_triage_candidates.py --pools gt,core,transform,control                        # full
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_baselines_A import RANSAC_PX, make_matcher  # noqa: E402

from cma.data import AmalgaMatchLoader  # noqa: E402
from cma.estimators import fit_transform  # noqa: E402
from cma.metrics import registration_metrics  # noqa: E402
from cma.triage import apply_input_transform, pair_seed, project, seed_everything  # noqa: E402

FIELDS = [
    "pair_id", "scene", "group", "subclass", "pool", "backbone", "mode", "transform", "seed",
    "h_s", "w_s", "h_t", "w_t", "n_gt", "n_matches", "n_inliers", "family", "H",
    "mu_ed", "med_ed", "runtime_s", "status", "error",
]
TRANSFORMS = ("invert", "histmatch", "clahe", "gradmag")


def build_configs(pools: set[str]) -> list[dict]:
    cfgs: list[dict] = []
    if "gt" in pools:
        cfgs += [dict(pool="gt", backbone="gt", mode=m, transform="none", seed=0)
                 for m in ("homography", "affine")]
    if "core" in pools:
        cfgs += [dict(pool="core", backbone=b, mode="direct", transform="none", seed=0)
                 for b in ("sift", "loftr", "roma", "ma_roma", "matchanything")]
        cfgs += [dict(pool="core", backbone=b, mode="pyramid_v2", transform="none", seed=0)
                 for b in ("roma", "ma_roma")]
    if "transform" in pools:
        cfgs += [dict(pool="transform", backbone=b, mode="direct", transform=t, seed=0)
                 for b in ("roma", "ma_roma") for t in TRANSFORMS]
    if "control" in pools:
        cfgs += [dict(pool="control", backbone=b, mode="direct", transform="none", seed=s)
                 for b in ("roma", "ma_roma") for s in range(1, 6)]
    return cfgs


def key(d: dict) -> tuple[str, str, str, str, str]:
    return (d["pair_id"], d["backbone"], d["mode"], d["transform"], str(d["seed"]))


def gt_fit(pair, mode: str) -> np.ndarray:
    """Least-squares transform fitted to ALL ground-truth points (no RANSAC): the best that
    a single global transform of this family can do on the pair."""
    import cv2
    tgt = pair.gt.tgt_xy.astype(np.float64)
    src = pair.gt.src_xy.astype(np.float64)
    if mode == "homography":
        H, _ = cv2.findHomography(tgt, src, method=0)
        if H is None:
            raise RuntimeError("homography fit failed")
        return H
    A = np.hstack([tgt, np.ones((len(tgt), 1))])
    X, *_ = np.linalg.lstsq(A, src, rcond=None)  # (3, 2)
    return np.vstack([X.T, [0.0, 0.0, 1.0]])


def run_one(pair, rec, cfg: dict, matcher) -> dict:
    h_s, w_s = pair.source.shape[:2]
    h_t, w_t = pair.target.shape[:2]
    row = dict(pair_id=rec.pair_id, scene=rec.pair_id.split("#")[0], group=rec.group,
               subclass=rec.subclass, pool=cfg["pool"], backbone=cfg["backbone"], mode=cfg["mode"],
               transform=cfg["transform"], seed=cfg["seed"], h_s=h_s, w_s=w_s, h_t=h_t, w_t=w_t,
               n_gt=len(pair.gt), status="ok", error="")
    seed_everything(pair_seed(int(cfg["seed"]), rec.pair_id))
    t0 = time.perf_counter()
    if cfg["pool"] == "gt":
        H = gt_fit(pair, cfg["mode"])
        row.update(n_matches=len(pair.gt), n_inliers=len(pair.gt), family=cfg["mode"])
    else:
        src_img, tgt_img = apply_input_transform(cfg["transform"], pair.source, pair.target)
        if cfg["mode"] == "pyramid_v2":
            from cma.pipeline import register_v2
            res = register_v2(src_img, tgt_img, matcher, pair.scale_ratio,
                              ransac_threshold_px=RANSAC_PX, certainty_threshold=None)
            H = res.H_target_to_source
            row.update(n_matches=res.n_correspondences, n_inliers=res.transform.n_inliers,
                       family=f"{res.transform.family}@{res.stage}")
        else:
            corr = matcher.match(src_img, tgt_img)
            row["n_matches"] = len(corr)
            if len(corr) < 4:
                raise RuntimeError(f"only {len(corr)} correspondences")
            est = fit_transform(src_xy=corr.b_xy, dst_xy=corr.a_xy, family="auto",
                                ransac_threshold_px=RANSAC_PX)
            H = est.as_3x3()
            row.update(n_inliers=est.n_inliers, family=est.family)
    row["runtime_s"] = f"{time.perf_counter() - t0:.2f}"
    m = registration_metrics(project(H, pair.gt.tgt_xy), pair.gt.src_xy)
    row.update(H=json.dumps([float(v) for v in np.asarray(H, dtype=np.float64).ravel()]),
               mu_ed=f"{m.mu_err:.3f}", med_ed=f"{m.med_err:.3f}")
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/AmalgaMatch")
    ap.add_argument("--out", default="results/triage/candidates.csv")
    ap.add_argument("--pools", default="gt,core,transform,control")
    ap.add_argument("--backbones", default="", help="optional comma filter, e.g. sift,roma")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="stop after N pairs (smoke)")
    args = ap.parse_args()

    cfgs = build_configs(set(args.pools.split(",")))
    if args.backbones:
        keep = set(args.backbones.split(",")) | {"gt"}
        cfgs = [c for c in cfgs if c["backbone"] in keep]
    by_backbone: dict[str, list[dict]] = {}
    for c in cfgs:
        by_backbone.setdefault(c["backbone"], []).append(c)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple] = set()
    if out.exists():
        with out.open(newline="", encoding="utf-8") as f:
            done = {key(r) for r in csv.DictReader(f)}
    write_header = not out.exists()
    loader = AmalgaMatchLoader(args.root)

    with out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        for backbone, bcfgs in by_backbone.items():
            matcher = None if backbone == "gt" else make_matcher(backbone, args.device)
            for n, (pair, rec) in enumerate(loader.iter()):
                if args.limit and n >= args.limit:
                    break
                for cfg in bcfgs:
                    probe = dict(pair_id=rec.pair_id, **{k: cfg[k] for k in ("backbone", "mode", "transform", "seed")})
                    if key(probe) in done:
                        continue
                    try:
                        row = run_one(pair, rec, cfg, matcher)
                    except Exception as e:  # noqa: BLE001 — a failed run is a data point, keep sweeping
                        traceback.print_exc()
                        row = {k: "" for k in FIELDS}
                        row.update(pair_id=rec.pair_id, scene=rec.pair_id.split("#")[0],
                                   group=rec.group, subclass=rec.subclass, pool=cfg["pool"],
                                   backbone=cfg["backbone"], mode=cfg["mode"],
                                   transform=cfg["transform"], seed=cfg["seed"],
                                   n_gt=len(pair.gt), status="failed", error=str(e)[:200])
                    writer.writerow(row)
                    f.flush()
                    print(f"[{backbone}|{cfg['mode']}|{cfg['transform']}|s{cfg['seed']}] "
                          f"{rec.pair_id}: {row['status']} mu_ed={row.get('mu_ed', '')} "
                          f"inl={row.get('n_inliers', '')}/{row.get('n_matches', '')} "
                          f"({row.get('runtime_s', '')}s)", flush=True)
            del matcher
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass


if __name__ == "__main__":
    main()
