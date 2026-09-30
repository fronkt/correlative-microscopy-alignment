"""Arm 1 runner: dense matchers with the burned-in overlay cropped away (pre-registered; see results/phaseA/synthesis.md).

Reuses scripts/run_triage_candidates.run_one unchanged (default behaviour of that script is untouched). For crop != none
the pair's images are cropped (cma.overlay), the GT keypoints are shifted into cropped coordinates (only matters for top
crops), run_one fits H on the cropped pair, and the reported H / mu_ed are then re-expressed in ORIGINAL coordinates
and evaluated at ALL GT points (none dropped; n_gt_in_band records how many fall inside the removed rows).

Plan (default --plan arm1): jobs = (pairs file, crop) in this order
  pairs_overlay x none, pairs_overlay x overlay, pairs_control_ti3alc2 x none, pairs_sham x none, pairs_sham x sham
Configurations per backbone: direct (+ pyramid_v2 for roma / ma_roma), seed 0, transform none.
Backbones run sequentially in ONE process (two RoMa-family models never share the GPU).
Resumable: key = (pair_id, backbone, mode, transform, seed, crop).

Smoke:  PYTHONPATH=src python scripts/run_arm1.py --backbones sift --device cpu --limit 2 --out results/arm1/smoke/sift.csv
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_triage_candidates as rtc  # noqa: E402
from run_baselines_A import make_matcher  # noqa: E402

from cma.data import AmalgaMatchLoader  # noqa: E402
from cma.data.types import KeypointSet  # noqa: E402
from cma.metrics import registration_metrics  # noqa: E402
from cma.overlay import (  # noqa: E402
    analyze,
    crop_overlay,
    overlay_rule,
    read_gray_u8,
    sham_rule,
    uncrop_homography,
)
from cma.triage import project  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXTRA = ["crop", "crop_side", "rows_src", "rows_tgt", "n_gt_in_band", "git_commit"]
FIELDS = rtc.FIELDS + EXTRA
PLAN_ARM1 = [("results/arm1/pairs_overlay.txt", "none"), ("results/arm1/pairs_overlay.txt", "overlay"),
             ("results/arm1/pairs_control_ti3alc2.txt", "none"),
             ("results/arm1/pairs_sham.txt", "none"), ("results/arm1/pairs_sham.txt", "sham")]


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def configs_for(backbone: str) -> list[dict]:
    modes = ["direct"] + (["pyramid_v2"] if backbone in ("roma", "ma_roma") else [])
    return [dict(pool="arm1", backbone=backbone, mode=m, transform="none", seed=0) for m in modes]


def key6(r: dict, crop: str | None = None) -> tuple:
    return (*rtc.key(r), crop if crop is not None else r["crop"])


def read_pairs(path: str) -> list[str]:
    return [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]


_ANALYSIS: dict[str, dict] = {}


def analysis(path) -> dict:
    k = str(path)
    if k not in _ANALYSIS:
        _ANALYSIS[k] = analyze(read_gray_u8(path))
    return _ANALYSIS[k]


def make_rule(crop: str, rec, pair):
    if crop == "none":
        return None
    a_s, a_t = analysis(rec.source_path), analysis(rec.target_path)
    if (a_s["h"], a_s["w"]) != tuple(pair.source.shape[:2]) or (a_t["h"], a_t["w"]) != tuple(pair.target.shape[:2]):
        raise RuntimeError("loader image shape differs from the detector's image shape")
    if crop == "overlay":
        rule = overlay_rule(a_s, a_t)
        if rule is None:
            raise RuntimeError("crop=overlay requested but no shared overlay detected")
        return rule
    return sham_rule(pair.source.shape[:2], pair.target.shape[:2])


def run_crop(pair, rec, cfg: dict, matcher, crop: str) -> dict:
    rule = make_rule(crop, rec, pair)
    if rule is None:
        row = rtc.run_one(pair, rec, cfg, matcher)
        row.update(crop="none", crop_side="", rows_src=0, rows_tgt=0, n_gt_in_band=0)
        return row
    src_c, tgt_c = crop_overlay(pair.source, pair.target, rule)
    (_, sdy), (_, tdy) = rule.offset_src, rule.offset_tgt
    gt_c = KeypointSet(src_xy=pair.gt.src_xy - np.array([0.0, sdy]), tgt_xy=pair.gt.tgt_xy - np.array([0.0, tdy]))
    row = rtc.run_one(dataclasses.replace(pair, source=src_c, target=tgt_c, gt=gt_c), rec, cfg, matcher)
    Hc = np.array(json.loads(row["H"])).reshape(3, 3)
    H = uncrop_homography(Hc, rule)
    m = registration_metrics(project(H, pair.gt.tgt_xy), pair.gt.src_xy)  # original coords, ALL GT points
    hs, ht = pair.source.shape[0], pair.target.shape[0]
    in_s = (pair.gt.src_xy[:, 1] >= hs - rule.rows_src) if rule.side == "bottom" else (pair.gt.src_xy[:, 1] < rule.rows_src)
    in_t = (pair.gt.tgt_xy[:, 1] >= ht - rule.rows_tgt) if rule.side == "bottom" else (pair.gt.tgt_xy[:, 1] < rule.rows_tgt)
    row.update(H=json.dumps([float(v) for v in H.ravel()]), mu_ed=f"{m.mu_err:.3f}", med_ed=f"{m.med_err:.3f}",
               crop=crop, crop_side=rule.side, rows_src=rule.rows_src, rows_tgt=rule.rows_tgt,
               n_gt_in_band=int((in_s | in_t).sum()))
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/AmalgaMatch")
    ap.add_argument("--out", default="results/arm1/arm1.csv")
    ap.add_argument("--backbones", default="roma,ma_roma")
    ap.add_argument("--pairs-file", default="", help="single job: pair_ids txt (requires --crop)")
    ap.add_argument("--crop", default="", choices=["", "none", "overlay", "sham"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="first N pairs of each job (smoke)")
    args = ap.parse_args()

    jobs = [(args.pairs_file, args.crop)] if args.pairs_file else PLAN_ARM1
    if args.pairs_file and not args.crop:
        ap.error("--pairs-file requires --crop")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple] = set()
    if out.exists():
        with out.open(newline="", encoding="utf-8") as f:
            done = {key6(r) for r in csv.DictReader(f)}
    write_header = not out.exists()
    commit = git_commit()
    loader = AmalgaMatchLoader(args.root)
    recs = {r.pair_id: r for r in loader.records}

    with out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        for backbone in args.backbones.split(","):
            matcher = make_matcher(backbone, args.device)
            cfgs = configs_for(backbone)
            for pairs_file, crop in jobs:
                pids = read_pairs(pairs_file)
                if args.limit:
                    pids = pids[: args.limit]
                for pid in pids:
                    todo = [c for c in cfgs if key6(dict(pair_id=pid, **{k: c[k] for k in ("backbone", "mode", "transform", "seed")}), crop) not in done]
                    if not todo:
                        continue
                    rec = recs[pid]
                    pair = loader.load_pair(rec)
                    for cfg in todo:
                        try:
                            row = run_crop(pair, rec, cfg, matcher, crop)
                        except Exception as e:  # noqa: BLE001 — a failed run is a data point
                            traceback.print_exc()
                            row = {k: "" for k in FIELDS}
                            row.update(pair_id=pid, scene=pid.split("#")[0], group=rec.group, subclass=rec.subclass,
                                       pool=cfg["pool"], backbone=cfg["backbone"], mode=cfg["mode"],
                                       transform=cfg["transform"], seed=cfg["seed"], n_gt=len(pair.gt),
                                       status="failed", error=str(e)[:200], crop=crop)
                        row["git_commit"] = commit
                        writer.writerow(row)
                        f.flush()
                        print(f"[{backbone}|{cfg['mode']}|crop={crop}] {pid}: {row['status']} mu_ed={row.get('mu_ed', '')} "
                              f"rows={row.get('rows_src', '')}/{row.get('rows_tgt', '')} ({row.get('runtime_s', '')}s)", flush=True)
            del matcher
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass
    print("run_arm1 done", time.strftime("%H:%M:%S"))


if __name__ == "__main__":
    main()
