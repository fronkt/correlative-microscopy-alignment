"""FOV ladder (Aim 3): per-backbone failure FOV on real cross-modal pairs.

Eligible pairs are those some direct backbone already registers at base
FOV (mean ED < 20 px in results/baselines_A.csv) — failure FOV is only
defined for pairs that are matchable at all. Each eligible pair's target
is cropped to a ladder of absolute area ratios; appearance, modality gap
and pixel sizes stay fixed while FOV shrinks.

All GT is kept for evaluation (out-of-crop points test the global
transform's extrapolation — see src/cma/data/fov_ladder.py). Analyses of
this CSV must use the transform-based mu_ed, NOT mu_ed_tps: TPS is fit on
in-crop inliers and extrapolates unreliably.

Writes one row per (pair, rung, backbone, mode) to results/fov_ladder.csv,
appending incrementally with resume (same convention as run_baselines_A).

Usage:
  python scripts/run_fov_ladder.py --backbones roma,ma_roma --modes direct,pyramid_v2
  python scripts/run_fov_ladder.py --backbones sift --limit 2   # smoke
"""

from __future__ import annotations

import argparse
import csv
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_baselines_A import FIELDS, make_matcher, run_pair  # noqa: E402

from cma.data.amalgamatch import AmalgaMatchLoader  # noqa: E402
from cma.data.fov_ladder import crop_target_to_area_ratio  # noqa: E402

RUNGS = (0.5, 0.25, 0.10, 0.05, 0.02)
LADDER_FIELDS = [*FIELDS, "rung", "area_ratio", "n_gt_inside"]


def eligible_pairs(baselines_csv: Path, max_ed: float = 20.0) -> set[str]:
    with baselines_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ok: set[str] = set()
    for r in rows:
        if r["mode"] != "direct" or r["status"] != "ok":
            continue
        v = r["mu_ed_tps"] or r["mu_ed"]
        if v and float(v) < max_ed:
            ok.add(r["pair_id"])
    return ok


def base_ratios(fov_csv: Path) -> dict[str, float]:
    with fov_csv.open(newline="", encoding="utf-8") as f:
        return {r["pair_id"]: float(r["fov_area_ratio"]) for r in csv.DictReader(f)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/AmalgaMatch")
    ap.add_argument("--baselines", default="results/baselines_A.csv")
    ap.add_argument("--fov-ratios", default="results/fov_ratios.csv")
    ap.add_argument("--out", default="results/fov_ladder.csv")
    ap.add_argument("--backbones", default="roma,ma_roma")
    ap.add_argument("--modes", default="direct,pyramid_v2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="max eligible pairs (smoke)")
    args = ap.parse_args()

    chosen = eligible_pairs(Path(args.baselines))
    ratios = base_ratios(Path(args.fov_ratios))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    done: set[tuple[str, str, str, str]] = set()
    if out.exists():
        with out.open(newline="", encoding="utf-8") as f:
            done = {(r["pair_id"], r["backbone"], r["mode"], r["rung"])
                    for r in csv.DictReader(f)}

    backbones = [b for b in args.backbones.split(",") if b]
    modes = [m for m in args.modes.split(",") if m]
    loader = AmalgaMatchLoader(args.root)

    recs = [r for r in loader.records if r.pair_id in chosen]
    if args.limit:
        recs = recs[: args.limit]
    print(f"eligible pairs: {len(recs)}")

    with out.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LADDER_FIELDS)
        if not done:
            writer.writeheader()
        for bb in backbones:
            matcher = None
            for rec in recs:
                pending = [(rung, mode) for rung in RUNGS for mode in modes
                           if (rec.pair_id, bb, mode, f"{rung:g}") not in done]
                if not pending:
                    continue
                pair = loader.load_pair(rec)
                base = ratios[rec.pair_id]
                cuts = {rung: crop_target_to_area_ratio(pair, base, rung)
                        for rung in RUNGS}
                for rung, mode in pending:
                    cut = cuts[rung]
                    base_row = {"pair_id": rec.pair_id, "group": rec.group,
                                "subclass": rec.subclass, "backbone": bb,
                                "mode": mode, "rung": f"{rung:g}"}
                    if cut is None:
                        writer.writerow({**base_row, "status": "skipped",
                                         "error": "rung_not_below_base"})
                        f.flush()
                        continue
                    if matcher is None:
                        matcher = make_matcher(bb, args.device)
                    try:
                        row = run_pair(cut.pair, rec, matcher, mode)
                    except Exception as e:  # noqa: BLE001 — record, keep sweeping
                        row = {**base_row, "status": "error",
                               "error": f"{type(e).__name__}: {e}"[:200]}
                        traceback.print_exc()
                    row.update(base_row,
                               area_ratio=f"{cut.area_ratio:.4f}",
                               n_gt_inside=cut.n_gt_inside)
                    writer.writerow(row)
                    f.flush()
                    print(f"[{bb}/{mode}] {rec.pair_id} rung={rung:g}: "
                          f"{row.get('status')} mu_ed={row.get('mu_ed', '-')} "
                          f"({row.get('runtime_s', '-')}s)")
    print("FOV LADDER DONE")


if __name__ == "__main__":
    main()
