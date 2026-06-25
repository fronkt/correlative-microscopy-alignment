"""Scale-axis pivot on the REAL AmalgaMatch FOV ladder.

The synthetic ladder saturates (RoMa solves same-modality natural warps
perfectly — ledger insight #1), so the make-or-break comparison must run on
real base-matchable pairs cropped to shrinking FOV, where RoMa genuinely fails
at low FOV (ledger: pyramid 0.23 vs direct 0.07 @ FOV 0.1).

Reuses the tested baselines infra (make_matcher / run_pair / eligible_pairs /
crop_target_to_area_ratio) unchanged. TTA is injected by adapting the matcher's
model in place, running the same pyramid_v2 path, then resetting:

  direct        — frozen matcher, no pyramid
  pyramid_v2    — frozen matcher + coarse-to-fine (Pivot-S)
  tta_scale     — per-pair multi-scale-consistency TTA + pyramid_v2

Writes results/scale_pivot_real.csv and prints SR@10/SR@20/med_ed per rung.
Usage: python scripts/run_scale_pivot_real.py --backbone ma_roma --limit 6
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run_baselines_A import make_matcher, run_pair  # noqa: E402
from run_fov_ladder import base_ratios, eligible_pairs  # noqa: E402

from cma.data.amalgamatch import AmalgaMatchLoader  # noqa: E402
from cma.data.fov_ladder import crop_target_to_area_ratio  # noqa: E402
from cma.tta import tta_adapt  # noqa: E402

RUNGS = (0.25, 0.10, 0.05)
METHODS = ("direct", "pyramid_v2", "tta_scale")


def _err(row: dict) -> float:
    if row.get("status") != "ok":
        return float("inf")
    v = row.get("mu_ed_tps") or row.get("mu_ed")
    return float(v) if v else float("inf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/AmalgaMatch")
    ap.add_argument("--baselines", default="results/baselines_A.csv")
    ap.add_argument("--fov-ratios", default="results/fov_ratios.csv")
    ap.add_argument("--out", default="results/scale_pivot_real.csv")
    ap.add_argument("--backbone", default="ma_roma")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--limit", type=int, default=6)
    args = ap.parse_args()

    chosen = eligible_pairs(Path(args.baselines))
    ratios = base_ratios(Path(args.fov_ratios))
    loader = AmalgaMatchLoader(args.root)
    recs = [r for r in loader.records if r.pair_id in chosen][: args.limit]
    print(f"eligible pairs (capped): {len(recs)}", flush=True)

    matcher = make_matcher(args.backbone, args.device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "rung", "method", "status", "err_ed", "runtime_s"])
        for rec in recs:
            pair = loader.load_pair(rec)
            base = ratios.get(rec.pair_id)
            if base is None:
                continue
            for rung in RUNGS:
                cut = crop_target_to_area_ratio(pair, base, rung)
                if cut is None:
                    continue
                for method in METHODS:
                    if method == "tta_scale":
                        _, _h, reset = tta_adapt(
                            matcher._model, cut.pair.source, cut.pair.target,
                            w_scale=1.0, w_appearance=0.0, steps=args.steps,
                            device=args.device)
                        try:
                            r = run_pair(cut.pair, rec, matcher, "pyramid_v2")
                        finally:
                            reset()
                    else:
                        r = run_pair(cut.pair, rec, matcher, method)
                    e = _err(r)
                    rows.append({"rung": rung, "method": method, "err": e})
                    w.writerow([rec.pair_id, f"{rung:g}", method,
                                r.get("status"), f"{e:.3f}", r.get("runtime_s", "")])
                    f.flush()
                    print(f"  {rec.pair_id} rung={rung:g} {method}: err={e:.2f}", flush=True)

    print("\n=== REAL SCALE-AXIS PIVOT (SR@10 / SR@20 | med_ed px) ===")
    print(f"{'rung':>6} | {'direct':>16} | {'pyramid_v2':>16} | {'tta_scale':>16}")
    for rung in RUNGS:
        cells = []
        for method in METHODS:
            errs = np.array([x["err"] for x in rows
                             if x["rung"] == rung and x["method"] == method])
            if len(errs) == 0:
                cells.append("—")
                continue
            cells.append(f"{np.mean(errs <= 10):.2f}/{np.mean(errs <= 20):.2f}|"
                         f"{np.median(errs):.0f}")
        print(f"{rung:>6} | {cells[0]:>16} | {cells[1]:>16} | {cells[2]:>16}")
    print(f"\nwrote {out}")
    print("VERDICT: tta_scale SR vs pyramid_v2 SR at rung 0.1 / 0.05 is the pivot")


if __name__ == "__main__":
    main()
