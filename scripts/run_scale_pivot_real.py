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
  tta_scale     — per-pair multi-scale-consistency TTA + pyramid_v2 (raw)
  tta_guarded   — tta_scale kept only if its RANSAC inlier RATIO >= pyramid_v2's,
                  else reverts to pyramid_v2. Label-free do-no-harm guard =
                  sub-claim (iii) in-domain neutrality, made a mechanism. This
                  catches single-instance adaptation collapse by construction.

Gentler defaults than the first run (lr 1e-4, 3 steps, anchor 1.0) to reduce
collapse frequency; the guard is the safety net for what remains. Pairs are
sampled across subclasses, not the first-N of one subclass.

Writes results/scale_pivot_real.csv and prints SR@10/SR@20/med_ed per rung.
Usage: python scripts/run_scale_pivot_real.py --backbone ma_roma --limit 24
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run_baselines_A import make_matcher, run_pair  # noqa: E402
from run_fov_ladder import base_ratios, eligible_pairs  # noqa: E402

from cma.data.amalgamatch import AmalgaMatchLoader  # noqa: E402
from cma.data.fov_ladder import crop_target_to_area_ratio  # noqa: E402
from cma.tta import tta_adapt  # noqa: E402

RUNGS = (0.25, 0.10, 0.05)
METHODS = ("direct", "pyramid_v2", "tta_scale", "tta_guarded")


def _err(row: dict) -> float:
    if row.get("status") != "ok":
        return float("inf")
    v = row.get("mu_ed_tps") or row.get("mu_ed")
    return float(v) if v else float("inf")


def _inlier_ratio(row: dict) -> float:
    """RANSAC inlier ratio — the label-free match-consistency proxy."""
    try:
        nm = float(row.get("n_matches") or 0)
        ni = float(row.get("n_inliers") or 0)
        return ni / nm if nm > 0 else 0.0
    except (TypeError, ValueError):
        return 0.0


def _diverse_recs(recs: list, limit: int) -> list:
    """Round-robin across subclasses so the sample isn't one narrow slice."""
    by_sub: dict = defaultdict(list)
    for r in recs:
        by_sub[r.subclass].append(r)
    order = sorted(by_sub)
    out: list = []
    i = 0
    while len(out) < limit and any(by_sub[s] for s in order):
        s = order[i % len(order)]
        if by_sub[s]:
            out.append(by_sub[s].pop(0))
        i += 1
    return out[:limit]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/AmalgaMatch")
    ap.add_argument("--baselines", default="results/baselines_A.csv")
    ap.add_argument("--fov-ratios", default="results/fov_ratios.csv")
    ap.add_argument("--out", default="results/scale_pivot_real.csv")
    ap.add_argument("--backbone", default="ma_roma")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--anchor", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=24)
    args = ap.parse_args()

    chosen = eligible_pairs(Path(args.baselines))
    ratios = base_ratios(Path(args.fov_ratios))
    loader = AmalgaMatchLoader(args.root)
    recs = _diverse_recs([r for r in loader.records if r.pair_id in chosen],
                         args.limit)
    subs = sorted({r.subclass for r in recs})
    print(f"eligible pairs: {len(recs)} across {len(subs)} subclasses", flush=True)

    matcher = make_matcher(args.backbone, args.device)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "subclass", "rung", "method", "status",
                    "err_ed", "inlier_ratio", "runtime_s"])
        for rec in recs:
            pair = loader.load_pair(rec)
            base = ratios.get(rec.pair_id)
            if base is None:
                continue
            for rung in RUNGS:
                cut = crop_target_to_area_ratio(pair, base, rung)
                if cut is None:
                    continue
                # base (unadapted) pyramid_v2 — the guard's reference
                r_pyr = run_pair(cut.pair, rec, matcher, "pyramid_v2")
                # adapted pyramid_v2
                _, _h, reset = tta_adapt(
                    matcher._model, cut.pair.source, cut.pair.target,
                    w_scale=1.0, w_appearance=0.0, steps=args.steps,
                    lr=args.lr, anchor_lambda=args.anchor, device=args.device)
                try:
                    r_tta = run_pair(cut.pair, rec, matcher, "pyramid_v2")
                finally:
                    reset()
                r_dir = run_pair(cut.pair, rec, matcher, "direct")

                # do-no-harm guard: keep tta only if it didn't lose inlier ratio
                keep_tta = _inlier_ratio(r_tta) >= _inlier_ratio(r_pyr)
                r_guard = r_tta if keep_tta else r_pyr
                results = {"direct": r_dir, "pyramid_v2": r_pyr,
                           "tta_scale": r_tta, "tta_guarded": r_guard}
                for method in METHODS:
                    r = results[method]
                    e = _err(r)
                    rows.append({"rung": rung, "method": method, "err": e})
                    w.writerow([rec.pair_id, rec.subclass, f"{rung:g}", method,
                                r.get("status"), f"{e:.3f}",
                                f"{_inlier_ratio(r):.3f}", r.get("runtime_s", "")])
                    f.flush()
                guard_tag = "kept-tta" if keep_tta else "reverted"
                print(f"  {rec.pair_id} rung={rung:g}: dir={_err(r_dir):.0f} "
                      f"pyr={_err(r_pyr):.0f} tta={_err(r_tta):.0f} "
                      f"guard={_err(r_guard):.0f} [{guard_tag}]", flush=True)

    print("\n=== REAL SCALE-AXIS PIVOT (SR@10 / SR@20 | med_ed px) ===")
    hdr = " | ".join(f"{m:>16}" for m in METHODS)
    print(f"{'rung':>6} | {hdr}")
    for rung in RUNGS:
        cells = []
        for method in METHODS:
            errs = np.array([x["err"] for x in rows
                             if x["rung"] == rung and x["method"] == method])
            if len(errs) == 0:
                cells.append(f"{'—':>16}")
                continue
            cells.append(f"{np.mean(errs <= 10):.2f}/{np.mean(errs <= 20):.2f}|"
                         f"{np.median(errs):.0f}".rjust(16))
        print(f"{rung:>6} | {' | '.join(cells)}")
    print(f"\nwrote {out}  (per-pair guard kept/reverted logged above)")
    print("VERDICT: tta_guarded SR vs pyramid_v2 SR at rung 0.1 / 0.05 is the pivot")


if __name__ == "__main__":
    main()
