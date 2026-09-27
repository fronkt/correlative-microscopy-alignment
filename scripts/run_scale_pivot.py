"""Scale-axis pivot: does TTA beat pyramid-only under pure FOV shift?

The make-or-break comparison for the paper, on the synthetic FOV ladder (no
download, pure scale shift, appearance fixed). Same MA-RoMa backbone and same
seeded pairs for all three methods:

  vanilla_direct  — frozen matcher, no pyramid (the OOD-on-scale failure)
  pyramid_only    — frozen matcher + pyramid (Pivot-S: the bar TTA must clear)
  tta_scale       — per-pair multi-scale-consistency TTA + pyramid

Per-pair error = median reprojection error of GT grid; SR@k = fraction of pairs
with error <= k px. Writes results/scale_pivot.csv and prints an aggregate table.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch

from cma.eval.methods import matcher_method, tta_method
from cma.eval.sweep import SweepConfig, fov_sweep
from cma.matchers.roma import RoMaMatcher
from cma.train.finetune import build_model

FOV_RATIOS = (0.5, 0.25, 0.1, 0.05)
N_PAIRS = 5
TTA_STEPS = 6


def _aggregate(rows) -> dict:
    """{(method, fov): {sr10, sr20, med, n, runtime}} from SweepRow list."""
    out: dict = {}
    by_key: dict = {}
    for r in rows:
        err = r.med_err if r.success else float("inf")
        by_key.setdefault((r.backbone, r.fov_ratio), []).append((err, r.runtime_s))
    for key, vals in by_key.items():
        errs = np.array([v[0] for v in vals])
        rts = np.array([v[1] for v in vals])
        out[key] = {
            "sr10": float(np.mean(errs <= 10)),
            "sr20": float(np.mean(errs <= 20)),
            "med": float(np.median(errs)),
            "n": len(errs),
            "runtime": float(np.nanmean(rts)),
        }
    return out


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = SweepConfig(fov_ratios=FOV_RATIOS, n_pairs=N_PAIRS, source="natural",
                      family="auto")

    model = build_model(torch.device(dev))  # shared MA-RoMa; tta resets per pair
    matcher = RoMaMatcher(variant="ma_outdoor", device=dev, model=model)

    methods = {
        "vanilla_direct": matcher_method(matcher, cfg, use_pyramid=False),
        "pyramid_only": matcher_method(matcher, cfg, use_pyramid=True),
        "tta_scale": tta_method(lambda: model, cfg, variant="ma_outdoor",
                                w_scale=1.0, w_appearance=0.0, steps=TTA_STEPS,
                                use_pyramid=True, device=dev),
    }

    all_rows = []
    for name, method in methods.items():
        t0 = time.perf_counter()
        rows = fov_sweep(name, method, cfg)
        all_rows.extend(rows)
        print(f"[{name}] {len(rows)} rows in {time.perf_counter() - t0:.0f}s", flush=True)

    agg = _aggregate(all_rows)
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / "scale_pivot.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "fov_ratio", "sr10", "sr20", "med_err", "n", "runtime_s"])
        for (m, fov), v in sorted(agg.items()):
            w.writerow([m, fov, f"{v['sr10']:.3f}", f"{v['sr20']:.3f}",
                        f"{v['med']:.2f}", v["n"], f"{v['runtime']:.2f}"])

    print("\n=== SCALE-AXIS PIVOT (SR@10 / SR@20 | med_err px) ===")
    print(f"{'FOV':>6} | {'vanilla':>18} | {'pyramid_only':>18} | {'tta_scale':>18}")
    for fov in FOV_RATIOS:
        cells = []
        for m in ("vanilla_direct", "pyramid_only", "tta_scale"):
            v = agg.get((m, fov))
            cells.append(f"{v['sr10']:.2f}/{v['sr20']:.2f}|{v['med']:.0f}" if v else "—")
        print(f"{fov:>6} | {cells[0]:>18} | {cells[1]:>18} | {cells[2]:>18}")
    print(f"\nwrote {csv_path}")
    print("PIVOT_VERDICT: compare tta_scale vs pyramid_only SR at low FOV (0.1, 0.05)")


if __name__ == "__main__":
    main()
