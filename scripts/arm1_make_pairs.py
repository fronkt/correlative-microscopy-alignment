"""Write the fixed Arm 1 pair lists to results/arm1/ and print the run count + GPU-time estimate."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "arm1"
SEED = 20260929


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pp = pd.read_csv(ROOT / "results/phaseA/a6_per_pair.csv", keep_default_na=False)
    ov = sorted(pp[pp.SHARED_OVERLAY.astype(str) == "True"].pair_id)
    ti = sorted(pp[pp.subclass.str.contains("Ti3AlC2")].pair_id)
    pool = sorted(pp[(pp.SHARED_OVERLAY.astype(str) == "False") & ~pp.subclass.str.contains("Ti3AlC2")
                     & (pp.src_overlay.astype(str) == "False") & (pp.tgt_overlay.astype(str) == "False")].pair_id)
    # 'overlay-free' = neither image flagged (a pair with one flagged image is neither shared nor clean)
    rng = np.random.default_rng(SEED)
    sham = sorted(rng.choice(pool, size=20, replace=False).tolist())
    for name, ids in (("pairs_overlay", ov), ("pairs_control_ti3alc2", ti), ("pairs_sham", sham)):
        (OUT / f"{name}.txt").write_text("\n".join(ids) + "\n", encoding="utf-8")
    print(f"overlay {len(ov)}, ti3alc2 {len(ti)}, sham {len(sham)} (drawn from {len(pool)} overlay-free pairs)")

    d = pd.read_csv(ROOT / "results/triage/candidates.csv", keep_default_na=False)
    d = d[(d.pool == "core") & d.backbone.isin(["roma", "ma_roma"]) & d["mode"].isin(["direct", "pyramid_v2"])
          & (d.seed.astype(int) == 0) & (d["transform"] == "none")].copy()
    d["rt"] = pd.to_numeric(d.runtime_s, errors="coerce")
    n_runs = 0
    total = 0.0
    print("runs (pair-sets x crops x 4 configs) and estimated GPU seconds from stored runtime_s:")
    for label, ids, ncrop in (("overlay", ov, 2), ("ti3alc2", ti, 1), ("sham", sham, 2)):
        sub = d[d.pair_id.isin(ids)]
        per_cfg = sub.groupby(["backbone", "mode"]).rt.agg(["count", "sum", "mean"])
        sec = float(sub.rt.sum()) * ncrop
        n = len(ids) * 4 * ncrop
        n_runs += n
        total += sec
        print(f"  {label}: {len(ids)} pairs x {ncrop} crop(s) x 4 cfg = {n} runs, ~{sec / 3600:.2f} h (stored rows {len(sub)}, "
              f"NaN runtimes {int(sub.rt.isna().sum())})")
        print(per_cfg.round(1).to_string())
    print(f"TOTAL runs {n_runs}, ~{total / 3600:.2f} GPU-h of matching time (excl. model load; candidates.csv hardware)")


if __name__ == "__main__":
    main()
