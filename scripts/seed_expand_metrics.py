"""Append SR@{5,10,20} + median ED for one fine-tune seed/config/mode to a summary CSV.

Reads an eval CSV produced by run_baselines_A.py restricted to the test split
(--only-pairs results/split.json --only-pairs-key test) and computes the
paper-protocol error per pair (mu_ed_tps with mu_ed fallback; inf when
status != ok), then appends a single summary row. Idempotent at the caller:
box_seed_expand.sh skips a (seed, config, mode) already present in the summary.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def ed(r: dict) -> float:
    if r.get("status") != "ok":
        return float("inf")
    v = r.get("mu_ed_tps") or r.get("mu_ed")
    try:
        return float(v) if v not in (None, "") else float("inf")
    except ValueError:
        return float("inf")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-csv", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", required=True, help="plain | l2sp")
    ap.add_argument("--lam", default="", help="anchor lambda as a string label")
    ap.add_argument("--mode", required=True, help="direct | pyramid_v2")
    ap.add_argument("--backbone", default="ma_roma_ft")
    ap.add_argument("--summary", default="results/seed_expand_summary.csv")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.eval_csv, newline="", encoding="utf-8")))
    eds = [ed(r) for r in rows
           if r.get("backbone") == a.backbone and r.get("mode") == a.mode]
    arr = np.asarray(eds, dtype=float)
    n = int(arr.size)
    fin = arr[np.isfinite(arr)]
    sr5 = float(np.mean(arr < 5)) if n else float("nan")
    sr10 = float(np.mean(arr < 10)) if n else float("nan")
    sr20 = float(np.mean(arr < 20)) if n else float("nan")
    med = float(np.median(fin)) if fin.size else float("inf")

    out = Path(a.summary)
    write_header = not out.exists()
    with open(out, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["seed", "config", "lambda", "mode", "n",
                        "sr5", "sr10", "sr20", "med_ed"])
        w.writerow([a.seed, a.config, a.lam, a.mode, n,
                    f"{sr5:.3f}", f"{sr10:.3f}", f"{sr20:.3f}", f"{med:.1f}"])

    print(f"[metrics] seed={a.seed} {a.config} {a.mode}: "
          f"n={n} SR@10={sr10:.3f} SR@20={sr20:.3f} medED={med:.1f}")


if __name__ == "__main__":
    main()
