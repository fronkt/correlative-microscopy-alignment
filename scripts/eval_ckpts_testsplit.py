"""Cheap per-checkpoint test-split readout for λ selection (Phase 9A).

The val C103 probe does not move under the plain ft, so it cannot rank the
L2-SP λ values; the forgetting shows up on the *test* C103 pairs. This runs
direct-match registration over the 28 test pairs for each checkpoint and
prints SR@10/20, median ED, and the per-pair ED for the C103 (retention) and
TEM (gain) pairs. These are parametric-fit µ-ED (no TPS), so they are a
*selection* signal, not the headline — the winner's headline numbers come
from the full TPS-protocol Phase 9B sweep. Compare against zero-shot ma_roma
in results/baselines_A.csv (test SR@20 0.393; C103 _0/_1 ~12 px).

Usage:
  python scripts/eval_ckpts_testsplit.py --ckpts \
      l0=/dev/shm/cma_ckpt/l0/ma_roma_ft.pth,l0p01=/dev/shm/cma_ckpt/l0p01/ma_roma_ft.pth,\
l0p1=/dev/shm/cma_ckpt/l0p1/ma_roma_ft.pth,l1=/dev/shm/cma_ckpt/l1/ma_roma_ft.pth
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cma.train.finetune import build_model, evaluate_direct


def _pp(errs: list[float], pids: list[str], substr: str) -> str:
    items = [(p.rsplit("_", 1)[-1], e) for p, e in zip(pids, errs) if substr in p]
    return ", ".join(f"{k}={'inf' if not np.isfinite(e) else round(e, 1)}"
                     for k, e in sorted(items))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", required=True, help="comma-sep tag=path pairs")
    ap.add_argument("--root", default="data/AmalgaMatch")
    ap.add_argument("--split", default="results/split.json")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    test = json.loads(Path(args.split).read_text())["test"]
    print(f"test split: {len(test)} pairs (parametric µ-ED, no TPS)\n")
    print(f"{'tag':6s} {'SR@10':>6s} {'SR@20':>6s} {'medED':>8s}")
    for item in args.ckpts.split(","):
        tag, path = item.split("=", 1)
        model = build_model(dev, weights_path=path)
        model.eval()
        v = evaluate_direct(model, args.root, test, dev)
        errs, pids = v["errs"], v["pair_ids"]
        fin = np.asarray(errs)[np.isfinite(errs)]
        med = float(np.median(fin)) if len(fin) else float("inf")
        print(f"{tag:6s} {v['sr10']:6.3f} {v['sr20']:6.3f} {med:8.1f}")
        print(f"   C103(retention): {_pp(errs, pids, 'C103')}")
        print(f"   TEM (gain)     : {_pp(errs, pids, 'TEM')}")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
