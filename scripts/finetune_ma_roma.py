"""Fine-tune MA-RoMa (decoder only) on the AmalgaMatch train split.

Usage:
  python scripts/finetune_ma_roma.py                          # full run
  python scripts/finetune_ma_roma.py --steps 2 --device cpu \
      --batch-size 1 --val-every 0 --num-workers 0            # smoke

Headline eval afterwards (test split analysis only):
  python scripts/run_baselines_A.py --backbones ma_roma_ft \
      --ft-weights checkpoints/ma_roma_ft.pth
"""

from __future__ import annotations

import argparse

from cma.train.finetune import finetune


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/AmalgaMatch")
    ap.add_argument("--split", default="results/split.json")
    ap.add_argument("--out-dir", default="checkpoints")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--val-every", type=int, default=100,
                    help="0 disables validation (smoke runs)")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", default="results/finetune_log.csv")
    args = ap.parse_args()

    finetune(
        root=args.root, split_path=args.split, out_dir=args.out_dir,
        steps=args.steps, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay, val_every=args.val_every,
        num_workers=args.num_workers, device=args.device, seed=args.seed,
        log_path=args.log,
    )


if __name__ == "__main__":
    main()
