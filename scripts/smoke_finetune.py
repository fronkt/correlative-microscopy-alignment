"""Smoke test for the MA-RoMa fine-tuning build. CPU, batch 1, 2 steps.

Checks, in order:
1. Dataset sample sanity — saves results/smoke_warp.png: the A-grid points
   (left) and where the densified GT warp sends them in B (right), colored
   by position so corresponding structure should line up visually.
2. Loss runs end-to-end on real decoder outputs.
3. Gradients flow ONLY in the decoder; loss decreases is NOT asserted
   (2 steps prove plumbing, not learning).

Usage: python scripts/smoke_finetune.py [--pair-index 0] [--steps 2]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cma.train.dataset import WARP_GRID, WarpPairDataset
from cma.train.finetune import build_model, forward_decoder_only, set_train_mode
from cma.train.loss import SparseGTRobustLoss

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def denorm(t: torch.Tensor) -> np.ndarray:
    img = t.permute(1, 2, 0).numpy() * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)


def visualize(sample: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    A = denorm(sample["im_A"])
    B = denorm(sample["im_B"])
    h, w = A.shape[:2]
    warp = sample["gt_warp"].numpy().reshape(-1, 2)
    prob = sample["gt_prob"].numpy().reshape(-1) > 0
    # A-grid in resized coords (uniform by construction)
    gx = np.linspace(0, w - 1, WARP_GRID)
    gy = np.linspace(0, h - 1, WARP_GRID)
    G = np.stack(np.meshgrid(gx, gy), axis=-1).reshape(-1, 2)
    # normalized warp -> resized-B pixels (stretch resize preserves
    # normalized coords): px = (x + 1) / 2 * size - 0.5
    bx = (warp[:, 0] + 1) / 2 * B.shape[1] - 0.5
    by = (warp[:, 1] + 1) / 2 * B.shape[0] - 0.5
    # color by A-grid position so structure correspondence is visible
    colors = np.stack([G[:, 0] / w, G[:, 1] / h, np.full(len(G), 0.5)], axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(A)
    axes[0].scatter(G[prob, 0], G[prob, 1], c=colors[prob], s=2)
    axes[0].set_title(f"A + supervised grid ({prob.sum()}/{len(prob)} pts), "
                      f"pair {sample['pair_id']}")
    axes[1].imshow(B)
    axes[1].scatter(bx[prob], by[prob], c=colors[prob], s=2)
    axes[1].set_title("B + warped grid (same colors = same point)")
    for ax in axes:
        ax.set_axis_off()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"warp visualization -> {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/AmalgaMatch")
    ap.add_argument("--split", default="results/split.json")
    ap.add_argument("--pair-index", type=int, default=0)
    ap.add_argument("--steps", type=int, default=2)
    args = ap.parse_args()

    split = json.loads(Path(args.split).read_text())

    # 1. dataset sample + visualization (no augmentation: GT check first)
    ds_plain = WarpPairDataset(args.root, split["train"], augment=False)
    sample = ds_plain[args.pair_index]
    visualize(sample, Path("results/smoke_warp.png"))
    ds_aug = WarpPairDataset(args.root, split["train"], augment=True, seed=1)
    visualize(ds_aug[args.pair_index], Path("results/smoke_warp_aug.png"))

    # 2 + 3. two training steps on CPU, decoder-only grads
    dev = torch.device("cpu")
    print("building model (CPU)...")
    model = build_model(dev)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"trainable {n_train / 1e6:.1f}M / total {n_total / 1e6:.1f}M params")

    objective = SparseGTRobustLoss()
    params = [p for p in model.decoder.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=2e-5, weight_decay=1e-4)
    set_train_mode(model)
    for i in range(args.steps):
        batch = {k: (v.unsqueeze(0) if torch.is_tensor(v) else v)
                 for k, v in ds_aug[(args.pair_index + i) % len(ds_aug)].items()}
        t0 = time.perf_counter()
        opt.zero_grad()
        corresps = forward_decoder_only(model, batch)
        loss = objective(corresps, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        print(f"step {i + 1}: loss {loss.item():.4f} "
              f"scales {sorted(corresps)} ({time.perf_counter() - t0:.0f}s)")

    enc_grads = [n for n, p in model.encoder.named_parameters()
                 if p.grad is not None]
    dec_grads = [p for p in model.decoder.parameters() if p.grad is not None]
    dec_nonzero = sum(int(p.grad.abs().sum() > 0) for p in dec_grads)
    assert not enc_grads, f"encoder received grads: {enc_grads[:5]}"
    assert dec_grads, "decoder received no grads"
    print(f"OK: encoder grad-free; decoder {len(dec_grads)} grad tensors, "
          f"{dec_nonzero} nonzero")


if __name__ == "__main__":
    main()
