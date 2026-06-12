"""Decoder-only fine-tuning of MA-RoMa on the AmalgaMatch train split.

The encoder (VGG + DINOv2) stays frozen and runs under no_grad in eval
mode — its BatchNorm running stats must not drift while we only train the
decoder. The forward mirrors RegressionMatcher.forward(batched=True):
encoder on cat(im_A, im_B), chunk(2), then decoder(f_q, f_s). Optimizer /
scaler / clip follow romatch.train.train_step (which we cannot import:
it logs to wandb unconditionally).

Validation = direct-match registration on the val split via the standard
RoMaMatcher path with the in-training model injected; selection metric is
the median over pairs of mu-ED (failed pairs count as inf). Headline
evaluation on the test split happens only once, after training, via
scripts/run_baselines_A.py --backbones ma_roma_ft.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cma.train.dataset import WarpPairDataset
from cma.train.loss import SparseGTRobustLoss

RANSAC_PX = 5.5  # paper protocol, same as run_baselines_A


def build_model(device: torch.device, weights_path: str | None = None):
    """MA-RoMa in the roma_outdoor arch, everything frozen but the decoder."""
    from romatch import roma_outdoor

    from cma.matchers.roma import _has_local_corr_kernel, _load_ma_roma_weights

    if weights_path is not None:
        weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    else:
        weights = _load_ma_roma_weights()
    model = roma_outdoor(device=device, weights=weights,
                         use_custom_corr=_has_local_corr_kernel())
    model.requires_grad_(False)
    model.decoder.requires_grad_(True)
    return model


def set_train_mode(model) -> None:
    """Decoder trains; frozen encoder stays in eval (BN stats, dropout)."""
    model.eval()
    model.decoder.train(True)


def forward_decoder_only(model, batch: dict) -> dict:
    """RegressionMatcher.forward(batched=True) with the encoder graph cut."""
    with torch.no_grad():
        feature_pyramid = model.extract_backbone_features(batch, batched=True)
    f_q = {s: f.chunk(2)[0] for s, f in feature_pyramid.items()}
    f_s = {s: f.chunk(2)[1] for s, f in feature_pyramid.items()}
    return model.decoder(f_q, f_s)


def evaluate_direct(model, root: str, pair_ids: list[str],
                    device: torch.device) -> dict:
    """Direct-match registration metrics over the given pairs.

    Returns median mu-ED (selection metric, inf for failed pairs), SR@10,
    SR@20, and the per-pair errors. Restores the model's train/eval state.
    """
    from cma.data import AmalgaMatchLoader
    from cma.estimators import fit_transform
    from cma.matchers.roma import RoMaMatcher
    from cma.metrics import registration_metrics

    matcher = RoMaMatcher(variant="ma_outdoor", device=str(device), model=model)
    was_training = model.decoder.training
    model.eval()
    loader = AmalgaMatchLoader(root)
    wanted = set(pair_ids)
    errs: list[float] = []
    for rec in loader.records:
        if rec.pair_id not in wanted:
            continue
        try:
            pair = loader.load_pair(rec)
            corr = matcher.match(pair.source, pair.target)
            if len(corr) < 4:
                raise RuntimeError(f"only {len(corr)} correspondences")
            est = fit_transform(src_xy=corr.b_xy, dst_xy=corr.a_xy,
                                family="auto", ransac_threshold_px=RANSAC_PX)
            H = est.as_3x3()
            h = np.hstack([pair.gt.tgt_xy, np.ones((len(pair.gt.tgt_xy), 1))])
            proj = (H @ h.T).T
            proj = proj[:, :2] / proj[:, 2:3]
            m = registration_metrics(proj, pair.gt.src_xy)
            errs.append(float(m.mu_err))
        except Exception:  # noqa: BLE001 — a failed pair is a data point, not a crash
            errs.append(float("inf"))
    if was_training:
        set_train_mode(model)
    arr = np.asarray(errs)
    return {
        "med_mu_ed": float(np.median(arr)),
        "sr10": float(np.mean(arr <= 10)),
        "sr20": float(np.mean(arr <= 20)),
        "n": len(errs),
        "errs": errs,
    }


def finetune(
    root: str = "data/AmalgaMatch",
    split_path: str = "results/split.json",
    out_dir: str = "checkpoints",
    steps: int = 1000,
    batch_size: int = 4,
    lr: float = 2e-5,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    val_every: int = 100,
    num_workers: int = 4,
    device: str = "cuda",
    seed: int = 0,
    log_path: str | None = None,
) -> Path:
    """Train and return the path of the best (val-selected) checkpoint."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    split = json.loads(Path(split_path).read_text())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    best_path = out / "ma_roma_ft.pth"

    model = build_model(dev)
    dataset = WarpPairDataset(root, split["train"], augment=True, seed=seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, drop_last=True,
                        persistent_workers=num_workers > 0)
    objective = SparseGTRobustLoss()
    params = [p for p in model.decoder.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    scaler = torch.amp.GradScaler(enabled=dev.type == "cuda")

    log_rows: list[dict] = []
    best = float("inf")
    step = 0
    t0 = time.perf_counter()
    set_train_mode(model)
    while step < steps:
        for batch in loader:
            if step >= steps:
                break
            batch = {k: v.to(dev) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            opt.zero_grad()
            corresps = forward_decoder_only(model, batch)
            loss = objective(corresps, batch)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            scaler.step(opt)
            scaler.update()
            # romatch.train.train_step keeps the scale from collapsing below 1
            if scaler.is_enabled() and scaler.get_scale() < 1.0:
                scaler._scale = torch.tensor(1.0).to(scaler._scale)
            sched.step()
            step += 1

            if step % 10 == 0 or step == 1:
                print(f"step {step}/{steps} loss {loss.item():.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
            if val_every > 0 and (step % val_every == 0 or step == steps):
                v = evaluate_direct(model, root, split["val"], dev)
                improved = v["med_mu_ed"] < best
                if improved:
                    best = v["med_mu_ed"]
                    torch.save(model.state_dict(), best_path)
                print(f"VAL step {step}: med_mu_ed {v['med_mu_ed']:.2f} "
                      f"SR@10 {v['sr10']:.3f} SR@20 {v['sr20']:.3f} "
                      f"(n={v['n']})" + (" *saved*" if improved else ""),
                      flush=True)
                log_rows.append({"step": step, "loss": f"{loss.item():.4f}",
                                 "med_mu_ed": f"{v['med_mu_ed']:.3f}",
                                 "sr10": f"{v['sr10']:.3f}",
                                 "sr20": f"{v['sr20']:.3f}",
                                 "saved": int(improved)})

    if val_every <= 0:  # no selection ran — keep the final state
        torch.save(model.state_dict(), best_path)
    if log_path:
        with Path(log_path).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f, fieldnames=["step", "loss", "med_mu_ed", "sr10", "sr20", "saved"])
            w.writeheader()
            w.writerows(log_rows)
    print(f"DONE: best val med_mu_ed {best:.2f} -> {best_path}", flush=True)
    return best_path
