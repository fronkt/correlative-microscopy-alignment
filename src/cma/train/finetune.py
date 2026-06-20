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
    pids: list[str] = []
    for rec in loader.records:
        if rec.pair_id not in wanted:
            continue
        pids.append(rec.pair_id)
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
        "pair_ids": pids,
    }


def _subset_sr(errs: list[float], pids: list[str], substr: str,
               thr: float = 20.0) -> float:
    """SR@thr over the val pairs whose id contains `substr` (nan if none).

    The forgetting probe: global val median hides a single modality's
    collapse, so λ selection needs the per-modality success rate. C103
    (SEM↔LOM, 0 train pairs) is the retention probe; TEM is the gain probe.
    """
    vals = [e for e, p in zip(errs, pids) if substr in p]
    if not vals:
        return float("nan")
    return float(np.mean(np.asarray(vals) <= thr))


def _l2sp_penalty(params: list, theta0: list) -> torch.Tensor:
    """L2-SP: ½·Σ(θ_dec − θ_dec⁰)² (weight-decay form, anchor at init)."""
    return 0.5 * sum((p - p0).pow(2).sum() for p, p0 in zip(params, theta0))


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
    anchor: str = "none",
    anchor_lambda: float = 0.0,
) -> Path:
    """Train and return the path of the best (val-selected) checkpoint.

    `anchor="l2sp"` adds `anchor_lambda * mean((θ_dec - θ_dec⁰)²)` to the
    loss, penalizing decoder drift from the zero-shot init to curb
    catastrophic forgetting of untrained modalities (§8.1). `anchor="none"`
    (or `anchor_lambda=0`) reproduces the plain decoder-only ft exactly, so
    it is the control arm of the λ sweep. Checkpoint selection is unchanged
    (min val med-µED) for comparability with §7; the per-modality retention
    probes are logged so the λ-level decision can apply a C103 floor.
    """
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
    use_anchor = anchor == "l2sp" and anchor_lambda > 0.0
    theta0 = [p.detach().clone() for p in params] if use_anchor else None
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
            task_loss = objective(corresps, batch)
            if use_anchor:
                pen = _l2sp_penalty(params, theta0)
                loss = task_loss + anchor_lambda * pen
                pen_val = float(pen.detach())
            else:
                loss = task_loss
                pen_val = 0.0
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
                print(f"step {step}/{steps} loss {task_loss.item():.4f} "
                      f"pen {pen_val:.3e} lr {sched.get_last_lr()[0]:.2e} "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
            if val_every > 0 and (step % val_every == 0 or step == steps):
                v = evaluate_direct(model, root, split["val"], dev)
                c103 = _subset_sr(v["errs"], v["pair_ids"], "C103")
                tem = _subset_sr(v["errs"], v["pair_ids"], "TEM")
                improved = v["med_mu_ed"] < best
                if improved:
                    best = v["med_mu_ed"]
                    torch.save(model.state_dict(), best_path)
                print(f"VAL step {step}: med_mu_ed {v['med_mu_ed']:.2f} "
                      f"SR@10 {v['sr10']:.3f} SR@20 {v['sr20']:.3f} "
                      f"[C103@20 {c103:.3f} TEM@20 {tem:.3f}] "
                      f"(n={v['n']})" + (" *saved*" if improved else ""),
                      flush=True)
                log_rows.append({"step": step, "loss": f"{task_loss.item():.4f}",
                                 "anchor_pen": f"{pen_val:.3e}",
                                 "med_mu_ed": f"{v['med_mu_ed']:.3f}",
                                 "sr10": f"{v['sr10']:.3f}",
                                 "sr20": f"{v['sr20']:.3f}",
                                 "c103_sr20": f"{c103:.3f}",
                                 "tem_sr20": f"{tem:.3f}",
                                 "saved": int(improved)})

    if val_every <= 0:  # no selection ran — keep the final state
        torch.save(model.state_dict(), best_path)
    if log_path:
        with Path(log_path).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f, fieldnames=["step", "loss", "anchor_pen", "med_mu_ed",
                               "sr10", "sr20", "c103_sr20", "tem_sr20", "saved"])
            w.writeheader()
            w.writerows(log_rows)
    print(f"DONE: best val med_mu_ed {best:.2f} -> {best_path}", flush=True)
    return best_path
