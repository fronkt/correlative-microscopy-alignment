"""Per-pair test-time adaptation of a frozen-encoder RoMa matcher.

We adapt ONLY the decoder's normalization-layer affine parameters (scale/shift)
with a frozen encoder (TENT-style minimal surface), regularized by an L2-SP
anchor to the initialization (mechanism validated offline in this repo). The
adaptation signal is label-free and axis-weighted: multi-scale self-consistency
for the scale axis, forward-backward cycle + CORAL feature alignment for the
appearance axis (see ``losses.py``).

``collect_norm_affine_params`` is pure and CPU-unit-tested. ``tta_adapt`` runs
the real RoMa forward and is validated on the GPU box; it reuses the
encoder-frozen forward from ``cma.train.finetune``.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from cma.tta.losses import (
    coral_loss,
    cycle_consistency_loss,
    multi_scale_consistency_loss,
)

_NORM_TYPES = (
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.LayerNorm, nn.GroupNorm,
    nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d,
)


def collect_norm_affine_params(module: nn.Module) -> list[nn.Parameter]:
    """Return the weight/bias affine params of every normalization layer.

    These are the only parameters TTA updates. Pure: no side effects on
    ``requires_grad`` (the caller decides that). Deterministic order.
    """
    params: list[nn.Parameter] = []
    for m in module.modules():
        if isinstance(m, _NORM_TYPES):
            if getattr(m, "weight", None) is not None:
                params.append(m.weight)
            if getattr(m, "bias", None) is not None:
                params.append(m.bias)
    return params


def _l2sp(params: list[torch.Tensor], theta0: list[torch.Tensor]) -> torch.Tensor:
    """½·Σ(θ − θ⁰)² anchor to init (matches cma.train.finetune)."""
    return 0.5 * sum((p - p0).pow(2).sum() for p, p0 in zip(params, theta0, strict=True))


def _build_pair_batch(source: np.ndarray, target: np.ndarray, res: int,
                      device: torch.device) -> dict:
    """RoMa-style {im_A, im_B} batch from a raw image pair, no GT."""
    from PIL import Image
    from romatch.utils.utils import get_tuple_transform_ops

    def _rgb8(a: np.ndarray) -> Image.Image:
        a = np.asarray(a)
        if a.ndim == 2:
            a = np.stack([a] * 3, -1)
        elif a.shape[2] == 1:
            a = np.repeat(a, 3, 2)
        if a.dtype != np.uint8:
            a = (np.clip(a, 0, 1) * 255).astype(np.uint8)
        return Image.fromarray(a)

    tf = get_tuple_transform_ops(resize=(res, res), normalize=True)
    im_a, im_b = tf((_rgb8(source), _rgb8(target)))
    return {"im_A": im_a.unsqueeze(0).to(device),
            "im_B": im_b.unsqueeze(0).to(device)}


def _decoder_forward(model, batch: dict) -> dict:
    """Encoder under no_grad, decoder with grad. Returns corresps dict."""
    with torch.no_grad():
        feats = model.extract_backbone_features(batch, batched=True)
    f_q = {s: f.chunk(2)[0] for s, f in feats.items()}
    f_s = {s: f.chunk(2)[1] for s, f in feats.items()}
    return model.decoder(f_q, f_s)


def tta_adapt(
    model,
    source: np.ndarray,
    target: np.ndarray,
    *,
    w_scale: float = 1.0,
    w_appearance: float = 1.0,
    anchor_lambda: float = 0.1,
    steps: int = 10,
    lr: float = 1e-3,
    res: int = 560,
    coral_ref: tuple[torch.Tensor, torch.Tensor] | None = None,
    device: str = "cuda",
    restore: bool = True,
):
    """Adapt ``model`` in place on one pair; return (model, history).

    ``w_scale`` weights the multi-scale-consistency term, ``w_appearance`` the
    cycle (+ optional CORAL) term — set by axis routing. Returns
    ``(model, history, reset)``: the model carries the adaptation (caller runs
    ``match`` on it), then calls ``reset()`` to restore the init norm-affine
    params, making the procedure stateless across pairs (no cross-pair
    forgetting by construction). If ``restore=False``, ``reset`` is a no-op.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model.requires_grad_(False)
    params = collect_norm_affine_params(model.decoder)
    if not params:
        raise RuntimeError("no norm-affine params found in decoder")
    for p in params:
        p.requires_grad_(True)
    theta0 = [p.detach().clone() for p in params]
    saved = [p.detach().clone() for p in params] if restore else None

    model.eval()
    model.decoder.train(True)  # norm layers use batch stats (TENT/AdaBN-style)
    opt = torch.optim.AdamW(params, lr=lr)

    batch_ab = _build_pair_batch(source, target, res, dev)
    batch_ba = {"im_A": batch_ab["im_B"], "im_B": batch_ab["im_A"]}

    history: list[float] = []
    for _ in range(steps):
        opt.zero_grad()
        corr_ab = _decoder_forward(model, batch_ab)
        loss = corr_ab[min(corr_ab)]["flow"].new_zeros(())
        if w_scale > 0:
            loss = loss + w_scale * multi_scale_consistency_loss(corr_ab)
        if w_appearance > 0:
            corr_ba = _decoder_forward(model, batch_ba)
            fin = min(corr_ab)
            loss = loss + w_appearance * cycle_consistency_loss(
                corr_ab[fin]["flow"], corr_ba[fin]["flow"],
                certainty_ab=corr_ab[fin].get("certainty"))
            if coral_ref is not None:
                feat = corr_ab[fin]["flow"].flatten(2).transpose(1, 2).reshape(
                    -1, corr_ab[fin]["flow"].shape[1])
                loss = loss + w_appearance * coral_loss(
                    feat, coral_ref[0], coral_ref[1])
        if anchor_lambda > 0:
            loss = loss + anchor_lambda * _l2sp(params, theta0)
        loss.backward()
        opt.step()
        history.append(float(loss.detach()))

    model.eval()  # adaptation retained; ready for the caller's match()

    def reset() -> None:
        if saved is None:
            return
        with torch.no_grad():
            for p, p0 in zip(params, saved, strict=True):
                p.copy_(p0)

    return model, history, reset
