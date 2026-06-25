"""Label-free test-time-adaptation objectives for the RoMa decoder.

Three signals, mapped to the two shift axes of the paper:

- ``multi_scale_consistency_loss`` — SCALE axis. The decoder emits a ``flow``
  estimate of the same A->B correspondence at every pyramid scale; under scale
  shift these disagree. Pulling them into agreement is a label-free signal that
  is informative exactly when scale is the problem.
- ``cycle_consistency_loss`` — APPEARANCE axis. Forward-backward (A->B->A)
  composition error, occlusion-masked and certainty-weighted, after UnFlow
  (Meister et al., AAAI 2018). Appearance/modality shift breaks the bijection
  the cycle assumes.
- ``coral_loss`` — APPEARANCE axis. Deep CORAL (Sun & Saenko, ECCV-W 2016):
  pull the second-order statistics of the decoder's features toward a source
  reference computed once on in-domain data ("align to what?" resolved offline).

All operate on the verified decoder contract: ``corresps[scale]`` carries
``flow`` (B,2,H,W in normalized [-1,1] grid coords) and ``certainty``
(B,1,H,W logits) for scale in {16,8,4,2,1}.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Robust L1 (Charbonnier) penalty, as used in unsupervised flow."""
    return torch.sqrt(x * x + eps * eps)


def identity_grid(h: int, w: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """(1,2,H,W) identity sampling grid (x,y order), pixel-center convention.

    Uses the ``align_corners=False`` convention ``coord = 2*(i+0.5)/n - 1`` so
    that ``grid_sample(..., align_corners=False)`` of a tensor at this grid is
    the identity. This matches RoMa's own flow convention
    (``to_pixel_coordinates``: ``x = 2*(px+0.5)/W - 1``).
    """
    yc = (2 * (torch.arange(h, device=device, dtype=dtype) + 0.5) / h) - 1
    xc = (2 * (torch.arange(w, device=device, dtype=dtype) + 0.5) / w) - 1
    ys, xs = torch.meshgrid(yc, xc, indexing="ij")
    return torch.stack((xs, ys), dim=0).unsqueeze(0)


def multi_scale_consistency_loss(
    corresps: dict,
    scales: list[int] | None = None,
    weight_by_certainty: bool = True,
    eps: float = 1e-3,
) -> torch.Tensor:
    """SCALE-axis signal: disagreement of per-scale ``flow`` vs the finest.

    Each coarser scale's flow is bilinearly upsampled to the finest scale's
    resolution and penalized (Charbonnier) against the finest flow, optionally
    weighted by the (sigmoid) certainty shared by both scales. Returns a scalar.
    """
    present = sorted(s for s in corresps if "flow" in corresps[s])
    if scales is not None:
        present = [s for s in present if s in scales]
    if len(present) < 2:
        # nothing to be consistent with
        ref_any = next(iter(corresps.values()))["flow"]
        return ref_any.new_zeros(())
    finest = min(present)  # smaller scale int == finer resolution in RoMa
    ref_flow = corresps[finest]["flow"]
    _, _, hf, wf = ref_flow.shape
    ref_cert = corresps[finest].get("certainty")

    total = ref_flow.new_zeros(())
    count = 0
    for s in present:
        if s == finest:
            continue
        flow_s = corresps[s]["flow"]
        flow_up = F.interpolate(flow_s, size=(hf, wf), mode="bilinear",
                                align_corners=False)
        resid = _charbonnier(flow_up - ref_flow, eps).mean(dim=1, keepdim=True)
        if weight_by_certainty and ref_cert is not None and "certainty" in corresps[s]:
            cert_s = F.interpolate(corresps[s]["certainty"], size=(hf, wf),
                                   mode="bilinear", align_corners=False)
            w = torch.sigmoid(ref_cert) * torch.sigmoid(cert_s)
            resid = resid * w
            total = total + resid.sum() / (w.sum() + 1e-6)
        else:
            total = total + resid.mean()
        count += 1
    return total / max(count, 1)


def cycle_consistency_loss(
    flow_ab: torch.Tensor,
    flow_ba: torch.Tensor,
    certainty_ab: torch.Tensor | None = None,
    eps: float = 1e-3,
) -> torch.Tensor:
    """APPEARANCE-axis signal: forward-backward composition error.

    ``flow_ab`` (B,2,H,W) maps the A grid into normalized B coords; ``flow_ba``
    maps the B grid into normalized A coords. Sampling ``flow_ba`` at the
    locations predicted by ``flow_ab`` should return the identity A grid.
    Out-of-bounds forward predictions are treated as occluded and masked.
    Optionally certainty-weighted. Returns a scalar.
    """
    b, _, h, w = flow_ab.shape
    grid_ab = flow_ab.permute(0, 2, 3, 1)  # (B,H,W,2) sampling grid in B
    flow_ba_at_ab = F.grid_sample(
        flow_ba, grid_ab, mode="bilinear", padding_mode="border",
        align_corners=False,
    )  # (B,2,H,W): A-coord predicted by going A->B->A
    ident = identity_grid(h, w, device=flow_ab.device, dtype=flow_ab.dtype)
    resid = _charbonnier(flow_ba_at_ab - ident, eps).mean(dim=1, keepdim=True)

    # occlusion mask: forward prediction must land inside the B frame
    inb = ((grid_ab[..., 0] >= -1) & (grid_ab[..., 0] <= 1)
           & (grid_ab[..., 1] >= -1) & (grid_ab[..., 1] <= 1))
    mask = inb.unsqueeze(1).to(resid.dtype)
    if certainty_ab is not None:
        mask = mask * torch.sigmoid(certainty_ab)
    return (resid * mask).sum() / (mask.sum() + 1e-6)


def coral_loss(
    feat: torch.Tensor,
    ref_mean: torch.Tensor,
    ref_cov: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """APPEARANCE-axis signal: Deep CORAL distance to a source reference.

    ``feat`` is (N, C) (flatten spatial dims before calling). ``ref_mean`` (C,)
    and ``ref_cov`` (C, C) are precomputed once on in-domain features. Returns
    the squared Frobenius covariance gap plus a mean-gap term, both normalized
    by feature dimension (Sun & Saenko, 2016).
    """
    if feat.dim() != 2:
        raise ValueError(f"feat must be (N, C); got {tuple(feat.shape)}")
    n, c = feat.shape
    mean = feat.mean(dim=0)
    centered = feat - mean
    cov = (centered.t() @ centered) / max(n - 1, 1) + eps * torch.eye(
        c, device=feat.device, dtype=feat.dtype)
    cov_gap = (cov - ref_cov).pow(2).sum() / (4 * c * c)
    mean_gap = (mean - ref_mean).pow(2).sum() / c
    return cov_gap + mean_gap
