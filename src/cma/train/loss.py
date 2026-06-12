"""RoMa robust loss adapted to precomputed dense GT warps.

Skeleton follows romatch.losses.robust_loss.RobustLosses, with two
changes: the GT warp/prob come from the batch (densified sparse GT, see
dataset.py) instead of depth+pose via get_gt_warp, and wandb logging is
removed. Branch coverage matches what roma_outdoor's decoder emits:
gm_cls at scale 16, delta regression at finer scales.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseGTRobustLoss(nn.Module):
    def __init__(
        self,
        ce_weight: float = 0.01,
        local_dist: dict | None = None,
        local_largest_scale: int = 8,
        alpha: float = 0.5,
        c: float = 1e-4,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.local_dist = local_dist or {1: 4, 2: 4, 4: 8, 8: 8}
        self.local_largest_scale = local_largest_scale
        self.alpha = alpha
        self.c = c

    def gm_cls_loss(self, x2, prob, scale_gm_cls, gm_certainty):
        with torch.no_grad():
            B, C, H, W = scale_gm_cls.shape
            device = x2.device
            cls_res = round(math.sqrt(C))
            G = torch.meshgrid(*[torch.linspace(
                -1 + 1 / cls_res, 1 - 1 / cls_res, steps=cls_res, device=device)
                for _ in range(2)], indexing="ij")
            G = torch.stack((G[1], G[0]), dim=-1).reshape(C, 2)
            GT = (G[None, :, None, None, :] - x2[:, None]).norm(dim=-1).min(dim=1).indices
        cls_loss = F.cross_entropy(scale_gm_cls, GT, reduction="none")[prob > 0.99]
        certainty_loss = F.binary_cross_entropy_with_logits(gm_certainty[:, 0], prob)
        if not torch.any(cls_loss):
            cls_loss = certainty_loss * 0.0
        return certainty_loss.mean(), cls_loss.mean()

    def regression_loss(self, x2, prob, flow, certainty, scale):
        epe = (flow.permute(0, 2, 3, 1) - x2).norm(dim=-1)
        ce_loss = F.binary_cross_entropy_with_logits(certainty[:, 0], prob)
        cs = self.c * scale
        x = epe[prob > 0.99]
        reg_loss = cs**self.alpha * ((x / cs) ** 2 + 1**2) ** (self.alpha / 2)
        if not torch.any(reg_loss):
            reg_loss = ce_loss * 0.0
        return ce_loss.mean(), reg_loss.mean(), epe

    def forward(self, corresps: dict, batch: dict) -> torch.Tensor:
        gt_warp = batch["gt_warp"]  # (B, Hg, Wg, 2) normalized B-coords
        gt_prob = batch["gt_prob"]  # (B, Hg, Wg) in {0, 1}
        tot_loss = 0.0
        prev_epe = None
        for scale in corresps:
            sc = corresps[scale]
            certainty, flow = sc["certainty"], sc["flow"]
            scale_gm_cls = sc.get("gm_cls")
            scale_gm_certainty = sc.get("gm_certainty")
            b, _, h, w = certainty.shape

            x2 = F.interpolate(gt_warp.permute(0, 3, 1, 2), size=(h, w),
                               mode="bilinear", align_corners=False
                               ).permute(0, 2, 3, 1).float()
            prob = F.interpolate(gt_prob[:, None], size=(h, w),
                                 mode="nearest-exact")[:, 0].float()

            if self.local_largest_scale >= scale and prev_epe is not None:
                prob = prob * (
                    F.interpolate(prev_epe[:, None], size=(h, w),
                                  mode="nearest-exact")[:, 0]
                    < (2 / 512) * (self.local_dist[scale] * scale))

            if scale_gm_cls is not None:
                cert_l, cls_l = self.gm_cls_loss(
                    x2, prob, scale_gm_cls, scale_gm_certainty)
                tot_loss = tot_loss + self.ce_weight * cert_l + cls_l
                prev_epe = (flow.permute(0, 2, 3, 1) - x2).norm(dim=-1).detach()
            else:
                ce_l, reg_l, epe = self.regression_loss(
                    x2, prob, flow, certainty, scale)
                tot_loss = tot_loss + self.ce_weight * ce_l + reg_l
                prev_epe = epe.detach()
        return tot_loss
