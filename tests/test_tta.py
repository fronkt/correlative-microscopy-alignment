"""CPU unit tests for the label-free TTA objectives + param selection.

These prove the math without touching RoMa weights (the real-model forward is
validated on the GPU box). Run: ``pytest tests/test_tta.py -q``.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cma.tta import (
    collect_norm_affine_params,
    coral_loss,
    cycle_consistency_loss,
    identity_grid,
    multi_scale_consistency_loss,
)


def test_identity_grid_shape_and_convention():
    g = identity_grid(5, 7)
    assert g.shape == (1, 2, 5, 7)
    # pixel-center (align_corners=False): symmetric, strictly inside [-1, 1]
    assert g.min() > -1.0 and g.max() < 1.0
    assert torch.isclose(g.min(), -g.max())


def test_cycle_loss_zero_for_identity():
    # A->B and B->A both identity => perfect cycle => floors at Charbonnier eps
    ident = identity_grid(8, 8)
    loss = cycle_consistency_loss(ident.clone(), ident.clone())
    assert loss.item() < 2e-3


def test_cycle_loss_positive_for_inconsistent():
    ident = identity_grid(8, 8)
    torch.manual_seed(0)
    bad = ident + 0.4 * torch.randn_like(ident)
    loss = cycle_consistency_loss(ident.clone(), bad)
    assert loss.item() > 1e-2


def test_multi_scale_consistency_zero_when_scales_agree():
    # constant flows at different resolutions but equal value => agree => ~0
    corr = {
        1: {"flow": torch.full((1, 2, 8, 8), 0.3),
            "certainty": torch.zeros(1, 1, 8, 8)},
        2: {"flow": torch.full((1, 2, 4, 4), 0.3),
            "certainty": torch.zeros(1, 1, 4, 4)},
    }
    loss = multi_scale_consistency_loss(corr)
    assert loss.item() < 2e-3  # floors at Charbonnier eps


def test_multi_scale_consistency_positive_when_scales_disagree():
    corr = {
        1: {"flow": torch.full((1, 2, 8, 8), 0.3),
            "certainty": torch.zeros(1, 1, 8, 8)},
        2: {"flow": torch.full((1, 2, 4, 4), -0.3),
            "certainty": torch.zeros(1, 1, 4, 4)},
    }
    loss = multi_scale_consistency_loss(corr)
    assert loss.item() > 1e-2


def test_multi_scale_single_scale_is_safe():
    corr = {1: {"flow": torch.zeros(1, 2, 4, 4)}}
    loss = multi_scale_consistency_loss(corr)
    assert loss.item() == 0.0


def test_coral_zero_against_own_stats():
    torch.manual_seed(1)
    feat = torch.randn(500, 16)
    mean = feat.mean(0)
    centered = feat - mean
    cov = (centered.t() @ centered) / (len(feat) - 1) + 1e-5 * torch.eye(16)
    loss = coral_loss(feat, mean, cov)
    assert loss.item() < 1e-3


def test_coral_positive_against_shifted_stats():
    torch.manual_seed(2)
    feat = torch.randn(500, 16)
    ref_mean = feat.mean(0) + 2.0
    ref_cov = 3.0 * torch.eye(16)
    loss = coral_loss(feat, ref_mean, ref_cov)
    assert loss.item() > 1e-2


def test_collect_norm_affine_params_picks_only_norms():
    module = nn.Sequential(
        nn.Conv2d(3, 4, 3),
        nn.BatchNorm2d(4),
        nn.ReLU(),
        nn.GroupNorm(2, 4),
        nn.Conv2d(4, 4, 1),
    )
    params = collect_norm_affine_params(module)
    # BatchNorm2d (w,b) + GroupNorm (w,b) = 4; conv params excluded
    assert len(params) == 4
    conv_weights = {id(module[0].weight), id(module[4].weight)}
    assert all(id(p) not in conv_weights for p in params)


def test_multi_scale_consistency_is_differentiable():
    flow1 = torch.zeros(1, 2, 8, 8, requires_grad=True)
    corr = {
        1: {"flow": flow1, "certainty": torch.zeros(1, 1, 8, 8)},
        2: {"flow": torch.full((1, 2, 4, 4), 0.2),
            "certainty": torch.zeros(1, 1, 4, 4)},
    }
    loss = multi_scale_consistency_loss(corr)
    loss.backward()
    assert flow1.grad is not None and flow1.grad.abs().sum() > 0
