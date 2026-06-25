"""Label-free test-time adaptation of dense matchers (TMLR paper).

Decomposes domain shift into a scale axis (multi-scale self-consistency) and an
appearance axis (forward-backward cycle + CORAL feature alignment); adapts only
the decoder norm-affine params of a frozen-encoder RoMa matcher, anchored to
init. See ``research/tmlr-tta/`` for the research brief and plan.
"""

from __future__ import annotations

from cma.tta.adapt import collect_norm_affine_params, tta_adapt
from cma.tta.losses import (
    coral_loss,
    cycle_consistency_loss,
    identity_grid,
    multi_scale_consistency_loss,
)

__all__ = [
    "collect_norm_affine_params",
    "tta_adapt",
    "coral_loss",
    "cycle_consistency_loss",
    "identity_grid",
    "multi_scale_consistency_loss",
]
