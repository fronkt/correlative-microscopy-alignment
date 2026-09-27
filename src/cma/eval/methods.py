"""Eval-harness Method factories for the TTA baseline ladder.

The sweep harness consumes ``Method = Callable[[ImagePair], MethodResult]``.
This module turns the (CPU-tested) ``cma.tta.tta_adapt`` adaptation into such a
Method and registers the comparison ladder. RoMa-coupled factories are
validated on the GPU box (local torch is CPU-only); the registry and the
MethodResult plumbing are import-checked here.

Ladder (see research/tmlr-tta/research_brief.md §3):
  vanilla · pyramid-only [Pivot-S] · DMP [nearest prior] · TENT/BN [generic TTA]
  · supervised-ft±L2SP [Pivot-A] · TTA(scale/appearance/both) · AdaBN.
"""

from __future__ import annotations

import numpy as np

from cma.data import ImagePair
from cma.eval.sweep import Method, MethodResult


def _result_from_pyramid(pair: ImagePair, matcher, cfg) -> MethodResult:
    """Run the pyramid pipeline with ``matcher`` and box it as a MethodResult."""
    from cma.pipeline import register

    res = register(
        pair.source, pair.target, matcher=matcher, scale_ratio=pair.scale_ratio,
        overlap=cfg.overlap, family=cfg.family,
        ransac_threshold_px=cfg.ransac_threshold_px,
    )
    return MethodResult(
        H_target_to_source=res.H_target_to_source,
        n_correspondences=res.n_correspondences,
        n_tiles=res.n_tiles,
        family=res.transform.family,
    )


def _result_from_direct(pair: ImagePair, matcher, cfg) -> MethodResult:
    """Direct (no-pyramid) match + robust fit, boxed as a MethodResult."""
    from cma.estimators import fit_transform

    corr = matcher.match(pair.source, pair.target)
    if len(corr) < 4:
        raise RuntimeError(f"only {len(corr)} correspondences")
    est = fit_transform(src_xy=corr.b_xy, dst_xy=corr.a_xy, family=cfg.family,
                        ransac_threshold_px=cfg.ransac_threshold_px)
    return MethodResult(
        H_target_to_source=est.as_3x3(),
        n_correspondences=len(corr),
        n_tiles=1,
        family=est.family,
    )


def tta_method(
    model_builder,
    cfg,
    *,
    variant: str = "ma_outdoor",
    w_scale: float = 1.0,
    w_appearance: float = 1.0,
    anchor_lambda: float = 0.1,
    steps: int = 10,
    lr: float = 1e-3,
    use_pyramid: bool = True,
    coral_ref: tuple | None = None,
    device: str = "cuda",
) -> Method:
    """Method that per-pair test-time-adapts then matches.

    ``model_builder()`` returns a fresh RoMa model with a grad-enabled decoder
    (e.g. ``cma.train.finetune.build_model``). The model is built once and
    reset after each pair (stateless). Axis weights ``w_scale`` / ``w_appearance``
    select the objective; set one to 0 for the single-axis ablations.
    """
    from cma.matchers.roma import RoMaMatcher
    from cma.tta import tta_adapt

    model = model_builder()

    def _run(pair: ImagePair) -> MethodResult:
        adapted, _hist, reset = tta_adapt(
            model, pair.source, pair.target,
            w_scale=w_scale, w_appearance=w_appearance,
            anchor_lambda=anchor_lambda, steps=steps, lr=lr,
            coral_ref=coral_ref, device=device, restore=True,
        )
        matcher = RoMaMatcher(variant=variant, device=device, model=adapted)
        try:
            if use_pyramid:
                return _result_from_pyramid(pair, matcher, cfg)
            return _result_from_direct(pair, matcher, cfg)
        finally:
            reset()  # stateless across pairs — no cross-pair forgetting

    return _run


def matcher_method(matcher, cfg, *, use_pyramid: bool = True) -> Method:
    """Non-adaptive baseline: a fixed matcher, pyramid or direct (vanilla / Pivot-S)."""

    def _run(pair: ImagePair) -> MethodResult:
        if use_pyramid:
            return _result_from_pyramid(pair, matcher, cfg)
        return _result_from_direct(pair, matcher, cfg)

    return _run


# --- baseline ladder registry --------------------------------------------------
# Factories filled in on the box (need a live RoMa model / their own optim loop).
# Keeping the names here makes the ladder explicit and the run scripts declarative.

LADDER_BASELINES: dict[str, str] = {
    "vanilla_direct": "matcher_method(use_pyramid=False)",
    "pyramid_only": "matcher_method(use_pyramid=True)  # Pivot-S",
    "tta_scale": "tta_method(w_scale=1, w_appearance=0)",
    "tta_appearance": "tta_method(w_scale=0, w_appearance=1)",
    "tta_both": "tta_method(w_scale=1, w_appearance=1)",
    "dmp": "TODO(box): Deep Matching Prior per-pair optimization [nearest prior]",
    "tent": "TODO(box): norm-affine adapt to minimize certainty-map entropy",
    "adabn": "TODO(box): recompute decoder BN stats on target, no optimization",
    "supervised_ft": "cma.train.finetune (Pivot-A ceiling, ±L2SP)",
}


def assert_ladder_known(name: str) -> None:
    if name not in LADDER_BASELINES:
        raise KeyError(
            f"unknown baseline {name!r}; known: {sorted(LADDER_BASELINES)}")


__all__ = ["tta_method", "matcher_method", "LADDER_BASELINES", "assert_ladder_known"]


def _smoke_method_result() -> MethodResult:
    """Construct a trivial MethodResult (import/plumbing sanity, no model)."""
    return MethodResult(H_target_to_source=np.eye(3), n_correspondences=0,
                        n_tiles=1, family="identity")
