"""FOV sensitivity sweep on synthetic correlative pairs.

For each FOV ratio in `config.fov_ratios`, generate `config.n_pairs` synthetic
ImagePairs (different seeds), run a registration method, and report per-pair
metrics on the ground-truth keypoint grid.

Outputs a tidy list of `SweepRow` records — one per (fov_ratio, seed, method).
A `method` is anything with the shape
    Callable[[ImagePair], (H_3x3_target_to_source, n_correspondences, n_tiles)]
which lets us plug in both `register(...)` (pyramid pipeline) and
`classical_register(...)` (Control B) without sweep code knowing about
either's internals.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np

from cma.data import ImagePair, synthesize_cross_modal_pair, synthesize_pair
from cma.matchers import SIFTMatcher
from cma.metrics import registration_metrics
from cma.pipeline import classical_register, register

MatcherFactory = Callable[[], "object"]
Method = Callable[[ImagePair], "MethodResult"]


@dataclass
class MethodResult:
    """Lightweight result tuple for the sweep — backend-agnostic."""

    H_target_to_source: np.ndarray
    n_correspondences: int
    n_tiles: int
    family: str


@dataclass
class SweepConfig:
    fov_ratios: tuple[float, ...] = (0.5, 0.25, 0.1, 0.05, 0.02)
    n_pairs: int = 10
    source_size: int = 1024
    target_size: int = 256
    rotation_deg: float = 5.0
    noise_sigma: float = 0.005
    family: str = "auto"
    ransac_threshold_px: float = 3.0
    overlap: float = 0.5
    cross_modal: str | None = None  # None | "invert" | "gamma" | "edge" | "smooth" | "stack"
    source: str = "noise"  # "noise" (layered noise) or "natural" (skimage.data.astronaut)


@dataclass
class SweepRow:
    backbone: str
    fov_ratio: float
    seed: int
    success: bool
    failure_reason: str = ""
    mu_err: float = float("nan")
    med_err: float = float("nan")
    p_match_at_1: float = float("nan")
    p_match_at_3: float = float("nan")
    p_match_at_5: float = float("nan")
    p_match_at_10: float = float("nan")
    n_tiles: int = 0
    n_correspondences: int = 0
    runtime_s: float = float("nan")
    extras: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = asdict(self)
        # Flatten extras
        for k, v in d.pop("extras").items():
            d[f"extra_{k}"] = v
        return d


def _apply_h(H: np.ndarray, xy: np.ndarray) -> np.ndarray:
    ones = np.ones((xy.shape[0], 1), dtype=xy.dtype)
    hom = np.concatenate([xy, ones], axis=1)
    proj = hom @ H.T
    return proj[:, :2] / proj[:, 2:3]


def pyramid_method(matcher, cfg: "SweepConfig") -> Method:
    """Adapt the pyramidal register() to the Method shape.

    `matcher` is a `Matcher` instance — kept across calls. Matchers must be
    stateless; the OracleMatcher's RNG state is the only exception and is
    not used inside the pyramid pipeline.
    """

    def _run(pair: ImagePair) -> MethodResult:
        res = register(
            pair.source,
            pair.target,
            matcher=matcher,
            scale_ratio=pair.scale_ratio,
            overlap=cfg.overlap,
            family=cfg.family,
            ransac_threshold_px=cfg.ransac_threshold_px,
        )
        return MethodResult(
            H_target_to_source=res.H_target_to_source,
            n_correspondences=res.n_correspondences,
            n_tiles=res.n_tiles,
            family=res.transform.family,
        )

    return _run


def classical_method(refine_with_mi: bool, cfg: "SweepConfig") -> Method:
    """Adapt classical_register() (Control B) to the Method shape."""

    def _run(pair: ImagePair) -> MethodResult:
        res = classical_register(
            pair.source,
            pair.target,
            matcher=SIFTMatcher(),
            family=cfg.family,
            ransac_threshold_px=cfg.ransac_threshold_px,
            refine_with_mi=refine_with_mi,
        )
        return MethodResult(
            H_target_to_source=res.H_target_to_source,
            n_correspondences=res.n_correspondences,
            n_tiles=1,  # classical does not tile
            family=res.transform.family,
        )

    return _run


def fov_sweep(
    method_name: str,
    method: Method,
    config: SweepConfig | None = None,
) -> list[SweepRow]:
    """Run the FOV sweep for a single method."""
    cfg = config or SweepConfig()
    rows: list[SweepRow] = []

    if cfg.source == "natural":
        from cma.data.synthetic import natural_source_image
        source_image = natural_source_image(cfg.source_size)
    elif cfg.source == "noise":
        source_image = None
    else:
        raise ValueError(f"unknown source '{cfg.source}', expected 'noise' or 'natural'")

    for fov_ratio in cfg.fov_ratios:
        for seed in range(cfg.n_pairs):
            if cfg.cross_modal:
                pair, _H_gt = synthesize_cross_modal_pair(
                    source_size=cfg.source_size,
                    fov_ratio=fov_ratio,
                    target_size=cfg.target_size,
                    seed=seed,
                    rotation_deg=cfg.rotation_deg,
                    noise_sigma=cfg.noise_sigma,
                    mode=cfg.cross_modal,  # type: ignore[arg-type]
                    source_image=source_image,
                )
            else:
                pair, _H_gt = synthesize_pair(
                    source_size=cfg.source_size,
                    fov_ratio=fov_ratio,
                    target_size=cfg.target_size,
                    seed=seed,
                    rotation_deg=cfg.rotation_deg,
                    noise_sigma=cfg.noise_sigma,
                    source_image=source_image,
                )
            t0 = time.perf_counter()
            try:
                result = method(pair)
            except RuntimeError as e:
                rows.append(
                    SweepRow(
                        backbone=method_name,
                        fov_ratio=fov_ratio,
                        seed=seed,
                        success=False,
                        failure_reason=str(e),
                        runtime_s=time.perf_counter() - t0,
                    )
                )
                continue
            runtime = time.perf_counter() - t0

            pred = _apply_h(result.H_target_to_source, pair.gt.tgt_xy)
            metrics = registration_metrics(pred, pair.gt.src_xy)
            rows.append(
                SweepRow(
                    backbone=method_name,
                    fov_ratio=fov_ratio,
                    seed=seed,
                    success=True,
                    mu_err=metrics.mu_err,
                    med_err=metrics.med_err,
                    p_match_at_1=metrics.p_match_at_1,
                    p_match_at_3=metrics.p_match_at_3,
                    p_match_at_5=metrics.p_match_at_5,
                    p_match_at_10=metrics.p_match_at_10,
                    n_tiles=result.n_tiles,
                    n_correspondences=result.n_correspondences,
                    runtime_s=runtime,
                    extras={"transform_family": result.family},
                )
            )

    return rows
