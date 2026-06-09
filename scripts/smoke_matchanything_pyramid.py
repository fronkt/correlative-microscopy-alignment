"""End-to-end smoke: MatchAnything through the pyramid pipeline."""

import time

import numpy as np

from cma.data import synthesize_pair, synthesize_cross_modal_pair
from cma.matchers import MatchAnythingMatcher
from cma.pipeline import register


def _apply_h(H, xy):
    ones = np.ones((xy.shape[0], 1))
    hom = np.concatenate([xy, ones], axis=1)
    proj = hom @ H.T
    return proj[:, :2] / proj[:, 2:3]


def _run(label, pair, matcher):
    t = time.perf_counter()
    result = register(
        pair.source, pair.target, matcher=matcher, scale_ratio=pair.scale_ratio,
    )
    rt = time.perf_counter() - t
    pred = _apply_h(result.H_target_to_source, pair.gt.tgt_xy)
    err = np.linalg.norm(pred - pair.gt.src_xy, axis=1)
    print(
        f"  {label:>14s}  n_tiles={result.n_tiles:>3d}  "
        f"n_corr={result.n_correspondences:>5d}  "
        f"family={result.transform.family:>10s}  "
        f"mean_err={err.mean():>8.3f}px  rt={rt:>5.2f}s"
    )


def main() -> None:
    print("loading MatchAnything ...")
    matcher = MatchAnythingMatcher()

    print("\n== same-modality, fov=0.10 ==")
    pair, _ = synthesize_pair(1024, 0.10, 256, seed=0, rotation_deg=5.0)
    _run("same-mod", pair, matcher)

    print("\n== cross-modal invert, fov=0.10 ==")
    pair, _ = synthesize_cross_modal_pair(1024, 0.10, 256, seed=0, rotation_deg=5.0, mode="invert")
    _run("invert", pair, matcher)

    print("\n== cross-modal edge, fov=0.10 ==")
    pair, _ = synthesize_cross_modal_pair(1024, 0.10, 256, seed=0, rotation_deg=5.0, mode="edge")
    _run("edge", pair, matcher)


if __name__ == "__main__":
    main()
