"""LoFTR smoke test: load weights, run on a synthetic pair, fit homography."""

import time

import numpy as np

from cma.data import synthesize_pair
from cma.matchers import LoFTRMatcher
from cma.pipeline import register


def _apply_h(H, xy):
    ones = np.ones((xy.shape[0], 1))
    hom = np.concatenate([xy, ones], axis=1)
    proj = hom @ H.T
    return proj[:, :2] / proj[:, 2:3]


def main() -> None:
    print("constructing LoFTR (will download weights on first run) ...")
    t0 = time.perf_counter()
    matcher = LoFTRMatcher()
    print(f"  loaded in {time.perf_counter() - t0:.1f}s on {matcher.device}")

    print("\nsynthesizing pair fov=0.1 ...")
    pair, _ = synthesize_pair(
        source_size=1024, fov_ratio=0.10, target_size=256, seed=0, rotation_deg=5.0
    )

    print("running register() with pyramid + LoFTR ...")
    t0 = time.perf_counter()
    result = register(
        pair.source, pair.target, matcher=matcher, scale_ratio=pair.scale_ratio,
    )
    runtime = time.perf_counter() - t0

    pred = _apply_h(result.H_target_to_source, pair.gt.tgt_xy)
    err = np.linalg.norm(pred - pair.gt.src_xy, axis=1)
    print(f"\nn_tiles={result.n_tiles} n_correspondences={result.n_correspondences}")
    print(f"transform_family={result.transform.family}")
    print(f"mean_err={err.mean():.3f}px median_err={np.median(err):.3f}px runtime={runtime:.2f}s")


if __name__ == "__main__":
    main()
