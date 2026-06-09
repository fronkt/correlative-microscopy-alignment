"""RoMa smoke: load + match a synthetic pair + classical_register on natural pair."""

import time

import numpy as np

from cma.data import synthesize_pair
from cma.data.synthetic import natural_source_image
from cma.matchers import RoMaMatcher
from cma.pipeline import classical_register, register


def _apply_h(H, xy):
    ones = np.ones((xy.shape[0], 1))
    hom = np.concatenate([xy, ones], axis=1)
    proj = hom @ H.T
    return proj[:, :2] / proj[:, 2:3]


def _err(result_H, gt):
    pred = _apply_h(result_H, gt.tgt_xy)
    return np.linalg.norm(pred - gt.src_xy, axis=1)


def main() -> None:
    print("loading RoMa (outdoor) ...")
    t0 = time.perf_counter()
    matcher = RoMaMatcher()
    print(f"  loaded in {time.perf_counter() - t0:.1f}s on {matcher.device}")

    # 1. Self-match sanity check
    src = natural_source_image(512)
    corr = matcher.match(src, src)
    if len(corr):
        print(f"\nself-match: {len(corr)} pairs  "
              f"mean(a-b)={np.linalg.norm(corr.a_xy - corr.b_xy, axis=1).mean():.3f} px")
    else:
        print("self-match: 0 pairs — wrapper bug")

    # 2. Natural-image synthetic pair, FOV=0.10
    src_full = natural_source_image(1024)
    pair, _ = synthesize_pair(
        source_size=1024, fov_ratio=0.10, target_size=256, seed=0,
        rotation_deg=5.0, source_image=src_full,
    )

    print("\nclassical_register (no pyramid) ...")
    t0 = time.perf_counter()
    cres = classical_register(pair.source, pair.target, matcher=matcher)
    rt = time.perf_counter() - t0
    err = _err(cres.H_target_to_source, pair.gt)
    print(f"  n_corr={cres.n_correspondences}  family={cres.transform.family}  "
          f"mean_err={err.mean():.3f}px  median={np.median(err):.3f}px  rt={rt:.2f}s")

    print("\nregister via pyramid ...")
    t0 = time.perf_counter()
    rres = register(pair.source, pair.target, matcher=matcher, scale_ratio=pair.scale_ratio)
    rt = time.perf_counter() - t0
    err = _err(rres.H_target_to_source, pair.gt)
    print(f"  n_tiles={rres.n_tiles}  n_corr={rres.n_correspondences}  "
          f"mean_err={err.mean():.3f}px  median={np.median(err):.3f}px  rt={rt:.2f}s")


if __name__ == "__main__":
    main()
