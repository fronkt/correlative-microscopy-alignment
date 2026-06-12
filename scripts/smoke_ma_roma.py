"""MA-RoMa smoke: weight load + identity check vs plain RoMa.

The discriminating test is contrast inversion: plain roma_outdoor fails on
inverted pairs (cross-modal battery, results/README.md) while the
cross-modal-trained MatchAnything-RoMa should keep matching. Run both
variants on the same inverted pair and compare.
"""

import time

import numpy as np

from cma.data.synthetic import natural_source_image
from cma.matchers import RoMaMatcher
from cma.pipeline import classical_register


def inverted_pair(size: int = 512):
    src = natural_source_image(size)
    return src, 1.0 - src


def report(tag: str, matcher: RoMaMatcher) -> None:
    src, tgt = inverted_pair()
    corr = matcher.match(src, tgt)
    if len(corr) == 0:
        print(f"{tag}: 0 matches on inverted pair")
        return
    # identity-aligned pair: residual = distance between matched coords
    res = np.linalg.norm(corr.a_xy - corr.b_xy, axis=1)
    t0 = time.perf_counter()
    reg = classical_register(src, tgt, matcher=matcher)
    rt = time.perf_counter() - t0
    # registration error: how far the fitted transform moves true identity points
    pts = np.array([[64.0, 64.0], [448.0, 64.0], [256.0, 256.0], [64.0, 448.0]])
    ones = np.ones((len(pts), 1))
    proj = np.concatenate([pts, ones], 1) @ reg.H_target_to_source.T
    proj = proj[:, :2] / proj[:, 2:3]
    reg_err = np.linalg.norm(proj - pts, axis=1)
    print(f"{tag}: n={len(corr)}  med_match_residual={np.median(res):.1f}px  "
          f"reg_err(mean)={reg_err.mean():.2f}px  n_inliers={reg.transform.n_inliers}  "
          f"rt={rt:.1f}s")


def main() -> None:
    print("loading MA-RoMa (ma_outdoor) ...")
    t0 = time.perf_counter()
    ma = RoMaMatcher(variant="ma_outdoor")
    print(f"  loaded in {time.perf_counter() - t0:.1f}s on {ma.device}  name={ma.name}")

    # self-match sanity first
    src = natural_source_image(512)
    corr = ma.match(src, src)
    print(f"self-match: {len(corr)} pairs  "
          f"mean(a-b)={np.linalg.norm(corr.a_xy - corr.b_xy, axis=1).mean():.3f} px")

    print("\n--- contrast-inversion identity check ---")
    report("ma_roma ", ma)
    del ma
    plain = RoMaMatcher()
    report("roma    ", plain)


if __name__ == "__main__":
    main()
