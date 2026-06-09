import numpy as np
import pytest

from cma.data import synthesize_pair
from cma.pipeline import classical_register


def _apply_h(H, xy):
    ones = np.ones((xy.shape[0], 1))
    hom = np.concatenate([xy, ones], axis=1)
    proj = hom @ H.T
    return proj[:, :2] / proj[:, 2:3]


def test_classical_recovers_homography_no_refine():
    pair, _ = synthesize_pair(
        source_size=1024, fov_ratio=0.25, target_size=256, seed=0, rotation_deg=5.0
    )
    res = classical_register(pair.source, pair.target, refine_with_mi=False)
    pred = _apply_h(res.H_target_to_source, pair.gt.tgt_xy)
    err = np.linalg.norm(pred - pair.gt.src_xy, axis=1)
    assert err.mean() < 2.0
    assert res.n_correspondences > 8


def test_classical_mi_refinement_does_not_regress():
    pair, _ = synthesize_pair(
        source_size=1024, fov_ratio=0.25, target_size=256, seed=1, rotation_deg=4.0
    )
    res = classical_register(
        pair.source, pair.target, refine_with_mi=True, mi_max_iter=20
    )
    # Refinement may or may not trigger, but if it does, MI must improve.
    if res.refined:
        assert res.mi_refined >= res.mi_initial


def test_classical_raises_when_no_matches():
    flat_a = np.zeros((128, 128), dtype=np.float32)
    flat_b = np.zeros((128, 128), dtype=np.float32)
    with pytest.raises(RuntimeError):
        classical_register(flat_a, flat_b)
