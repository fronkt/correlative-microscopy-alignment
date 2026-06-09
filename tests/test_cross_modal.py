"""Cross-modal synthesizer preserves geometry; only the target's intensities change."""

import numpy as np
import pytest

from cma.data import synthesize_cross_modal_pair, synthesize_pair


@pytest.mark.parametrize("mode", ["invert", "gamma", "edge", "smooth", "stack"])
def test_cross_modal_preserves_keypoints(mode):
    base, H_base = synthesize_pair(
        source_size=512, fov_ratio=0.2, target_size=128, seed=0, rotation_deg=3.0
    )
    cm, H_cm = synthesize_cross_modal_pair(
        source_size=512, fov_ratio=0.2, target_size=128, seed=0, rotation_deg=3.0, mode=mode
    )
    # Geometry must be identical
    np.testing.assert_allclose(H_base, H_cm)
    np.testing.assert_allclose(base.gt.src_xy, cm.gt.src_xy)
    np.testing.assert_allclose(base.gt.tgt_xy, cm.gt.tgt_xy)
    # Target intensities must differ
    assert not np.allclose(base.target, cm.target, atol=1e-3)
    # Output stays in [0, 1]
    assert cm.target.min() >= 0.0 and cm.target.max() <= 1.0


def test_cross_modal_invert_is_exact_complement():
    base, _ = synthesize_pair(
        source_size=256, fov_ratio=0.2, target_size=64, seed=1, rotation_deg=0.0, noise_sigma=0.0
    )
    cm, _ = synthesize_cross_modal_pair(
        source_size=256, fov_ratio=0.2, target_size=64, seed=1, rotation_deg=0.0,
        noise_sigma=0.0, mode="invert",
    )
    np.testing.assert_allclose(cm.target, 1.0 - base.target, atol=1e-6)
