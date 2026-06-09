"""Natural-source synthetic pairs route through the same harness."""

import numpy as np

from cma.data import synthesize_pair
from cma.data.synthetic import natural_source_image


def test_natural_source_image_shape_and_range():
    img = natural_source_image(512)
    assert img.shape == (512, 512)
    assert img.dtype == np.float32
    assert 0.0 <= img.min() and img.max() <= 1.0


def test_synthesize_pair_with_natural_source():
    src = natural_source_image(512)
    pair, H_gt = synthesize_pair(
        source_size=512, fov_ratio=0.2, target_size=128, seed=0,
        rotation_deg=5.0, source_image=src,
    )
    assert pair.source.shape == (512, 512)
    assert pair.target.shape == (128, 128)
    np.testing.assert_allclose(pair.source, src)
