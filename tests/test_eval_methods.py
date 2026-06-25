"""Interface/plumbing tests for the TTA eval-method factories.

The RoMa-coupled execution paths are validated on the box; here we check the
registry, factory construction, and MethodResult plumbing import-cleanly.
"""

from __future__ import annotations

import numpy as np
import pytest

from cma.eval.methods import (
    LADDER_BASELINES,
    _smoke_method_result,
    assert_ladder_known,
    matcher_method,
)
from cma.eval.sweep import SweepConfig


def test_ladder_registry_membership():
    assert_ladder_known("pyramid_only")  # no raise
    assert_ladder_known("tta_both")
    with pytest.raises(KeyError):
        assert_ladder_known("does_not_exist")


def test_ladder_has_the_pivots():
    # the two make-or-break baselines must be present by name
    assert "pyramid_only" in LADDER_BASELINES
    assert "supervised_ft" in LADDER_BASELINES
    assert "dmp" in LADDER_BASELINES  # nearest prior


def test_matcher_method_constructs_callable():
    # construction must not touch the matcher (no match() until a pair arrives)
    method = matcher_method(object(), SweepConfig(), use_pyramid=False)
    assert callable(method)


def test_method_result_plumbing():
    r = _smoke_method_result()
    np.testing.assert_array_equal(r.H_target_to_source, np.eye(3))
    assert r.n_tiles == 1 and r.family == "identity"
