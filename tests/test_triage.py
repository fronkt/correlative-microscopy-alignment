"""CPU tests for cma.triage (label-free triage study)."""

import numpy as np
import pytest

from cma import triage


def _img(h=40, w=50, c=None, seed=0):
    rng = np.random.default_rng(seed)
    shape = (h, w) if c is None else (h, w, c)
    return rng.random(shape).astype(np.float32)


@pytest.mark.parametrize("name", triage.INPUT_TRANSFORMS)
@pytest.mark.parametrize("channels", [None, 3])
def test_input_transforms_keep_geometry_and_range(name, channels):
    s, t = _img(60, 80, channels, 1), _img(30, 40, channels, 2)
    s2, t2 = triage.apply_input_transform(name, s, t)
    assert s2.shape[:2] == s.shape[:2] and t2.shape[:2] == t.shape[:2]
    for a in (s2, t2):
        assert np.isfinite(a).all() and a.min() >= 0.0 and a.max() <= 1.0


def test_invert_is_involution_and_leaves_source():
    s, t = _img(seed=3), _img(seed=4)
    s2, t2 = triage.apply_input_transform("invert", s, t)
    assert s2 is s
    np.testing.assert_allclose(1.0 - t2, t, atol=1e-6)


def test_histmatch_matches_reference_quantiles():
    rng = np.random.default_rng(5)
    s = rng.beta(5, 2, (64, 64)).astype(np.float32)
    t = rng.beta(2, 5, (64, 64)).astype(np.float32)
    s2, t2 = triage.apply_input_transform("histmatch", s, t)
    q = [0.1, 0.5, 0.9]
    np.testing.assert_allclose(np.quantile(t2, q), np.quantile(s2, q), atol=0.02)


def test_unknown_transform_raises():
    with pytest.raises(ValueError):
        triage.apply_input_transform("sharpen", _img(), _img())


def test_pair_seed_is_order_independent_and_distinct():
    assert triage.pair_seed(0, "a#1") == triage.pair_seed(0, "a#1")
    assert triage.pair_seed(0, "a#1") != triage.pair_seed(1, "a#1")
    assert triage.pair_seed(0, "a#1") != triage.pair_seed(0, "a#2")


def test_grid_points_span_image():
    p = triage.grid_points(11, 21, n=5)
    assert p.shape == (25, 2)
    assert p[:, 0].min() == 0 and p[:, 0].max() == 20 and p[:, 1].max() == 10


def test_transform_distance_translation():
    pts = triage.grid_points(10, 10)
    T = np.eye(3)
    T[0, 2] = 3.0
    T[1, 2] = 4.0
    assert triage.transform_distance(np.eye(3), T, pts) == pytest.approx(5.0)
    assert triage.transform_distance(T, T, pts) == 0.0


def test_agreement_prefers_the_consensus_and_handles_missing():
    pts = triage.grid_points(10, 10)
    def shift(dx):
        H = np.eye(3)
        H[0, 2] = dx
        return H
    Hs = {"a": shift(0), "b": shift(1), "c": shift(0.5), "far": shift(100), "dead": None}
    s = triage.agreement_scores(Hs, pts, voters=["a", "b", "c", "far", "dead"])
    assert s["dead"] == float("-inf")
    assert s["far"] < min(s["a"], s["b"], s["c"])
    assert s["c"] == max(s.values())  # the medoid of a, b, c


def test_auroc_basic_and_ties():
    assert triage.auroc(np.array([1, 2, 3, 4]), np.array([0, 0, 1, 1])) == 1.0
    assert triage.auroc(np.array([4, 3, 2, 1]), np.array([0, 0, 1, 1])) == 0.0
    assert triage.auroc(np.array([1, 1, 1, 1]), np.array([0, 1, 0, 1])) == 0.5
    assert np.isnan(triage.auroc(np.array([1, 2]), np.array([1, 1])))
    # -inf scores (failed runs) rank lowest
    assert triage.auroc(np.array([-np.inf, 1.0, 2.0]), np.array([0, 1, 1])) == 1.0


def test_auroc_matches_sklearn_formula_on_random_data():
    rng = np.random.default_rng(0)
    s = rng.normal(size=200)
    y = (s + rng.normal(size=200) > 0).astype(int)
    # brute-force pairwise definition
    pos, neg = s[y == 1], s[y == 0]
    brute = ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg))
    assert triage.auroc(s, y) == pytest.approx(brute)


def test_cluster_bootstrap_resamples_whole_clusters():
    clusters = np.array([0, 0, 1, 1, 2, 2])
    seen = []
    boot = triage.cluster_bootstrap(lambda idx: seen.append(len(idx)) or float(len(idx)),
                                    clusters, B=50, seed=1)
    assert len(boot) == 50 and all(n == 6 for n in seen)


def test_mcnemar_exact():
    assert triage.mcnemar_exact(0, 0) == 1.0
    assert triage.mcnemar_exact(10, 0) == pytest.approx(2 * 0.5**10)
    assert triage.mcnemar_exact(5, 5) == 1.0


def test_youden_and_coverage_helpers():
    s = np.array([0.1, 0.2, 0.8, 0.9])
    y = np.array([0, 0, 1, 1])
    assert triage.youden_cutoff(s, y) == pytest.approx(0.8)
    assert triage.accepted_success(s, y, 0.5) == 1.0
    assert triage.accepted_success(s, y, 1.0) == 0.5
    assert triage.aurc(s, y) < triage.aurc(-s, y)
