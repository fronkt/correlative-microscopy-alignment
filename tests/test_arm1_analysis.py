"""CPU tests for scripts/analyze_arm1.py (lock definition, McNemar direction, sham >10% rule), tiny synthetic frames."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_arm1 as A  # noqa: E402

from cma.triage import grid_points  # noqa: E402

GRID = grid_points(1000, 1000, 5)


def _shift(dx, dy=0.0):
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1.0]])


# ------------------------------------------------------------------ lock definition


def test_lock_requires_all_three_conditions():
    assert A.is_lock(np.eye(3), 100.0, 300.0, GRID)                # identity, wrong, GT far -> lock
    assert A.is_lock(_shift(10), 100.0, 300.0, GRID)               # 10 px from identity
    assert not A.is_lock(_shift(25), 100.0, 300.0, GRID)           # 25 px from identity: not near-identity
    assert not A.is_lock(np.eye(3), 15.0, 300.0, GRID)             # success (mu <= 20): not a lock
    assert not A.is_lock(np.eye(3), 100.0, 30.0, GRID)             # GT itself near identity: not a lock
    assert not A.is_lock(None, 100.0, 300.0, GRID)                 # failed run
    assert not A.is_lock(np.eye(3), float("nan"), 300.0, GRID)


def test_lock_boundaries_are_strict():
    assert not A.is_lock(_shift(20), 100.0, 300.0, GRID)           # exactly 20 px away: not "< 20"
    assert not A.is_lock(np.eye(3), 20.0, 300.0, GRID)             # mu_ed must be > 20
    assert not A.is_lock(np.eye(3), 100.0, 40.0, GRID)             # GT must be > 40


def test_parse_H_and_s1():
    assert A.parse_H("") is None and A.parse_H(None) is None
    assert A.parse_H(json.dumps(list(np.eye(3).ravel()))).shape == (3, 3)
    assert A.s1_score(5, 20, True) == 0.25
    assert A.s1_score(5, 20, False) == float("-inf")


# ------------------------------------------------------------------ McNemar direction


def test_directional_p_matches_binomial_tail():
    assert A.mcnemar_directional(0, 0) == 1.0
    assert abs(A.mcnemar_directional(5, 0) - 1 / 32) < 1e-12
    assert A.mcnemar_directional(0, 5) == 1.0
    assert abs(A.mcnemar_directional(3, 1) - (4 + 1) / 16) < 1e-12


def test_success_direction_more_when_cropped():
    base = np.array([0] * 10 + [1] * 5, bool)
    treat = base.copy()
    treat[:8] = True                                  # 8 discordant pairs, all gained by cropping
    t = A.paired_test(base, treat, "more")
    assert (t["only_treat"], t["only_base"]) == (8, 0) and t["direction_ok"]
    assert t["p_one_sided"] < 0.05 and t["supported"]
    rev = A.paired_test(treat, base, "more")          # the same data reversed: significant but the WRONG way
    assert rev["p_two_sided"] < 0.05 and not rev["direction_ok"] and not rev["supported"] and rev["p_one_sided"] > 0.9


def test_lock_and_false_accept_direction_is_fewer():
    base = np.array([1] * 8 + [0] * 4, bool)          # locks uncropped
    treat = np.array([0] * 8 + [0] * 4, bool)         # all gone when cropped
    t = A.paired_test(base, treat, "fewer")
    assert t["direction_ok"] and t["supported"] and t["n_pos"] == 8
    more = A.paired_test(treat, base, "fewer")        # locks appearing: not supported
    assert not more["supported"]


def test_holm_monotone_and_capped():
    adj = A.holm([0.01, 0.04, 0.03])
    assert adj[0] == 0.03 and abs(adj[2] - 0.06) < 1e-12 and abs(adj[1] - 0.06) < 1e-12
    assert all(0 <= a <= 1 for a in A.holm([0.9, 0.8, 0.7]))


# ------------------------------------------------------------------ sham > 10 % rule


def _sham_frame(n, n_changed):
    succ_base = np.zeros(n, bool)
    succ_treat = succ_base.copy()
    succ_treat[:n_changed] = True
    return pd.DataFrame(dict(cfg="roma|direct", succ_base=succ_base, succ_treat=succ_treat))


def test_sham_rule_more_than_ten_percent():
    assert not A.analyse_sham(_sham_frame(80, 8))["crop_itself_matters"]     # exactly 10 %: not "more than"
    assert A.analyse_sham(_sham_frame(80, 9))["crop_itself_matters"]         # 11.25 %
    assert not A.analyse_sham(_sham_frame(80, 0))["crop_itself_matters"]


def test_sham_counts_both_directions():
    n = 80
    b = np.zeros(n, bool)
    t = np.zeros(n, bool)
    b[:5] = True                                       # 5 lost
    t[5:10] = True                                     # 5 gained
    r = A.analyse_sham(pd.DataFrame(dict(cfg="x", succ_base=b, succ_treat=t)))
    assert r["discordant"] == 10 and r["successes_uncropped"] == r["successes_sham"] and r["crop_itself_matters"]


def test_ha2_reported_as_general_when_sham_trips():
    rows = []
    for i in range(12):
        for bb, mo in A.CONFIGS:
            rows.append(dict(pair_id=f"p{i}", backbone=bb, mode=mo, cfg=f"{bb}|{mo}", succ_base=False, succ_treat=True))
    P = pd.DataFrame(rows)
    ok = A.analyse_ha2(P, sham_flag=False)["primary"]
    assert ok["supported"] and ok["reported_as"] == "supported"
    gen = A.analyse_ha2(P, sham_flag=True)["primary"]
    assert "NOT attributed to the overlay" in gen["reported_as"]
    assert all(v["supported"] for v in A.analyse_ha2(P, False)["secondary"].values())


def test_false_accept_indicator_uses_fixed_cutoff():
    fa = A.fa_indicator(np.array([0.1711, 0.17, 0.5, -np.inf]), np.array([50.0, 50.0, 5.0, np.inf]))
    assert fa.tolist() == [True, False, False, False]
