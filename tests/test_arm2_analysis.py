"""CPU tests for scripts/analyze_arm2.py on tiny synthetic frames (no images, no matchers, no Arm 2 data)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_arm2 as A  # noqa: E402

IC = A.VOTERS.index(A.H21_CAND)
IB = A.VOTERS.index(A.BEST_SINGLE)


# ------------------------------------------------------------------ constants carried over from the prereg
def test_carried_constants_and_order():
    assert A.S1_CUT == 0.1711 and A.BEST_SINGLE == ("ma_roma", "pyramid_v2", "none")
    assert len(A.VOTERS) == 15 and A.VOTERS[0] == ("sift", "direct", "none")
    assert A.VOTERS[:7] == [("sift", "direct", "none"), ("loftr", "direct", "none"), ("roma", "direct", "none"),
                            ("ma_roma", "direct", "none"), ("matchanything", "direct", "none"),
                            ("roma", "pyramid_v2", "none"), ("ma_roma", "pyramid_v2", "none")]


# ------------------------------------------------------------------ R: max n_inliers, ties to first listed
def test_R_tie_goes_to_first_listed():
    ninl = np.full((1, 15), 3.0)
    ninl[0, 4], ninl[0, 9] = 50.0, 50.0                 # tie between candidate 4 and 9
    succ = np.zeros((1, 15), bool)
    succ[0, 9] = True                                   # only the LATER tied candidate is right
    assert not pick_R(ninl, succ)[0]                    # first-listed (4) is picked -> failure
    succ[0, 4], succ[0, 9] = True, False
    assert pick_R(ninl, succ)[0]


def test_R_picks_the_max_and_all_failed_row_fails():
    ninl = np.array([[1.0, 40.0] + [2.0] * 13, [-np.inf] * 15])
    succ = np.zeros((2, 15), bool)
    succ[0, 1] = True
    succ[1, 0] = True                                   # would "win" by argmax over -inf ties, but every run failed
    out = pick_R(ninl, succ)
    assert out[0] and not out[1]


def pick_R(ninl, succ):
    return A.pick_success(ninl, succ)


def test_pick_by_S1_ties_to_first_and_nan_is_minus_inf():
    s1 = np.array([[0.2, 0.5, 0.5] + [0.1] * 12])
    succ = np.zeros((1, 15), bool)
    succ[0, 2] = True
    assert not A.pick_success(s1, succ)[0]
    s1n = s1.copy()
    s1n[0, 1] = np.nan                                  # NaN never wins
    assert A.pick_success(s1n, succ)[0]


# ------------------------------------------------------------------ one-sided exact McNemar, direction
def test_mcnemar_direction():
    better = np.array([True] * 8 + [False] * 4)
    other = np.array([False] * 8 + [False] * 4)
    t = A.paired_test(better, other)                    # 8 discordant, all for "better"
    assert t["only_better"] == 8 and t["only_other"] == 0
    assert t["p_one_sided"] == pytest.approx(1 / 256) and t["supported"]
    rev = A.paired_test(other, better)                  # wrong direction: p = 1, never supported
    assert rev["p_one_sided"] == 1.0 and not rev["supported"]
    assert rev["p_two_sided"] == pytest.approx(2 / 256)  # the two-sided p is small, but the direction is wrong


def test_mcnemar_needs_p_below_005_and_more_successes():
    better = np.array([True] * 5 + [False] * 3)
    other = np.array([False] * 5 + [True] * 3)          # 5 vs 3 discordant: p = 0.363
    t = A.paired_test(better, other)
    assert t["p_one_sided"] == pytest.approx(sum(__import__("math").comb(8, i) for i in range(5, 9)) / 256)
    assert not t["supported"]
    assert A.mcnemar_one_sided(0, 0) == 1.0


# ------------------------------------------------------------------ H2-1 untestable rule and criteria
def test_h21_untestable_below_8_per_class():
    rng = np.random.default_rng(0)
    y7 = np.array([True] * 7 + [False] * 30)
    r = A.h21(rng.random(len(y7)), y7, np.arange(len(y7)), B=50)
    assert r["status"] == "UNTESTABLE" and r["supported"] is None and r["ci95"] is None
    y7f = np.array([True] * 30 + [False] * 7)
    assert A.h21(rng.random(len(y7f)), y7f, np.arange(len(y7f)), B=50)["status"] == "UNTESTABLE"
    y88 = np.array([True] * 8 + [False] * 8)
    assert A.h21(rng.random(16), y88, np.arange(16), B=50)["status"] != "UNTESTABLE"


def test_h21_supported_with_perfect_score_and_not_with_noise():
    y = np.array([True] * 12 + [False] * 12)
    clusters = np.repeat(np.arange(12), 2)
    s = np.where(y, 1.0, 0.0) + np.linspace(0, 0.01, 24)
    r = A.h21(s, y, clusters, B=500)
    assert r["status"] == "SUPPORTED" and r["auroc"] == 1.0 and r["ci95"][0] > 0.5
    inv = A.h21(-s, y, clusters, B=500)
    assert inv["status"] == "NOT SUPPORTED" and inv["auroc"] == 0.0


def test_h22_no_accepted_pairs_and_positive_difference():
    y = np.array([True] * 10 + [False] * 10)
    none = A.h22(np.full(20, 0.05), y, np.arange(20), B=100)
    assert none["n_accepted"] == 0 and not none["supported"]
    s = np.where(y, 0.9, 0.0)                           # accepted = exactly the successes: rate 1.0 vs base 0.5
    r = A.h22(s, y, np.repeat(np.arange(10), 2), B=500)
    assert r["accepted_rate"] == 1.0 and r["difference"] == pytest.approx(0.5) and r["supported"]
    edge = A.h22(np.full(20, 0.1711), y, np.arange(20), B=100)   # cut-off is inclusive
    assert edge["n_accepted"] == 20


# ------------------------------------------------------------------ thresholds: 20 px and 1% of the source diagonal
def _arrays(n=6, mu=25.0):
    mu_a = np.full((n, 15), mu)
    return mu_a, np.full((n, 15), 0.3), np.full((n, 15), 10.0)


def test_one_percent_of_diagonal_threshold():
    mu, s1, ni = _arrays(mu=25.0)
    cl = np.arange(6)
    r20 = A.hypotheses(mu, s1, ni, np.full(6, A.TH_PX), cl, B=20)
    assert r20["SR"]["oracle_best_of_15"] == 0          # 25 px > 20 px
    diag = np.hypot(2000.0, 1500.0)                      # 2500 -> 1% = 25 px
    rp = A.hypotheses(mu, s1, ni, A.DIAG_FRAC * np.full(6, diag), cl, B=20)
    assert A.DIAG_FRAC * diag == pytest.approx(25.0)
    assert rp["SR"]["oracle_best_of_15"] == 6 and rp["SR"]["R"] == 6     # mu = 25 <= 25 (inclusive)
    small = A.hypotheses(mu, s1, ni, A.DIAG_FRAC * np.full(6, 2000.0), cl, B=20)   # 1% of 2000 = 20 px < 25
    assert small["SR"]["oracle_best_of_15"] == 0


def test_oracle_and_hypothesis_bundle():
    n = 10
    mu = np.full((n, 15), 500.0)
    mu[:, 0] = np.where(np.arange(n) < 4, 5.0, 500.0)    # SIFT succeeds on 4 pairs
    mu[:, 6] = np.where(np.arange(n) < 8, 5.0, 500.0)    # MA-RoMa pyramid_v2 (best single) on 8
    ni = np.zeros((n, 15))
    ni[:, 6] = 100.0                                     # R always picks MA-RoMa pyramid_v2
    s1 = np.zeros((n, 15))
    s1[:, 0] = 0.9                                       # pick-by-S1 always picks SIFT (sparse-matcher bias)
    h = A.hypotheses(mu, s1, ni, np.full(n, 20.0), np.arange(n), B=50)
    assert h["SR"] == dict(R=8, pick_by_S1=4, best_single=8, ma_roma_direct=0, oracle_best_of_15=8, n=10)
    assert h["H2-3"]["only_better"] == 4 and h["H2-3"]["only_other"] == 0
    assert h["H2-4"]["only_better"] == 0 and not h["H2-4"]["supported"]


# ------------------------------------------------------------------ GT gate
def test_gate_errors_in_target_pixels():
    H = np.diag([2.0, 2.0, 1.0])                         # target -> source: source px are 2x target px
    left = np.array([[10.0, 10.0]])
    right = np.array([[26.0, 20.0]])                     # GT maps left to (20, 20): 6 source px off = 3 target px
    e_tgt, e_src = A.gate_errors(H, left, right)
    assert e_src[0] == pytest.approx(6.0) and e_tgt[0] == pytest.approx(3.0)


def test_gate_pass_fail_boundary_and_unverified_type():
    errors = {"p1": np.array([4.0, 10.0, 10.0]), "p2": np.array([10.0, 30.0]),   # T1: median of all 5 = 10 -> pass
              "p3": np.array([11.0, 12.0, 13.0])}                                 # T2: median 12 -> fail
    types = {"p1": "T1", "p2": "T1", "p3": "T2", "p4": "T3"}                      # T3: never clicked
    g = A.evaluate_gate(errors, types, ["T1", "T2", "T3"], limit=10.0)
    assert g["T1"]["passes"] and g["T1"]["median_target_px"] == 10.0
    assert not g["T2"]["passes"]
    assert not g["T3"]["passes"] and "cannot be verified" in g["T3"]["reason"]


def test_gate_pools_points_not_pair_medians():
    errors = {"a": np.array([1.0, 1.0]), "b": np.array([50.0, 50.0, 50.0])}   # pooled median 50, pair medians 1 and 50
    g = A.evaluate_gate(errors, {"a": "T", "b": "T"}, ["T"], limit=10.0)
    assert g["T"]["median_target_px"] == 50.0 and not g["T"]["passes"]


# ------------------------------------------------------------------ end-to-end on in-memory frames
def _manifest(rows):
    df = pd.DataFrame(rows, columns=["pair_id", "component", "cluster", "subclass", "h_s", "w_s", "overlay_src", "overlay_tgt"])
    df["diag_src"] = np.hypot(df.h_s, df.w_s)
    df["group"] = "g"
    return df


def _cand(pair_ids, mu_of):
    recs = []
    for p in pair_ids:
        for m in A.GT_MODES:
            recs.append(dict(pair_id=p, pool="gt", backbone="gt", mode=m, transform="none", seed="0", status="ok",
                             mu_ed="1.0", n_inliers="25", n_matches="25", error=""))
        for k, (b, mo, t) in enumerate(A.VOTERS):
            mu = mu_of(p, k)
            recs.append(dict(pair_id=p, pool="core" if t == "none" else "transform", backbone=b, mode=mo, transform=t,
                             seed="0", status="ok" if np.isfinite(mu) else "failed", mu_ed=str(mu) if np.isfinite(mu) else "",
                             n_inliers="10" if np.isfinite(mu) else "", n_matches="20" if np.isfinite(mu) else "", error=""))
    return pd.DataFrame(recs).astype(str)


def test_end_to_end_gate_excludes_failing_type_and_diag_threshold():
    man = _manifest([("P1#0", "A", "c1", "A-x", 1000, 1000, "none", "none"),
                     ("N1#0", "NIST", "n1", "GOOD", 2000, 1500, "none", "none"),
                     ("N2#0", "NIST", "n2", "BAD", 2000, 1500, "none", "none")])
    cand = _cand(list(man.pair_id), lambda p, k: 25.0 if p.startswith("N") else 5.0)   # NIST mu = 25 px
    gate = dict(passing_types=["GOOD"], failing_types=["BAD"], types={})
    S = A.analyse(cand, man, gate, B=20, csv_path="x", man_path="y", dry_run=False)
    assert S["row_counts"]["ok"] and S["row_counts"]["expected"] == 3 * A.ROWS_PER_PAIR
    assert S["primary_n"] == 2 and S["primary_n_N"] == 1 and S["n_types_excluded"] == ["BAD"]      # BAD pairs excluded
    assert S["primary_20px"]["SR"]["oracle_best_of_15"] == 1                                      # P1 only
    assert S["primary_1pct"]["SR"]["oracle_best_of_15"] == 2                                      # NIST 25 <= 1% of 2500
    assert S["gt_ceiling"]["homography"] == 2
    dry = A.analyse(cand, man, None, B=20, csv_path="x", man_path="y", dry_run=True)
    assert dry["primary_n"] == 3


def test_row_count_mismatch_and_control_rows_are_flagged():
    man = _manifest([("P1#0", "A", "c1", "A-x", 1000, 1000, "none", "none")])
    cand = _cand(list(man.pair_id), lambda p, k: 5.0)
    extra = cand.iloc[[2]].copy()
    extra["pool"], extra["seed"] = "control", "1"
    S = A.analyse(pd.concat([cand, extra]), man, None, B=10, csv_path="x", man_path="y", dry_run=True)
    assert S["row_counts"]["stray"] == 1 and not S["row_counts"]["ok"]
    short = A.analyse(cand.iloc[:-1], man, None, B=10, csv_path="x", man_path="y", dry_run=True)
    assert not short["row_counts"]["ok"]


def test_failed_run_counts_as_failure_and_missing_rows_too():
    man = _manifest([("P1#0", "A", "c1", "A-x", 1000, 1000, "none", "none"),
                     ("P2#0", "A", "c2", "A-x", 1000, 1000, "none", "none")])
    cand = _cand(list(man.pair_id), lambda p, k: 5.0 if p == "P1#0" else float("inf"))   # P2: every candidate failed
    S = A.analyse(cand, man, None, B=10, csv_path="x", man_path="y", dry_run=True)
    assert S["primary_20px"]["SR"]["R"] == 1 and S["primary_20px"]["SR"]["oracle_best_of_15"] == 1
    gone = A.analyse(cand[cand.pair_id != "P2#0"], man, None, B=10, csv_path="x", man_path="y", dry_run=True)
    assert gone["primary_20px"]["SR"]["R"] == 1 and not gone["row_counts"]["ok"]


# ------------------------------------------------------------------ the script refuses without the hand-check
def test_main_refuses_without_handcheck_unless_dry_run(tmp_path, monkeypatch):
    man = _manifest([("P1#0", "A", "c1", "A-x", 1000, 1000, "none", "none")])
    mp = tmp_path / "manifest.csv"
    man.to_csv(mp, index=False)
    cp = tmp_path / "cand.csv"
    _cand(list(man.pair_id), lambda p, k: 5.0).to_csv(cp, index=False)
    base = ["analyze_arm2.py", "--manifest", str(mp), "--candidates", str(cp), "--out-dir", str(tmp_path / "out"),
            "--handcheck", str(tmp_path / "missing.csv"), "--boot", "10"]
    monkeypatch.setattr(sys, "argv", base)
    with pytest.raises(SystemExit) as e:
        A.main()
    assert "REFUSING" in str(e.value) and not (tmp_path / "out" / "arm2_report.md").exists()
    monkeypatch.setattr(sys, "argv", base + ["--no-gate-dry-run"])
    A.main()
    assert (tmp_path / "out" / "arm2_DRYRUN_report.md").read_text(encoding="utf-8").lstrip().startswith("# Arm 2")
    assert "DRY RUN" in (tmp_path / "out" / "arm2_DRYRUN_report.md").read_text(encoding="utf-8")
    assert not (tmp_path / "out" / "arm2_report.md").exists()
