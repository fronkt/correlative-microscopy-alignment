"""Fill paper/cjsj/paper.template.md from results/triage/summary.json → paper/cjsj/paper.md.

Placeholders are {{name}}. Plain names come from NUMBERS below, each a function of the summary, so every
number in the paper traces to summary.json. {{ABSTRACT_RESULTS}} and {{RESULTS_AND_DISCUSSION}} are
filled from paper/cjsj/results_section.template.md, which may itself use {{name}} placeholders.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/triage/summary.json"
TEMPLATE = ROOT / "paper/cjsj/paper.template.md"
RESULTS = ROOT / "paper/cjsj/results_section.template.md"
OUT = ROOT / "paper/cjsj/paper.md"


def sgn(x: float) -> str:
    """Signed value with a true minus sign (U+2212), as the CJSJ template asks."""
    return format(x, "+.2f").replace("-", "−")


def pct(x: float, nd: int = 0) -> str:
    return f"{100 * x:.{nd}f} %"


def numbers(s: dict) -> dict[str, str]:
    h1, h2, h3, tri = s["H1"], s["H2"], s["H3"], s["triage"]
    tr = h1["transfer"]
    a20 = h1["auroc_S1_at_20"]
    n = s["n_pairs"]
    sel = {k: h3[f"select_{k}"] for k in ("S1", "S2", "S3")}
    return {
        "n_pairs": str(n), "n_scenes": str(s["n_scenes"]), "n_subclasses": str(s["n_subclasses"]),
        "n_voters": str(len(s["voters"])),
        "ceiling20": str(s["ceiling"]["homography_le_20"]),
        "auc_s1": f"{a20['auroc']:.2f}", "auc_s1_lo": f"{a20['lo']:.2f}", "auc_s1_hi": f"{a20['hi']:.2f}",
        "n_success_primary": str(a20["n_success"]),
        "auc_s1_roma": f"{h1['roma_auroc_S1_at_20']['auroc']:.2f}",
        "auc_s1_roma_lo": f"{h1['roma_auroc_S1_at_20']['lo']:.2f}",
        "auc_s1_roma_hi": f"{h1['roma_auroc_S1_at_20']['hi']:.2f}",
        "auc_s1_8": f"{h1['auroc_S1_at_8']['auroc']:.2f}",
        "cutoff": f"{tr['cutoff']:.3f}", "design_pairs": str(tr["design_pairs"]), "held_pairs": str(tr["heldout_pairs"]),
        "held_base": pct(tr["heldout_base_rate"]), "held_acc_n": str(tr["heldout_accepted"]),
        "held_acc_succ": pct(tr["heldout_accepted_success"]), "held_recall": pct(tr["heldout_recall"]),
        "held_far": pct(tr["heldout_false_accept_rate"]),
        "held_diff": f"{100 * tr['diff']:.0f}", "held_diff_lo": f"{100 * tr['diff_ci']['lo']:.0f}",
        "held_diff_hi": f"{100 * tr['diff_ci']['hi']:.0f}",
        "auc_s2": f"{h2['primary_auroc_S2']['auroc']:.2f}", "auc_s3": f"{h2['primary_auroc_S3']['auroc']:.2f}",
        "d_s3": sgn(h2['primary_delta_S3_minus_S1']['delta']),
        "d_s3_lo": sgn(h2['primary_delta_S3_minus_S1']['lo']), "d_s3_hi": sgn(h2['primary_delta_S3_minus_S1']['hi']),
        "pool_s1": f"{h2['pooled_auroc_S1']['auroc']:.2f}", "pool_s2": f"{h2['pooled_auroc_S2']['auroc']:.2f}",
        "pool_s3": f"{h2['pooled_auroc_S3']['auroc']:.2f}",
        "pd_s2": sgn(h2['pooled_delta_S2_minus_S1']['delta']),
        "pd_s2_lo": sgn(h2['pooled_delta_S2_minus_S1']['lo']), "pd_s2_hi": sgn(h2['pooled_delta_S2_minus_S1']['hi']),
        "pd_s3": sgn(h2['pooled_delta_S3_minus_S1']['delta']),
        "pd_s3_lo": sgn(h2['pooled_delta_S3_minus_S1']['lo']), "pd_s3_hi": sgn(h2['pooled_delta_S3_minus_S1']['hi']),
        "best_single": h3["best_single"], "best_single_n": str(round(h3["best_single_sr20"] * n)),
        "oracle_n": str(round(h3["oracle_sr20"] * n)),
        **{f"sel_{k}_n": str(round(v["sr20"] * n)) for k, v in sel.items()},
        **{f"sel_{k}_won": str(v["won"]) for k, v in sel.items()},
        **{f"sel_{k}_lost": str(v["lost"]) for k, v in sel.items()},
        **{f"sel_{k}_p": f"{v['mcnemar_p']:.3g}" for k, v in sel.items()},
        "rerun_bar_n": str(round(h3["rerun_bar"] * n)),
        **{f"tri_{lab}_{c}": pct(tri[lab][f"accepted_success_at_{c}"]["value"])
           for lab in ("primary_S1", "primary_S3", "selected_S3") for c in (25, 50, 75)},
        **{f"tri_{lab}_{c}_lo": pct(tri[lab][f"accepted_success_at_{c}"]["lo"])
           for lab in ("primary_S1", "primary_S3", "selected_S3") for c in (25, 50, 75)},
        **{f"tri_{lab}_{c}_hi": pct(tri[lab][f"accepted_success_at_{c}"]["hi"])
           for lab in ("primary_S1", "primary_S3", "selected_S3") for c in (25, 50, 75)},
        **{f"tri_{lab}_base": pct(tri[lab]["base_rate"]) for lab in ("primary_S1", "selected_S3")},
        "aurc": f"{tri['primary_S1']['aurc']:.2f}", "aurc_lo": f"{tri['primary_S1']['aurc_ci']['lo']:.2f}",
        "aurc_hi": f"{tri['primary_S1']['aurc_ci']['hi']:.2f}", "aurc_random": f"{1 - tri['primary_S1']['base_rate']:.2f}",
        "auc_s1_10": f"{h1['auroc_S1_at_10']['auroc']:.2f}", "auc_s1_10_lo": f"{h1['auroc_S1_at_10']['lo']:.2f}",
        "auc_s1_10_hi": f"{h1['auroc_S1_at_10']['hi']:.2f}",
        "auc_s1_8_lo": f"{h1['auroc_S1_at_8']['lo']:.2f}", "auc_s1_8_hi": f"{h1['auroc_S1_at_8']['hi']:.2f}",
        "d_s2": sgn(h2['primary_delta_S2_minus_S1']['delta']),
        "d_s2_lo": sgn(h2['primary_delta_S2_minus_S1']['lo']), "d_s2_hi": sgn(h2['primary_delta_S2_minus_S1']['hi']),
        **{f"sel_{k}_gain_lo": f"{100 * v['gain_ci']['lo']:.0f}".replace("-", "−") for k, v in sel.items()},
        **{f"sel_{k}_gain_hi": f"{100 * v['gain_ci']['hi']:.0f}".replace("-", "−") for k, v in sel.items()},
        "rerun_roma_n": str(round(h3["rerun_control"]["roma"]["gain"] * n)),
        "rerun_ma_n": str(round(h3["rerun_control"]["ma_roma"]["gain"] * n)),
        "design_prec": pct(tr["design_accepted_success"]),
    }


def fill(text: str, nums: dict[str, str]) -> str:
    def rep(m):
        k = m.group(1)
        if k not in nums:
            raise KeyError(f"unknown placeholder {{{{{k}}}}}")
        return nums[k]
    return re.sub(r"\{\{(\w+)\}\}", rep, text)


def exploratory(x: dict, n: int) -> dict[str, str]:
    e1, e2, e3, e4 = x["E1"], x["E2"], x["E3"], x["E4"]
    return {
        "ex_none": str(e1["pairs_no_candidate_succeeds"]), "ex_oracle_only": str(e1["oracle_only_pairs"]),
        "ex_oracle_le2": str(e1["oracle_only_with_at_most_2_successes"]),
        "ex_oracle_gt10": str(e1["oracle_only_best_error_over_10px"]),
        "ex_flip": str(e1["ma_roma_rerun_flip"]), "ex_mixed": str(e2["mixed_pairs"]),
        "ex_within": f"{e2['within_pair_auroc_median']:.2f}",
        "ex_sift_picks": str(e3["sift_picks"]), "ex_lost": str(e3["lost"]), "ex_lost_sift": str(e3["lost_sift_picks"]),
        "ex_sift_med": f"{e3['sift_S1_median']:.2f}", "ex_dense_med": f"{e3['dense_S1_median']:.2f}",
        "ex_dense_k": str(e4["n_candidates"]), "ex_dense_n": str(e4["n_ok"]), "ex_dense_won": str(e4["won"]),
        "ex_dense_lost": str(e4["lost"]), "ex_dense_p": f"{e4['mcnemar_p']:.2f}",
        "ex_sift_nm": f"{e3['sift_matches_median']:.0f}",
        "ex_grp_k": str(x["E5"]["n_testable"]), "ex_grp_min": f"{x['E5']['min_auroc']:.2f}",
        "ex_grp_max": f"{x['E5']['max_auroc']:.2f}",
        "ex_slip_n": str(x["E5"]["groups"]["SlipPartitioning"]["n"]),
        "ex_frac_n": str(x["E5"]["groups"]["FractureSurfaces"]["n"]),
        "ex_cons": str(x["E6"]["consensus_pairs"]), "ex_cons_fail": str(x["E6"]["consensus_all_fail"]),
        "ex_fa": str(x["E7"]["false_accepts"]),
        "ex_fa_tem": str(x["E7"]["by_group"].get("DislocationCharacterization", 0)),
        "ex_tem_fail": str(x["E7"]["tem_failures_heldout"]),
        "ex_fa_min": f"{x['E7']['fa_err_min']:.0f}", "ex_fa_med": f"{x['E7']['fa_err_median']:.0f}",
        "ex_fa_max": f"{x['E7']['fa_err_max']:.0f}", "ex_fa_scenes": str(x["E7"]["fa_scenes"]),
        "ex_fa_alloy_n": str(x["E7"]["fa_alloys"].get("MoTaTiZrHf", 0)),
        "pilot_ma": f"{x['E0']['pilot_auroc_ma_roma']:.2f}", "pilot_roma": f"{x['E0']['pilot_auroc_roma']:.2f}",
        "ex_topq_n": str(x["E8"]["top_quarter_n"]), "ex_topq_fail": str(x["E8"]["top_quarter_failures"]),
        "ex_topq_fail_tem": str(x["E8"]["top_quarter_failures_tem"]),
        "ex_bottom_fail": str(x["E8"]["bottom_all_fail_n"]), "ex_lowest_succ": f"{x['E8']['lowest_success_rank_pct']:.0f} %",
        "ex_held_scenes": str(x["E9"]["heldout_scenes"]), "ex_tem_scenes": str(x["E9"]["tem_scenes"]),
        "ex_tem_n": str(x["E9"]["tem_pairs"]), "ex_tem_alloy_n": str(x["E9"]["tem_pairs_top_alloy"]),
        "ex_dense_voters": str(x["E9"]["dense_voters"]), "ex_ma_ever": str(x["E9"]["ma_roma_ever"]),
        "ex_seed_min": str(x["E9"]["ma_roma_seed_min"]), "ex_seed_max": str(x["E9"]["ma_roma_seed_max"]),
        "ex_w_min": f"{x['E9']['wide_width_min']:,}", "ex_w_max": f"{x['E9']['wide_width_max']:,}",
        "ex_loftr_nm": f"{x['E9']['loftr_matches_median']:.0f}",
        "ex_mael_nm": f"{x['E9']['matchanything_matches_median']:.0f}",
    }


def main() -> None:
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    nums = numbers(s)
    ex = ROOT / "results/triage/exploratory.json"
    if ex.exists():
        nums.update(exploratory(json.loads(ex.read_text(encoding="utf-8")), s["n_pairs"]))
    res = RESULTS.read_text(encoding="utf-8") if RESULTS.exists() else "ABSTRACT_RESULTS:\n\n---\n"
    abs_part, _, body_part = res.partition("\n---\n")
    nums["ABSTRACT_RESULTS"] = fill(abs_part.replace("ABSTRACT_RESULTS:", "").strip(), nums)
    nums["RESULTS_AND_DISCUSSION"] = fill(body_part.strip(), nums)
    OUT.write_text(fill(TEMPLATE.read_text(encoding="utf-8"), nums), encoding="utf-8")
    body = OUT.read_text(encoding="utf-8")
    main_text = body.split("# References")[0]
    print(f"wrote {OUT}: {len(main_text.split())} words before References", file=sys.stderr)


if __name__ == "__main__":
    main()
