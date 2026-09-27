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
        "d_s3": f"{h2['primary_delta_S3_minus_S1']['delta']:+.2f}",
        "d_s3_lo": f"{h2['primary_delta_S3_minus_S1']['lo']:+.2f}", "d_s3_hi": f"{h2['primary_delta_S3_minus_S1']['hi']:+.2f}",
        "pool_s1": f"{h2['pooled_auroc_S1']['auroc']:.2f}", "pool_s2": f"{h2['pooled_auroc_S2']['auroc']:.2f}",
        "pool_s3": f"{h2['pooled_auroc_S3']['auroc']:.2f}",
        "pd_s2": f"{h2['pooled_delta_S2_minus_S1']['delta']:+.2f}",
        "pd_s2_lo": f"{h2['pooled_delta_S2_minus_S1']['lo']:+.2f}", "pd_s2_hi": f"{h2['pooled_delta_S2_minus_S1']['hi']:+.2f}",
        "pd_s3": f"{h2['pooled_delta_S3_minus_S1']['delta']:+.2f}",
        "pd_s3_lo": f"{h2['pooled_delta_S3_minus_S1']['lo']:+.2f}", "pd_s3_hi": f"{h2['pooled_delta_S3_minus_S1']['hi']:+.2f}",
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
    }


def fill(text: str, nums: dict[str, str]) -> str:
    def rep(m):
        k = m.group(1)
        if k not in nums:
            raise KeyError(f"unknown placeholder {{{{{k}}}}}")
        return nums[k]
    return re.sub(r"\{\{(\w+)\}\}", rep, text)


def main() -> None:
    s = json.loads(SUMMARY.read_text(encoding="utf-8"))
    nums = numbers(s)
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
