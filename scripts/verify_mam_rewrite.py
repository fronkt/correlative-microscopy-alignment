# -*- coding: utf-8 -*-
"""Gate for the Microscopy and Microanalysis rewrite (after MAM-26-246).

    python scripts/mam_rewrite_numbers.py   # first: recompute every number from source
    python scripts/verify_mam_rewrite.py    # then: check the text against it

Why this replaces verify_mam_draft.py: that gate asserted strings copied from an earlier
manuscript, so it could only confirm that two texts agreed, never that either agreed
with the data. It passed 109 checks on a paper that described a GPU crash as 106
estimator failures. Here every expected number is FORMATTED FROM
results/mam_rewrite_numbers.json at run time; no numeric claim is typed into this file.

Groups: (1) numeric claims vs the recomputed numbers, (2) retired claims that must not
return, (3) journal rules (abstract, section order, abbreviations, references, figure and
table numbering, alt text), (4) cross-references between the manuscript and the
supplement, (5) worked-example legends vs the cached repeat runs, when present.
Exits non-zero on any failure.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

ROOT = pathlib.Path(__file__).resolve().parents[1]
MS = (ROOT / "paper/mam/manuscript.md").read_text(encoding="utf-8")
SUP = (ROOT / "paper/mam/supplementary.md").read_text(encoding="utf-8")
N = json.loads((ROOT / "results/mam_rewrite_numbers.json").read_text(encoding="utf-8"))

fails: list[str] = []
checks = 0


def need(label: str, s: str, hay: str = MS, where: str = "MS") -> None:
    global checks
    checks += 1
    if s not in hay:
        fails.append(f"MISSING  [{where}] {label:<44} {s!r}")


def forbid(label: str, pattern: str, hay: str = MS, where: str = "MS", regex: bool = False) -> None:
    global checks
    checks += 1
    hit = re.search(pattern, hay) if regex else (pattern in hay)
    if hit:
        fails.append(f"PRESENT  [{where}] {label:<44} {pattern!r}")


def c(x: int) -> str:
    """Thousands separator as the text writes it."""
    return f"{x:,}"


def pct(x: float, nd: int = 1) -> str:
    return f"{100 * x:.{nd}f} %"


# ---------------------------------------------------------------- (1) numbers
t, v1, fov, app = N["table187"], N["v1"], N["fov"], N["appearance"]
con, lad, orc, gtf = N["contrasts"], N["ladder"], N["oracle"], N["gt"]
tl, pd_ = v1["tiled"], v1["padded"]

# Abstract and the pooled-tiling headline
need("abstract: tiled median error", f"from {tl['median_ed_direct']:.0f} to {tl['median_ed_v1']:.0f} pixels")
need("benchmark size", "187 correlative image pairs")
# Table 3, every cell
for key in ("sift", "sift_mi", "loftr", "ma_eloftr", "roma", "roma_v2", "ma_roma", "ma_roma_v2"):
    r = t[key]
    need(f"Table 3 row {key}", f"| {r['k10']} | {r['k20']} | {r['median_ed']:.1f} | {r['fits']} |")
need("Table 3 ceiling row", f"| {orc['either_k10']} | {orc['either_k20']} | – | – |")
need("ceiling in text", f"registers {orc['either_k10']} of the 187 pairs within 10 pixels and {orc['either_k20']} within 20")
need("GT affine floor", f"median mean residual of {gtf['affine_floor_median']} pixels")
need("GT points per pair", f"a median of {gtf['points_median']} per pair (range {gtf['points_min']} to {gtf['points_max']})")
need("image sizes", f"from {N['image_sizes']['min_long_side']} to {c(N['image_sizes']['max_long_side'])} pixels")
need("10,000 cap on all pairs", "It returned exactly 10,000 on every one of the 187 pairs")
assert N["cap_hit"] == {"roma": 187, "ma_roma": 187}
# Table 4, both rows
T4 = {  # row label -> (tiled cell, padded cell); the table is transposed, one column per group
    "Pairs": (f"{tl['n']}", f"{pd_['n']}"),
    "Tiles per pair": (f"{tl['tiles_min']} to {tl['tiles_max']} (median {tl['tiles_median']:.0f})", "1"),
    "Median error (pixels)": (f"{tl['median_ed_direct']:.1f} → {tl['median_ed_v1']:.1f}",
                              f"{pd_['median_ed_direct']:.1f} → {pd_['median_ed_v1']:.1f}"),
    "Registered within 20 pixels": (f"{tl['k20_direct']} → {tl['k20_v1']}", f"{pd_['k20_direct']} → {pd_['k20_v1']}"),
    "Registered within 10 pixels": (f"{tl['k10_direct']} → {tl['k10_v1']}", f"{pd_['k10_direct']} → {pd_['k10_v1']}"),
    "Share kept by robust fitting": (f"{tl['inlier_frac_direct']:.2f} → {tl['inlier_frac_v1']:.4f}",
                                     f"{pd_['inlier_frac_direct']:.3f} → {pd_['inlier_frac_v1']:.4f}"),
    "Pairs on which pooling raised the error": (f"{tl['n_worse']} of {tl['n']}", f"{pd_['n_worse']} of {pd_['n']}"),
}
for row, (a, b) in T4.items():
    need(f"Table 4 {row}", f"| {row} | {a} | {b} |")
need("Table 4 not-evaluated note", f"The other {v1['crashed']} of the 187 pairs were not evaluated")
need("v1 evaluated / tiled / padded", f"evaluated on {v1['evaluated']} pairs: {tl['n']} that were genuinely tiled")
need("v1 crash position", f"stopped at the {v1['crash_starts_at']}nd of the 187 pairs")
need("v1 padded share of 187", f"This happens on {v1['padded_pairs_all187']} of the 187 pairs")
need("v1 largest pool", f"{c(v1['max_pooled'])} from {v1['max_pooled'] // 10_000} tiles")
need("v1 owed tile matches", f"{c(v1['owed_tile_matches'])} further tile matches")
need("v1 owed direct successes", f"include {v1['owed_direct_successes']} pairs that RoMa registers directly")
need("v1 worse count text", f"the error rose on {tl['n_worse']} of the {tl['n']} pairs")
need("v1 inlier share text", f"from a median of {tl['inlier_frac_direct']:.2f} for the whole image to {tl['inlier_frac_v1']:.4f}")
# Field of view
need("FOV groups", f"below 0.05 ({fov['strata_gt'][0]} pairs), 0.05 to 0.25 ({fov['strata_gt'][1]}), "
                   f"0.25 to 0.5 ({fov['strata_gt'][2]}) and 0.5 or more ({fov['strata_gt'][3]})")
need("FOV median", f"median of {fov['median_ratio_gt']}")
need("FOV below half", f"Only {fov['below_half_gt']} pairs fall below 0.5")
need("FOV minimum", f"it covers {100 * fov['min_ratio_gt']:.2f} %")
need("metadata inconsistent count", f"For {fov['metadata_inconsistent']} pairs in three series")
need("below-half leverage", f"Below a ratio of 0.5 there are {fov['below_half_gt']} pairs, and no method in Table 3 "
                            f"registers more than {N['below_half_max_registered']} of them")
for key, label in (("sift", "SIFT"), ("loftr", "LoFTR"), ("roma", "RoMa"), ("ma_roma_v2", "MatchAnything-RoMa + checked search")):
    need(f"Table 5 row {key}", f"| {label} | " + " | ".join(s.split("/")[0] for s in N["strata_sr10"][key]) + " |")
need("Table 5 pairs row", "| Pairs | " + " | ".join(str(x) for x in fov["strata_gt"]) + " |")
need("Table 5 NMI row", "| Median NMI | " + " | ".join(f"{x:.3f}" for x in app["median_nmi_by_stratum"]) + " |")
need("NMI correlation", f"Pearson *r* = +{app['pearson_r']:.2f}, *p* = {app['pearson_p']:.3f}")
need("NMI rank correlation", f"Spearman's rank correlation +{app['spearman_rho']:.2f}, *p* = {app['spearman_p']:.3f}")
need("NMI medians text", "the median NMI is " + ", ".join(f"{x:.3f}" for x in app["median_nmi_by_stratum"][:3])
                         + f" and {app['median_nmi_by_stratum'][3]:.3f}")
sh = N["wrapper_shift_by_stratum"]
assert sh[2] == {"gained": 1, "lost": 0} and sh[3] == {"gained": 0, "lost": 1}, sh
need("wrapper shift text", "registered one pair in the 0.25 to 0.5 field-of-view group that the direct match missed, "
                           "and lost one in the group of 0.5 and above")
# Contrasts on 187 pairs
w10, w20, wm = con["wrapper_sr10_raw"], con["wrapper_sr20_raw"], con["wrapper_median_raw"]
need("wrapper SR10 CI", f"(difference 0.000, 95 % CI {w10['ci_lo']:+.3f} to {w10['ci_hi']:+.3f}, *p* = {w10['p_two_sided']:.2f})".replace("+-", "−").replace("-0", "−0"))
need("wrapper SR20", f"{round(w20['sr_b'] * 187)} against {round(w20['sr_a'] * 187)} (*p* = {w20['p_two_sided']:.2f})")
need("wrapper median", f"({wm['value']:.1f} pixels, 95 % CI {wm['ci_lo']:.1f} to +{wm['ci_hi']:.1f}, *p* = {wm['p_two_sided']:.2f})".replace("-", "−"))
b10 = con["backbone_sr10_raw"]
need("backbone SR10", f"(+{b10['value']:.3f}, 95 % CI {b10['ci_lo']:.3f} to +{b10['ci_hi']:.3f}, *p* = {b10['p_two_sided']:.2f})".replace("-", "−"))
need("backbone median p", f"(*p* = {con['backbone_median_raw']['p_two_sided']:.2f})")
need("refined wrapper p", f"(+{con['wrapper_sr10_tps']['value']:.3f}, *p* = {con['wrapper_sr10_tps']['p_two_sided']:.3f})")
need("refined backbone p", f"(+{con['backbone_sr10_tps']['value']:.3f}, *p* = {con['backbone_sr10_tps']['p_two_sided']:.3f})")
vp = N["v2_per_pair"]
need("wrapper per-pair", f"it lowered the error on {vp['better']} pairs and raised it on {vp['worse']}")
need("limitation: raised error", f"raised the registration error on {vp['worse']} of 187 pairs")
stg = vp["stage_counts"]
need("stage counts", f"on {stg['direct']} of the 187 pairs the direct match was kept; on {stg['direct+zoom']} a zoom step")
assert stg.get("tile", 0) == 1 and N["v2_tile_trigger"]["roma"] == 0
need("tile trigger", "every one of the 187 fits kept at least 50")
# Ladder
m1, mh, mc, rr = lad["ma_roma_r010"], lad["ma_roma_r010_heldout"], lad["ma_roma_r010_consistent_only"], lad["roma_r010"]
need("ladder headline", f"registers {m1['k_v2']} of {m1['n']} at that ratio, three times as many ({m1['sr_direct']:.3f} to "
                        f"{m1['sr_v2']:.3f}, difference +{m1['delta']:.3f}, 95 % CI +{m1['ci'][0]:.3f} to +{m1['ci'][1]:.3f}, *p* = {m1['p']:.4f})")
need("ladder direct count", f"at a field-of-view ratio of 0.1 it registers {m1['k_direct']} of {m1['n']}")
need("ladder held-out", f"on the {mh['n']} of these pairs that were not used for fine-tuning ({mh['k_direct']} to {mh['k_v2']}, *p* = {mh['p']:.3f})")
need("ladder consistent-only", f"on the {mc['n']} whose metadata is consistent ({mc['k_direct']} to {mc['k_v2']}, *p* = {mc['p']:.4f})")
need("ladder RoMa", f"({rr['k_direct']} to {rr['k_v2']} of {rr['n']}, *p* = {rr['p']:.2f})")
need("ladder base-matchable", f"RoMa registers {lad['base_matchable']['roma']} within 20 pixels at full size and MatchAnything-RoMa registers {lad['base_matchable']['ma_roma']}")
need("ladder testbed", f"The candidate pairs form a fixed set of {lad['testbed_pairs']}")
need("ladder inconsistent", f"for the {lad['testbed_metadata_inconsistent']} of the {lad['testbed_pairs']} pairs whose metadata is inconsistent")
ls = N["ladder_v2_success_stages_ma_roma_r010"]
need("ladder stages", f"Of those {m1['k_v2']} successes, {ls['direct+zoom']} came from a zoom step, {ls['tile+zoom']} from the tile search")
# Certainty share
cs = N["certainty_share"]
need("certainty share n", f"on {cs['n']} pairs that run kept the direct match as its answer")
need("certainty registered", f"All {cs['registered']} of those pairs that were registered had at least {100 * cs['registered_min']:.0f} % above the cut-off (median {100 * cs['registered_median']:.0f} %)")
need("certainty failed", f"Of the {cs['failed_100']} that missed by more than 100 pixels, {cs['failed_below_registered_min']} had less than "
                         f"{100 * cs['registered_min']:.0f} % (median {100 * cs['failed_median']:.0f} %), but {cs['failed_over_half']} still had more than half")
need("certainty rho", f"Spearman's rank correlation {cs['spearman_rho']:.2f})".replace("-", "−"))
gk = N["gate_kept"]
need("gate kept", f"a median of {c(gk['median'])} of the 10,000 correspondences cleared the cut-off")
ab = N["ablations"]
need("zoom variant", f"registered {round(ab['zoom_sr10_raw']['sr_b'] * 187)} pairs within 10 pixels instead of {round(ab['zoom_sr10_raw']['sr_a'] * 187)}, "
                     f"and {round(ab['zoom_sr20_raw']['sr_b'] * 187)} within 20 instead of {round(ab['zoom_sr20_raw']['sr_a'] * 187)} "
                     f"(*p* = {ab['zoom_sr10_raw']['p_two_sided']:.2f} and {ab['zoom_sr20_raw']['p_two_sided']:.2f})")
need("gate variant", f"registered {round(ab['gate_sr20_raw']['sr_b'] * 187)} pairs within 20 pixels against "
                     f"{round(ab['gate_sr20_raw']['sr_a'] * 187)} without the discard (*p* = {ab['gate_sr20_raw']['p_two_sided']:.2f})")
# Fine-tuning
ft = N["finetune"]
need("ft TEM medians", f"from {ft['tem_test']['zs_median']:.0f} to {ft['tem_test']['ft_median']:.0f} pixels")
need("ft TEM count", f"on the {ft['tem_test']['n']} TEM pairs of the test set")
need("ft test within 20", f"fell from {ft['raw']['zs_k20']} to {ft['raw']['ft_k20']} of {ft['raw']['n']}")
need("ft train size", f"on {ft['train_pairs']} of the 187 pairs, drawn from {ft['train_subclasses']} of the 19 series")
assert not ft["c103_in_train"]
r48 = ft["runs_4to8"]
need("ft runs range", f"between {100 * r48['plain_direct_sr20'][0]:.1f} % and {100 * r48['plain_direct_sr20'][1]:.1f} %")
assert r48["plain_direct_sr20"][:2] == r48["l2sp_direct_sr20"][:2]
# Refinement coverage and regressions
cov = N["tps_coverage"]
need("coverage text", f"on {cov['loftr']} for LoFTR, on {cov['sift']} for SIFT and on none for SIFT with mutual information")
assert cov["sift_mi"] == 0 and all(cov[k] == 187 for k in ("roma", "roma_v2", "ma_roma", "ma_roma_v2"))
need("TPS regressions", f"turned {N['tps_regressions']['count']} registrations into failures across the {N['tps_regressions']['configs']} method configurations")
# Transform family (supplement)
fam = N["family"]
need("H3 counts", f"Among the {fam['n_well_registered']} results within 20 pixels", SUP, "SUP")
need("H3 affine", f"for {fam['affine']} ({fam['affine_pct']} %) and a homography for {fam['homography']}", SUP, "SUP")
need("H3 medians", f"(median error {fam['median_homography']} against {fam['median_affine']} pixels)", SUP, "SUP")
# Supplement tables
need("S1 total row", f"| **Total** | **{v1['evaluated']}** | **{v1['crashed']}** | **{c(sum(x[2] for x in N['v1_by_subset'].values()))}** |", SUP, "SUP")
for sub, (ev, cr, nt) in N["v1_by_subset"].items():
    need(f"S1 row {sub[:28]}", f"| {ev} | {cr} | {c(nt)} |", SUP, "SUP")
af = N["af9628_scene3"]
need("S2 AF9628 width", f"{c(af['source_width_px'])} pixels wide, and the metadata gives a pixel size of {c(round(af['source_px_meta_nm']))} nm", SUP, "SUP")
need("S2 AF9628 implied", f"imply a pixel size of {af['source_px_implied_nm']} nm", SUP, "SUP")
need("S2 AF9628 mm", f"{af['source_width_meta_mm']} mm wide", SUP, "SUP")
need("S2 metadata groups", "hold " + ", ".join(str(x) for x in fov["strata_meta"][:3]) + f" and {fov['strata_meta'][3]} pairs instead of "
     + ", ".join(str(x) for x in fov["strata_gt"][:3]) + f" and {fov['strata_gt'][3]}, and {fov['stratum_changes']} pairs change group", SUP, "SUP")
lt = lad["ma_roma_r010_tps"]
need("S3 ladder refined", f"| refined | {lt['sr_direct']:.3f} | {lt['sr_v2']:.3f} | +{lt['delta']:.3f} |", SUP, "SUP")
need("S3 ladder refined counts", f"from {m1['k_v2']} of {m1['n']} to {lt['k_v2']}", SUP, "SUP")
need("S4 gate refined CI", f"−{abs(ab['gate_sr20_tps']['value']):.3f} (95 % CI −{abs(ab['gate_sr20_tps']['ci_lo']):.3f} to +{ab['gate_sr20_tps']['ci_hi']:.3f}, *p* = {ab['gate_sr20_tps']['p_two_sided']:.2f})", SUP, "SUP")
need("S4 gate under 1000", f"fewer than 1,000 on only {gk['under_1000']} pairs", SUP, "SUP")
for pid, (z, f_) in ft["c103_test"].items():
    need(f"S5 C103 {pid[-3:]}", f"{z} and {f_}", SUP, "SUP")

# ---------------------------------------------------------------- (2) retired claims
RETIRED = {
    "crash described as estimator failure": "cannot return a transform at all",
    "old mismatched headline": "80 to 2708",
    "old mismatched headline (px)": "2707.6",
    "old inlier fraction over mismatched sets": "0.114",
    "old factor": "factor of 22",
    "old 'GT-implied' claim for metadata bins": "GT-implied",
    "old extreme-stratum claim": "0 of 4",
    "old confounded gate p": r"\*p\* = 0\.002(?!\d)",
    "old name for the checked search": "checked tiling",
    "unsupported material name": "tungsten",
    "one-sided p": "one-sided",
    "seeds not runs": "eight seeds",
    "old coverage claim": "187/187 for every dense",
    "old monotonicity figure": "94 of 187",
    "abbreviated unit": r"\bpx\b",
}
for label, pat in RETIRED.items():
    regex = pat.startswith("\\b") or pat.startswith("\\*")
    forbid(label, pat, MS, "MS", regex)
    forbid(label, pat, SUP, "SUP", regex)

# ---------------------------------------------------------------- (3) journal rules
m = re.search(r"^## Abstract\s*\n(.*?)\n---", MS, re.S | re.M)
checks += 1
if not m:
    fails.append("STRUCT   abstract block not found")
else:
    abstract = m.group(1).strip()
    nwords = len(abstract.split())
    checks += 3
    if nwords > 200:
        fails.append(f"LIMIT    abstract is {nwords} words (max 200)")
    if re.search(r"\(\s*[A-Z][A-Za-z\-']+,?\s+(et al\.,?\s+)?\d{4}", abstract):
        fails.append("LIMIT    abstract contains a reference citation")
    ab_caps = sorted(set(re.findall(r"\b[A-Z]{2,}\b", abstract)))
    if ab_caps:
        fails.append("LIMIT    abstract contains abbreviations: " + ", ".join(ab_caps))
    print(f"abstract: {nwords} words")

ORDER = ["## Abstract", "## 1. Introduction", "## 2. Materials and Methods", "## 3. Results", "## 4. Discussion",
         "## 5. Conclusions", "## Supplementary Material", "## Acknowledgments", "## Competing Interests",
         "## Author Contributions", "## Data Availability", "## References", "## Tables", "## Figure Legends"]
pos = [MS.find(h) for h in ORDER]
checks += 1
if -1 in pos or pos != sorted(pos):
    fails.append("ORDER    section order: " + ", ".join(f"{h}={p}" for h, p in zip(ORDER, pos)))

body = MS[MS.find("## 1. Introduction"):MS.find("## Supplementary Material")]
legends = MS[MS.find("## Tables"):]
# Abbreviations: every all-caps token must be defined "(TOKEN)" at or before first use,
# or be a proper name. Checked over the body, then over tables and legends.
NAMES = {"MAX", "SP", "MAGSAC", "CC", "BY", "VGG", "CUDA"}
defined = set(re.findall(r"\(([A-Z]{2,})\)", MS))
for tok in sorted(set(re.findall(r"\b[A-Z]{2,}\b", body + legends)) - NAMES):
    checks += 1
    first = (body + legends).find(tok)
    dm = re.search(r"\(" + tok + r"\)", body + legends)
    if tok not in defined or (dm and dm.start() > first + len(tok) + 400):
        fails.append(f"ABBREV   {tok} is not defined at or near first use")

# References: every entry cited, every citation listed, no "et al." inside the list.
refs = MS[MS.find("## References"):MS.find("## Tables")]
ref_keys = set(re.findall(r"^([A-Z][A-Za-zÀ-ÿ\-]+), [A-Z]", refs, re.M))
cited = set(re.findall(r"([A-Z][A-Za-zÀ-ÿ\-]+)(?: et al\.| & [A-Z][A-Za-z\-]+)?,? \(?\d{4}[ab]?\)?", body + legends))
for k in ref_keys:
    checks += 1
    if k not in body + legends:
        fails.append(f"REF      listed but never cited: {k}")
for k in {"Durmaz", "Lowe", "Maes", "Sun", "Wang", "Edstedt", "He", "Fischler", "Barath", "Bookstein", "Kirkpatrick", "Li"}:
    checks += 1
    if k not in ref_keys:
        fails.append(f"REF      cited but not listed: {k}")
checks += 1
if "et al." in refs:
    fails.append("REF      'et al.' in the reference list")
checks += 1
if re.search(r"\bOquab|\bHu, E\.J\.|\bLindenberger|\bSarlin", refs):
    fails.append("REF      an entry for a paper no longer cited is still listed")

# Figures and tables: legends 1..N, each cited in the body, each legend has alt text.
fig_leg = [int(x) for x in re.findall(r"^\*\*Figure (\d+)\.\*\*", MS, re.M)]
checks += 1
if fig_leg != list(range(1, len(fig_leg) + 1)):
    fails.append(f"FIGS     legends not 1..N: {fig_leg}")
for k in fig_leg:
    checks += 2
    if not re.search(rf"Figure {k}[A-E]?\b|Figures [0-9, and]*\b{k}\b", body):
        fails.append(f"FIGS     Figure {k} never cited in the body")
    leg = re.search(rf"^\*\*Figure {k}\.\*\*.*?(?=^\*\*Figure {k + 1}\.\*\*|\Z)", MS, re.M | re.S).group(0)
    if "*Alt text:*" not in leg:
        fails.append(f"FIGS     Figure {k} has no alt text")
    checks += 1
    if not (ROOT / f"paper/mam/figures/Figure{k}.pdf").exists():
        fails.append(f"FIGS     paper/mam/figures/Figure{k}.pdf missing")
checks += 1
if (ROOT / f"paper/mam/figures/Figure{len(fig_leg) + 1}.pdf").exists():
    fails.append(f"FIGS     orphan file Figure{len(fig_leg) + 1}.pdf with no legend")
tab_leg = [int(x) for x in re.findall(r"^\*\*Table (\d+)\.\*\*", MS, re.M)]
checks += 1
if tab_leg != list(range(1, len(tab_leg) + 1)):
    fails.append(f"TABLES   legends not 1..N: {tab_leg}")
for k in tab_leg:
    checks += 1
    if not re.search(rf"Tables? [0-9, and]*\b{k}\b", body):
        fails.append(f"TABLES   Table {k} never cited in the body")

# ---------------------------------------------------------------- (4) cross-references
heads = set(re.findall(r"^### (\d\.\d+)\.", MS, re.M))
for sec in sorted(set(re.findall(r"Sections? (\d\.\d+)", MS + SUP))):
    checks += 1
    if sec not in heads:
        fails.append(f"XREF     Section {sec} is referenced but does not exist")
sup_heads = set(re.findall(r"^## (S\d)\.", SUP, re.M))
for s in sorted(set(re.findall(r"Supplementary Section (S\d)", MS + SUP))):
    checks += 1
    if s not in sup_heads:
        fails.append(f"XREF     Supplementary Section {s} is referenced but does not exist")
for s in sorted(sup_heads):
    checks += 1
    if f"Supplementary Section {s}" not in MS and s not in ("S7",):
        fails.append(f"XREF     Supplementary Section {s} is never cited from the manuscript")
sup_tabs = set(re.findall(r"^\*\*Table (S\d)\.\*\*", SUP, re.M))
for s in sorted(set(re.findall(r"Supplementary Table (S\d)|Table (S\d)", MS + SUP)) ):
    for x in s:
        if x:
            checks += 1
            if x not in sup_tabs:
                fails.append(f"XREF     Table {x} referenced but not in the supplement")
sup_figs = set(re.findall(r"^\*\*Figure (S\d)\.\*\*", SUP, re.M))
for x in sorted(set(re.findall(r"Figure (S\d)", MS + SUP))):
    checks += 2
    if x not in sup_figs:
        fails.append(f"XREF     Figure {x} referenced but has no legend")
    if not (ROOT / f"paper/mam/figures/Figure{x}.pdf").exists():
        fails.append(f"XREF     paper/mam/figures/Figure{x}.pdf missing")

# ---------------------------------------------------------------- (5) worked examples
cache = ROOT / "results/mam_examples"
if (cache / "pipeline_5842.npz").exists():
    import numpy as np

    def ex(job, key):
        return np.load(cache / f"{job}.npz")[key]

    def share(job):
        return float((ex(job, "conf") >= 0.999).mean())

    need("Fig 2 legend error", f"their mean, {float(ex('pipeline_5842', 'mu_ed')):.1f} pixels")
    need("Fig 2 kept/rejected", f"kept {c(int(ex('pipeline_5842', 'inliers').sum()))} of the 10,000 correspondences and rejected "
                                f"{c(10_000 - int(ex('pipeline_5842', 'inliers').sum()))}")
    need("Fig 3 inside share", f"{100 * share('tile_inside'):.0f} % of them with certainty above the 0.05 cut-off")
    need("Fig 3 outside share", f"Only {100 * share('tile_outside'):.0f} % of those cleared the cut-off")
    need("Fig 3B tile alone", f"the inside tile of Figure 3B misses by {round(float(ex('tile_inside', 'mu_ed')))} pixels")
    need("S6 tile rows", f"| RoMa | {round(float(ex('tile_inside', 'mu_ed')))} (this tile alone) | – | "
                         f"{share('tile_inside'):.2f} |", SUP, "SUP")
    need("S6 tile rows", f"| RoMa | {c(round(float(ex('tile_outside', 'mu_ed'))))} (this tile alone) | – | "
                         f"{share('tile_outside'):.2f} |", SUP, "SUP")
    for job, fig in (("pipeline_5842", "2"), ("gal_appearance", "4A"), ("gal_fov", "4B"), ("gal_c103_zs", "4C"), ("gal_c103_ft", "4C")):
        need(f"S6 share {job}", f"| {share(job):.2f} |", SUP, "SUP")
    need("Fig 4A error", f"{c(round(float(ex('gal_appearance', 'mu_ed'))))} pixels from where it belongs")
    need("Fig 4B error", f"places it {round(float(ex('gal_fov', 'mu_ed')))} pixels away")
    need("Fig 4C errors", f"{round(float(ex('gal_c103_zs', 'mu_ed')))} pixels before, {round(float(ex('gal_c103_ft', 'mu_ed')))} after")
    need("Fig 4C ft share", f"After fine-tuning, {100 * share('gal_c103_ft'):.0f} % of the 10,000")
    need("inliers on the two failures", f"found {int(ex('gal_appearance', 'inliers').sum())} and {int(ex('gal_fov', 'inliers').sum())} of them")
    for job in ("gal_appearance", "gal_fov"):
        checks += 1
        if float(ex(job, "frac_above_thresh")) != 0.0:
            fails.append(f"EXAMPLE  {job}: text says no pixel cleared the cut-off, cache says {float(ex(job, 'frac_above_thresh'))}")
else:
    print("note: results/mam_examples/ not present; worked-example checks skipped "
          "(regenerate with scripts/mam_examples_run.py)")

# ---------------------------------------------------------------- report
em = [ln for ln in MS.splitlines() if "—" in ln and "Writing —" not in ln]
print(f"{checks} checks, {len(fails)} failures")
if em:
    print(f"note: {len(em)} line(s) with an em dash outside the CRediT roles")
for f_ in fails:
    print("  " + f_)
sys.exit(1 if fails else 0)
