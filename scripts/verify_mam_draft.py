# -*- coding: utf-8 -*-
"""Parity and journal-compliance check on the Microscopy & Microanalysis draft.

Run from the repo root after any edit to paper/mam/manuscript.md:

    python scripts/verify_mam_draft.py

Gates fall into three groups:
  (1) numeric parity with the settled TMLR text (paper/tmlr/main.tex);
  (2) banned language -- TPS-era claims the forensic audit falsified;
  (3) Microscopy & Microanalysis author-instruction compliance.

Exits non-zero on any failure."""
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MS = ROOT / "paper/mam/manuscript.md"
doc = MS.read_text(encoding="utf-8")

fails = []
checks = 0


def need(label, s, hay=doc):
    global checks
    checks += 1
    if s not in hay:
        fails.append("MISSING  %-38s %r" % (label, s))


def forbid(label, s, hay=doc):
    global checks
    checks += 1
    if s in hay:
        fails.append("PRESENT  %-38s %r" % (label, s))


# --- (1) Numeric parity with the settled result files ------------------------
MUST = {
    "Table 1 RoMa raw median":        "80.2",
    "Table 1 v1 raw median":          "2707.6",
    "Table 1 v2 raw median":          "69.9",
    "Table 1 MA-RoMa raw median":     "81.0",
    "Table 1 MA-RoMa+v2 raw median":  "77.6",
    "inlier frac direct":             "0.114",
    "inlier frac pyramid":            "0.005",
    "v1 hard failures":               "106 of 187",
    "pooled match max":               "9,420,000",
    "tiles worth":                    "942 tiles",
    "match cap":                      "10,000 correspondences per",
    "wrapper SR@10 null":             "0.091 → 0.091",
    "wrapper median delta":           "−10.3 px",
    "wrapper SR@20 nominal loss":     "0.225 → 0.219",
    "wrapper stratum gain":           "1/33 → 2/33",
    "wrapper stratum loss":           "15/126 → 14/126",
    "ladder rung 0.1":                "0.075 to 0.225",
    "ladder p":                       "0.0028",
    "ladder held-out":                "0.045 → 0.227",
    "backbone raw delta":             "+0.016",
    "backbone raw p":                 "*p* = 0.33",
    "backbone stratum":               "15/126 → 19/126",
    "NMI r":                          "+0.215",
    "NMI strata":                     "0.0196, 0.0021, 0.1648 and 0.0725",
    "sub-0.5 pair count":             "61 pairs",
    "sub-0.5 leverage":               "more than three",
    "H3 affine count":                "82 (69 %)",
    "H3 homography count":            "36 (31 %)",
    "H3 median comparison":           "6.5 against 11.3",
    "H3 well-registered n":           "118 well-registered",
    "certainty gate raw null":        "0.219 → 0.219",
    "certainty gate refined":         "−0.037",
    "TPS coverage LoFTR":             "93/187",
    "TPS coverage Control A":         "70/187",
    "TPS coverage Control B":         "0/187",
    "TPS destroyed successes":        "13 configuration-pair",
    "metric-manufactured pair":       "37.5 px",
    "ft regression both metrics":     "0.393 → 0.250",
    "ft zero-shot SR@20":             "0.393 to 0.264",
    "GT floor":                       "10.3 px",
    "ladder denominators":            "38, 41 and 53",
    "wrapper worst case":             "60.6 → 1481.7",
    "wrapper raises error count":     "94 of 187",
    "L2-SP retention":                "one of three runs against three of three",
    "draw A SR@10":                   "0.095 ± 0.021",
    "fine-tune ladder held-out":      "0.111 → 0.278",
    "checkpoint step":                "step 900 of 1500",
    "MAGSAC threshold":               "5.5 px",
    "tensor parity":                  "603 of 603",
}
for label, s in MUST.items():
    need(label, s)

# --- (2) Language that must not survive the metric switch --------------------
BANNED = {
    "old wrapper gain claim":        "0.10 to 0.12",
    "old wrapper p-value":           "*p* = 0.034)  without",
    "old 'without losing a pair'":   "without losing a pair",
    "withdrawn H3 claim":            "homography confers no accuracy advantage, ",
    "appearance-dominance claim":    "appearance dominates",
    "old severe-stratum first":      "first non-zero result in the severe",
    "one-sided p":                   "one-sided",
    "seeds not runs":                "eight seeds",
    "anonymised repo":               "anonymous.4open.science",
    "TMLR boilerplate":              "Under review as submission to TMLR",
    "stale publisher":               "Cambridge University Press",
}
for label, s in BANNED.items():
    forbid(label, s)

# --- (3) Microscopy & Microanalysis compliance -------------------------------

# 3a. Abstract: <= 200 words, no citations, no abbreviations.
m = re.search(r"^## Abstract\s*\n(.*?)\n---", doc, re.S | re.M)
checks += 1
if not m:
    fails.append("STRUCT   abstract block not found")
else:
    abstract = m.group(1).strip()
    words = abstract.split()
    checks += 1
    if len(words) > 200:
        fails.append("LIMIT    abstract is %d words (max 200)" % len(words))

    checks += 1
    if re.search(r"\(\s*[A-Z][A-Za-z\-']+\s+et al\.|\(\s*[A-Z][A-Za-z\-']+\s*,?\s*\d{4}", abstract):
        fails.append("LIMIT    abstract contains a reference citation")

    # Any run of 2+ capitals is an abbreviation for M&M's purposes.
    abbrevs = sorted(set(re.findall(r"\b[A-Z]{2,}\b", abstract)))
    checks += 1
    if abbrevs:
        fails.append("LIMIT    abstract contains abbreviations: %s" % ", ".join(abbrevs))

# 3b. Mandated section order.
REQUIRED_ORDER = [
    "## 1. Introduction",
    "## 2. Materials and Methods",
    "## 3. Results",
    "## 4. Discussion",
    "## 5. Conclusions",
    "## Acknowledgments",
    "## References",
]
pos = -1
for head in REQUIRED_ORDER:
    checks += 1
    i = doc.find(head)
    if i == -1:
        fails.append("STRUCT   missing mandated section %r" % head)
    elif i < pos:
        fails.append("STRUCT   section out of mandated order: %r" % head)
    else:
        pos = i

# 3c. Reference list must not use "et al." (M&M: all authors required).
refs = doc.split("## References", 1)
checks += 1
if len(refs) < 2:
    fails.append("STRUCT   no References section")
else:
    reflist = refs[1].split("## Tables", 1)[0]
    checks += 1
    if "et al." in reflist:
        fails.append("STYLE    'et al.' appears in the reference list (M&M forbids it)")
    # Author-date style: every entry should carry a parenthesised year.
    entries = [ln for ln in reflist.splitlines() if ln.strip().startswith(("Barath", "Bookstein",
               "Durmaz", "Edstedt", "Fischler", "He,", "Hu,", "Kirkpatrick", "Li,", "Lindenberger",
               "Lowe", "Maes", "Oquab", "Sarlin", "Sun,", "Wang"))]
    checks += 1
    if len(entries) != 17:
        fails.append("STYLE    expected 17 reference entries, found %d" % len(entries))
    for e in entries:
        checks += 1
        if not re.search(r"\(\d{4}[ab]?\)\.", e):
            fails.append("STYLE    reference lacks (year). form: %s" % e[:60])

# 3c-bis. No orphan references (listed but never cited) and none dangling.
if len(refs) >= 2:
    body = refs[0]
    reflist = refs[1].split("## Tables", 1)[0]
    listed = set(re.findall(r"^([A-Z][A-Za-zÀ-ɏ\-]+),.*?\((\d{4}[ab]?)\)\.",
                            reflist, re.M))
    cited = set()
    for grp in re.findall(r"\(([^()]*\d{4}[ab]?[^()]*)\)", body):
        for part in grp.split(";"):
            mm = re.search(r"([A-Z][A-Za-zÀ-ɏ\-]+)"
                           r"(?:\s+et al\.|\s*&\s*[A-Za-z\-]+)?,\s*(\d{4}[ab]?)", part)
            if mm:
                cited.add((mm.group(1), mm.group(2)))
    for mm in re.findall(r"([A-Z][A-Za-zÀ-ɏ\-]+)\s+et al\.\s*\((\d{4}[ab]?)\)", body):
        cited.add(mm)

    checks += 1
    orphans = listed - cited
    if orphans:
        fails.append("STYLE    reference listed but never cited: %s"
                     % ", ".join("%s %s" % o for o in sorted(orphans)))
    checks += 1
    dangling = cited - listed
    if dangling:
        fails.append("STYLE    cited but missing from reference list: %s"
                     % ", ".join("%s %s" % d for d in sorted(dangling)))

# 3d. Required statements.
for head in ["## Conflict of Interest", "## Author Contributions", "## Data Availability"]:
    need("required statement", head)

# 3e. Every figure legend carries alt text (M&M requires it).
legends = re.findall(r"\*\*Figure (\d+)\.\*\*", doc)
alts = re.findall(r"\*Alt text:\*", doc)
checks += 1
if len(legends) != len(alts):
    fails.append("STYLE    %d figure legends but %d alt-text blocks" % (len(legends), len(alts)))

# 3e-bis. Legends numbered 1..N with no gap, and no body reference to a figure
# that has no legend. Renumbering after a figure is cut breaks both silently:
# nothing else in this file would notice a body citing a Figure 5 that is gone.
nums = [int(n) for n in legends]
checks += 1
if nums != list(range(1, len(nums) + 1)):
    fails.append("STRUCT   figure legends are numbered %s, expected 1..%d"
                 % (nums, len(nums)))
body_figs = set(int(n) for n in re.findall(r"Figure (\d+)", doc.split("## Figure Legends")[0]))
checks += 1
missing = sorted(body_figs - set(nums))
if missing:
    fails.append("STRUCT   text cites Figure %s but no such legend exists"
                 % ", ".join(str(m) for m in missing))
checks += 1
uncited = sorted(set(nums) - body_figs)
if uncited:
    fails.append("STRUCT   Figure %s has a legend but is never cited in the text"
                 % ", ".join(str(u) for u in uncited))

# 3f. Table 2 strata must sum to the Table 1 SR@10 column.
T1 = {
    "SIFT (Control A)": 0.016, "SIFT + mutual information (Control B)": 0.016,
    "LoFTR": 0.064, "MatchAnything-ELoFTR": 0.011, "RoMa (zero-shot)": 0.091,
    "RoMa + pyramid v1": 0.005, "RoMa + pyramid v2": 0.091,
    "MatchAnything-RoMa": 0.107, "MatchAnything-RoMa + pyramid v2": 0.118,
}
for line in doc.splitlines():
    if not line.startswith("| ") or "/4 |" not in line:
        continue
    cells = [c.strip() for c in line.strip("|").split("|")]
    name = cells[0]
    if name not in T1:
        continue
    got = sum(int(c.split("/")[0]) for c in cells[1:5])
    checks += 1
    if abs(got / 187 - T1[name]) > 0.001:
        fails.append("PARITY   Table 2 strata for %r sum to %d (=%.3f), Table 1 says %.3f"
                     % (name, got, got / 187, T1[name]))

# --- Report ------------------------------------------------------------------
print("verify_mam_draft: %d checks" % checks)
if fails:
    print("\n%d FAILURE(S):\n" % len(fails))
    for f in fails:
        print("  " + f)
    sys.exit(1)
print("all checks passed")
