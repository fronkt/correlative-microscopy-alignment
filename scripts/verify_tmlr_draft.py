# -*- coding: utf-8 -*-
"""Parity, anonymity and structural check on the TMLR submission.

Run from the repo root after any edit to paper/tmlr/main.tex:

    python scripts/verify_tmlr_draft.py

Requires main.txt next to main.pdf (pdftotext main.pdf main.txt) for the
rendered-PDF anonymity check. Exits non-zero on any failure."""
import pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
tex = (ROOT / "paper/tmlr/main.tex").read_text(encoding="utf-8")
txt = (ROOT / "paper/tmlr/main.txt").read_text(encoding="utf-8", errors="replace")

fails = []

# --- Values that MUST appear (all recomputed from source this session) --------
MUST = {
    "Table 1 RoMa raw median":        "80.2",
    "Table 1 v1 raw median":          "2707.6",
    "Table 1 v2 raw median":          "69.9",
    "Table 1 MA-RoMa raw median":     "81.0",
    "Table 1 MA-RoMa+v2 raw median":  "77.6",
    "inlier frac direct":             "0.114",
    "inlier frac pyramid":            "0.005",
    "v1 hard failures":               "106 of 187",
    "pooled match max":               "9{,}420{,}000",
    "wrapper SR@10 null":             r"0.091\rightarrow0.091",
    "wrapper median delta":           "-10.3",
    "ladder rung 0.1":                r"0.075\rightarrow0.225",
    "ladder p":                       "0.0028",
    "ladder held-out":                r"0.045\rightarrow0.227",
    "backbone raw delta":             "+0.016",
    "backbone raw p":                 "p=0.33",
    "NMI r":                          "+0.215",
    "NMI strata":                     "0.0196, 0.0021, 0.1648, 0.0725",
    "H3 affine count":                "82 (69",
    "H3 homography count":            "36 (31",
    "H3 median comparison":           "6.5 against 11.3",
    "certainty gate raw null":        r"0.219\rightarrow0.219",
    "MA-RoMa v2 gain pair 1":         r"11.8\rightarrow8.3",
    "MA-RoMa v2 gain pair 2":         r"325.8\rightarrow8.1",
    "TPS coverage Control B":         "0/187",
    "TPS destroyed successes":        "13 configuration-pair",
    "metric pair 3.3 -> 37.5":        "37.5",
    "ft regression both metrics":     r"0.393\rightarrow0.250",
    "GT floor":                       "10.3",
    "base-matchable denominators":    "38 pairs for RoMa",
    "wrapper stratum shift":          "1/33",
    "sub-0.5 leverage":               "more than three",
}
for label, s in MUST.items():
    if s not in tex:
        fails.append("MISSING in tex: %-30s %r" % (label, s))

# --- Language that MUST NOT survive (TPS-era claims now false or reframed) ----
BANNED = {
    "old wrapper gain claim":       r"0.10\rightarrow0.12",
    "old 'without losing a pair'":  "without losing a pair",
    "old two-gained-two-lost":      "two pairs gained and two lost",
    "old 84.8 recovery":            "84.8",
    "old cert-gate as primary":     r"is significantly \emph{worse} than plain v2 (SR@20 $-0.037$",
    "old cap wording":              "which is the cap we impose",
    "seed framing":                 "seed-robust",
    "author name":                  "Frank",
    "affiliation":                  "Purdue",
    "real repo":                    "fronkt",
    "orcid":                        "ORCID",
    "zenodo doi":                   "zenodo",
}
for label, s in BANNED.items():
    if s in tex:
        fails.append("BANNED still in tex: %-28s %r" % (label, s))

# --- Withdrawn claims: may appear ONLY inside an explicit retraction ----------
# Both phrases survive in the text because the paper withdraws them by name.
# Assert the retraction framing is present, so a future edit that reinstates the
# bare claim trips this check.
RETRACTED = {
    "severe-stratum from-zero": ("first non-zero",
                                 "claim was an artefact"),
    "H3 no-advantage":          ("no accuracy advantage",
                                 "we cannot add, as an earlier version"),
}
for label, (claim, retraction) in RETRACTED.items():
    if claim in tex and retraction not in tex:
        fails.append("withdrawn claim reinstated without retraction: %s" % label)
    if tex.count(claim) > 1:
        fails.append("withdrawn claim %s appears %d times; expected 1 (the retraction)"
                     % (label, tex.count(claim)))

# --- Anonymity of the RENDERED pdf -------------------------------------------
for s in ("Frank", "Cai", "Purdue", "fronkt", "ORCID", "zenodo", "frankyc"):
    if re.search(re.escape(s), txt, re.I):
        fails.append("DEANONYMISED in rendered pdf: %r" % s)

# --- Structural LaTeX sanity --------------------------------------------------
if tex.count("{") != tex.count("}"):
    fails.append("brace imbalance: %d vs %d" % (tex.count("{"), tex.count("}")))
if tex.count(r"\begin{") != tex.count(r"\end{"):
    fails.append("begin/end imbalance")
bare = re.findall(r"(?<!\\)%", tex)
com = [l for l in tex.splitlines() if l.lstrip().startswith("%")]
if len(bare) != len(com):
    fails.append("possible unescaped %% outside comments: %d marks, %d comment lines"
                 % (len(bare), len(com)))
if re.search(r"\\cite\{", tex):
    fails.append("bare \\cite{} found; TMLR expects \\citep/\\citet")
nonascii = [(i + 1, l) for i, l in enumerate(tex.splitlines())
            if any(ord(c) > 127 for c in l)]
for i, l in nonascii:
    fails.append("non-ASCII on tex line %d: %r" % (i, l[:70]))

print("checks run: %d must-appear, %d banned, structural"
      % (len(MUST), len(BANNED)))
if fails:
    print("\n%d PROBLEM(S):" % len(fails))
    for f in fails:
        print("  " + f)
    sys.exit(1)
print("\nALL CHECKS PASS")
