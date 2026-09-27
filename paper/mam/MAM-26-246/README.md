# MAM-26-246 as submitted (archived 2026-09-25)

Copies of the manuscript and cover letter submitted to *Microscopy and Microanalysis* on
2026-08-28 and rejected on 2026-09-11 (Editor Joseph Michael; no reviewer reports). The
full submitted package, including its four figures and the DOCX builds, is the tip of
branch `mam-submission` (commit 35589a7). Kept here, next to the rewrite, so the cover
letter's list of changes can be checked against the text that was actually submitted.

Two statements in this version are wrong and are corrected in the rewrite:
- "106 of 187 pairs become hard failures, meaning the estimator cannot return a transform
  at all": the 106 rows are a GPU crash (`CUDA error: unknown error`), not estimator
  failures; see `scripts/pyramid_v1_audit.py`.
- "We bin pairs by the GT-implied target-to-source area ratio": the bins used the
  pixel-size metadata, which disagrees with the annotated points on 13 pairs; see
  `scripts/fov_ratios_gt.py`.

The decision letter's only comment: "Although this manuscript may contain important new
information for the microscopy community, it is extremely difficult to read. I suggest that
it be written in a more tutorial way so that terms and methods are better described and
examples of the failures are shown. The use of the existing known database is useful."
