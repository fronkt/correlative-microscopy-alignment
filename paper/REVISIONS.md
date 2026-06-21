# Revision log — round 1 (simulated peer review)

Status of the editorial revision roadmap from the `academic-paper-reviewer`
panel (decision: Major Revision, favourable). Applied to `main.tex` + `paper.md`
(and rebuilt into `paper.docx`).

## Applied (no GPU required)

| # | Roadmap item | What changed |
|---|---|---|
| 2 | Scope the claims to the benchmark | Title now names the AmalgaMatch benchmark; "binding constraint" → "dominant constraint"; "structural conclusion" → "the conclusion we draw, for this benchmark". |
| 2 | `p = 0.91` ≠ no regression | Abstract + Results now state this is a **low-power null** on 28 pairs ("not detectably worse", not proven equivalence); a formal equivalence test on a larger set is named as planned. |
| 2 | Trim over-claiming intensifiers | Removed "provably effective", "the cheapest possible", "demonstrably", "catastrophic forgetting … solved"; softened summary sentences. |
| 3 | Appearance-control caveat (Devil's Advocate) | New Discussion paragraph: scale is decoupled by a controlled experiment; appearance is implicated **by elimination**, since the FOV ladder runs only on base-matchable pairs. Framed as the dominant *uncontrolled* factor, not a law. |
| 3 | FOV-ladder ecological-validity caveat | Limitations: cropping a target ≠ native low-FOV acquisition (texture/noise/resolution); wrapper gains are a controlled lower bound. |
| 4 | L2-SP framed as a minimal floor | Results: L2-SP is a deliberate zero-cost lower bound; one of two off-distribution C103 scenes stays unrecoverable; stronger methods (replay, EWC, LoRA) expected to improve on it. |
| 5 | Practical recommendations | New Discussion ordering (i–iv): backbone choice, automatic affine/homography, forgetting-robust fine-tune with compute (~100M params, ~90 min on one RTX 5090, 445 MB ckpt, λ = 0.01), wrapper for multiscale only — plus the modality-support envelope. |

## Deferred — require a GPU box (flagged in Limitations, not faked)

| # | Roadmap item | Why deferred |
|---|---|---|
| 1 | Multi-seed the fine-tune + L2-SP tables (≥3 seeds, report variance) | Needs re-training on a 5090-class box. |
| 4 | Run one stronger continual-learning baseline (LoRA) | Needs training runs. Framed in-text as expected-to-help, not claimed. |
| 3 | A native-low-FOV / appearance-swept control experiment | Needs new matcher inference / data manipulation runs. |
| 6 | Quantitative "never-abstain" figure (matches vs tile-overlap) | Needs per-tile match-count logging from matcher runs; current claim kept textual (inlier fraction 0.114 → 0.005). |

All deferred items are disclosed in the manuscript's Limitations as "planned and
not yet run"; no result is reported as completed that was not actually run.

## Style pass (round 2, applied)

- Full em-dash / sentence-rhythm de-AI pass across the whole body of `main.tex`
  and `paper.md`. Prose em-dashes went from ~45 to 0 (the only remaining `—`
  are the two "not applicable" cells in Table 1). Antithesis cadence
  ("not X but Y") broken up, paragraph-ending aphorisms varied, intensifiers
  trimmed. All numbers, citations, and tables unchanged. `paper.docx` rebuilt.
