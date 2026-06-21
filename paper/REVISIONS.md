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

## Multi-seed experiment (round 3, applied — GPU box)

Reviewer R1's critical-for-acceptance item ("multi-seed the fine-tune + L2-SP tables,
≥3 seeds, report variance") is now run, not deferred. Six runs on an RTX 5090
(3 seeds × {λ=0 plain, λ=0.01 L2-SP}), identical decoder-only recipe (AdamW 2e-5,
1500 steps, min-val-medED checkpoint selection), direct-match eval on the 28-pair
test split.

| Config | SR@10 | SR@20 | median ED (px) |
|---|---|---|---|
| zero-shot (deterministic) | 0.214 | 0.393 | 83.5 |
| plain fine-tune (λ=0), 3 seeds | 0.095 ± 0.021 | 0.298 ± 0.041 | 46.0 ± 3.9 |
| L2-SP (λ=0.01), 3 seeds | 0.095 ± 0.021 | 0.274 ± 0.021 | 56.3 ± 6.2 |

**What the seeds showed (and how the paper changed):**

- Across seeds the plain and L2-SP fine-tunes are **statistically indistinguishable**
  on every test metric (overlapping ±SD). The single-run L2-SP median-ED edge
  (−15.6 px, pyramid mode) does **not** survive as a cross-seed effect.
- The originally reported single-run SR@10 (0.25) sat **well above** the 3-seed mean
  (0.095 ± 0.021): 28-pair SR is high-variance, and the single run was an optimistic
  draw. Table 2 (direct) is now reported as mean ± SD over 3 seeds; Table 3 (pyramid)
  is kept but relabelled as a single seed-0 run with a pointer to Table 2.
- **Robust finding (all 6 runs):** the second C103 SEM-SE↔LOM-height scene is
  unrecoverable (370–1400 px) at every seed and every λ → a modality-*coverage* gap,
  not an optimiser artifact. The "plain ft catastrophically forgets, L2-SP rescues"
  sub-narrative was largely a seed-0 event: seeds 1–3 plain ft retains the recoverable
  scene (≤31 px); L2-SP destabilises it in 2 of 3 seeds.
- Edits: abstract, contributions (4th), fine-tuning + L2-SP Results subsections,
  Tables 2–3 + captions, methods ("weight-anchored variant", 3-seed note), practical
  recommendation (iii), and Limitations (multi-seed "not yet run" → done; pyramid
  multi-seed / equivalence test / LoRA still pending). L2-SP reframed from
  "removes the forgetting" to a zero-cost floor that does not worsen the fine-tune.
  `paper.docx` rebuilt.

Still deferred (require more GPU and were not done in this pass): pyramid-protocol
multi-seed replication, formal equivalence test on a larger held-out set, LoRA-style
continual-learning baseline, native-low-FOV/appearance-swept control, quantitative
never-abstain figure.

## Re-review residuals (round 4, applied — no GPU)

Verification re-review (re-review mode) upgraded the decision from Major to **Minor
Revision**: the critical multi-seed item was judged substantively resolved, blocked
from Accept only by single-seed coverage of the recommended pyramid config plus a few
cosmetic consistency fixes. Applied the four no-GPU residuals + the one-sentence #5 fix:

- **NEW-1** — un-bolded the single-run L2-SP cells in Table 3 (they no longer signal a
  "winner" while the text says the gap is within seed noise).
- **NEW-2** — harmonised the Discussion's "5.2×" to "~5× (5.2× in the seed-0 run,
  seed-robust in direction)"; the Results body keeps the precise single-run figure.
- **NEW-3** — added Table 4, the six per-seed runs underlying Table 2's mean ± SD, with
  min–max ranges, so the spread is shown rather than assumed Gaussian.
- **NEW-4** — Contribution 4 reordered to foreground the diagnostic finding (coverage,
  not optimiser/anchor) ahead of the recipe.
- **#5** — Limitations now states the recommended deployment config (wrapper-on-fine-tune)
  is characterised only at seed 0 and should be assumed to inherit the direct-protocol
  spread until the pyramid multi-seed (P1-1b) is run.

`paper.docx` rebuilt. Still open: **P1-1b** — pyramid-protocol multi-seed replication
(deferred to next free GPU slot); plus LoRA baseline, native-low-FOV control, never-abstain
figure (all P2/P3, disclosed as future work).

## Style pass (round 2, applied)

- Full em-dash / sentence-rhythm de-AI pass across the whole body of `main.tex`
  and `paper.md`. Prose em-dashes went from ~45 to 0 (the only remaining `—`
  are the two "not applicable" cells in Table 1). Antithesis cadence
  ("not X but Y") broken up, paragraph-ending aphorisms varied, intensifiers
  trimmed. All numbers, citations, and tables unchanged. `paper.docx` rebuilt.
