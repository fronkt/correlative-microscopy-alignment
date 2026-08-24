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

## P1-1b: pyramid-protocol multi-seed replication (round 5, applied — GPU)

Closed the last open item. Retrained the plain (λ=0) and L2-SP (λ=0.01) decoder-only
fine-tunes at seeds 1–3 and evaluated each checkpoint under **both** the direct and the
pyramid-v2 protocol on the 28-pair held-out test split (`run_p11b.sh`, RTX 5090 box;
metric identical to `ft_test_analysis.py` — mu_ed_tps→mu_ed fallback, SR@k = mean(ed<k),
medED = median of finite ED over 28 pairs). The original multi-seed checkpoints had been
deleted, so these are a fresh independent draw of the same recipe.

Three-seed pyramid result (seeds 1–3, mean ± SD):

- plain ft:  SR@10 0.226 ± 0.021, SR@20 0.262 ± 0.021, medED 49.6 ± 3.1 px
- L2-SP:     SR@10 0.250 ± 0.000, SR@20 0.274 ± 0.021, medED 50.9 ± 3.6 px
- direct (same checkpoints): plain SR@20 0.262 ± 0.021, L2-SP SR@20 0.274 ± 0.021

Two findings folded into the paper:

- **The seed-0 L2-SP pyramid advantage does not replicate.** Earlier-draft seed-0 numbers
  (SR@20 0.321, medED 41 px) were one favourable draw; across seeds 1–3 L2-SP and the plain
  fine-tune are indistinguishable, matching the direct-protocol conclusion.
- **The wrapper does not stack on the fine-tune** on the real test split (pyramid SR@20 =
  direct SR@20 within each config), consistent with the FOV-ladder finding that scale and
  appearance failures are confounded in the real distribution.

**Reproducibility caveat surfaced as a result, not hidden.** Fixed-seed decoder fine-tuning
is not bit-reproducible on this stack (non-deterministic CUDA kernels + multi-worker data
loading): this independent draw lands at direct SR@10 0.250 vs Table 2's 0.095 ± 0.021, i.e.
run-to-run scatter exceeds seed-to-seed scatter at 28 pairs. Table 3 is therefore reported as
a self-contained block where the controlled comparison is *within* the table (each direct/
pyramid pair shares a checkpoint); Table 2 (orig multiseed, direct) is left untouched.

Edits: Table 3 rebuilt as a 3-seed direct+pyramid block with full caption; §"weight anchor"
prose de-claims the seed-0 stacking/CI wins; Limitations records the non-determinism and drops
the pyramid multi-seed from the "not yet run" list. `paper.docx` rebuilt. Decision item
blocking Accept (single-seed pyramid coverage) now resolved.

Still open: LoRA baseline, native-low-FOV control, never-abstain figure (all P2/P3, disclosed
as future work).

## Round 6 (2026-08-24) — Scientific Reports rejection, applied

Submission 46c1e084-d0e4-4fb1-b217-a2b79135181f was **rejected** (Editorial Board
Member sided with a negative R2 against a positive R1). All six reviewer points were
audited against the result files and git history before any text changed. **All six are
correct.** The audit also surfaced three problems neither reviewer raised, and those
drive the largest changes here.

### Found by audit, not by the reviewers

1. **Both native-pair headline results are metric-contingent.** Accuracy is reported on
   TPS-refined error. On raw parametric error the wrapper gain over direct matching is
   exactly flat (0.0909 -> 0.0909, p = 1.0000, against the published +0.021, p = 0.0340)
   and the MatchAnything-RoMa gain falls to +0.016, CI [-0.011, +0.043], p = 0.3338
   (published +0.032, p = 0.0350). Refinement converts 13 otherwise-successful fits into
   failures. TPS coverage is also non-uniform: 1.000 for the dense RoMa-family configs,
   0.000 for Control B, so Table 1 partly compared configurations on different metrics.
   The FOV ladder is the most robust result (+0.150 raw, p = 0.0028; strengthening to
   +0.182 on held-out pairs, p = 0.0234), but it is **not** a both-metrics result: rescored
   under TPS it halves to +0.075 and loses significance (p = 0.0868). An earlier draft of
   this entry claimed it survived both metrics; that claim was made on one-sided p-values
   and is withdrawn (see round 6b).

   *All p-values in this entry are the corrected two-sided values. They were one-sided
   when this round was first written; round 6b below explains why and what changed.* New: `scripts/metric_sensitivity.py`,
   `results/metric_sensitivity.csv`, `reports/metric_sensitivity.md`.

2. **The title claim is not supported by a measured appearance axis.** NMI on the
   GT-aligned overlap (187/187 pairs, ~3.5 min CPU, no GPU) confirms the confound
   (Pearson r = +0.215 vs log area ratio, p = 0.003; median NMI 0.0022 below ratio 0.25
   vs 0.0725 at >= 0.5) but the 2x2 does not rank appearance above FOV: at fixed FOV the
   appearance effect is -0.002 (RoMa) and +0.076 (MA-RoMa), neither significant, while at
   fixed appearance the FOV effect is +0.114 (p = 0.040) and +0.169 (p = 0.045).
   New: `scripts/appearance_axis.py`, `results/appearance_nmi.csv`,
   `reports/appearance_axis.md`. **Title changed accordingly.**

3. **There is no seeding.** `grep` for `manual_seed`, `torch.Generator`, `worker_init_fn`
   returns nothing; `--seed` reaches only the augmentation sampler. The eight "seeds" are
   eight unseeded runs, and the Limitations sentence blaming non-determinism "at fixed
   seed" was a category error. Reworded throughout from "seeds" to "runs".

   Also recorded: the FOV-ladder testbed is 39 train / 13 val / 11 test, so the
   fine-tuned-vs-zero-shot ladder comparison ran on training-enriched, per-backbone
   differing denominators (n = 51 / 40 / 36). Disclosed in Results, Methods and Limitations.

### Reviewer points, as applied

| # | Point | Applied |
|---|---|---|
| R1.1 / R2.3 | Dropped seed run | Table 3 rebuilt with a **Draw** column carrying draw A's six per-run rows next to draw B's ten, recovered from `git show 9b4c2fb^`. Non-retention explained plainly (GPU instance released, checkpoints deleted with it). Added the rebuttal R2's arithmetic invites: the gap is 4 pairs in 28 = **2.0 binomial SE**, not nine, and draw A was *better* on SR@20 (0.297 vs 0.264) and median ED (46.0 vs 49.6), so dropping it made the fine-tune look worse on the metrics the section turns on. |
| R1.2 | Forgetting claim too broad | Evidence base now stated before it is interpreted: three scenes (two test / four pairs, one validation), scenes not pairs as independent units, and the **validation scene of the same untrained combination is undamaged**, which cuts against the coverage reading. Softened to "consistent with" at first use, in the abstract, in contribution 4 and in Results. |
| R2.1 | lambda selected on test | Conceded, not defended. "Test set touched once" withdrawn. Validation ranking now reported (lambda=1.0 best at 15.8 px; **lambda=0.01 worst at 18.9**), so test selection overrode validation and picked the validation-worst value. Single-run CIs at L114 marked as not-inference. Methods records it as a protocol defect with the fix a replication should use. The eight-run null is unaffected and we say why: the bias runs toward L2-SP and L2-SP still does not separate. |
| R2.2 | Appearance by elimination; no leverage | Title changed; measured axis added (above); leverage disclosed (61 of 187 pairs below area ratio 0.5, **at most 3 ever registered** by any zero-shot or wrapped config). The broken causal leg at L81 ("severe-FOV pairs fail on appearance first") is **retracted in place**: at the area ratios those pairs exhibit (0.00013, 0.019) the wrapper already fails on the ladder where appearance is fixed (SR@10 0.025 at rung 0.05, 0.000 at rung 0.02), so scale alone suffices to explain them. |
| R2.4 | Severe-stratum inconsistency | Root cause was undefined terms: "severe" meant 0.05-0.25 at L66, <0.05 at L72/L89/L137, and a ladder rung at L95. Bin edges now stated once in Results and Methods with n = 4 / 33 / 24 / 126, ratio direction corrected to target/source to match the Fig. 5 axis, and the word "severe" retired for **extreme** (<0.05) and **low** (0.05-0.25) FOV. The "only non-zero severe-stratum result (0.00 -> 0.03)" claim is **withdrawn**: direct RoMa already registers that pair to 3.3 px pre-refinement. |
| R2.5 | Figures below standard | All five figures regenerated at 300 dpi. Prose axis labels (no `mu_ed`, no raw config tokens). Per-panel and per-stratum **n**. **Wilson 95 % CIs** on every rate. Config set aligned to Table 1's nine, including Control B, which the figures had silently dropped. `ma_roma_ft` explicitly excluded from all 187-pair panels — it had entered `baselines_A.csv` and would have contributed **train-contaminated** rows (train SR@10 0.389 vs test 0.250). Fig. 5 switched to small multiples so the RoMa + pyramid v2 curve is no longer fully occluded. Result text removed from the Fig. 1 methods schematic. Supplementary `figs/sr_bars_raw.png` shows the raw-metric version. |

### Round 6b — every p-value was one-sided while Methods declared two-sided

Found during post-revision verification, not by any reviewer, and it is the most
consequential single correction in this round.

`scripts/bootstrap_ci.py`, `scripts/fov_ladder_bootstrap.py` and
`scripts/plot_fov_ladder.py` all computed the bootstrap p as a single tail mass,
`(d_boot <= 0).mean()`. The code was honest about it (`bootstrap_ci.py` prints
"p(one-sided)"), but the manuscript's Statistics subsection stated "reported
p-values are two-sided bootstrap probabilities". Every p-value in the submitted
paper was therefore half its declared value.

Fixed by making the code two-sided, `min(1, 2 * min(P(d<=0), P(d>=0)))`, which is
the conservative reading and matches what the manuscript already claimed. CIs, deltas,
n and success rates are unchanged; only p-values move.

| Contrast | one-sided (submitted) | two-sided (now) |
|---|---:|---:|
| RoMa direct to pyramid v2, SR@10, TPS | 0.0170 | **0.0340** |
| RoMa direct to pyramid v2, SR@10, raw | 0.6491 | **1.0000** |
| RoMa to MA-RoMa direct, SR@10, TPS | 0.0175 | **0.0350** |
| RoMa to MA-RoMa direct, SR@10, raw | 0.1669 | **0.3338** |
| FOV ladder rung 0.1, MA-RoMa, raw | 0.0014 | **0.0028** |
| FOV ladder rung 0.1, MA-RoMa, TPS | 0.0434 | **0.0868** |
| FOV ladder rung 0.1, MA-RoMa-ft, raw | 0.0154 | **0.0308** |

No headline conclusion inverts except one: **the FOV ladder no longer clears 0.05 under
the TPS metric** (0.087). The abstract and the H1 verdict had claimed the ladder was
"the one headline that holds under both error metrics"; that claim is withdrawn and
replaced with the accurate version, which is that the ladder carries much the largest
effect in the study, is significant on the unrefined metric it is computed on
(p = 0.0028), and halves to non-significance under TPS refinement because refinement is
close to useless at 10 % FOV, where too few correspondences survive for a spline to help.

Two single-run L2-SP values (median-ED p = 0.0046 and the SR@20 deficit p = 0.91) could
not be recomputed, because the seed-0 checkpoints they came from were deleted. Rather
than publish a p-value we cannot regenerate, both are now reported as their confidence
intervals only, with the interval's relation to zero stated. Those two numbers were
already marked as not-inference because lambda had been selected on the test split.

The Statistics subsection now states the two-sided formula explicitly and records that
success-rate proportions use Wilson score intervals, which matters here because several
strata sit at or near zero.

### Round 6c — forensic number audit, 17 corrections

Every numeric claim in the revised manuscript was re-derived from the result files,
the scripts and git history. Seventeen were wrong, several of them introduced by
round 6 itself. All are now fixed and every replacement value was independently
recomputed before it was written in.

**Errors round 6 introduced**

- **"survives restriction to held-out pairs"** was false for the backbone the p-value
  tests. The fine-tuned ladder contrast held out is +0.167, CI [+0.000, +0.333],
  p = 0.077, n = 18: direction preserved, significance not.
- **The Limitations held-out sentence quoted the wrong backbone.** It discussed the
  fine-tuned ladder but cited the zero-shot numbers (0.045 -> 0.227). Both are now
  given, correctly attributed.
- **A fabricated mechanism.** The H1 verdict blamed the TPS ladder's non-significance
  on "too few correspondences surviving for a spline to help". False: TPS coverage at
  rung 0.1 is 61/61, median inliers 322/523, and refinement *lowers* median ED by
  15.4 px there. The real cause is near-threshold pairs crossing the 10 px line.
- **Limitations contradicted Results** on the between-draw gap: "within the sampling
  error" against "2.0 standard errors". The latter is right.
- **Two abstract overclaims:** "most robust effect" (it is the largest, but under TPS
  it loses significance) and "any configuration" for the sub-0.5 leverage count, which
  is 3 only for zero-shot and wrapped configurations and 32 including the fine-tune.

**Pre-existing errors the audit caught**

- **"zero pairs lost" is true only at SR@10 under TPS.** Per pair the wrapper raises
  TPS error on **94 of 187** pairs, materially on 22, worst case 60.6 -> 1481.7 px.
  The verifier is monotone in its own MI score, not in GT error. Now stated.
- **H2's "dominates ... including every low-FOV stratum"** is false: RoMa and
  MatchAnything-ELoFTR are tied at 0.000 in both low-FOV strata. The support comes
  from the two higher-FOV strata.
- **"36-40 pairs per backbone"** matched neither pool; the true sizes are 38 / 41 / 53.
- **"all straddling the 7-13 px band"**: three of four do; the fourth is 84.8 -> 8.3 px.
- **"validation median 17.6 px"** was a double rounding of 17.546; and step 900 of 1500
  is not "an early checkpoint".
- **"370-1400 px"**: no value above 1032.5 exists in the released files.
- **"80/1220 -> 16/24 px"** comes from the parametric lambda-selection probe, not the
  full pipeline, and the released files show a different shape. Now flagged in place.
- **"unrecoverable in every run"** rests on the three runs examined per scene, not eight.
- **The 69 % affine figure no longer regenerated.** `scripts/h3_family_readout.py` over
  the current `baselines_A.csv` printed 64 %, because `ma_roma_ft` rows had entered the
  file and its well-registered set is mostly its own training data. The script now
  excludes it and reproduces 82/118 = 69 % exactly, restoring the Data-availability claim.
- **The extreme-FOV ladder citation was mis-anchored.** Those pairs sit at area ratios
  0.019 and 0.00013; rung 0.05 is above both and three of the four are more than two
  orders of magnitude below the ladder's floor. The conclusion survives; the citation
  is now accurate.
- **The NMI/FOV relationship is not monotone.** Per-stratum median NMI is 0.0196,
  0.0021, 0.1648, 0.0725, so the 0.25-0.5 stratum is the most mutually informative.
  Disclosed rather than left to imply a monotone trend.

**What the audit confirmed.** Every cell of Table 1; both Table 2 zero-shot rows; all 16
Table 3 per-run values against `git show 9b4c2fb^` and `seed_expand_summary.csv`; every
p-value under the new two-sided convention; the whole appearance-axis analysis; the
pyramid v1 collapse mechanism (106/187 failures, inlier fraction 0.1135 -> 0.0052); the
GT affine residual re-derived from the dataset at median 10.26 px; the 603-tensor and
445 MB checkpoint facts; and the lambda sweep showing 0.01 was validation-worst.

Still not recomputable, and disclosed as such: the per-run values of draw-B runs 1-3,
and the per-scene C103 readouts from the deleted seed-0 checkpoints.

### Not done / open

- **The metric decision is Frank's.** Both metrics are now reported and the contingency is
  disclosed, with TPS kept as primary because it is the declared pipeline. Choosing to
  headline the raw metric instead would null two Table 1 claims outright and is a
  judgement call, not a correction.
- The never-abstain figure (matches vs tile overlap) is still textual. It is inference-only
  logging and remains the single best missing figure.
- LoRA continual-learning baseline; native-low-FOV acquisition control; formal equivalence
  test on a larger held-out set. All still disclosed as future work.
- No new GPU compute was used in this round. Everything above is text, re-analysis of
  existing result files, or CPU-only measurement.
