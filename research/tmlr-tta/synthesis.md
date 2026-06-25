# Synthesis & Gap Analysis — Stage 1 / Phase 3

Integrates the verified corpus (`bibliography.md`) into the thematic narrative that
positions the paper, with contradictions surfaced and the gap stated.

## Thematic synthesis

**T1 — TTA is mature for classification, near-absent for dense correspondence.**
TENT, TTT/TTT++, CoTTA, EATA, SAR, MEMO form a strong line, but all target
classification/segmentation. The *only* test-time work on dense correspondence is
**Deep Matching Prior** (per-pair optimization), which neither adapts a *pretrained
SOTA* matcher nor distinguishes shift types. This is the opening we occupy.

**T2 — Appearance/modality robustness for matchers has been bought only with supervised
retraining.** **MatchAnything** fixes cross-modal matching by large-scale supervised
pretraining. Our repo's own evidence shows the cheaper version of this — supervised
*decoder finetuning* — works on appearance but **catastrophically forgets** (needing
L2-SP to claw back to neutral). So the appearance axis currently costs either a massive
pretrain (MatchAnything) or a forgetting tax (finetuning). A label-free, stateless,
per-pair route is unclaimed.

**T3 — The two self-supervised signals already exist, but separately and at training
time.** Forward-backward consistency (UnFlow, CRW) is a label-free correspondence signal;
CORAL/AdaBN align feature statistics for domain adaptation; scale-adaptability/consistency
(Scale-Net, PRISM, Jin et al. 2026) is a matching *training* signal. **None is deployed as
a test-time signal on a frozen pretrained matcher, and none is paired axis-specifically.**

**T4 — Forgetting-free finetuning points at our appearance tool and our severity metric.**
EWC/L2-SP/LwF and especially **Proxy-FDA** (ICML 2025) reduce forgetting via
feature-distribution alignment, and Proxy-FDA shows *forgetting correlates with a
distributional (not L2) distance*. This independently motivates (a) feature-distribution
alignment as our appearance signal and (b) a feature-space distance (Fréchet DINOv2 / OTDD)
as the severity x-axis.

## Contradictions / tensions (surfaced, not hidden)

- **Single-instance TTA can be unstable** (TTT++ failure modes; SAR exists precisely to fix
  small-batch instability). We adapt on *one pair* — a real risk. Mitigation is structural:
  the pyramid manufactures a many-tile batch from one pair, plus the L2-SP anchor. This must
  be demonstrated, not assumed.
- **Is affine adaptation even needed over AdaBN's parameter-free stat replacement?** Honest
  open question → resolved by the BN-stats-only (AdaBN-like) vs +affine ablation already in
  the ladder.
- **"Without forgetting" is partly structural.** Stateless per-pair TTA has no persisted
  weights to forget, so the claim is *not* a forgetting-mechanism breakthrough. The honest,
  defensible contribution is **accuracy parity with supervised finetuning on the appearance
  axis at no forgetting cost and no labels** — we will frame it exactly that way and account
  compute fairly (per-pair TTA time vs one-time ft training).

## Gap statement (final)

> TTA is mature for classification but, for dense correspondence, exists only as untrained
> per-pair optimization (Deep Matching Prior); cross-modal matcher robustness has been
> achieved only through supervised retraining (MatchAnything) or forgetting-prone
> finetuning. No prior work (i) decomposes matcher domain shift into distinct **scale** and
> **appearance/modality** axes, (ii) shows these axes require *different* label-free signals
> — multi-scale self-consistency vs. forward-backward cycle + feature-distribution alignment
> — or (iii) deploys these as *test-time* adaptation of a *frozen pretrained* matcher's
> decoder norm-affine parameters that approaches supervised accuracy on appearance without
> its forgetting tax. We close this gap, using a frozen-encoder feature-distance metric to
> place each target on the two axes and select the objective.

## Checkpoint 2 — Devil's Advocate (synthesis)

| Check | Finding |
|---|---|
| Cherry-picking? | No — surfaced TTT++/SAR instability, AdaBN-as-degenerate-case, and the structural triviality of the forgetting claim. |
| Confirmation bias? | Counter-evidence (single-instance TTA fragility) is named with a concrete, testable mitigation, not waved away. |
| Logic chain | T1–T4 → gap is valid; each premise is a verified Tier-A/B source. |
| Alternative explanation | "Maybe pyramid inference alone suffices" → that's exactly Pivot-S, defended hardest. "Maybe AdaBN suffices" → ablation. |

**Verdict: PASS.** Stage 1 deliverables complete (RQ Brief, Methodology, Bibliography,
Synthesis). Ready to hand off to Stage 2 (WRITE) on user confirmation.

## Handoff package → academic-paper (Stage 2)
1. RQ Brief + Methodology Blueprint — `stage1_scoping.md`
2. Annotated Bibliography — `bibliography.md`
3. Synthesis + Gap — `synthesis.md`
4. Brief (scope/guardrails) — `research_brief.md` (v2)
