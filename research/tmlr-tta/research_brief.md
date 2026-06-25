# Research Brief — TMLR TTA Paper

**Working title:** *Multi-Scale Self-Consistency as Test-Time Supervision: Label-Free Adaptation of Dense Matchers under Backbone Domain Shift*

**Target venue:** Transactions on Machine Learning Research (TMLR) — rigorous, correctness-based, fast turnaround. CV-method work, judged on soundness not impact.

**Relationship to prior work in this repo:** This is the *follow-up* to the original pyramid-wrapper paper (`paper/`). That paper established that a training-free coarse-to-fine **inference** wrapper recovers dense-matcher accuracy on out-of-domain microscopy. This paper asks whether a genuine **per-pair test-time adaptation** step adds value *on top of* that wrapper, and why.

> Status: Stage 1 (RESEARCH) of the academic-pipeline. This brief is the input artifact for `deep-research`. Items marked **[deep-research]** are deferred to that stage.

---

## 1. Motivation

Dense feature matchers (RoMa, ELoFTR, MatchAnything) inherit a backbone pretrained on natural images. Applied to imagery far from that pretraining distribution (materials microscopy, biomedical, remote sensing), accuracy collapses. Retraining the backbone needs target-domain labels that often do not exist. We ask: **can a matcher repair itself, per image pair, at test time, with no labels and no backbone retraining?**

## 2. Central claim (3-part, falsifiable)

> Per-pair test-time adaptation driven by **multi-scale self-consistency** (+ forward-backward cycle consistency) recovers dense-matcher accuracy under backbone domain shift, such that:
>
> **(i) Practicality** — no target labels, no backbone retraining (encoder frozen).
> **(ii) Dose-response law** — the accuracy gain *scales monotonically with the measured severity of the domain shift* (x-axis = distance between target imagery and backbone-pretraining distribution in frozen-encoder feature space; report correlation r).
> **(iii) In-domain neutrality** — the method does **not** hurt when the backbone is already in-domain, explaining the prior observation that the wrapper "earns its keep only on out-of-domain backbones."

The pivotal empirical question the whole paper defends: **does adaptation beat pyramid-only inference?** If not, there is no paper.

## 3. Method

Built on the existing `cma/` pipeline (pyramid → matcher → consensus). Per test pair:

1. **Pyramid as batch generator.** The coarse-to-fine tiling already produces many tiles from one pair, manufacturing a batch that stabilizes single-instance adaptation.
2. **Self-supervised objective (joint):**
   - **A — Multi-scale consistency (headline):** adapt so that correspondence estimates of the same points *agree across pyramid levels*. The pyramid stops being mere inference and becomes the *source of the TTA signal* — this unifies the paper's two halves.
   - **C — Forward-backward cycle consistency:** match A→B and B→A, penalize disagreement. Folded in as a joint term and as an ablation.
3. **Adapted parameters:** decoder **norm-affine params + certainty head only** (TENT-style), with a **KL/L2 anchor-to-initialization** regularizer to prevent single-pair collapse. Encoder stays frozen. Full-decoder adaptation reported as an ablation ("more params != better under shift").
4. **Cost:** seconds/pair; report honestly. No real-time claims.

## 4. Experimental design

### 4.1 Baseline ladder (pyramid-only defended hardest)

| Tier | Baseline | Proves |
|---|---|---|
| Floor | SIFT+MAGSAC | non-learned reference |
| Floor | Vanilla RoMa / ELoFTR / MatchAnything (no pyramid, no TTA) | the OOD failure |
| **Pivot** | **Pyramid-only inference (no adaptation)** | **isolates TTA's marginal value — make-or-break** |
| Rival | Generic TTA: TENT / BN-stats (on pyramid) | consistency objective beats off-the-shelf TTA |
| Ablation | A-only / C-only / A+C | decomposes the objective |
| Ablation | Full-decoder vs norm-affine | params vs robustness |
| Ceiling | Supervised decoder finetune on target (uses labels) | gap to supervised |
| Rival | AmalgaMatch (existing pipeline) | beats incumbent in home domain |

### 4.2 Shift-severity axis (defends claim ii)

- **Synthetic axis:** graded domain corruptions (modality-gap / contrast / texture, ImageNet-C-style) at increasing strength → clean monotone dose-response curve (internal validity). **[deep-research]** finalize corruption protocol.
- **Backbone-swap axis:** hold imagery fixed, swap backbones pretrained on progressively more distant domains (matches "out-of-domain *backbones*" literally).
- **Shift metric (x-axis):** quantitative distance between target imagery and backbone-pretraining distribution in frozen-encoder feature space. **[deep-research]** pick the exact metric (e.g., FID-like in DINOv2 feature space, MMD, OTDD).

### 4.3 Domains (exactly 3)

- **Home:** materials microscopy (AmalgaMatch + correlative pairs — already in repo).
- **+2 public OOD domains with existing correspondence GT** — **[deep-research]** to select best fit and source (candidates to evaluate: biomedical/histology/retina registration e.g. FIRE; remote-sensing / multimodal optical–SAR; thermal–RGB). Must have public GT to avoid hand-labeling.

### 4.4 Metrics

Existing harness: `P_match@k`, `mu_err`, `med_err`, `success_rate`. Central figure = gain (Δ success_rate or Δ med_err vs pyramid-only) plotted against the measured shift metric.

## 5. Guardrails (non-goals)

- No new architecture; adapt existing RoMa-family decoder.
- Encoder frozen — no encoder adaptation.
- Exactly 3 domains; ≤3 backbones.
- No real-time / latency claims.
- No 3D / video / multi-view — pairwise 2D registration only.
- Single GPU class (RTX 5090, vast.ai) is the compute envelope.

## 6. Deferred to deep-research (Stage 1 deliverables)

- [ ] Lock the 2 public OOD datasets (license, GT format, access).
- [ ] Prior-work positioning: test-time adaptation (TENT/TTT/EATA), dense matching (RoMa/LoFTR/ELoFTR/MatchAnything), multi-scale matching, domain-shift / OOD distance metrics, self-supervised matching (cycle consistency, photometric).
- [ ] Exact severity-corruption protocol + the feature-space shift metric.
- [ ] Novelty gap statement vs nearest prior (esp. any existing "consistency-based TTA for correspondence").
- [ ] Methodology section + bibliography + synthesis.
