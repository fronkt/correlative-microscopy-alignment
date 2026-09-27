# Research Brief — TMLR TTA Paper (v2, two-axis)

**Working title:** *Scale and Appearance are Different: Label-Free Test-Time Adaptation of Dense Matchers Decomposes Domain Shift into Two Axes*

**Target venue:** Transactions on Machine Learning Research (TMLR) — rigorous, correctness-based, fast turnaround.

**Relationship to prior work in this repo:** Follow-up to the original pyramid-wrapper paper (`paper/`). That paper showed a coarse-to-fine **inference** wrapper recovers accuracy under **scale/FOV** shift but is **flat on appearance**, and that appearance OOD required **supervised** decoder finetuning (MA-RoMa) which **catastrophically forgot** (mitigated by an L2-SP anchor). This paper turns that scale-vs-appearance split into its thesis and asks whether a single **label-free, per-pair test-time adaptation** framework can address *both* axes — beating supervised finetuning on appearance *without* its forgetting.

> Stage 1 (RESEARCH) input artifact for `deep-research`. Items marked **[deep-research]** are deferred. **v2 supersedes v1** after re-plan against the project ledger (scale ≠ appearance; see §0).

---

## 0. Why v2 (the collision v1 ignored)

The project ledger establishes — with significant results — that **domain shift is not one axis**:
- **Scale/FOV axis:** pyramid/multi-scale processing recovers matching (FOV 0.1: SR@10 **0.23 vs 0.07** direct, p=0.0014). Multi-scale consistency is a *scale* signal.
- **Appearance/modality axis:** the pyramid is **flat**; "severe-FOV pairs fail on appearance first." Appearance OOD was only fixed by **supervised** finetuning, which **forgot** (net SR@20 0.393→0.250) until L2-SP clawed it back to neutral.

v1's headline ("multi-scale self-consistency recovers *backbone-domain* shift") conflated these and risked a **null central figure**. v2 separates them.

## 1. Central claim (two-axis, falsifiable)

> Domain shift for dense matchers decomposes into a **scale** axis and an **appearance** axis that demand **different label-free test-time signals**:
>
> **(S) Scale axis** — **multi-scale self-consistency** as test-time supervision recovers matching under scale/FOV shift (extends the prior inference-only result into a per-pair *adaptation*).
> **(A) Appearance axis** — **forward-backward cycle + feature-distribution alignment** as test-time supervision recovers matching under appearance/modality shift, **approaching supervised-finetuning accuracy without its catastrophic forgetting**.
> **(X) Cross-term** — the two signals are **complementary, not redundant**: each helps its own axis and neither alone covers both; combined TTA tracks a 2-D dose-response surface (gain vs measured scale-severity × appearance-severity).

Pivotal questions defended hardest: (1) per-pair TTA beats pyramid-only inference on the **scale** axis; (2) label-free cycle/feature TTA beats **forgetting-prone supervised finetuning** on the **appearance** axis (matched compute, no labels).

## 2. Method

Per test pair, built on existing `cma/` (pyramid → matcher → consensus):

1. **Pyramid as batch generator** — many tiles from one pair → stabilizes single-instance adaptation.
2. **Two self-supervised signals, mapped to two axes:**
   - **Scale:** multi-scale consistency — adapt so correspondences agree across pyramid levels.
   - **Appearance:** forward-backward cycle consistency + feature-distribution alignment (pull target-image features toward the backbone's source-domain statistics). **[deep-research]** confirm the exact feature-alignment objective.
3. **Adapted parameters:** decoder **norm-affine params only**, with an **L2-SP / KL anchor-to-init** (mechanism already validated offline in this repo). **Certainty head is NOT adapted by default** — the ledger shows certainty-head collapse under adaptation; include it only as a documented ablation.
4. **Cost:** seconds/pair; reported honestly. No real-time claims.

## 3. Experimental design

### 3.1 Baseline ladder (two pivots)

| Tier | Baseline | Proves |
|---|---|---|
| Floor | SIFT+MAGSAC | non-learned reference |
| Floor | Vanilla RoMa / ELoFTR / MatchAnything | the OOD failure |
| **Pivot-S** | **Pyramid-only inference (no adaptation)** | **TTA's marginal value on the scale axis** |
| **Pivot-A** | **Supervised decoder ft (MA-RoMa ± L2-SP)** | **label-free TTA approaches it without forgetting** |
| Rival | Generic TTA: TENT / BN-stats | consistency/alignment objectives beat off-the-shelf TTA |
| Ablation | Scale-signal only / appearance-signal only / both | the cross-term (X): complementarity |
| Ablation | norm-affine vs full-decoder; +certainty-head | params vs robustness; document collapse |
| Rival | AmalgaMatch | beats incumbent in home domain |

### 3.2 Two shift-severity axes (defend the (S)(A)(X) claim)

- **Scale severity:** FOV ratio — **already implemented** (`src/cma/data/fov_ladder.py`, crops base-matchable pairs holding appearance fixed). Strong existing evidence.
- **Appearance severity:** quantitative distance between target imagery and backbone-pretraining distribution in **frozen-encoder (DINOv2) feature space**. **[deep-research]** pick exact metric (FID-like / MMD / OTDD).
- **Central figure:** 2-D dose-response surface — gain vs (scale-severity × appearance-severity) — showing each signal lights up its own axis.

### 3.3 Domains (exactly 3)

- **Home:** materials microscopy (AmalgaMatch + correlative pairs, in repo; appearance shift is real here — cross-modal SEM/EBSD/AFM/TEM).
- **+2 public OOD domains with correspondence GT** — **[deep-research]** select for best *appearance*-axis coverage (candidates: biomedical/histology/retina e.g. FIRE; multimodal optical–SAR / thermal–RGB). Must have public GT.

### 3.4 Metrics

Existing harness: `P_match@k`, `mu_err`, `med_err`, `success_rate`. Forgetting measured as retention on in-domain pairs (as in the L2-SP study).

## 4. Guardrails (non-goals)

- No new architecture; adapt existing RoMa-family decoder. Encoder frozen.
- Two axes, exactly 3 domains, ≤3 backbones.
- No real-time / latency claims. No 3D / video / multi-view.
- Single GPU class (RTX 5090, vast.ai) is the compute envelope.

## 5. Deferred to deep-research (Stage 1 deliverables)

- [ ] Lock 2 public OOD datasets emphasizing the **appearance** axis (license, GT, access).
- [ ] Exact **feature-alignment** test-time objective + **appearance-severity** metric.
- [ ] Prior work: TTA (TENT/TTT/EATA), dense matching, multi-scale matching, **feature/statistic alignment for domain adaptation**, cycle-consistent matching, OOD-distance metrics, **forgetting-free finetuning** (the supervised-ft rival's literature).
- [ ] Novelty gap vs nearest prior (esp. consistency/alignment-based TTA for correspondence; any work decomposing shift into scale vs appearance).
- [ ] Methodology section + bibliography + synthesis.
