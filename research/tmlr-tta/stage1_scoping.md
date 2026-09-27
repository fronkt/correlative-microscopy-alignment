# Stage 1 — Scoping Deliverables (deep-research Phase 1)

Generated 2026-06-24. Inputs: `research_brief.md` (v2). Backed by two verified
literature/dataset research passes (see §5 citations).

---

## A. Research Question Brief

**Main RQ:** Does dense-matcher domain shift factorize into a **scale** axis and an
**appearance/modality** axis that demand *different* label-free test-time-adaptation
(TTA) signals — and can axis-aware TTA approach **supervised** finetuning accuracy on
the appearance axis **without its catastrophic forgetting**?

**Sub-questions:**
- **SQ-S (scale):** Does multi-scale self-consistency TTA improve matching under
  scale/FOV shift *beyond pyramid-only inference* (the prior result)?
- **SQ-A (appearance):** Does forward-backward cycle + feature-distribution-alignment
  TTA improve matching under appearance/modality shift, *approaching supervised
  decoder finetuning* while avoiding its forgetting?
- **SQ-X (cross-term):** Are the two signals *axis-specific* — each helps its own axis,
  neither alone covers both — as a function of *measured* scale- and appearance-severity?

**FINER scoring:**
| Criterion | Verdict | Note |
|---|---|---|
| Feasible | ✅ | cma pipeline, FOV ladder, MA-RoMa ft, L2-SP already built; 2 public datasets; single RTX 5090 envelope |
| Interesting | ✅ | TTA for dense matching is underexplored; decomposition is a fresh lens |
| Novel | ✅ | No prior scale/appearance decomposition for matchers; nearest prior (DMP) is distinct |
| Ethical | ✅ | Public datasets; ANHIR is CC-BY-NC-SA (research OK); note 3MOS license before release |
| Relevant | ✅ | Cross-modal matching matters for scientific/medical/remote-sensing imaging |

**In scope:** pairwise 2D registration; frozen encoder; decoder norm-affine TTA;
3 domains; ≤3 backbones; the 2-D severity surface.
**Out of scope:** new architectures; encoder adaptation; 3D/video/multi-view;
real-time claims; certainty-head adaptation (collapse-prone — ablation only).

## B. Methodology Blueprint

- **Paradigm:** positivist / empirical ML; controlled benchmark + ablation.
- **Backbones (≤3):** RoMa, ELoFTR as primary adaptees; **MatchAnything** as the
  *supervised cross-modal reference/ceiling* (it tackles appearance via retraining).
- **Domains (3):** AmalgaMatch (home, materials microscopy) · **ANHIR** (cross-stain
  histology) · **3MOS** (optical–SAR remote sensing).
- **Severity axes (the central 2-D figure):**
  - *Scale severity* = FOV ratio — controllable, appearance-fixed, via existing
    `src/cma/data/fov_ladder.py`.
  - *Appearance severity* = distance between target imagery and the backbone's
    natural-image pretraining proxy in **frozen DINOv2 feature space** — use
    **Fréchet DINOv2 Distance (FDD)** as primary, **OTDD** as cross-check.
- **TTA method:** frozen encoder; adapt **decoder norm-affine params only**; **L2-SP
  anchor-to-init** (mechanism validated offline in this repo).
  - Scale signal: multi-scale consistency (correspondences agree across pyramid levels).
  - Appearance signal: forward-backward cycle consistency **+ CORAL-style alignment of
    decoder feature statistics toward a *precomputed source-domain reference*** (mean/cov
    stored once from in-domain data — resolves "align to what?" at test time).
- **Baseline ladder (updated):**
  | Tier | Baseline | Proves |
  |---|---|---|
  | Floor | SIFT+MAGSAC; vanilla RoMa/ELoFTR/MatchAnything | the OOD failure |
  | **Pivot-S** | Pyramid-only inference | TTA's marginal value on scale |
  | **Pivot-A** | Supervised decoder ft (MA-RoMa ± L2-SP) | label-free TTA ≈ it, no forgetting |
  | **Nearest prior** | **Deep Matching Prior (DMP, ICCV 2021)** | beats per-pair TTO prior |
  | Rival | TENT / BN-stats TTA | axis signals beat generic TTA |
  | Ceiling | MatchAnything (supervised cross-modal) | gap to supervised cross-modal |
  | Ablation | scale-only / appearance-only / both; norm-affine vs full vs +certainty | cross-term + param study |
  | Incumbent | AmalgaMatch | beats incumbent in home domain |
- **Metrics:** `success_rate`, `med_err`, `P_match@k`, `mu_err`; forgetting = source
  retention (as in the L2-SP study).
- **Validity (ledger-driven):** paired bootstrap CIs (`bootstrap_ci.py`); **multi-seed
  from the start** — the ledger shows run-to-run scatter > seed scatter at n≈28, so
  report scatter honestly and size test sets accordingly.

## C. Devil's Advocate — Checkpoint 1

| # | Challenge | Severity | Resolution |
|---|---|---|---|
| 1 | DMP (ICCV 2021) does per-pair TTO for correspondence — is the delta enough? | High | **Add DMP as a baseline** (done in ladder). Delta = SOTA-matcher target, axis-specific signals, decomposition, forgetting claim. |
| 2 | Feature-alignment at test time with frozen encoder — align to *what*? | High | Precompute a **source-domain feature reference** (mean/cov) once; CORAL-align per pair. Specified in §B. |
| 3 | "Axes separable" is an assumption; real datasets entangle scale+appearance | Med | Test it: FOV ladder = scale-only (appearance fixed); fixed-resolution cross-stain ANHIR = appearance-only. The 2-D surface IS the test of separability. |
| 4 | "Without forgetting" is near-trivial for stateless per-pair TTA | Med | Reframe as **accuracy parity with supervised ft, without paying its forgetting tax**; account compute fairly (per-pair TTA vs one-time ft). |
| 5 | 3MOS license unverified; ANHIR non-commercial | Low | Confirm 3MOS repo license; ANHIR NC is fine for a benchmark — note in data statement. |

**Verdict: PASS (with required revisions 1–2 already folded in).** No CRITICAL/blocking
flaw. Proceed to Phase 2 (bibliography + source verification) on user confirmation.

## D. Open items carried into Phase 2

- [ ] Verify 3MOS license on the GitHub repo; re-fetch FIRE in-browser (TLS issue) if promoted.
- [ ] Reconfirm author orderings (LoFTR, EWC, L2-SP, OTDD, TTT++, MEMO) and the two
      scale-consistency matcher arXiv IDs flagged as unverifiable.
- [ ] Confirm MatchAnything venue status (arXiv Jan 2025).

## E. Verified load-bearing citations (nearest neighbors + tools)

1. Wang et al. **TENT**, ICLR 2021 — norm-affine-only TTA precedent.
2. Hong & Kim. **Deep Matching Prior**, ICCV 2021 — *nearest prior; new baseline.*
3. Edstedt et al. **RoMa**, CVPR 2024 — frozen-DINOv2 matcher (design assumption).
4. Wang et al. **Efficient LoFTR**, CVPR 2024.
5. Sun et al. **LoFTR**, CVPR 2021.
6. He et al. **MatchAnything**, arXiv:2501.07556, 2025 — supervised cross-modal rival/ceiling.
7. Meister et al. **UnFlow**, AAAI 2018 — forward-backward consistency.
8. Sun & Saenko. **Deep CORAL**, ECCV 2016 — feature-distribution alignment.
9. Niu et al. **EATA**, ICML 2022 — anti-forgetting TTA.
10. Kirkpatrick et al. **EWC** (PNAS 2017); Xuhong et al. **L2-SP** (ICML 2018).
11. Apple. **Proxy-FDA**, ICML 2025 — forgetting-free ft via distribution alignment.
12. Alvarez-Melis & Fusi. **OTDD**, NeurIPS 2020; + Fréchet DINOv2 Distance practice.
