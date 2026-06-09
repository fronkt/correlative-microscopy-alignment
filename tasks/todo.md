# TODO — Correlative Microscopy Alignment

Track in this file. Check items off as completed.
Source docs: `docs/context.md`, `docs/research_plan.md`, `docs/task_plan.md`.

## Phase 0 — Setup
- [x] 0.1 Repo scaffold + lint (ruff configured; CI deferred)
- [x] 0.2 Pinned env (`pyproject.toml` w/ dev + torch extras)
- [ ] 0.3 AmalgaMatch acquired + checksummed
- [~] 0.4 RoMa / ELoFTR / MatchAnything weights load + smoke-test (RoMa + MatchAnything done; ELoFTR still shell)

## Phase 1 — Baselines
- [x] 1.1 Dataset loader yields (I_s, I_t, K_gt, scale_meta, group, subclass) — interface, awaits real data
- [x] 1.2 Metric harness (P_match@{1,3,5,10}, mu_err, med_err, success) — runtime/mem deferred
- [ ] 1.3 Control A: zero-shot each backbone, persist to `results/baselines_A.parquet`
- [x] 1.4 Control B: SIFT + MMI (classical_register) — synthetic numbers logged; AmalgaMatch run pending
- [ ] 1.5 Baseline plots in `reports/figs/baselines/`

## Phase 2 — Pyramidal Wrapper
- [x] 2.1 `pyramid.build(...)` + back-projection metadata
- [ ] 2.2 Scale-metadata fallback estimator
- [~] 2.3 `Matcher` ABC + SIFT/RoMa/MatchAnything done; ELoFTR still a shell
- [ ] 2.4 Tile-batched fp16 inference + memory guard
- [x] 2.5 Correspondence aggregator (currently in pipeline.register; refactor if needed)

## Phase 3 — Consensus + Transform
- [x] 3.1 MAGSAC++ wrapper (cv2 USAC_MAGSAC)
- [x] 3.2 Affine + Homography fitters + per-pair selection (BIC-style score)
- [x] 3.3 `register(I_s, I_t, backbone)` end-to-end
- [x] 3.4 Synthetic-pair test recovers H to <2 px via SIFT (oracle path consensus-only)

## Phase 4 — Full Evaluation
- [ ] 4.1 Experimental run on test split for all 3 backbones
- [ ] 4.2 Headline table (Control A / B / Exp) per group
- [ ] 4.3 Paired-bootstrap CIs + significance markers
- [ ] 4.4 Draft results section

## Phase 5 — Ablations + Sensitivity
- [~] 5.1 FOV sweep {50, 25, 10, 5, 2}% — SIFT placeholder done on synthetic; re-run per backbone on AmalgaMatch
- [ ] 5.2 Pyramid depth {1, 2, 3, 4}
- [ ] 5.3 Overlap {25, 50, 75}%
- [ ] 5.4 RANSAC threshold sweep
- [ ] 5.5 Affine vs homography
- [ ] 5.6 Scale-metadata error +/-20%

## Phase 6 — Writeup + Release
- [ ] 6.1 Tech report (figures + tables + failure modes)
- [ ] 6.2 README single-command reproduction
- [ ] 6.3 v0.1 tag + archived results bundle

---

## Success Gates (block release until met)
- [ ] P_match @ 5 px > 85% on AmalgaMatch test
- [ ] mu_err < 1.5 px on AmalgaMatch test
- [ ] >= 35% relative mu_err improvement vs best zero-shot at FOV <= 5%
- [ ] FOV breakdown curves logged for every backbone

## Open Questions / Risks (track here, resolve before release)
- [ ] AmalgaMatch GT correspondence coverage — confirm density per group
- [ ] Modality pairs with near-zero mutual information — separate reporting agreed?
- [ ] Scale metadata missingness rate — does fallback estimator hold up?

## Review (fill at end)
- Summary of what shipped:
- Results vs hypotheses (H1, H2, H3):
- Surprises / follow-ups:
