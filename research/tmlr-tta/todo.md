# TODO — TMLR TTA Paper (academic-pipeline)

Plan for the follow-up paper. Branch: `tmlr-tta`. Brief: `research_brief.md`.
Pipeline: deep-research → write → integrity → review → revise → re-review → final integrity → finalize.

## Stage 0 — Setup ✅
- [x] Harden scope via /grilling (8 decisions resolved)
- [x] Write research brief (`research_brief.md`)
- [x] Write this plan
- [ ] Push branch to origin

## RE-PLAN (2026-06-24): brief v2 = two-axis decomposition
Ledger collision caught: multi-scale = SCALE axis, but "backbone domain shift" = APPEARANCE
axis where pyramid is flat & supervised ft forgot. v2 splits shift into scale (multi-scale
consistency) + appearance (cycle + feature-alignment, vs forgetting-prone supervised ft).

## Stage 1 — RESEARCH (deep-research, IN PROGRESS)
- [ ] Lock 2 public OOD datasets emphasizing the APPEARANCE axis (license + access)
- [ ] Exact feature-alignment test-time objective + appearance-severity metric
- [ ] Prior-work map: TTA (TENT/TTT/EATA), dense matching, multi-scale, feature/stat alignment for DA, cycle matching, OOD-distance metrics, forgetting-free finetuning
- [ ] Novelty gap (consistency/alignment TTA for correspondence; scale-vs-appearance decomposition)
- [ ] Deliverables: RQ Brief, Methodology, Bibliography, Synthesis
- [ ] **MANDATORY checkpoint** — confirm before Stage 2

## Stage 2 — WRITE (academic-paper)
- [ ] Implement TTA module on existing `cma/` decoder (norm-affine + certainty head, anchor-to-init)
- [ ] Implement multi-scale-consistency + cycle objectives
- [ ] Run baseline ladder (pyramid-only defended hardest)
- [ ] Build dose-response curve (gain vs measured shift)
- [ ] Draft paper

## Stage 2.5 / 4.5 — INTEGRITY (mandatory gates)
- [ ] Reference/citation/data 100% verification + 7-mode AI failure checklist

## Stage 3 / 3' — REVIEW / RE-REVIEW
- [ ] 5-reviewer review + revision roadmap

## Stage 4 / 4' — REVISE
- [ ] Address roadmap; R&R traceability

## Stage 5 / 6 — FINALIZE + PROCESS SUMMARY
- [ ] Format-convert (TMLR style), PDF from LaTeX
- [ ] Process record

## Key risks (from grilling)
- **Pivotal:** TTA must beat pyramid-only — design experiments to defend this hardest.
- Single-pair adaptation collapse → mitigated by pyramid-as-batch + anchor-to-init.
- "Scales with severity" needs a *measured* shift metric, not asserted.
- TMLR generality bar → 3 domains, not microscopy-only.
