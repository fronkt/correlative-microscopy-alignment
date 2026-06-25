# TODO — TMLR TTA Paper (academic-pipeline)

Plan for the follow-up paper. Branch: `tmlr-tta`. Brief: `research_brief.md`.
Pipeline: deep-research → write → integrity → review → revise → re-review → final integrity → finalize.

## Stage 0 — Setup ✅
- [x] Harden scope via /grilling (8 decisions resolved)
- [x] Write research brief (`research_brief.md`)
- [x] Write this plan
- [ ] Push branch to origin

## Stage 1 — RESEARCH (deep-research, IN PROGRESS)
- [ ] Lock 2 public OOD datasets with correspondence GT (license + access verified)
- [ ] Prior-work map: TTA (TENT/TTT/EATA), dense matching, multi-scale, OOD-distance metrics, self-supervised/cycle matching
- [ ] Novelty gap vs nearest prior (consistency-based TTA for correspondence)
- [ ] Finalize severity-corruption protocol + feature-space shift metric choice
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
