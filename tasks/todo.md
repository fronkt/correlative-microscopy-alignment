# TODO — Correlative Microscopy Alignment

Track in this file. Check items off as completed.
Source docs: `docs/context.md`, `docs/research_plan.md`, `docs/task_plan.md`.

## Phase 0 — Setup
- [x] 0.1 Repo scaffold + lint (ruff configured; CI deferred)
- [x] 0.2 Pinned env (`pyproject.toml` w/ dev + torch extras)
- [x] 0.3 AmalgaMatch acquired + checksummed + extracted (2026-06-09)
      zip on disk: `data/AmalgaMatch/AmalgaMatch_Dataset.zip` (4,228,037,938 bytes)
      sha256: `084516E37F865B616619ADB40E2100A91BF2F30177AED803EE31A797A0F8AAFB`
      source: https://fordatis.fraunhofer.de/handle/fordatis/478 (DOI 10.24406/fordatis/436, CC-BY-4.0)
      extracted: 19 subsets, 187 pairs confirmed
- [~] 0.4 RoMa / ELoFTR / MatchAnything weights load + smoke-test (RoMa + MatchAnything done; ELoFTR still shell)

## Phase 1 — Baselines
- [x] 1.1 Dataset loader yields (I_s, I_t, K_gt, scale_meta, group, subclass) — rewritten 2026-06-09 for real layout; 187 pairs load, integration tests green
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
- [x] AmalgaMatch GT correspondence coverage — 8-61 hand-annotated points per pair (see results/README.md 2026-06-09)
- [ ] **GT-quality ceiling (NEW, blocking the μ_err<1.5px gate):** GT only fits a
      global affine to median 10.3 px residual (p90 31 px, max 58 px). Either the
      true deformation is non-affine, or annotation noise is ~10 px. Re-derive the
      success gates relative to the GT-fit residual floor, or adopt the original
      paper's evaluation protocol. `scripts/check_gt_consistency.py` has per-pair numbers.
- [ ] Modality pairs with near-zero mutual information — separate reporting agreed?
- [ ] Scale metadata missingness rate — does fallback estimator hold up?

## Review (fill at end)
- Summary of what shipped:
- Results vs hypotheses (H1, H2, H3):
- Surprises / follow-ups:

---

## Resume from here (handoff 2026-06-09, evening)

**Last action:** AmalgaMatch extracted (19 subsets, 187 pairs — exact match to
paper counts). Loader rewritten from scratch for the real layout
(`src/cma/data/amalgamatch.py`); all 33 default tests green including 2 new
real-data integration tests. One real pair ran end-to-end through
`register(..., SIFTMatcher())` — pipeline plumbing confirmed working.

**Real-layout facts (validated against data):**
- `eval_indexs/eval_*.npz` are *pickled dicts* (np.load falls back to pickle),
  keys: `dataset_name, image_paths, image_metadata, pair_infos, gt_2D_matches`.
- GT arrays are (N,4) `[x_i, y_i, x_j, y_j]` in `pair_infos` index order.
- Loader orients pairs so source = wider physical FOV (swaps GT cols; `flipped`
  flag in record). GT-implied scale matches pixel-size ratio (median 2.8% dev).
- Windows MAX_PATH: some release paths are >260 chars; loader uses `\\?\`-prefixed
  absolute paths + cv2.imdecode-on-bytes. LongPathsEnabled NOT required.
- `pair_infos` flag is always 1 in this release. `val_list.txt` lists eval names
  (validation split) — not yet consumed by the loader.

**Key new finding (see Open Questions):** GT fits a global affine only to
median 10.3 px residual → the μ_err < 1.5 px gate as written is likely
unattainable; needs re-derivation against the GT-fit floor or the original
paper's protocol. Read the Durmaz et al. AmalgaMatch paper evaluation section
before running Phase 1.3 headline numbers.

**Concrete next steps, in order:**
1. Read the AmalgaMatch paper's evaluation protocol; align `registration_metrics`
   usage (which direction, which points, what error normalization) and revisit
   the success gates in this file.
2. Phase 1.3 Control A: zero-shot SIFT / LoFTR / RoMa / MatchAnything over all
   187 pairs → `results/baselines_A.parquet`. GPU box recommended for RoMa/MA
   (CPU pyramid+RoMa took ~47 min/pair locally). Mind lessons.md tar exclude
   pattern when shipping data; better: re-download the zip on the box.
3. Phase 1.4 Control B on real data (classical_register).
4. Baseline plots (1.5) + first real FOV-stratified breakdown.

**Other live state at handoff:**
- gh 2.93.0 installed. To use, run: `gh auth login -h github.com -p ssh -w`
- RoMa wired: `src/cma/matchers/roma.py` + `tests/test_roma.py` (marked `slow`).
  CPU smoke ~49s, pyramid warped-pair ~47 min. GPU box will be much faster.
- Slow pytest marker registered in `pyproject.toml`; default suite (`pytest`)
  excludes it, run with `pytest -m slow` to include.
- 31 default tests + 2 slow tests, all green at handoff.
- Open backbone: ELoFTR remains a shell. MatchAnything wrapper already covers
  the ELoFTR architecture via HF transformers, so this may not need wiring.
