# TODO — Correlative Microscopy Alignment

Track in this file. Check items off as completed.
Source docs: `docs/context.md`, `docs/research_plan.md`, `docs/task_plan.md`.

## Phase 0 — Setup
- [x] 0.1 Repo scaffold + lint (ruff configured; CI deferred)
- [x] 0.2 Pinned env (`pyproject.toml` w/ dev + torch extras)
- [~] 0.3 AmalgaMatch acquired + checksummed
      zip on disk: `data/AmalgaMatch/AmalgaMatch_Dataset.zip` (4,228,037,938 bytes)
      sha256: `084516E37F865B616619ADB40E2100A91BF2F30177AED803EE31A797A0F8AAFB`
      source: https://fordatis.fraunhofer.de/handle/fordatis/478 (DOI 10.24406/fordatis/436, CC-BY-4.0)
      NOT YET EXTRACTED — do this next session
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

---

## Resume from here (handoff 2026-06-09)

**Last action:** AmalgaMatch zip downloaded but not extracted. RoMa wiring shipped
to GitHub (initial commit `a87efc6` on `fronkt/correlative-microscopy-alignment`,
public). Pushed via SSH; `gh` CLI now installed (v2.93.0) but not yet authenticated.

**Critical: the loader does NOT match the real layout.** Current
`src/cma/data/amalgamatch.py` assumes `manifest.csv` + `images/<group>/<subclass>/<pair_id>/source.png` etc.
The actual AmalgaMatch layout (from `unzip -l`) is:

```
<SubsetName>/                            # one per subset, e.g. CoNi-AM67_OM-SEM_Multiscale
    eval_indexs/
        eval_<SubsetName>_<N>.npz        # canonical pair lists / GT — NPZ not CSV
        val_list.txt
    image_metadata.json                  # per-image metadata (pixel sizes etc.)
    scenes/<SubsetName>_<idx>/
        <modality>_<frame>.tif           # e.g. BSE_000.tif, SE_000.tif,
                                         # EBSD_000_CI.tiff, EBSD_000_IQ.tiff, EBSD_000_PRIAS.tiff,
                                         # CoNi67_BSE.tif, CoNi67_high_OM.tif, CoNi67_mid_OM.tif
```

Total: 348 files across the subsets we've seen so far (multiple subsets enumerated
in zip listing). Modalities encoded in filename, not directory. GT comes via NPZ
`eval_indexs/`, not a flat keypoints CSV. Pixel/scale metadata lives in per-subset
`image_metadata.json`.

**Concrete next steps, in order:**

1. Extract: `cd data/AmalgaMatch && unzip AmalgaMatch_Dataset.zip` (~5 GB unpacked).
2. Inspect one full subset end-to-end: `image_metadata.json` schema, what's in
   the eval_*.npz (likely pair indices + GT homography / keypoints), how
   `val_list.txt` maps to splits.
3. Rewrite `src/cma/data/amalgamatch.py` from scratch to match this layout.
   Key design decisions: how to enumerate the 187 pairs (modality × scene
   combinations? eval_*.npz rows?), how to map subsets to (group, subclass) in
   the existing `AmalgaMatchRecord` schema, and how to load NPZ GT into the
   existing `KeypointSet` type.
4. Keep `tests/test_amalgamatch_loader.py` (synthetic fixture) passing as a unit
   test of the schema, and add a real-data integration test gated on
   `data/AmalgaMatch/<some-subset>/` being present (skip otherwise).
5. Run one real pair through `register(..., matcher=SIFTMatcher(), ...)` end-to-end
   to confirm the loader + pipeline meet at the seam.
6. Then Phase 1.3 (Control A zero-shot baselines on real data) becomes unblocked.

**Other live state at handoff:**
- gh 2.93.0 installed. To use, run: `gh auth login -h github.com -p ssh -w`
- RoMa wired: `src/cma/matchers/roma.py` + `tests/test_roma.py` (marked `slow`).
  CPU smoke ~49s, pyramid warped-pair ~47 min. GPU box will be much faster.
- Slow pytest marker registered in `pyproject.toml`; default suite (`pytest`)
  excludes it, run with `pytest -m slow` to include.
- 31 default tests + 2 slow tests, all green at handoff.
- Open backbone: ELoFTR remains a shell. MatchAnything wrapper already covers
  the ELoFTR architecture via HF transformers, so this may not need wiring.
