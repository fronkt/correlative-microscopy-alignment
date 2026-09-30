# Pre-registration, Arm 2: does label-free triage replicate on independent materials data?

Status: BINDING from the commit that pushes it, which comes before any Arm 2 matcher run. The commit
hash is the timestamp. The final pair list goes in Addendum A, committed before the GPU run.

## Data (materials only, per Frank's decision of 2026-09-29; none of it overlaps AmalgaMatch)
- **P, the public hand-annotated pool: 14 pairs, 12 region clusters.** Built by `scripts/arm2_assemble.py`; manifest
  at `C:\Users\frank\Documents\materials-bench\manifest.csv`. All components are CC BY 4.0.
  - **A, DefDAP HR-DIC vs EBSD: 11 pairs** (Zenodo 8383311, 21218524, 14245480, 16633511, 14532401, 13755208). Each
    pair is EBSD band contrast vs the DIC max-shear map, with the authors' hand-picked homologous points (6–23 per
    pair).
  - **D, refodat.86 BSE vs EBSD band contrast: 3 pairs.** Hand point pairs from QGIS (8–11 per pair).
  - **Excluded before any run, with reasons in `results/arm2/assembly_log.md`:**
    - Li & Shaffer: its landmarks cannot be mapped onto the shipped optical crops, and its LOO residual is 1–6% of
      the diagonal.
    - Ånes 2023: the images are not shipped.
    - Ånes 2022: GPL-3.0.
    - refodat.90: no pairwise points.
- **N, NIST AM Bench 2022 serial sectioning: up to 81 pairs.** The sources are mds2-2767 (IN718) and mds2-2765
  (IN625), under the NIST open-data licence.
  - **Design:** every 25th section. IN718 sections 25–500 give three pair types (BSE1 vs optical, BSE2 vs BSE1, BSE2
    vs optical). IN625 sections 100–600 give one (BSE2 vs BSE1).
  - **Preprocessing:** optical images are downsampled ×4.
  - **GT:** same-modality raw→registered homographies chained through the authors' registered frame, so it inherits
    the authors' mutual-information registration error.
  - **GT points:** a fixed 5×5 grid in the target specimen mask.
  - **Clusters:** alloy × block of 100 sections.
  - **Missing data:** pairs whose files cannot be downloaded (NIST was returning HTTP 524 on 2026-09-30) are left out,
    and the count is reported. No other pair is dropped.
  - **NIST GT gate, decided by a human before the analysis.** Frank hand-clicks 6 corresponding points on each of 20
    pairs, stratified by pair type and depth (list: `tools/handcheck/nist_handcheck_pairs.json`, seed 20260930). For
    each pair type, the median over clicked points of the distance between the GT-mapped click and Frank's click, in
    target px of the pool images, must be ≤ 10 px (half the 20 px success threshold). A type that fails is excluded
    from every Arm 2 analysis. The gate result goes in an addendum before the analysis is run.
- **Overlay check.** The frozen Arm 1 detector (`src/cma/overlay.py`) runs on every image before matching. It found 0
  shared overlays in P and in the 10 NIST pairs built so far. Any shared-overlay pair found later is analysed as it
  is and reported separately.

## Candidates (fixed; the same definitions as the CJSJ run)
- **The 15 voting candidates:**
  - Core: SIFT, LoFTR, RoMa, MA-RoMa and MatchAnything direct, plus RoMa and MA-RoMa pyramid_v2.
  - Transform pool: RoMa and MA-RoMa direct on {invert, histmatch, CLAHE, gradmag}.
- **Settings:** seed 0, MAGSAC++ at 5.5 px, family auto, the same pinned environment.
- **Also:** the GT-fit homography and affine rows (ceiling). No rerun controls.

## Metric
- **Error:** `mu_ed` = the unrefined mean error at GT points in the source frame, from the fitted global H.
- **Success:** `mu_ed` ≤ 20 px (primary). A secondary threshold is 1% of the source-image diagonal, reported for
  every hypothesis, because pixel sizes differ.
- **Failed runs:** a failed run counts as a failure.

## Parameters carried over unchanged from AmalgaMatch (never re-fitted here)
- **The S1 cut-off:** 0.1711, the Youden point on SameSlice + SerialSectioning in the CJSJ run.
- **The best single candidate:** MA-RoMa pyramid_v2, chosen on AmalgaMatch.
- **The scores:**
  - S1 = n_inliers / n_matches.
  - Pick rule R = the voting candidate with the maximum n_inliers (Phase A4/A5; ties go to the candidate listed
    first).

## Hypotheses (one-sided; α = 0.05; primary set = P ∪ the N types that pass the gate)
- **H2-1: S1 predicts success (primary).**
  - Measure: AUROC of S1 for MA-RoMa direct success.
  - Supported if AUROC ≥ 0.80 AND the lower bound of the cluster-bootstrap 95% CI (B = 10,000, clusters as above)
    is > 0.5.
  - Declared untestable, not failed, if there are fewer than 8 successes or fewer than 8 failures.
- **H2-2: the cut-off transfers.**
  - Measure: with S1 ≥ 0.1711, the success rate among accepted MA-RoMa direct pairs vs the base rate.
  - Supported if the cluster-bootstrap CI on the difference is above 0.
- **H2-3: R beats pick-by-S1.**
  - Measure: SR@20 of R vs picking the maximum S1 among the same 15 candidates.
  - Test: exact McNemar, one-sided.
  - Supported if p < 0.05 and R has more successes.
- **H2-4: R beats the best single candidate (secondary; low power is stated in advance).**
  - Measure: SR@20 of R vs MA-RoMa pyramid_v2.
  - Test: exact McNemar, one-sided.
- **Reported for each hypothesis:** P alone and N alone (descriptive), both thresholds, and the oracle best-of-15.

## Honest priors (written before any run)
- **H2-1:** likely, if both classes occur. On AmalgaMatch S1 had AUROC 0.85–0.94, and 0.81–1.00 within every group
  with both classes.
- **H2-2:** uncertain. The cut-off was loose on AmalgaMatch, where precision was 31% on held-out groups.
- **H2-3:** likely, if SIFT returns few matches here too (the mechanism is S1's bias toward sparse matchers).
- **H2-4:** unlikely to reach significance at this n.
- **Floor effect:** DefDAP EBSD vs strain map may be near-total failure, which would make H2-1 untestable in P.

## Exclusions and stopping
- **Infrastructure errors:** rerun once; both attempts are logged.
- **Fixed design:** no candidates, pairs or thresholds are added after Addendum A. Everything else is exploratory.
- **Duplicated sources:** the pools share no images with AmalgaMatch. This was checked by source record; the sha256
  of every image is in the manifest.

## Cost
About 95 pairs × 15 candidates × 1 ≈ 1,400 runs plus the GT rows. By analogy with the CJSJ run, that is about 3–4
GPU-h, or about $2 on one RTX 4090. Credit was $17.29 on 2026-09-30.

## Addendum A: final pair list, written before any Arm 2 matcher run (2026-09-30)
Added after the pre-registration commit a59cb01. NIST's file server came back on the morning of 2026-09-30. No Arm 2
candidate has been run, and no Arm 2 result file exists.
- **Final pool: 95 pairs, 24 clusters.** The pair IDs, clusters, subclasses and image sha256s are frozen in
  `prereg/arm2_addendumA_pairs.csv`.
  - P = 14 (DefDAP 11, refodat.86 3), unchanged.
  - N = 81 of 81. None was lost to the download outage. IN718: 20 BSE1-vs-OM, 20 BSE2-vs-BSE1, 20 BSE2-vs-OM.
    IN625: 21 BSE2-vs-BSE1. There are 12 NIST clusters (alloy × block of 100 sections).
- **Pack.** `matpool.tar.gz` is 762,732,961 bytes, sha256 `cb5a6571249db410dc4f21891b5bee989bdb30e34d7e6ca42821a344070821c8`.
  The manifest sha256 is `d050fca4285d6ff113051d43135a23597df5166e47bcae067b9c69e9a0096c51`. The pack's round trip
  loaded 95 of 95 pairs.
- **NIST GT build.** 102 of 102 same-modality fits are OK. Two optical fits (IN718 s425 and s475) first failed with
  MISSING, because a second copy of the download was still writing the file. Both were re-run on the finished
  files. The GT-to-homography refit residual is at most 1.2e-4 px, and every NIST pair has all 25 grid points inside
  the mask and the source image.
- **Overlay check (frozen Arm 1 detector).** No pair has a SHARED overlay. Five pairs have an overlay on one side only:
  - NIST_625_s200_bse2-bse1 (target bottom 95 px);
  - NIST_718_s300_bse2-bse1 and NIST_718_s300_bse2-om (target top 86 px);
  - defdap_14245480_Ti64_bulk (source top 55 px);
  - defdap_16633511_Ti_CS1 (target bottom 106 px).
  
  As pre-registered, these stay in the pool uncropped. They are listed descriptively and are not a separate analysis
  arm.
- **GT-quality flags.** These carry over from P: affine LOO > 20 px on defdap Ti64_bulk, refodat86 site3 and site5.
  The pairs stay in the pool and are reported. A dataset author told us on 2026-09-30 that the refodat.86 hand
  alignment is "by no means perfect", which is consistent with these flags.
- **Hand-check list.** The list is `tools/handcheck/nist_handcheck_pairs.json`: 20 pairs, seed 20260930, IN718
  BSE1-vs-OM 6, BSE2-vs-BSE1 5, BSE2-vs-OM 5, and IN625 BSE2-vs-BSE1 4. It was generated after all 81 pairs existed,
  and before any matcher run. The gate outcome goes in Addendum B, before `analyze_arm2.py` is run. The GPU run may
  go ahead before the gate, because the gate only decides which NIST types enter the analysis.

## Addendum B: NIST GT gate outcome, written before `analyze_arm2.py` is run (2026-09-30)
The Arm 2 candidate rows exist (3b724f7) but have NOT been analysed. This addendum records the gate, which was
computed with `analyze_arm2.py --gate-only` (`results/arm2/gate_result.json`). That run reads only Frank's click file
and the GT homographies.
- **Hand-check.** Frank clicked 17 of the 20 listed pairs, 6 points each (102 points). He skipped 3 pairs, all with
  optical images: IN718 s025 BSE2-vs-OM, s100 BSE1-vs-OM and s150 BSE1-vs-OM, each recorded as "no confident shared
  features". He clicked blind: the GT was never drawn on the images.
- **Gate result.** The measure is the pooled median, per pair type, of the target-px distance between the GT-mapped
  click and the click. The limit is 10 px.

  | Pair type | Pairs | Points | Median (target px) | Median (source px) | Verdict |
  |---|---|---|---|---|---|
  | IN625 BSE2-vs-BSE1 | 4 | 24 | 8.0 | 4.0 | **PASS** |
  | IN718 BSE2-vs-BSE1 | 5 | 30 | 9.8 | 4.9 | **PASS** |
  | IN718 BSE1-vs-OM | 4 | 24 | 12.5 | 7.8 | **EXCLUDED** |
  | IN718 BSE2-vs-OM | 4 | 24 | 25.4 | 7.9 | **EXCLUDED** |
- **Primary set.** It is P (14) ∪ IN625 BSE2-vs-BSE1 (21) ∪ IN718 BSE2-vs-BSE1 (20) = **55 pairs**. The 40 optical
  pairs are excluded from every Arm 2 analysis, as pre-registered.
- **Descriptive notes (these do not change any verdict):**
  - IN718 s175 BSE2-vs-OM is off by about 400 source px in a consistent direction (mean offset 370 px). The same
    section's BSE2-vs-BSE1 pair agrees within 4 px, and the optical fits have 0.07 px residuals. So this is most
    likely a mislocated patch in the optical image, not a GT error. Without that pair, BSE2-vs-OM still fails (17.4
    target px over 18 points).
  - The gate is measured in target px, which scales source-px click scatter by the inverse of the pair's scale factor
    (×2 for BSE2-vs-BSE1, ×1.6 for BSE1-vs-OM, ×3.2 for BSE2-vs-OM). So the optical exclusions partly reflect how
    precisely clicks can be placed on low-contrast etched optical images. The rule was fixed in advance and is applied
    as written.
  - None of the clicks were redone after the error printout.
