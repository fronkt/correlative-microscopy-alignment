# Pre-registration, Arm 1: does cropping burned-in overlays remove the "no motion" lock?

Status: BINDING from the commit that pushes it, which comes before any Arm 1 GPU run. The commit
hash is the timestamp.

## Why this is confirmatory although AmalgaMatch has been seen
Phase A (`results/phaseA/synthesis.md`) found the lock on already-seen runs: a metadata banner sits at the same place
in both images of 67 of the 187 pairs, and matchers lock onto it and return close to "no motion". That is an
association. What happens when the dense matchers (RoMa, MA-RoMa) run on banner-free images has **not** been
observed. The only intervention so far used SIFT, which lost its locks but still did not succeed. The predictions
below concern that unobserved outcome.

## Frozen inputs
- **Overlay detector and crop rule:** `src/cma/overlay.py` at the pre-registration commit. It is a faithful port of the
  Phase A6 detector, checked to reproduce A6's per-image flags on all 156 images. It removes the union band (maximum
  of the two detected heights, plus a 2% safety margin) from the same side of both images.
- **Coordinates:** GT points are never dropped. Error is `mu_ed`, the unrefined mean error at all GT points in
  original coordinates, from the fitted global H.
- **Pair lists** (in `results/arm1/`):
  - `pairs_overlay.txt`: the 67 shared-overlay pairs.
  - `pairs_control_ti3alc2.txt`: the 3 TEM pairs without a banner.
  - `pairs_sham.txt`: 20 overlay-free non-TEM pairs drawn with `default_rng(20260929)`.
- **Configurations:** RoMa direct, MA-RoMa direct, RoMa pyramid_v2 and MA-RoMa pyramid_v2, all with seed 0, transform
  none and the same pinned environment as the CJSJ run. Voting-candidate definitions and RANSAC (MAGSAC++, 5.5 px) are
  unchanged.
- **Runs:**
  - Overlay pairs: uncropped and cropped.
  - Ti3AlC2 pairs: uncropped.
  - Sham pairs: uncropped and sham-cropped (the same fractional bottom band removed from both images).
  - Every comparison uses the **fresh** uncropped run from the same box, never the stored CJSJ rows.
- **Definitions:**
  - Success means `mu_ed` ≤ 20 px.
  - A near-identity lock means H lies within 20 px of the identity map (mean over a 5×5 grid on the target) AND
    `mu_ed` > 20 px AND the GT homography lies more than 40 px from the identity.

## Hypotheses (one-sided and directional; α = 0.05)
- **H-A1: the lock goes away.** On the 67 overlay pairs, pooled over the 4 configurations (268 paired runs), the
  number of near-identity locks is lower when cropped.
  - Test: exact McNemar on the paired lock indicator.
  - Supported if p < 0.05 and locks fall.
- **H-A2: success rises (primary).** For MA-RoMa direct on the 67 overlay pairs, SR@20 is higher when cropped.
  - Test: exact McNemar.
  - Supported if p < 0.05 and there are more successes.
  - The other 3 configurations are secondary and reported the same way, with Holm correction across the 3.
- **H-A3: triage improves.** With the S1 cut-off **fixed at 0.1711**, MA-RoMa direct is applied to the 4 held-out
  groups, using the cropped rows for the overlay pairs and the fresh uncropped rows for all other held-out pairs.
  - Prediction: fewer false accepts than the same computation done entirely on uncropped fresh runs.
  - Test: exact McNemar on the per-pair false-accept indicator.
  - Also reported, not tested: the S1 AUROC on DislocationCharacterization, cropped vs uncropped, with a paired
    bootstrap CI.

## Controls and validity checks (reported whatever the outcome)
- **Sham crop.** On the 20 sham pairs × 4 configurations, the change in successes with its McNemar p.
  - If sham cropping changes success on more than 10% of sham runs, the crop itself matters.
  - In that case H-A2 is reported as "cropping changes results generally" and is not attributed to the overlay.
- **Ti3AlC2.** The 3 TEM pairs without a banner succeed in the CJSJ run. Their fresh reruns show whether the
  environment reproduces.
- **Reproduction.** A fresh uncropped run should match the stored CJSJ `mu_ed` within 1 px on at least 95% of runs.
  - If it does not, this is reported; comparisons stay fresh-vs-fresh.
- **The confound, stated in advance.** In AmalgaMatch, overlay, identical image size and TEM modality coincide
  perfectly. Arm 1 can show that removing the band changes the outcome; it cannot separate the band from anything
  else about these TEM images.
- **Clustering.** The 67 pairs come from 4 scenes and about 5 GT-offset configurations, too few clusters for a
  clustered bootstrap. So the tests are pair-level, and results are always also shown per scene and per offset
  configuration. A pair-level p-value treats the pairs as independent, which they are not, and the paper says so.

## Exclusions and stopping
- **Failed runs.** A run that fails (no H) counts as a failure.
- **Infrastructure errors.** A run that stops with a CUDA/OOM/timeout error is rerun once. Both attempts are logged,
  and a second failure counts as a failure.
- **Fixed design.** No configurations, pairs or thresholds are added after the pre-registration commit. Anything
  else is labelled exploratory.

## Honest priors (written before the run)
- H-A1: very likely.
- H-A2: uncertain.
  - Against: SIFT did not recover after cropping.
  - For: RoMa already succeeds on some of these pairs uncropped (113 correct rows among the 42 false accepts' other
    candidates).
- A null on H-A2 alongside a supported H-A1 would still be a useful result: the banner explains the confident
  false accepts, but not the whole failure.

## Implementation details fixed at this commit
- **Detector check.** `scripts/arm1_verify_detector.py` confirms that the detector reproduces A6's flags on
  156/156 images and the shared-overlay flag on 67/67 pairs. Tests: `pytest tests/test_overlay.py
  tests/test_triage.py` gives 30 passed.
- **Crops** (`results/arm1/crop_plans.csv`), all from the bottom:
  - 48 pairs: 188 of 2,192 rows.
  - 18 pairs: 375 of 4,384 rows.
  - The C103 caption pair has different image sizes, so it gets the same height fraction removed from each image
    (the maximum band fraction + 2%): 184 of 2,188 rows (source) and 285 of 3,388 rows (target).
- **Sham crop.** Fixed at 8.6% of the height from the bottom, about the median overlay crop fraction.
- **Sham draw pool.** 112 pairs: overlay-free in both images, excluding Ti3AlC2.
- **Pyramid.** `pyramid_v2` keeps its `scale_ratio`. It is a pixel-size ratio, and cropping changes no pixel size.
- **Runner.** `scripts/run_arm1.py` wraps the unchanged `run_triage_candidates.run_one`. Each row records the crop
  mode, the removed rows and the git commit.
- **Box.** `scripts/box_arm1.sh` runs one process, RoMa then MA-RoMa, so two models never share the card.
- **Known before the run.** On CPU, SIFT reproduces the stored `mu_ed` on 4 of 6 smoke rows. The two that differ have
  only 3–6 RANSAC inliers, which is run-to-run nondeterminism in the stored file versus now. This is why comparisons
  are fresh-vs-fresh.

## Cost
708 runs: 536 overlay, 12 Ti3AlC2, 160 sham. From the stored runtimes that is about 0.5 GPU-h of matching; about 1 h
wall-clock with setup. One RTX 4090 on vast.ai at about $0.4/h comes to under $1. Credit was $17.79 on 2026-09-29.
