# Phase A synthesis: exploratory, on the already-seen CJSJ runs (2026-09-29)

Everything below was found on the 187 AmalgaMatch pairs that the CJSJ study already analysed, so none of it counts
as confirmed. Each item's own log (`aN_log.md`) lists every variant tried. Opus spot-checked one headline number per item
by recomputing it from `results/triage/candidates.csv`, and every check matched:
- A1: 29/42 near-identity transforms.
- A2: AUROC of −D = 0.870.
- A3: TEM within-subclass AUROC = 0.598.
- A4: pick by n_inliers gives 56/187.
- A5: 10 of the 14 wins are in one TEM scene.
- A6: 29 of the 50 S1 false accepts are banner locks.

## What was found

**A1 + A6. Burned-in metadata banners lock the matchers onto "no motion". This is the main finding.**
- **The banner.** Every MoTaTiZrHf STEM image has a data bar burned into its bottom 9% (Size / Dwell / HT / Scan fov /
  Mag / Pixel size / CL / scale bar). The bar sits at the same position in both images of a pair and is identical
  apart from the detector label. Opus viewed it directly.
- **The lock.** Matchers match the bar and return a transform close to the identity, while the true offset is
  50–500 px.
- **Prevalence.** 67 of the 187 pairs share an overlay at the same position: 66 of the 69 DislocationCharacterization
  pairs and 1 C103 pair. No other group has one.
- **Share of failures.** 475 of the 2,315 failed candidate rows (20.5%) are near-identity locks, and every one is in
  a shared-overlay pair. None occurs in an overlay-free pair.
- **Share of false accepts.** 29 of the 50 S1 false accepts (58%) are locks. Among the 42 held-out TEM false
  accepts, 29 (69%) are.
- **Causal test so far, SIFT only.**
  - Cropping the band removes every near-identity lock: 42 → 0, and 24 → 0 on the other shared pairs.
  - SIFT still succeeds on none of them. So the banner explains the lock, not every SIFT failure.
  - The dense matchers are untested, because that needs a GPU.
- **Confound.** Overlay, identical image size and TEM modality coincide perfectly. The only TEM control without a
  banner is Ti3AlC2 (3 pairs).
- **Rejected.** Repeating structure was rejected: there is no 2-D periodicity, and lattice p ≈ 1.0.
- **Clustering.** The 42 false accepts come from about 5 GT-offset configurations, so any statistic over them has to
  be clustered by configuration.

**A2. Seed disagreement D.**
- On its own, D predicts success as well as S1 does: AUROC 0.870 vs 0.854 for MA-RoMa, and 0.917 vs 0.944 for RoMa.
- It adds nothing reliable beyond S1: +0.022 [−0.003, +0.051] at 6× compute, and +0.005 with one extra run.
- The banner locks are consistent across seeds (median D 7.6 px), which is why D cannot catch them.
- **Dropped.**

**A3. Per-subclass cut-offs from k hand-checked pairs.**
- Rule A (the midpoint rule) saves 6 of about 42 false accepts at k = 3 and does nothing in TEM, where the
  within-subclass AUROC is 0.60 and 0.65.
- Rule B (reject a subclass if all k checked pairs fail) takes the false-accept rate from 0.38 to 0.075, at the cost
  of recall 0.89 → 0.70.
- Rule B works because some image types never succeed, not because it separates good pairs from bad ones.
- **Kept only as a practical protocol recommendation, not as a hypothesis.**

**A4 + A5. A selection score that is fair across matchers.**
- S1 favours matchers with few matches: pick-by-S1 chose SIFT on 79 pairs.
- The a-contrario log-NFA saturates and ends up tracking raw n_inliers (Spearman 0.995 within the dense matchers).
- The Wilson bound does not fix the bias: it still makes 65 SIFT picks.
- **Picking the maximum n_inliers** among the 15 voting candidates gives:
  - 56/187 vs 42 for pick-by-S1: 14 won, 0 lost, p = 0.0001.
  - 56 vs 47 for the best single candidate: 14 won, 5 lost, p = 0.064.
  - Exactly the same outcome as the "RoMa-family-only" rule (E4), with no hand-made list of which matchers to use.
- **The gain over the best single candidate is fragile.**
  - 10 of the 14 wins come from one TEM scene.
  - The CI is [−1.6, +12.4] points.
  - 16 rules have now been tried on these pairs, and Bonferroni gives p = 0.19 even for the best one.

## Proposed Phase B hypotheses (for Frank's approval)

**Arm 1: banner intervention on AmalgaMatch.**
- **Why it can be confirmatory:** the outcome, how the dense matchers behave with the banner removed, has not been
  observed.
- **Freeze before running:** the A6 detector and crop rule (band height as detected), committed and pushed.
- **Runs:**
  - Rerun RoMa and MA-RoMa (direct and pyramid_v2; seed 0) on the 67 shared-overlay pairs, uncropped (a
    reproduction check) and cropped.
  - Controls: the 3 Ti3AlC2 pairs plus a sham crop of the same band on 20 overlay-free pairs, drawn at random with a
    fixed seed. These show that cropping alone changes nothing.
- **Predictions (directional):**
  - P1: near-identity locks among the dense runs fall.
  - P2: MA-RoMa direct SR@20 on the 67 pairs rises (McNemar; clustered bootstrap by scene and offset configuration).
  - P3: S1 false accepts in the held-out groups fall, and S1 AUROC on DislocationCharacterization rises.
- **An honest prior:** P1 is very likely. P2 is uncertain, because SIFT did not recover after cropping. Unlike SIFT,
  RoMa already succeeds on some of these pairs (113 correct rows among the 42).
- **Cost:** about 90 pairs × 8 runs, under 2 GPU-h, about $1 at 4090 rates. The vast.ai credit must be checked first.

**Arm 2: materials replication on new data (Frank: materials only).**
- **H1-rep:**
  - MA-RoMa direct S1 AUROC ≥ 0.80, with a specimen-clustered CI excluding 0.5.
  - The AmalgaMatch cut-off 0.171 applied unchanged gives precision above the base rate.
- **H-pick:**
  - Pick by maximum n_inliers beats pick-by-S1 (the large effect). This is only testable if SIFT is in the pool.
  - Secondary: it beats the AmalgaMatch best single candidate, MA-RoMa pyramid_v2. This is underpowered at about 50
    pairs.
- **Overlay check:** the frozen A6 detector runs on the new data first, and shared-overlay pairs are reported
  separately.
- **Data** (`cma-cjsj/research/materials_benchmark_scan.md`):
  - A pooled set of about 50 hand-annotated pairs from Manchester HR-DIC/EBSD, NTNU EBSD/BSE, Weimar and Imperial,
    about 3–4 GPU-h. It is mostly CC BY 4.0.
  - Optionally, NIST AM Bench serial sections with hand-checked points. Validating its automatic registration on
    about 20 pairs would take Frank about 1 h. Annotating about 150 pairs fully would take 5–8 h.

## Consequences outside this paper (flag only; nothing edited)
- **CJSJ:** nothing in the paper is wrong, since it calls the TEM false accepts "consistent-but-wrong". Under the
  separation rule the paper is submitted unchanged.
- **M&M:** the reported TEM success rates may be partly a benchmark artefact caused by the banners. Check this before
  resubmission.
- **AmalgaMatch authors:** they may want to know. That is Frank's call, and nothing has been sent.
