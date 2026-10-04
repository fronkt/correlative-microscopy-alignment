---
title: "Burned-in data bars cause confident false registrations in correlative microscopy, and the retained-inlier fraction flags failures on new materials data"
running_head: "Data bars and label-free registration triage"
article_type: "Original Article"
---

# Burned-in data bars cause confident false registrations in correlative microscopy, and the retained-inlier fraction flags failures on new materials data

**Frank Cai**

Purdue University, West Lafayette, IN, USA

Correspondence: Frank Cai, frankyc11223@gmail.com. ORCID: 0009-0003-0041-1459

**Running head:** Data bars and label-free registration triage

**Article type:** Original Article

**Keywords:** correlative microscopy; image registration; feature matching; quality control; pre-registration; electron microscopy

---

## Abstract

Pretrained image matchers can register correlative micrographs automatically, but they do not say when they have failed. The share of point matches kept by robust fitting (the retained fraction, S1) has been proposed as a label-free check. On the AmalgaMatch benchmark it tracks which registrations are correct, but a cut-off fixed at S1 = 0.1711 also accepted 42 failed pairs, all scanning transmission electron micrographs of one alloy. This paper reports two pre-registered tests. In Arm 1, an instrument data bar burned into the same rows of both images made the matchers return near-identity transforms. Cropping the bar from both images removed all 110 such locks among 268 paired runs, raised MatchAnything-RoMa success from 6 to 16 of 67 pairs (one-sided exact McNemar p = 0.006) and cut false accepts at the fixed cut-off from 42 to 14 (p = 1.2 × 10⁻⁷); a sham crop of overlay-free pairs changed 8.8% of runs. In Arm 2, the fixed cut-off was carried, without refitting, to 55 materials pairs from three independent sources. S1 predicted success (AUROC 0.983, cluster-bootstrap 95% CI 0.950–1.000) and accepting pairs above the cut-off raised the success rate by 0.10 (0.03–0.21). Picking among 15 registrations by inlier count did not beat the alternatives. Two limits apply. Forty-one of the 55 pairs are same-modality backscattered-electron pairs from NIST on which nearly every method succeeds, so the AUROC is largely a contrast between data sources. Both optical-microscope pair types were excluded by a ground-truth check fixed in advance, so the result does not cover optical-to-electron registration.

## Lay description

Materials scientists often image the same piece of metal or ceramic with several instruments, for example an electron microscope that shows the shape of grains and a second technique that shows how each grain is oriented or how it stretched under load. To combine the pictures, one image has to be laid exactly over the other. This step, called registration, can now be done by computer programs that were trained on everyday photographs. The trouble is that these programs always return an answer, even when it is wrong, and a researcher processing hundreds of image pairs cannot check every one by eye.

A number the programs already compute, the fraction of their proposed point matches that agree with the final answer, can serve as a warning sign: low values usually mean a wrong answer. But on a public benchmark the warning failed on one group of electron microscope images, where the program was confidently wrong. This paper shows why. The microscope had printed a strip of text and a scale bar along the bottom of every image, in the same place. The program matched the two strips of text instead of the material and concluded that the images were not shifted at all. Cutting the strip off both images removed every one of these mistakes and more than doubled the number of correct overlays.

The paper then checked the warning sign on new images from three other laboratories, without retuning it. It still separated right answers from wrong ones well. Both tests were written down and made public before the computer runs were done.

## 1. Introduction

Correlative microscopy combines measurements of one region of a specimen made with different instruments: grain orientations from electron backscatter diffraction (EBSD) with strain fields from high-resolution digital image correlation (HR-DIC), or backscattered-electron (BSE) contrast with phase maps.^1,2^ Before the measurements can be combined, the images must be registered, that is, brought into one coordinate frame. Image matchers pretrained on large photograph collections, such as RoMa^3^ and its cross-modality retrained version MatchAnything-RoMa (MA-RoMa),^4^ can do this without any hand-placed points, but on the AmalgaMatch benchmark of 187 correlative materials pairs^1,5^ they register only about a quarter of the pairs correctly.^6^ They also give no sign of failure: as configured, the dense matchers always output a fixed 10,000 correspondences, even for image pairs that do not overlap.

Automatic registration of a batch therefore needs a label-free check: a score, computed without annotated points, that tells which results to keep. Durmaz et al.^2^ proposed the share of correspondences that survive robust fitting for this purpose, here called the retained fraction S1. Baseline runs on all 187 AmalgaMatch pairs, made with the pipeline of Section 2.2 before the present study, set the starting point. S1 separated MA-RoMa successes from failures; a cut-off of S1 = 0.1711, chosen by Youden's index^7^ on two task groups, was fixed for later use; and on the four remaining (held-out) groups that cut-off accepted 42 failed registrations, all of them dislocation images of a single MoTaTiZrHf alloy. Why it did so was untested. A related manuscript under review at *Microscopy and Microanalysis*^6^ studies a different question, how field-of-view mismatch and image tiling affect the same matchers, and does not address label-free quality checks.

An exploratory audit of the baseline runs, done before the work reported here, suggested a cause. Each of these MoTaTiZrHf scanning transmission electron microscope (STEM) images (labelled TEM in the benchmark) carries an instrument data bar burned into its bottom rows: pixel count, dwell time, camera length, collection angle, detector and scale bar (Fig. 1a). The bar sits at the same pixel position in both images of a pair, while the specimen is offset by 50 to 500 pixels. A matcher that matches the bar finds a large set of mutually consistent correspondences supporting the identity transform; robust fitting keeps them, S1 is high, and the registration is wrong. The failure resembles shortcut learning,^8^ in which a model satisfies its objective through a cue unrelated to the intended structure. The audit found a shared overlay in 67 of the 187 pairs (66 of the 69 dislocation pairs and one fracture-surface pair), but this was a correlation on runs already seen, and its only intervention used SIFT, not the dense matchers.

This paper reports two pre-registered tests. Arm 1 asks whether cropping the shared data bar removes the near-identity locks, raises MA-RoMa success and reduces S1 false accepts on AmalgaMatch. Arm 2 asks whether S1 and the fixed 0.1711 cut-off still work on independent materials data from other laboratories, and whether picking one of 15 candidate registrations per pair by inlier count beats simpler rules. Both protocols were pushed to a public repository ahead of the runs they govern.^9,10^

## 2. Materials and methods

### 2.1 Pre-registration and analysis plan

Each arm has a pre-registration file in the project repository (`prereg/arm1_banner_crop.md`, `prereg/arm2_materials_replication.md`),^10^ binding from the commit that pushed it (91c8b67 for Arm 1, a59cb01 for Arm 2). Each fixes the data, configurations, success criterion, directional hypotheses, tests, support rules and controls. Addenda committed before the relevant runs settled ambiguities: Arm 1 Addendum A (ae57397) before any Arm 1 run; Arm 2 Addendum A (f328135), the frozen pair list, before any matcher run; and Arm 2 Addendum B (dc71e54), the ground-truth gate outcome, before the analysis script was run. The analysis scripts write every pre-registered quantity and list their interpretive choices. Section 2.6 gives the one departure from the original wording. Anything else is labelled exploratory.

### 2.2 Registration pipeline (shared by both arms)

All runs used one code base and pinned software environment. The baseline runs (the candidates below on all 187 AmalgaMatch pairs, made before this study) supply the S1 cut-off and the stored rows referred to below. A registration maps the narrow-field (target) image into the wide-field (source) image. Correspondences come from one of five matchers applied to the whole images: SIFT,^11^ LoFTR,^12^ RoMa,^3^ MA-RoMa^4^ and the MatchAnything version of Efficient LoFTR (MA-ELoFTR).^4,13^ RoMa and MA-RoMa additionally ran within a coarse-to-fine zoom search ("zoom search" below; `pyramid_v2` in the code^6^) and on four label-free image transformations: contrast inversion, histogram matching, contrast-limited adaptive histogram equalisation (CLAHE)^14^ and gradient magnitude. Together these are the 15 voting candidates. Each correspondence set was fitted with MAGSAC++^15^ at a 5.5-pixel inlier threshold, selecting an affine transform or a homography automatically, with seed 0. S1 = (inliers) / (proposed correspondences); a failed run has no transform and counts as a failure.

Registration error (μ) is the mean distance, in source-image pixels, between the annotated source points and the target points mapped by the fitted global transform, with no refinement. A registration succeeds if μ ≤ 20 px.

### 2.3 Arm 1: cropping the shared data bar

**Data.** The 67 AmalgaMatch pairs flagged by the audit as sharing an overlay at the same image edge (CC BY 4.0).^1,5^ A frozen detector (`src/cma/overlay.py`) re-implements the audit's rule and reproduced its flags on 156 of 156 images and 67 of 67 pairs before the pre-registration commit.

**Intervention.** The crop rule removes the union of the two detected bands, plus a 2% safety margin, from the same edge of both images. All crops were at the bottom edge and removed 8.4–8.6% of the image height (for 48 pairs, 188 of 2,192 rows; for 18 pairs, 375 of 4,384; for the one fracture-surface pair, whose images differ in size, the same fraction of each). Annotated points are never dropped: the transform fitted on the cropped images is mapped back to the original coordinates and μ is computed on all points.

**Runs.** RoMa and MA-RoMa, each direct and in the zoom search (4 configurations), on the 67 pairs uncropped and cropped: 536 runs. All comparisons use the fresh uncropped runs from the same GPU instance, not stored baseline rows.

**Definitions.** A near-identity lock is a run whose transform lies within 20 px of the identity (mean displacement over a 5 × 5 grid on the target image), with μ > 20 px, on a pair whose ground-truth transform lies more than 40 px from the identity.

**Hypotheses** (one-sided, α = 0.05, exact McNemar test^16^ on paired indicators):

- **H-A1.** Over the 268 paired runs, cropping reduces near-identity locks.
- **H-A2 (primary).** For MA-RoMa direct on the 67 pairs, cropping raises the number of successes. The other three configurations are secondary, with Holm correction^17^ across the three.
- **H-A3.** With the S1 cut-off fixed at 0.1711, MA-RoMa direct registrations of the 119 pairs in the four held-out groups have fewer false accepts (S1 ≥ 0.1711 and μ > 20 px) when the overlay pairs are cropped than when all pairs are uncropped. Reported but not tested: the S1 AUROC^18^ on the dislocation group, with a pair-level paired bootstrap CI.^19^

**Controls and validity checks** (reported whatever the outcome):

- **Sham crop.** On 20 overlay-free, non-TEM pairs drawn with a fixed seed, the same fraction of the image (8.6%, about the median overlay crop) was removed from the bottom of both images. If this changed success on more than 10% of the 80 sham runs, H-A2 would be reported as "cropping changes results generally" rather than attributed to the bar.
- **Ti3AlC2 control.** The 3 TEM pairs without a bar (Ti3AlC2), run uncropped, check that the environment reproduces the baseline result.
- **Reproduction.** Fresh uncropped runs had to match the stored baseline μ within 1 px on at least 95% of runs.

### 2.4 Arm 2: independent materials replication

**Data.** Pairs were taken only from materials-science sources that share no images with AmalgaMatch; this was checked by source record and image SHA-256. The pool was assembled and frozen before any matcher run (Addendum A).

- **Component A, HR-DIC vs EBSD: 11 pairs.** Six public Zenodo records from DefDAP-based HR-DIC and EBSD studies^20–25^ (CC BY 4.0), processed as in the authors' DefDAP notebooks.^26^ They cover Zircaloy-4 (hydrided; irradiated, four regions), additively manufactured Ti-6Al-4V (two regions), a titanium alloy, CoCrFeNi (coarse and fine grained) and a Ni-base superalloy. Each pair is the EBSD band-contrast map against the HR-DIC maximum-shear-strain map. The ground truth is the authors' own hand-picked homologous points, 6 to 23 per pair.
- **Component D, BSE vs EBSD: 3 pairs.** Hydrated alite (C3S) from the refodat.86 dataset^27^ (CC BY 4.0): a BSE mosaic window against the EBSD band contrast at three sites. The ground truth is the authors' QGIS georeferencing points (8 to 11 per pair).
- **Component N, NIST AM Bench 2022 serial sectioning: 81 pairs.** Data from the AM Bench 2022 IN718 and IN625 serial-sectioning records^28,29^ (NIST open-data licence), described in the AM Bench 2022 overview.^30^ Every 25th section was used. IN718 sections 25 to 500 give three pair types: BSE2 vs BSE1 (two BSE images of the section at pixel sizes about a factor of two apart), BSE1 vs optical micrograph (OM) and BSE2 vs OM. IN625 sections 100 to 600 give BSE2 vs BSE1. The ground truth chains same-modality raw-to-registered homographies through the authors' registered frame. It therefore inherits the error of the authors' mutual-information registration,^31^ and it is sampled on a fixed 5 × 5 grid inside the specimen mask.

Two further sources were excluded before any matcher run (ground-truth frame not recoverable; images not shipped). The frozen pool has 95 pairs in 24 independence clusters (specimen or region for components A and D; alloy × block of 100 sections for N). Component A has 11 clusters, D has 1 and N has 12.

**Ground-truth gate for NIST.** Because the NIST ground truth is derived rather than hand-annotated, the pre-registration required a human check before any NIST pair type could enter the analysis. The author clicked six corresponding points on each of 20 NIST pairs, drawn by a fixed seed and stratified by pair type and depth. He clicked without seeing the ground truth and skipped pairs with no confident shared features. A pair type passed if the pooled median distance between the ground-truth-mapped click and the click was ≤ 10 target pixels, half the success threshold. A failing type is excluded from every Arm 2 analysis.

**Runs.** The 15 voting candidates on all 95 pairs, plus ground-truth-fitted homography and affine rows as a ceiling (1,615 rows). The Arm 1 overlay detector found no shared overlay; five pairs with a one-sided flag were left uncropped, as pre-registered.

**Constants carried over unchanged from AmalgaMatch** (none refitted): the S1 cut-off 0.1711; the best single candidate, MA-RoMa in the zoom search; and the pick rule R, which chooses, among the 15 candidates, the one with the most inliers (ties to the first listed).

**Hypotheses** (primary set = all of components A and D plus the NIST types that pass the gate; success at μ ≤ 20 px is primary, and 1% of the source-image diagonal is secondary, reported for every hypothesis):

- **H2-1 (primary).** S1 predicts MA-RoMa direct success. Supported if AUROC ≥ 0.80 and the lower bound of the cluster-bootstrap 95% CI (B = 10,000) exceeds 0.5. Declared untestable with fewer than 8 successes or 8 failures.
- **H2-2.** The cut-off transfers. Supported if the cluster-bootstrap CI of (success rate among pairs with S1 ≥ 0.1711) minus (base success rate) lies above 0.
- **H2-3.** R registers more pairs than picking the candidate with the highest S1. Tested by one-sided exact McNemar.
- **H2-4 (secondary, low power stated in advance).** R registers more pairs than the best single candidate.

### 2.5 Statistics

McNemar tests are exact and one-sided in the predicted direction (two-sided values are in the analysis files). Arm 1 tests are at pair level, as pre-registered: the 67 overlay pairs come from 4 scenes and about 5 offset configurations, too few clusters for a clustered test, so the p-values overstate the evidence and per-scene results are shown beside them (Table 1). Arm 2 intervals are 95% percentile intervals over cluster resamples; replicates in which a statistic is undefined are dropped and counted.

### 2.6 Deviation from the pre-registration (H-A3)

The body of the Arm 1 pre-registration says H-A3 uses "the fresh uncropped rows for all other held-out pairs". Arm 1 re-ran only the overlay, Ti3AlC2 and sham pairs. Addendum A, committed before any Arm 1 run, therefore specified that the remaining held-out pairs use their stored baseline MA-RoMa direct rows on both sides of the comparison. In the analysis, 77 of the 119 held-out pairs have fresh rows and 42 use stored rows; none of the 42 is an overlay pair. Each such pair contributes the same row to both arms, so it is concordant and cannot change the McNemar statistic, but it does enter the absolute false-accept counts and the dislocation-group AUROC. It is reported here as a deviation. Fresh and stored rows agreed within 1 px on all 360 re-run comparisons (Section 3.1), which bounds its likely effect.

### 2.7 Use of AI tools

The author used an AI assistant (Claude, Anthropic) to help edit the manuscript text and the analysis code. The author reviewed and verified all AI-assisted output and takes full responsibility for the content of this article.

## 3. Results

### 3.1 Validity checks (Arm 1)

All 708 Arm 1 runs completed (536 overlay, 12 Ti3AlC2, 160 sham), with no failed runs. Fresh uncropped runs matched the stored baseline errors within 1 px on 360 of 360 comparisons, and all 12 Ti3AlC2 runs succeeded, as they had in the baseline runs. The sham crop changed the success indicator on 7 of 80 runs (8.8%; 7 successes uncropped, 4 sham-cropped; two-sided McNemar p = 0.45), below the 10% rule. So H-A2 can be attributed to the removal of the data bar rather than to cropping as such.

### 3.2 Cropping the data bar removes the locks (H-A1)

Uncropped, 110 of the 268 overlay runs were near-identity locks: 35 for each MA-RoMa configuration and 20 for each RoMa configuration (Fig. 2a). Cropped, there were none. All 110 discordant pairs went in the predicted direction (one-sided exact McNemar p = 7.7 × 10⁻³⁴). **H-A1 is supported.** Figure 1 shows a typical case. Uncropped, MA-RoMa placed every annotated point about 187 px from its true position, displaced by the ground-truth offset, because it had aligned the two identical data bars. After cropping, the same matcher on the same pair had a mean error of 11 px.

### 3.3 Success rises for MA-RoMa (H-A2)

MA-RoMa direct registered 6 of the 67 pairs uncropped and 16 cropped: 12 pairs gained and 2 lost, one-sided p = 0.0065 (Fig. 2b). **H-A2 is supported.** Among the secondary configurations, MA-RoMa in the zoom search gave the same counts (6 to 16, Holm-adjusted p = 0.019); on these equal-size pairs the search mostly kept the direct result. RoMa, direct or in the zoom search, went from 14 to 16 (5 gained, 3 lost; Holm-adjusted p = 0.73), not significant. RoMa locked less often than MA-RoMa (20 against 35 locks per configuration), so it had less to recover.

Removing the lock did not make the pairs easy. Of the 35 MA-RoMa direct runs that locked when uncropped, the cropped run succeeded on 10. The other 25 still failed, 9 of them narrowly (21 to 30 px) and the rest by 45 to 2,680 px (Fig. 2c). The effect depended on the scene (Table 1). Successes rose from 4 to 13 of 32 in the MoTaTiZrHf 1100 °C scene. In the 900 °C scene, 26 locks disappeared but successes stayed at 0 of 16, and the large-field-of-view 900 °C scene went from 1 to 2 of 18. The bar explains the confident near-identity failures; it does not explain why the matchers fail on the 900 °C images once it is gone.

**Table 1.** Arm 1 by scene (67 overlay pairs; locks pooled over the 4 configurations; success for MA-RoMa direct, μ ≤ 20 px). Descriptive, not tested.

| Scene | Pairs | Locks uncropped | Locks cropped | MA-RoMa direct successes, uncropped | MA-RoMa direct successes, cropped |
|---|---|---|---|---|---|
| MoTaTiZrHf DDRX 1100 °C, dislocations | 32 | 34 | 0 | 4 | 13 |
| MoTaTiZrHf DDRX 900 °C, dislocations | 16 | 26 | 0 | 0 | 0 |
| MoTaTiZrHf DDRX 900 °C, large field of view | 18 | 50 | 0 | 1 | 2 |
| C103, fracture surface (SEM vs optical) | 1 | 0 | 0 | 1 | 1 |
| **Total** | **67** | **110** | **0** | **6** | **16** |

### 3.4 Triage false accepts fall (H-A3)

At the fixed cut-off, MA-RoMa direct produced 42 false accepts among the 119 held-out pairs when every pair was uncropped. With the overlay pairs cropped, it produced 14. Thirty pairs stopped being false accepts and 2 became false accepts, one-sided p = 1.2 × 10⁻⁷ (Fig. 2d). **H-A3 is supported**, subject to the deviation in Section 2.6. All 42 and all 14 false accepts were overlay pairs; no other held-out pair was falsely accepted in either arm. Ranking did not improve within the dislocation group: S1 AUROC on its 69 pairs was 0.811 uncropped and 0.786 cropped (difference −0.025, paired-bootstrap 95% CI −0.151 to 0.106; reported, not tested). Cropping removed most confident failures without making S1 better at ordering the remaining ones.

### 3.5 Arm 2: ground-truth gate and analysed set

The two same-modality NIST types passed the gate: IN625 BSE2 vs BSE1 with a median of 8.0 target px over 24 clicked points, and IN718 BSE2 vs BSE1 with 9.8 px over 30 points. Both optical types failed it, IN718 BSE1 vs OM at 12.5 px and IN718 BSE2 vs OM at 25.4 px (Fig. 3a). (Three optical pairs were skipped as having no confident shared features.) All 40 optical pairs were therefore excluded, leaving a primary set of 55 pairs in 24 clusters: 11 HR-DIC/EBSD, 3 BSE/EBSD and 41 NIST BSE pairs. Seven of 1,615 candidate rows failed with fewer than four correspondences (six MA-ELoFTR, one SIFT) and count as failures. The ground-truth-fitted homography registered all 55 pairs.

### 3.6 S1 predicts success on the new data (H2-1, H2-2)

MA-RoMa direct registered 46 of the 55 pairs. S1 separated its successes from its 9 failures with AUROC 0.983 (cluster-bootstrap 95% CI 0.950 to 1.000; Fig. 3b, c). **H2-1 is supported.** At the fixed cut-off, 48 pairs were accepted, of which 45 were correct (0.938, against a base rate of 0.836). The difference is +0.101 (95% CI 0.028 to 0.208). **H2-2 is supported.** Under the secondary threshold (1% of the source diagonal) MA-RoMa had only 7 failures, so H2-1 was untestable by the pre-registered rule (AUROC 0.991, no CI computed), and H2-2 was again supported (+0.106, CI 0.029 to 0.217).

The three accepted failures were one HR-DIC/EBSD pair (the Ti-6Al-4V pore region, error 349 px) and two of the three alite BSE/EBSD pairs (sites 5 and 3, errors 21 and 34 px). Both alite sites carry pre-registered flags for noisy hand annotation, with affine leave-one-out residuals of about 30 px. Of the 7 pairs below the cut-off, 6 were failures; the one success below it was a NIST IN718 pair registered to 4 px.

The descriptive split shows where the AUROC comes from. On the 41 NIST pairs, MA-RoMa succeeded 39 times and S1 ranked both failures below every success (AUROC 1.000). On the 14 hand-annotated pairs from DefDAP and refodat.86, it succeeded 7 times, and S1 gave AUROC 0.959. These are point estimates with no test, and the 14 pairs come from 12 clusters.

### 3.7 Choosing among 15 candidates (H2-3, H2-4)

Rule R registered 50 of 55 pairs. Picking by highest S1 registered 49 (R alone succeeded on 1 pair, the comparator alone on none; p = 0.5). The best single candidate, MA-RoMa in the zoom search, registered 47 (3 against 0; p = 0.125). **Neither H2-3 nor H2-4 is supported.** An oracle choosing a correct candidate whenever one existed reached 51, so at most one pair separated R from the best possible choice. Results at the 1% threshold were the same (R 51 vs 51 and vs 49). On the 14 hand-annotated pairs alone, R registered 9, pick-by-S1 8 and the best single candidate 7, against an oracle of 10.

## 4. Discussion

### 4.1 A burned-in data bar is a ground-truth-free failure mode

When two micrographs carry the same instrument data bar in the same rows, MA-RoMa and RoMa often register the bar instead of the specimen, and confidently, so S1 accepts the result. Removing the bar from both images removed every such lock, and the sham crop rules out the simplest alternative, that losing the bottom 9% of any image changes results this much. The 42 false accepts that the baseline runs could only attribute to dislocation images of one alloy have at least one specific, fixable cause.

The mechanism is not specific to these matchers. Robust fitting rewards the largest consistent set of correspondences, and identical text and scale bars at fixed pixel positions supply one that supports the identity transform. Because this failure produces a high retained fraction, S1 cannot catch it; the overlay has to be removed before matching. In practice: export images without burned-in annotation or crop it from both images, and treat any automatic result close to the identity on a pair known to be offset as suspect. Benchmark results on overlay-bearing pairs, including the author's own,^6^ partly measure how matchers handle the overlay and are best reported with and without it.

Arm 1 does not show that the bar is the only problem. As stated in the pre-registration, the bar, identical image size and STEM imaging coincide perfectly in AmalgaMatch, so the experiment shows that removing the band changes the outcome but cannot separate the band from everything else about these images. In the 900 °C scenes all locks disappeared but almost nothing succeeded, RoMa gained little, and the S1 ranking within the dislocation group did not improve. Cropping turned confident failures into ordinary failures for some pairs and into successes for others.

### 4.2 The retained fraction transfers, with a strong caveat

On three independent sources the AmalgaMatch cut-off, used unchanged, still separated good registrations from bad ones and raised the precision of accepted results; this is the first check of the score outside the benchmark it was proposed and tested on.^2^ The caveat is the composition of the analysed set. Forty-one of the 55 pairs are NIST BSE pairs on which MA-RoMa failed twice, so most of the AUROC comes from HR-DIC/EBSD and alite failures scoring below easy BSE successes. That is useful to a user, but it is weaker evidence about ranking within a difficult modality than the headline suggests. The 14 hand-annotated pairs alone (7 successes, 7 failures) give AUROC 0.959, a point estimate consistent with the within-group range of the baseline runs, not a test. Two of the three accepted failures missed by 21 and 34 px on pairs whose own annotations have leave-one-out residuals near 30 px, so part of the cut-off's apparent error on new data is ground-truth error.

### 4.3 Choosing among candidates did not help

Rule R, suggested by exploratory analysis of the baseline runs, did not significantly beat picking by S1 or the best single method. With an oracle ceiling of 51 against R's 50 there was almost nothing to gain, and H2-4 was declared low-powered in advance. On data this easy for the strongest single candidate, running 15 registrations is not worth the computation; whether it helps on harder pairs remains open.

### 4.4 Limitations

- **Arm 1.** The bar is confounded with STEM imaging, image size and one alloy. The 67 pairs come from 4 scenes, so pair-level p-values overstate the evidence, and most of the gain is in one scene (Table 1). H-A3 used stored rows for 42 non-overlay pairs (Section 2.6).
- **Arm 2 near-ceiling set.** Forty-one of 55 pairs are same-modality NIST BSE pairs with 39 MA-RoMa successes; the AUROC is largely a between-source contrast.
- **Excluded optical types.** Both NIST optical-vs-BSE types (40 pairs) failed the ground-truth gate, so Arm 2 says nothing about optical-to-electron registration. Because the gate is measured in target pixels, which scales click scatter by the pair's scale factor (about ×1.6 and ×3.2 for these types), the exclusions partly reflect how precisely points can be clicked on etched optical images.
- **Ground truth.** The NIST ground truth inherits the authors' mutual-information registration and was built by fitting raw to registered images of the same modality with SIFT and dense optical flow. That is a different task from the one scored, but not independent of feature matching. The hand-annotated pool is 14 pairs, three with leave-one-out residuals above 20 px.
- **Single runs.** Every candidate was run once (seed 0); the baseline runs showed MA-RoMa run-to-run variation of a few pairs, larger than the H2-3 and H2-4 differences.

## 5. Conclusions

Two pre-registered tests address when automatic correlative registration can be trusted. A data bar burned into the same rows of both images of a correlative pair makes dense matchers lock onto the identity transform with high confidence. Cropping it removed all 110 locks, raised MA-RoMa success from 6 to 16 of 67 pairs and cut false accepts at the fixed S1 cut-off from 42 to 14. On 55 pairs from three independent materials sources, the retained fraction predicted success and the cut-off carried over without refitting. Most of that evidence comes from near-ceiling same-modality BSE pairs, and optical-to-electron pairs could not be tested. Microscopists running automatic registration should remove burned-in annotation before matching, record the retained fraction for every result, and check accepted results by eye on any new kind of image pair.

## Acknowledgements

The author thanks the creators of AmalgaMatch (Durmaz et al.), the DefDAP-based HR-DIC/EBSD datasets, refodat.86 (F. Kleiner) and the NIST AM Bench 2022 serial-sectioning data for releasing their data openly.

## Conflict of interest

The author declares no conflict of interest.

## Funding

This research received no specific grant from any funding agency in the public, commercial or not-for-profit sectors.

## AI-use statement

AI tool use is declared in Section 2.7.

## Data and code availability

Code, pre-registrations with addenda, analysis scripts, per-run result tables (`results/arm1/arm1.csv`, `results/arm2/candidates.csv`), analysis reports and the figure script for this paper are in the project repository, branch `triage-ext` (https://github.com/fronkt/correlative-microscopy-alignment/tree/triage-ext).^10^ No new images were acquired. All images are from public datasets and are not redistributed; scripts rebuild the pairs from the sources:

- AmalgaMatch,^1,5^ CC BY 4.0, doi:10.24406/fordatis/436.
- The six HR-DIC/EBSD Zenodo records,^20–25^ all CC BY 4.0.
- refodat.86,^27^ CC BY 4.0, doi:10.71758/refodat.86.
- NIST AM Bench 2022 records mds2-2767 and mds2-2765,^28,29^ NIST open-data licence (https://www.nist.gov/open/license).

The Arm 2 pair manifest, with source URLs and SHA-256 hashes of every image, is frozen in `prereg/arm2_addendumA_pairs.csv`.

## References

1. Durmaz, A. R., Lamb, J. D., Zaripova, K., Vailhe, M., Pürstl, J. T., Schulte, J., Ackermann, M., Echlin, M. P., & Pollock, T. M. (2026). A correlative microscopy dataset for multimodal data fusion and image matching in materials science. *Scientific Data*. 10.1038/s41597-026-07961-2
2. Durmaz, A. R., Lamb, J. D., Echlin, M. P., & Pollock, T. M. (2026). Foundation models for multimodal image data fusion in materials science. *Frontiers in Materials*, 13, 1815017. 10.3389/fmats.2026.1815017
3. Edstedt, J., Sun, Q., Bökman, G., Wadenbäck, M., & Felsberg, M. (2024). RoMa: Robust dense feature matching. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 19790–19800). 10.1109/CVPR52733.2024.01871
4. He, X., Yu, H., Peng, S., Tan, D., Shen, Z., Bao, H., & Zhou, X. (2025). MatchAnything: Universal cross-modality image matching with large-scale pre-training. *arXiv*, 2501.07556. 10.48550/arXiv.2501.07556
5. Durmaz, A. R. (2026). *AmalgaMatch: A correlative microscopy dataset for multimodal data fusion and image matching in materials science* (Version 1.0.0) [Data set]. Fordatis, Fraunhofer-Gesellschaft. 10.24406/fordatis/436
6. Cai, F. (2026). Image tiling does not solve field-of-view mismatch in correlative microscopy registration. Manuscript MAM-26-277, under review at *Microscopy and Microanalysis*.
7. Youden, W. J. (1950). Index for rating diagnostic tests. *Cancer*, 3(1), 32–35. 10.1002/1097-0142(1950)3:1<32::AID-CNCR2820030106>3.0.CO;2-3
8. Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W., Bethge, M., & Wichmann, F. A. (2020). Shortcut learning in deep neural networks. *Nature Machine Intelligence*, 2(11), 665–673. 10.1038/s42256-020-00257-z
9. Nosek, B. A., Ebersole, C. R., DeHaven, A. C., & Mellor, D. T. (2018). The preregistration revolution. *Proceedings of the National Academy of Sciences*, 115(11), 2600–2606. 10.1073/pnas.1708274114
10. Cai, F. (2026). Pre-registrations, code and results for the triage extension (branch `triage-ext`; Arm 1 pre-registration commit 91c8b67, Arm 2 commit a59cb01). GitHub. https://github.com/fronkt/correlative-microscopy-alignment/tree/triage-ext
11. Lowe, D. G. (2004). Distinctive image features from scale-invariant keypoints. *International Journal of Computer Vision*, 60(2), 91–110. 10.1023/B:VISI.0000029664.99615.94
12. Sun, J., Shen, Z., Wang, Y., Bao, H., & Zhou, X. (2021). LoFTR: Detector-free local feature matching with transformers. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 8918–8927). 10.1109/CVPR46437.2021.00881
13. Wang, Y., He, X., Peng, S., Tan, D., & Zhou, X. (2024). Efficient LoFTR: Semi-dense local feature matching with sparse-like speed. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 21666–21675). 10.1109/CVPR52733.2024.02047
14. Zuiderveld, K. (1994). Contrast limited adaptive histogram equalization. In P. S. Heckbert (Ed.), *Graphics Gems IV* (pp. 474–485). Academic Press. 10.1016/B978-0-12-336156-1.50061-6
15. Barath, D., Noskova, J., Ivashechkin, M., & Matas, J. (2020). MAGSAC++, a fast, reliable and accurate robust estimator. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 1301–1309). 10.1109/CVPR42600.2020.00138
16. McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*, 12(2), 153–157. 10.1007/BF02295996
17. Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70. https://www.jstor.org/stable/4615733
18. Hanley, J. A., & McNeil, B. J. (1982). The meaning and use of the area under a receiver operating characteristic (ROC) curve. *Radiology*, 143(1), 29–36. 10.1148/radiology.143.1.7063747
19. Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall/CRC. 10.1201/9780429246593
20. Thomas, R. (2023). *Dataset for: The role of hydrides and precipitates on the strain localisation behaviour in a zirconium alloy* [Data set]. Zenodo. 10.5281/zenodo.8383311
21. Thomas, R., & Lunt, D. (2026). *Dataset for: The effect of loading direction on slip and twinning in an irradiated zirconium alloy* [Data set]. Zenodo. 10.5281/zenodo.21218524
22. Thomas, R., Lunt, D., Smith, A., Donoghue, J., & Cao, S. (2024). *Dataset: The effect of a keyhole defect on strain localisation in an additive manufactured titanium alloy* [Data set]. Zenodo. 10.5281/zenodo.14245480
23. Smith, A., Lunt, D., Thomas, R., Taylor, M., Davis, A., Martinez, F., Candeias, A., Gholinia, A., Preuss, M., & Donoghue, J. (2025). *Dataset for "A new approach to SEM in-situ thermomechanical experiments through automation"* [Data set]. Zenodo. 10.5281/zenodo.16633511
24. Yang, B., Xu, X., Lunt, D., Zhang, F., Atkinson, M. D., Li, Y., LLorca, J., & Zhou, X. (2024). *Dataset used in the publication entitled "Grain size dependence of microscopic strain distribution in a high entropy alloy at the onset of plastic deformation"* [Data set]. Zenodo. 10.5281/zenodo.14532401
25. Hu, D. (2024). *HRDIC and EBSD data for the study of early stage plasticity of a Ni-base superalloy* [Data set]. Zenodo. 10.5281/zenodo.13755208
26. Atkinson, M. D., Thomas, R., Crowther, P., Fullwood, D., Quinta da Fonseca, J., & Harte, A. (2023). *MechMicroMan/DefDAP: v0.93.6* [Computer software]. Zenodo. 10.5281/zenodo.10160238
27. Kleiner, F. (2026). *BSE and EBSD measurements of 7d hydrated alite* [Data set]. refodat. 10.71758/refodat.86
28. Schwalbach, E. J., Chapman, M. G., Shah, M. N., Uchic, M. D., Hrabe, N., Kafka, O., Moser, N., Lane, B., Carson, R., Belak, J., & Levine, L. E. (2024). *AM Bench 2022: IN718 serial sectioning and X-ray computed tomography measurement data* (Version 1.2.1) [Data set]. National Institute of Standards and Technology. 10.18434/mds2-2767
29. Schwalbach, E. J., Chapman, M. G., Shah, M. N., Uchic, M. D., Levine, L. E., Hrabe, N., Kafka, O., Moser, N., Carson, R., & Belak, J. (2023). *AM Bench 2022 IN625 3D microstructure reconstructions* (Version 1.1.1) [Data set]. National Institute of Standards and Technology. 10.18434/mds2-2765
30. Levine, L. E., Lane, B., Becker, C., Belak, J., Carson, R., Deisenroth, D., et al. (2024). Outcomes and conclusions from the 2022 AM Bench measurements, challenge problems, modeling submissions, and conference. *Integrating Materials and Manufacturing Innovation*, 13(3), 598–621. 10.1007/s40192-024-00372-4
31. Maes, F., Collignon, A., Vandermeulen, D., Marchal, G., & Suetens, P. (1997). Multimodality image registration by maximization of mutual information. *IEEE Transactions on Medical Imaging*, 16(2), 187–198. 10.1109/42.563664

## Figure legends

**Figure 1.** A burned-in data bar causes a near-identity lock. AmalgaMatch pair MoTaTiZrHf DDRX 1100 °C, dislocation characterisation, scene 0, pair 4 (STEM; CC BY 4.0^1,5^). (a) Narrow (target) image. The instrument data bar, at the same rows in both images, is shaded; the dashed line marks the pre-registered crop (bottom 188 of 2,192 rows, removed from both images). (b) Wide (source) image, uncropped. Circles are annotated positions; crosses are where the MA-RoMa direct registration puts the corresponding target points; yellow lines join each pair. The registration aligned the two data bars and is off by the ground-truth offset (mean error 187 px). (c) The same matcher and settings after the bar was cropped from both images (cropped band darkened): mean error 11 px.

**Figure 2.** Arm 1 results on the 67 AmalgaMatch pairs that share a data bar. (a) Near-identity locks per configuration, uncropped and cropped (H-A1; 110 to 0 over 268 paired runs). (b) Pairs registered within 20 px (H-A2; primary configuration MA-RoMa direct, 6 to 16). (c) Mean error of MA-RoMa direct before and after cropping, one point per pair; orange points were locked when uncropped; dashed lines mark 20 px; dotted line marks no change. (d) Left: false accepts at the fixed cut-off S1 ≥ 0.1711 among the 119 held-out pairs, all uncropped vs overlay pairs cropped (H-A3; see Section 2.6 for the row-source deviation). Right: sham-crop control, successes over 20 overlay-free pairs × 4 configurations.

**Figure 3.** Arm 2 results. (a) NIST ground-truth gate: pooled median distance between the ground-truth-mapped hand click and the hand click, per pair type; the 10 px limit was fixed in advance, and both optical types were excluded. (b) Retained fraction S1 against registration error for MA-RoMa direct on the 55-pair primary set, coloured by source; dashed line, the cut-off carried over from AmalgaMatch; dotted line, 20 px. (c) ROC curve of S1 for MA-RoMa direct success (H2-1), with the 14 hand-annotated pairs shown separately (descriptive). (d) Pairs registered within 20 px by MA-RoMa direct, the best single candidate (MA-RoMa zoom search), pick by highest S1, rule R (most inliers), the oracle best of 15, and the homography fitted to the ground truth.
