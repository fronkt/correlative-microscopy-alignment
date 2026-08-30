# Image tiling does not solve field-of-view mismatch in correlative microscopy registration

**Frank Cai**

Purdue University, West Lafayette, IN 47907, USA

Correspondence: frankyc11223@gmail.com · ORCID 0009-0003-0041-1459

**Running head:** Image tiling and field-of-view mismatch

**Keywords:** correlative microscopy, image registration, dense feature matching, field of view, multimodal imaging, benchmarking

---

## Abstract

Correlative microscopy pairs images of one specimen taken on different instruments. They share little visual appearance and can differ in field of view by more than an order of magnitude, and bringing them into a common coordinate frame is the bottleneck. Pretrained dense image matchers are the strongest tools now available for this, but they fail when the field of view is badly mismatched. An appealing remedy needs no training: cut the wider image into tiles, match each tile, pool the correspondences, and fit one transform. Across all 187 pairs of a public correlative microscopy benchmark, we show this makes matters far worse, and identify the reason. Dense matchers never decline to match, so a tile that does not overlap the target still returns thousands of confident correspondences, and robust estimation is left with no consensus to find. Median error rises from 80 to 2708 pixels. A verified coarse-to-fine wrapper removes the collapse but gains nothing overall on this benchmark, while on a controlled ladder that shrinks field of view with appearance held fixed it triples success. We also find that scoring the same experiments after an optional refinement stage turns two null results into significant ones.

---

## 1. Introduction

Correlative microscopy spatially associates measurements of a single specimen acquired on different instruments: scanning electron microscopy (SEM), electron backscatter diffraction (EBSD), transmission electron microscopy (TEM), light optical microscopy (LOM), and derived modalities such as image-quality and inverse-pole-figure maps. Linking a dislocation configuration seen in TEM to a grain orientation measured by EBSD, or slip partitioning observed in SEM to the underlying microstructure, requires that the two images be brought into a common coordinate frame to sub-grain precision. That step is image registration, and it is the practical bottleneck (Durmaz et al., 2026b).

The regime couples two compounding difficulties. The modalities share little mutual information, because a secondary-electron contrast image and a crystallographic orientation map are produced by entirely different physics. And the field of view (FOV) can differ by more than an order of magnitude, with the narrow-FOV image covering as little as 2 % of the wide-FOV frame. Classical intensity- and feature-based registration assumes shared appearance and comparable scale, and fails outright in this regime (Durmaz et al., 2026a).

Learned dense feature matchers have transformed correspondence estimation for natural images, and are the obvious candidate remedy. Detector-free transformer matchers such as LoFTR (Sun et al., 2021) and its efficient successor (Wang et al., 2024) produce semi-dense matches in texture-poor scenes. RoMa (Edstedt et al., 2024) couples frozen DINOv2 features (Oquab et al., 2024) with a fine-feature branch and a match-classification decoder to produce a pixel-dense warp together with a certainty map. MatchAnything (He et al., 2025) retrains such backbones on large-scale synthetic cross-modal signal and reports strong transfer across unseen modality pairs from a single set of weights. Durmaz et al. (2026a) introduced the AmalgaMatch benchmark and showed that classical registration fails on it; Durmaz et al. (2026b) evaluated foundation matchers on the same data and found the MatchAnything-RoMa configuration strongest across most subsets.

One property of this family of matchers is central to what follows. Sparse learned matchers built on keypoint detection and attention-based assignment (Sarlin et al., 2020; Lindenberger et al., 2023) are able to decline to match. Detector-free dense matchers instead emit a confident correspondence at essentially every pixel, and never abstain.

The natural way to spend that density on a FOV gap is to tile: crop the wide-FOV image into target-resolution tiles, match each tile against the narrow-FOV image, pool all resulting correspondences, and fit a single global transform by robust estimation. The approach is attractive because it requires no training and treats the matcher as a black box. This paper reports what happens when that idea is executed carefully across all 187 pairs of AmalgaMatch, and the outcome is diagnostic rather than methodological.

Our contributions are four.

1. **A mechanism.** Tiling does not merely fail to help; it destroys the matcher it wraps, and it does so because dense matchers cannot abstain (Section 3.2). Non-overlapping tiles contribute confident but spurious correspondences, the inlier fraction collapses by a factor of 22, and the robust estimator has no consensus left to find. The argument predicts failure for any pool-then-fit tiling scheme over a non-abstaining matcher, independent of tile size or overlap.

2. **A wrapper that survives, and an honest account of what it buys.** A verified coarse-to-fine design removes the collapse (Section 3.3), but on the native benchmark its aggregate effect is exactly zero. What it does is change which pairs succeed, in the direction its scale mechanism predicts.

3. **A controlled decoupling.** A FOV ladder that crops real pairs with appearance held fixed shows the scale mechanism is sound where scale is the isolated variable (Section 3.4), and a directly measured appearance axis shows why the native benchmark cannot show this: scale and appearance are confounded on it, and it carries too little leverage below an area ratio of 0.5 to separate them (Section 3.6).

4. **A methodological hazard.** An optional refinement stage with non-uniform coverage across configurations converts both of our null results into significant ones (Section 3.8). We report the consequences for our own claims in full, and argue the failure mode is general.

We additionally report a domain fine-tuning study (Section 3.7), which cuts in-distribution error roughly fivefold while regressing performance on the one modality combination absent from training, a form of catastrophic forgetting (Kirkpatrick et al., 2017) that a weight-anchoring remedy (Li et al., 2018) does not repair.

The original project hypotheses were **(H1)** that tiling yields a 35 % relative gain at FOV below 5 % over the best zero-shot baseline; **(H2)** that RoMa's dense flow tolerates partial overlap better than the ELoFTR family at low FOV; and **(H3)** that an affine model suffices for AmalgaMatch, with full homography offering no significant benefit. Verdicts on all three are stated in Section 5.

## 2. Materials and Methods

### 2.1. Benchmark

AmalgaMatch (Durmaz et al., 2026a) contains 187 image pairs spanning six correlative-microscopy task groups (orientation mapping, serial sectioning, multiscale, slip partitioning, fracture surfaces, dislocation characterisation) across 19 material subsets. Each pair carries ground-truth (GT) point correspondences. The dataset is publicly available from the Fraunhofer Fordatis repository.

### 2.2. Registration pipeline

For each pair we compute correspondences with the matcher under test, fit a transform with MAGSAC++ (Barath et al., 2020) at a 5.5 px reprojection threshold, and select between a 6-degree-of-freedom affine and an 8-degree-of-freedom homography by a Bayesian-information-criterion-style penalty on inlier residuals. MAGSAC++ is a marginalising estimator in the random sample consensus lineage (Fischler & Bolles, 1981) that replaces a hard inlier threshold with marginalisation over noise scale.

Accuracy is the mean Euclidean distance (ED) between GT target points mapped into source coordinates and their GT source positions. Aggregates are the median ED over pairs and the success rate at 5, 10 and 20 px, written SR@5, SR@10 and SR@20, defined as the fraction of the 187 pairs registered to within that mean error.

Matches are capped at 10,000 per matcher invocation, a cap RoMa reaches on every pair. Tiled modes pool across invocations and are not capped in aggregate.

### 2.3. Error metric

The pipeline optionally refines the fitted transform on its inliers with a thin-plate spline (TPS) (Bookstein, 1989). **We report the unrefined parametric error as the primary metric and the refined error alongside.** This is a deliberate reversal of our original protocol, and Section 3.8 gives the evidence for it: TPS coverage ranges from 0.000 to 1.000 across configurations, so a TPS-scored table compares some configurations under refined error and others under raw error. The unrefined metric scores every configuration identically. We note the direction of the change: it removes statistical significance from two of our results rather than creating it.

### 2.4. Field-of-view strata

We bin pairs by the GT-implied target-to-source area ratio, with edges below 0.05 (n = 4), 0.05–0.25 (n = 33), 0.25–0.5 (n = 24) and above 0.5 (n = 126), calling the first two the extreme- and low-FOV strata. Only four pairs sit below ratio 0.05 under any definition, so no claim resting on that stratum alone can carry weight.

### 2.5. A protocol floor

The GT correspondences themselves fit a global affine only to a median residual of 10.3 px. Threshold metrics are therefore floored well above the sub-pixel precision our original plan assumed, and any sub-5 px claim on this benchmark measures the GT more than it measures the method. We report SR@5 for completeness and rest no conclusion on it.

### 2.6. Statistics

Method-versus-method comparisons use a paired bootstrap over per-pair errors (B = 10,000 resamples) with 95 % percentile confidence intervals (CIs) on the difference of the relevant aggregate. Reported *p*-values are two-sided, computed as twice the smaller tail mass of the bootstrap difference distribution and capped at 1. Success-rate proportions carry Wilson score intervals, which behave correctly near zero; several strata here sit at or near zero. The bootstrap is seeded and its seed is in the released scripts.

### 2.7. Matchers and weights

We evaluate SIFT (Lowe, 2004), LoFTR (Sun et al., 2021), the MatchAnything-ELoFTR configuration (Wang et al., 2024; He et al., 2025), RoMa (Edstedt et al., 2024), and MatchAnything-RoMa (He et al., 2025). A classical control adds mutual-information intensity refinement (Maes et al., 1997) to SIFT. The released MatchAnything-RoMa checkpoint is key-for-key compatible with the RoMa outdoor architecture (603 of 603 tensors), so it loads into the identical pipeline without modification.

### 2.8. Tiling wrappers

**Pyramid v1** tiles the source into a sliding-window pyramid at 50 % overlap, whose levels match the target's nominal physical resolution; it matches every tile, back-projects and pools all correspondences, then fits one global transform.

**Pyramid v2** computes a direct match and an incumbent transform first, then evaluates candidate stages — tile search over a scaled source grid, followed by zoom refinement on the best-scoring region — only when direct support is weak, and accepts a candidate only if a mutual-information-on-overlap verifier scores it above the incumbent. Ablations cover iterated versus single zoom, and certainty gating at 0.5.

### 2.9. Controlled field-of-view ladder

For each backbone we take its base-matchable pairs (direct mean ED below 20 px) and crop the target to absolute area ratios of 0.5, 0.25, 0.1, 0.05 and 0.02, holding appearance, modality and pixel size fixed. Full GT is retained so that out-of-crop correspondences test extrapolation of the fitted transform. The fine-tuned-backbone ladder is restricted to a fixed 63-pair testbed (39 training, 13 validation, 11 test pairs) so that direct rows do not expand the eligible set.

### 2.10. Domain fine-tuning and weight anchoring

Fine-tuning is decoder-only: the VGG and DINOv2 encoder is held in evaluation mode with normalisation statistics fixed, while 100 million decoder parameters train on a 131-pair training split covering 12 subclasses. Supervision densifies sparse GT correspondences into a dense warp via thin-plate splines and supervises inside the GT-support region under a sparse-GT-robust loss, with FOV-crop and photometric augmentation. Optimisation is AdamW at a learning rate of 2 × 10⁻⁵ for 1500 steps; the checkpoint is selected by minimum validation median ED, which chose step 900 of 1500 (validation median 17.5 against 21.7 px zero-shot). The anchored variant adds an L2-SP penalty (Li et al., 2018) of the form λ⁄2 ‖θ_dec − θ⁰_dec‖², with λ swept over 0, 0.01, 0.1 and 1.0. Only the augmentation sampler is seeded.

### 2.11. Appearance measure

For each pair we warp the target into source coordinates under the GT-implied global affine, take the valid overlap region, quantise both images to 64 grey levels, and compute the normalised mutual information (NMI) over that region. The measure uses no matcher output, no fitted transform, and no error metric.

## 3. Results

### 3.1. Baseline registration accuracy

Table 1 reports the headline comparison across all 187 pairs, and Figure 1 plots the same data with Wilson intervals. The classical baselines reproduce the benchmark's own finding: SIFT succeeds on 3 of 187 pairs, and mutual-information intensity refinement moves the median ED by 80 px without flipping a single pair to success. Among zero-shot foundation matchers RoMa is the strongest single backbone, and MatchAnything-RoMa is nominally stronger still.

### 3.2. Pooling across tiles breaks matchers that cannot decline

The direct implementation of the scale hypothesis, pyramid v1, does not help. It destroys the backbone.

RoMa's median error rises from 80.2 to 2707.6 px, and 106 of 187 pairs become hard failures, meaning the estimator cannot return a transform at all (Table 1). The mechanism is structural. RoMa returns exactly 10,000 correspondences per invocation on all 187 pairs — a fixed sampling cap, hit every time — and it returns them whether or not the tile overlaps the target at all, because it has no mechanism for declining. Pooling over a pyramid therefore contributes one honest tile's worth of signal and *k* − 1 tiles' worth of confident noise.

The pooled counts make the scale of the flood concrete: pyramid v1 reaches 9,420,000 correspondences on a single pair, or 942 tiles' worth, of which at most one tile's worth can be correct. Measured over all 187 pairs, the median inlier fraction falls from 0.114 for the direct match to 0.005 for the tiled variant, a factor of 22. MAGSAC++ is robust to a minority of outliers; at a 0.5 % inlier fraction there is no consensus set left to find.

We stress that this is a property of the class, not a tuning artefact. The argument makes no reference to tile size, overlap fraction, pyramid depth, or the particular backbone: any pool-then-fit scheme over a matcher that cannot abstain will contribute outliers in proportion to the number of non-overlapping tiles, and the inlier fraction will fall roughly as the reciprocal of the tile count. The remedy is not a better pyramid, but a mechanism that decides which tile to believe before pooling anything.

### 3.3. A verified coarse-to-fine wrapper, and what it does not buy

We redesigned the wrapper so that it never pools blindly (pyramid v2). Because it accepts a candidate only if a mutual-information verifier scores it above the incumbent, the design is monotone with respect to its own verifier: tile noise cannot displace a direct fit the verifier prefers.

This removes the collapse completely. RoMa with pyramid v2 restores all 187 fits and improves the median error from 80.2 to 69.9 px (−10.3 px, 95 % CI [−48.2, +23.2], *p* = 0.37). But on the native benchmark that is the whole story: the aggregate success rate is unchanged, SR@10 0.091 → 0.091 (95 % CI [−0.016, +0.016], *p* = 1.00), and SR@20 is nominally lower (0.225 → 0.219, *p* = 0.83). The wrapper buys nothing in aggregate.

What it does is change *which* pairs succeed, and the direction is what its mechanism predicts. Counting successes by FOV stratum (Table 2), the wrapper converts one low-FOV failure into a success (1/33 → 2/33 in the 0.05–0.25 stratum) and loses one high-FOV success (15/126 → 14/126). Seventeen pairs succeed either way: the composition shifts and the total does not. A wrapper whose only mechanism is scale search should help where FOV is the binding constraint and cost a little where it is not, on a benchmark where 126 of 187 pairs are already at comparable scale. The effect is two pairs wide and we draw no inference from its statistical significance, of which it has none; we report it because the sign structure is a prediction the mechanism makes and could have failed.

**Ablations.** Two ablations show the design is at its useful limit. Iterated zoom, which chains refinements, is not significantly better than a single zoom and is borderline worse, because each stage compounds verifier error. Certainty gating at 0.5, which discards low-certainty matches before fitting, does not help either: on the unrefined metric it is exactly null (SR@20 0.219 → 0.219, 95 % CI [−0.021, +0.021], *p* = 1.00), and on the refined metric it is significantly worse than plain v2 (−0.037, 95 % CI [−0.070, −0.011], *p* = 0.002). The gate never improves on either metric, and it starves the estimator on exactly the hard pairs where every match is low-certainty. This is worth stating plainly given Section 3.2: RoMa's certainty map is the obvious candidate for the abstention signal whose absence causes the collapse, and thresholding it does not recover the missing behaviour.

### 3.4. A controlled field-of-view ladder

The native benchmark confounds scale with appearance, so we built a testbed where scale is the only variable. Cropping base-matchable real pairs to shrinking FOV ratios, with appearance, modality and pixel size fixed, shows direct matching collapsing between the 0.25 and 0.1 rungs, while the verified wrapper restores success at the 0.1 rung: SR@10 rises from 0.075 to 0.225 (*p* = 0.0028), a threefold increase (Figure 2). On pairs held out of fine-tuning the effect is stronger still, 0.045 → 0.227 (*p* = 0.023).

The ladder is measured on the unrefined metric, which is the only one meaningful at every rung: at 10 % FOV, refinement has almost no inliers to work with and near-zero effect.

The scale mechanism is therefore sound where scale is isolated. The benchmark simply never presents scale as an isolated failure mode.

### 3.5. The backbone is the largest available lever, and it is not significant

Swapping stock RoMa for MatchAnything-RoMa is the largest single off-the-shelf change we found, and on the unrefined metric it is not statistically significant: SR@10 0.091 → 0.107, Δ = +0.016, 95 % CI [−0.011, +0.043], *p* = 0.33. Median error is unchanged (+0.8 px, *p* = 0.86). Under the refined metric the same contrast reads +0.032, *p* = 0.035 (Section 3.8).

The stratum breakdown (Table 2) shows where the nominal gain sits: entirely in the above-0.5 stratum, 15/126 → 19/126. Cross-modal training repairs appearance on pairs that were nearly matchable already; it does not reach pairs whose FOV is the binding constraint, where it registers 0/33 against stock RoMa's 1/33. This ordering — MatchAnything-RoMa strongest overall — agrees with the independent evaluation of Durmaz et al. (2026b). Our contribution is to locate the gain (high FOV), to bound it (not significant on unrefined error), and to show what it cannot reach.

### 3.6. A directly measured appearance axis

Both preceding sections lean on the claim that low FOV and appearance divergence are confounded on this benchmark. Rather than infer that by elimination, we measure it.

The confound is real: across all 187 pairs the correlation between log₁₀ area ratio and NMI is *r* = +0.215 (*p* = 0.003). Pairs with less field of view in common also share less mutual information, so on native pairs the two axes cannot be separated. But the relationship is not monotone, and we report this because it limits how the correlation may be used: per-stratum median NMI runs 0.0196, 0.0021, 0.1648 and 0.0725 across the four bins, so the 0.25–0.5 stratum is in fact the most mutually informative of the four, and the coefficient above carries a weak overall trend rather than a clean ordering.

**The benchmark cannot rank scale against appearance.** Below area ratio 0.5 — 61 pairs — no zero-shot or wrapped configuration in this study registers more than three (Table 2). A regime in which the best available method succeeds on 3 of 61 cases carries essentially no discriminative power: there is no room for one axis to explain variance that the other does not. We therefore state, as a finding about the benchmark rather than about the methods, that AmalgaMatch has too little leverage below area ratio 0.5 to establish whether appearance or scale is the dominant constraint, and that an earlier version of this work which claimed appearance dominance by elimination was not entitled to that conclusion.

### 3.7. Domain fine-tuning trades in-distribution error for an untrained modality combination

The backbone swap suggests appearance is an operative constraint; fine-tuning attacks it directly. In distribution the recipe works: median TEM error falls roughly fivefold, consistently in direction across runs.

Out of distribution it costs. On the held-out 28-pair test split, averaged over eight training runs, SR@20 regresses from the zero-shot 0.393 to 0.264 ± 0.019 (Table 3). The regression is not diffuse. It falls on C103, the single subclass whose modality combination (SEM secondary-electron against LOM height) our split leaves untrained, and within it on the four test pairs from two scenes. This is catastrophic forgetting (Kirkpatrick et al., 2017) in the narrow sense of the term: the model loses a capability it had, on a combination it was never shown, while improving on what it was shown. We use the term at first mention with that scope stated, because it is often used more loosely.

We tested the cheapest available remedy. Over eight runs, L2-SP does not fix the regression: plain and anchored are statistically indistinguishable on every metric (Table 3), and on the one C103 scene that is recoverable at all the anchor is if anything less stable, retaining it in one of three runs against three of three for λ = 0. The second C103 test scene is unrecoverable in every run and at every λ. That rules out an optimisation or anchor-strength explanation and is consistent with a modality-coverage gap, but the validation scene of the same untrained combination survives intact, so four pairs from two scenes cannot establish coverage as the cause. We tested L2-SP because it is a one-line penalty with no deployment cost; parameter-efficient alternatives such as low-rank adaptation (Hu et al., 2022) constrain the update differently and are not evaluated here, and a formal equivalence test on a larger held-out set remains the obvious next step.

**Two protocol defects we disclose.** First, λ was selected by held-out test-split retention rather than on validation, because the validation split cannot resolve the C103 damage. This biases every λ-conditioned effect size toward L2-SP and leaves the eight-run null unaffected — the null is the result we report, and it is the one a biased selection procedure was least able to produce. A replication should select λ on a fold constructed to contain the untrained modality combination. Second, these are independent runs, not seed replicates: only the augmentation sampler is seeded, so run-to-run variation includes non-deterministic parallel reduction order on the graphics processor and is not reproducible from a seed alone. We report "runs" rather than "seeds" throughout for that reason.

**An earlier draw, and why we report it.** An initial draw of three runs, whose checkpoints were deleted when the compute instance was released, sat lower at direct SR@10 0.095 ± 0.021. We tabulate it in full rather than dropping it. It was not discarded for being unfavourable — on SR@20 (0.297 against 0.264) and on median ED (46.0 against 47.9 px) it is the better draw, and only on SR@10 is it worse. The gap of about four pairs in 28 is roughly two standard errors of a 28-pair success rate, which is the scale of variation this test split can resolve.

**A metric caveat specific to this section.** Table 3 is scored on the TPS-refined error, not the unrefined error used everywhere else, because the per-pair outputs of the eight runs were not retained and only their summary rows survive; rescoring is therefore impossible. For the one fine-tuned run whose per-pair rows do survive, the held-out SR@20 regression is identical under both metrics (0.393 → 0.250 raw, 0.393 → 0.250 refined), so we do not expect this result to be metric-contingent in the way the wrapper and backbone contrasts are. We flag it rather than assume it.

### 3.8. Refinement is not a neutral post-process

Everything above is scored on the unrefined parametric error. Our original protocol scored it after TPS refinement, and we changed primary metric during revision. This section reports what that change does, because the effect is large enough to be a finding in its own right rather than a housekeeping note.

**Both of our aggregate null results become significant under refinement.** The wrapper contrast on RoMa reads Δ = 0.000, *p* = 1.00 unrefined and +0.021, 95 % CI [+0.005, +0.043], *p* = 0.034 refined. The backbone contrast reads +0.016, *p* = 0.33 unrefined and +0.032, 95 % CI [+0.005, +0.064], *p* = 0.035 refined (Table 4). Two publishable-looking results and two null results, from the same runs, the same pairs and the same bootstrap, differing only in whether an optional spline was applied before measuring. Figure 4 plots the nine configurations under the refined metric; set beside Figure 1, which is the same nine under the unrefined metric, it is the whole of the effect in one view.

**Why we headline the unrefined metric.** The decisive reason is comparability, not effect size. A TPS refinement is only defined when the fit has enough surviving inliers to constrain a spline, so the refined column is populated for some configurations and not others. Coverage across Table 1's nine configurations ranges from 187/187 for every dense RoMa-family row down to 93/187 for LoFTR, 70/187 for the SIFT control and 0/187 for the SIFT-plus-mutual-information control. Where the column is blank the pipeline falls back to the unrefined value. A "TPS-refined" table therefore scores the dense rows entirely on refined error and the weak rows almost entirely on raw error: it is not one metric but two, interleaved by configuration, and part of the apparent gap between the strong and weak families is a difference in scoring rather than in registration. The unrefined metric scores all nine identically. Two further considerations point the same way. Refinement is an optional stage downstream of the object we are studying, which is the matcher and the wrapper; and it is not monotone — across the 16 configurations in our result files, 13 configuration-pair combinations register below 10 px unrefined and above it after refinement, so refinement destroys successes as well as creating them.

**Consequences we accept.** Three claims in earlier versions of this work do not survive the change and we withdraw them. (i) The verified wrapper is not an accuracy improvement on native pairs; it is a collapse fix with a null aggregate effect. (ii) The backbone gain is suggestive, not established. (iii) A "first non-zero result in the extreme stratum" claim was an artefact: under refinement the 0.05–0.25 stratum reads 0/33 → 1/33, which looks like a qualitative first; unrefined it reads 1/33 → 2/33, because the pair that supplies the "zero" is one that direct RoMa already registers to 3.3 px and that refinement then pushes to 37.5 px. The wrapper gains one pair in that stratum either way. Only the from-zero framing was metric-manufactured.

**The general hazard.** We think this generalises beyond our pipeline. Any evaluation with an optional post-processing stage whose applicability depends on the quality of the thing being evaluated will produce a metric that is silently non-uniform across configurations, and will tend to flatter methods whose outputs are dense enough for the stage to fire. Refinement stages, outlier filters and fallback heuristics all have this character. Our recommendation is narrow and cheap: **report post-processing coverage per configuration alongside the metric, and check headline contrasts with the stage disabled.** Had we done so at first submission, three of our claims would have been stated differently.

## 4. Discussion

The practical message for a correlative-microscopy workflow is short. A pretrained dense matcher, used directly, is the best zero-shot option now available, and MatchAnything-RoMa is the strongest of those we tested. Tiling that matcher to bridge a field-of-view gap — the first thing most practitioners will try, and the thing we tried first — is actively harmful, and no amount of tuning the tile size or overlap will repair it. If a tiled search is nevertheless wanted, it must verify candidates against an incumbent rather than pool correspondences, which costs nothing in aggregate accuracy and removes the failure mode entirely.

The mechanism deserves emphasis because it is not specific to our implementation. A detector-free dense matcher is trained to produce a correspondence field, not a decision about whether one exists. Presented with a tile of specimen that does not overlap the target region at all, it returns a full complement of confident correspondences describing a relationship that does not exist. Pooling those with the correspondences from the one honest tile is not a small perturbation of a robust estimation problem; it changes the inlier fraction by more than an order of magnitude, and robust estimators have a breakdown point. Any scheme that pools before it decides inherits this, which is why we present it as a property of the class.

Two boundaries on the negative results should be stated clearly. First, **the negative results are bounded by the benchmark, not by the mechanism.** Sections 3.3 and 3.5 report nulls on native pairs, and Section 3.4 shows the wrapper mechanism working when scale is isolated. The correct reading is that AmalgaMatch has 126 of 187 pairs at area ratio above 0.5 and only 4 below 0.05, so it is not an instrument for measuring field-of-view-adaptation methods. A benchmark stratified for FOV, with appearance controlled, would settle in one experiment what we could only approach by construction. This is, we think, the most useful thing the community could build next.

Second, **the ladder denominators are not equal.** Base-matchability is defined per backbone, so the three ladders rest on different and partly training-enriched pair sets (38, 41 and 53 pairs). Cross-backbone comparisons read off the ladder are not like-for-like. The wrapper contrast, which is within-backbone, is unaffected, and its direction is preserved on held-out pairs for both backbones tested, though with different force: for zero-shot MatchAnything-RoMa it strengthens (0.045 → 0.227, *p* = 0.023), while for the fine-tuned backbone the direction holds but significance does not (0.111 → 0.278, *p* = 0.077, n = 18).

**Monotonicity is narrower than it sounds.** The verifier is monotone in its own mutual-information score, not in GT error. Per pair, the wrapper raises refined error on 94 of 187 pairs, materially on 22, with a worst case of 60.6 → 1481.7 px. What it reliably avoids is losing threshold crossings, not regressing in error.

**Not everything is reproducible from the released files.** The per-run values of three fine-tuning runs, and the per-scene C103 readouts from the deleted checkpoints, cannot be regenerated. Everything else in this paper regenerates from the released result files.

**Single benchmark, single domain.** Every measurement here is on AmalgaMatch. The non-abstention argument of Section 3.2 is analytic and should transfer to any non-abstaining matcher under any tiling scheme; we have not tested that, and it is the most valuable replication this work invites.

## 5. Conclusions

Wrapping a pretrained dense matcher in an image pyramid is an appealing, training-free answer to extreme field-of-view mismatch, and the direct version of it does not work — not because of tuning, but because dense matchers cannot abstain, so every non-overlapping tile contributes confident noise and robust estimation loses its consensus. A verified coarse-to-fine design repairs that, and on a controlled ladder where scale is the only variable it triples success at 10 % field of view. On the native benchmark it buys nothing, because that benchmark almost never presents scale as an isolated problem: its low-FOV pairs are also its most appearance-divergent, and it carries too little leverage in that regime to say which constraint dominates. We report both outcomes, and we report that the choice of whether to score before or after an optional refinement stage was, on our data, sufficient to turn two null results into significant ones — which is why we headline the metric that scores every configuration the same way.

Against the three hypotheses stated in Section 1:

**H1** (35 % relative gain at FOV below 5 %) is **refuted**, decisively and on both metrics. Every configuration registers 0 of 4 pairs in the extreme-FOV stratum. No wrapper we built moves that number, and with n = 4 the stratum could not have supported the claim even had it.

**H2** (RoMa's dense flow tolerates partial overlap better than the ELoFTR family at low FOV) is **supported, weakly**. RoMa beats MatchAnything-ELoFTR in aggregate (SR@10 0.091 against 0.011) and in each stratum where either registers anything: 1/33 against 0/33 at 0.05–0.25, and 1/24 against 0/24 at 0.25–0.5. The direction is consistent, but the low-FOV evidence is a single pair, and we would not defend the claim on that alone.

**H3** (affine suffices; homography adds nothing) is **supported as stated, with one caveat we did not anticipate**. Among the 118 well-registered pairs across all zero-shot backbones (unrefined error below 20 px), the selector chooses affine on 82 (69 %) and homography on 36 (31 %). Affine is sufficient for roughly seven pairs in ten, which is the claim. The caveat is that the pairs on which the selector does reach for a homography are the more accurate ones (median ED 6.5 against 11.3 px for affine-selected), so we cannot add, as an earlier version of this work did, that homography confers no accuracy advantage. That comparison conditions on the selector's own decision and licenses no causal reading in either direction; separating the two would require fitting both families on every pair and comparing, which we did not do.

## Acknowledgments

The author thanks the authors of the AmalgaMatch benchmark for releasing the dataset under an open licence.

## Conflict of Interest

The author declares no conflict of interest.

## Author Contributions

**Frank Cai:** Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing — original draft, Writing — review and editing, Visualization.

## Data Availability

The AmalgaMatch dataset is publicly available from the Fraunhofer Fordatis repository (doi:10.24406/fordatis/436). The analysis code, evaluation harness, wrapper implementations, field-of-view-ladder construction, fine-tuning trainer and every analysis script are openly available at https://github.com/fronkt/correlative-microscopy-alignment and archived at Zenodo (doi:10.5281/zenodo.20819649). All tables and figures regenerate from the released per-pair result files via the scripts referenced therein. The exceptions are the per-run values of three fine-tuning runs and the per-scene C103 readouts from non-retained checkpoints, which are the only quantities in this paper that cannot be regenerated from the release. Fine-tuned model checkpoints (445 MB) are available from the author on reasonable request.

## References

Barath, D., Noskova, J., Ivashechkin, M. & Matas, J. (2020). MAGSAC++, a fast, reliable and accurate robust estimator. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 1304–1312.

Bookstein, F.L. (1989). Principal warps: Thin-plate splines and the decomposition of deformations. *IEEE Trans. Pattern Anal. Mach. Intell.* **11**, 567–585.

Durmaz, A.R., Lamb, J.D., Echlin, M.P. & Pollock, T.M. (2026a). AmalgaMatch: A benchmark dataset for cross-modal image matching in correlative materials microscopy. *Fordatis, Fraunhofer Research Data Repository*.

Durmaz, A.R., Lamb, J.D., Echlin, M.P. & Pollock, T.M. (2026b). Foundation models for multimodal image data fusion in materials science. *Front. Mater.* **13**.

Edstedt, J., Sun, Q., Bökman, G., Wadenbäck, M. & Felsberg, M. (2024). RoMa: Robust dense feature matching. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 19790–19800.

Fischler, M.A. & Bolles, R.C. (1981). Random sample consensus: A paradigm for model fitting with applications to image analysis and automated cartography. *Commun. ACM* **24**, 381–395.

He, X., Yu, H., Peng, S., Dong, D., Tan, D., Zhou, X., Bao, H. & Shen, Z. (2025). MatchAnything: Universal cross-modality image matching with large-scale pre-training. *arXiv preprint* arXiv:2501.07556.

Hu, E.J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L. & Chen, W. (2022). LoRA: Low-rank adaptation of large language models. In *International Conference on Learning Representations (ICLR)*.

Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A.A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., Hassabis, D., Clopath, C., Kumaran, D. & Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks. *Proc. Natl. Acad. Sci. U. S. A.* **114**, 3521–3526.

Li, X., Grandvalet, Y. & Davoine, F. (2018). Explicit inductive bias for transfer learning with convolutional networks. In *Proceedings of the 35th International Conference on Machine Learning (ICML)*, vol. 80, pp. 2825–2834.

Lindenberger, P., Sarlin, P.-E. & Pollefeys, M. (2023). LightGlue: Local feature matching at light speed. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 17627–17638.

Lowe, D.G. (2004). Distinctive image features from scale-invariant keypoints. *Int. J. Comput. Vis.* **60**, 91–110.

Maes, F., Collignon, A., Vandermeulen, D., Marchal, G. & Suetens, P. (1997). Multimodality image registration by maximization of mutual information. *IEEE Trans. Med. Imaging* **16**, 187–198.

Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P.-Y., Li, S.-W., Misra, I., Rabbat, M., Sharma, V., Synnaeve, G., Xu, H., Jegou, H., Mairal, J., Labatut, P., Joulin, A. & Bojanowski, P. (2024). DINOv2: Learning robust visual features without supervision. *Trans. Mach. Learn. Res.*

Sarlin, P.-E., DeTone, D., Malisiewicz, T. & Rabinovich, A. (2020). SuperGlue: Learning feature matching with graph neural networks. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 4938–4947.

Sun, J., Shen, Z., Wang, Y., Bao, H. & Zhou, X. (2021). LoFTR: Detector-free local feature matching with transformers. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 8922–8931.

Wang, Y., He, X., Peng, S., Tan, D. & Zhou, X. (2024). Efficient LoFTR: Semi-dense local feature matching with sparse-like speed. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.

## Tables

**Table 1.** Registration accuracy on all 187 AmalgaMatch pairs, unrefined parametric error. "Fits" counts pairs on which the estimator returned a transform at all; the remainder are scored as failures at every threshold. Classical methods reproduce the benchmark's near-total failure. Tiling (pyramid v1) destroys RoMa. The verified wrapper (v2) restores it but adds nothing in aggregate. No difference among the four dense rows is statistically significant on this metric (Table 4); under the refined metric two of them are. The fine-tuned backbone is excluded from this table because it was trained on 131 of these 187 pairs; it appears only in Table 3.

| Configuration | Median ED (px) | SR@5 | SR@10 | SR@20 | Fits |
|---|---|---|---|---|---|
| SIFT (Control A) | 903.1 | 0.011 | 0.016 | 0.021 | 169/187 |
| SIFT + mutual information (Control B) | 823.6 | 0.011 | 0.016 | 0.021 | 169/187 |
| LoFTR | 270.1 | 0.037 | 0.064 | 0.091 | 183/187 |
| MatchAnything-ELoFTR | 510.1 | 0.011 | 0.011 | 0.032 | 177/187 |
| RoMa (zero-shot) | 80.2 | 0.059 | 0.091 | 0.225 | 187/187 |
| RoMa + pyramid v1 | 2707.6 | 0.000 | 0.005 | 0.021 | 81/187 |
| RoMa + pyramid v2 | 69.9 | 0.059 | 0.091 | 0.219 | 187/187 |
| MatchAnything-RoMa | 81.0 | 0.059 | 0.107 | 0.241 | 187/187 |
| MatchAnything-RoMa + pyramid v2 | 77.6 | 0.053 | 0.118 | 0.230 | 187/187 |

**Table 2.** Success at 10 px by native field-of-view stratum, unrefined error. Every configuration is at the floor in the extreme-FOV stratum. The wrapper's low-FOV gain and high-FOV loss on RoMa are visible as 1 → 2 and 15 → 14. Across the 61 pairs below area ratio 0.5, no configuration registers more than three. Figure 3 plots these with intervals.

| Configuration | <0.05 (n = 4) | 0.05–0.25 (n = 33) | 0.25–0.5 (n = 24) | >0.5 (n = 126) |
|---|---|---|---|---|
| SIFT (Control A) | 0/4 | 0/33 | 1/24 | 2/126 |
| SIFT + mutual information (Control B) | 0/4 | 0/33 | 1/24 | 2/126 |
| LoFTR | 0/4 | 0/33 | 1/24 | 11/126 |
| MatchAnything-ELoFTR | 0/4 | 0/33 | 0/24 | 2/126 |
| RoMa (zero-shot) | 0/4 | 1/33 | 1/24 | 15/126 |
| RoMa + pyramid v1 | 0/4 | 0/33 | 0/24 | 1/126 |
| RoMa + pyramid v2 | 0/4 | 2/33 | 1/24 | 14/126 |
| MatchAnything-RoMa | 0/4 | 0/33 | 1/24 | 19/126 |
| MatchAnything-RoMa + pyramid v2 | 0/4 | 1/33 | 1/24 | 20/126 |

**Table 3.** Decoder-only fine-tuning on the held-out 28-pair test split: mean ± standard deviation over eight training runs. Scored on TPS-refined ED — the exception to this paper's primary metric, forced by non-retention of per-run outputs and discussed in Section 3.7. Zero-shot rows are deterministic. SR@5 is omitted as 0 at the 10.3 px GT floor. Both protocols share per-run checkpoints. Plain and L2-SP are statistically indistinguishable on every metric, so the anchor is a zero-cost floor rather than a reliable improvement; both regress SR@20 against zero-shot while cutting in-distribution TEM median ED roughly fivefold. The verified wrapper does not stack on the fine-tune (within-checkpoint SR@20 within ±0.01).

| Method (8-run mean ± SD) | Protocol | SR@10 | SR@20 | Median ED (px) |
|---|---|---|---|---|
| MatchAnything-RoMa (zero-shot) | direct | 0.214 | 0.393 | 83.5 |
| MatchAnything-RoMa (zero-shot) | pyramid v2 | 0.214 | 0.393 | 69.1 |
| Plain fine-tune (λ = 0) | direct | 0.236 ± 0.019 | 0.264 ± 0.019 | 49.6 ± 3.7 |
| Plain fine-tune (λ = 0) | pyramid v2 | 0.228 ± 0.027 | 0.264 ± 0.019 | 49.5 ± 3.1 |
| L2-SP (λ = 0.01) | direct | 0.250 ± 0.000 | 0.268 ± 0.019 | 50.8 ± 8.1 |
| L2-SP (λ = 0.01) | pyramid v2 | 0.245 ± 0.023 | 0.272 ± 0.019 | 50.2 ± 6.6 |

**Table 4.** The same three contrasts under both error metrics. Paired bootstrap, B = 10,000 resamples, two-sided. The two native-pair contrasts are null unrefined and significant refined; the controlled ladder result is the reverse. No contrast is significant under both.

| Contrast | Metric | A | B | Δ | 95 % CI | *p* |
|---|---|---|---|---|---|---|
| RoMa + pyramid v2 vs. RoMa, SR@10 | unrefined | 0.0909 | 0.0909 | +0.0000 | [−0.0160, +0.0160] | 1.000 |
| RoMa + pyramid v2 vs. RoMa, SR@10 | refined | 0.0963 | 0.1176 | +0.0214 | [+0.0053, +0.0428] | 0.034 |
| MatchAnything-RoMa vs. RoMa, SR@10 | unrefined | 0.0909 | 0.1070 | +0.0160 | [−0.0107, +0.0428] | 0.334 |
| MatchAnything-RoMa vs. RoMa, SR@10 | refined | 0.0963 | 0.1283 | +0.0321 | [+0.0053, +0.0642] | 0.035 |
| FOV ladder, MatchAnything-RoMa at rung 0.1 | unrefined | 0.075 | 0.225 | +0.150 | [+0.050, +0.275] | 0.0028 |
| FOV ladder, MatchAnything-RoMa at rung 0.1 | refined | 0.025 | 0.100 | +0.075 | [+0.000, +0.175] | 0.087 |

## Figure Legends

**Figure 1.** Success rates across all nine configurations, unrefined parametric error, all 187 pairs. Error bars are 95 % Wilson score intervals. Pyramid v1 collapses RoMa; pyramid v2 restores it to parity. The intervals overlap throughout the dense-matcher rows, which is the honest summary of the native-pair comparison: nothing here is separated. Figure 4 shows the same panel under the refined metric, where two contrasts become significant.

*Alt text:* Grouped bar chart of success rate at 5, 10 and 20 pixels for nine registration configurations, with 95 percent Wilson confidence intervals. The classical SIFT baselines sit near zero. The RoMa-family bars are the tallest but their intervals overlap one another. The pyramid v1 bar collapses to near zero, far below the other dense-matcher bars.

**Figure 2.** Controlled field-of-view ladder, appearance held fixed, unrefined error. Cropping base-matchable pairs to shrinking field-of-view ratios shows direct matching collapsing between the 0.25 and 0.1 rungs, while the verified wrapper restores success at 0.1 (0.075 → 0.225, *p* = 0.0028). The ladder is measured on the unrefined metric, which is the only one meaningful at every rung: at 10 % field of view, refinement has almost no inliers to work with and near-zero effect.

*Alt text:* Line plot of success rate at 10 pixels against field-of-view area ratio on a descending axis from 0.5 to 0.02. The direct-matching curve falls steeply between the 0.25 and 0.1 rungs and reaches zero by 0.05. The wrapper curve stays above it, peaking at the 0.1 rung at roughly three times the direct value, before both collapse at the smallest ratios.

**Figure 3.** Success at 10 px versus native field-of-view stratum, unrefined error, with per-stratum sample size and Wilson 95 % intervals. The extreme-FOV stratum holds four pairs, so its interval is very wide and nothing may be read from its point estimate. Every configuration is at the floor there.

*Alt text:* Plot of success rate at 10 pixels across four field-of-view strata for each configuration, with Wilson confidence intervals. All configurations sit at zero in the smallest stratum, where the interval is extremely wide because it contains only four pairs. Success rises only in the largest-field-of-view stratum.

**Figure 4.** The same nine configurations under the TPS-refined metric. Compare Figure 1. The dense rows separate more here than they do unrefined, and part of that separation is the coverage artefact described in Section 3.8: the four RoMa-family rows have 187/187 refinement coverage while the SIFT-plus-mutual-information control has 0/187 and is scored entirely on the unrefined fallback.

*Alt text:* The same grouped bar chart as Figure 1 but computed after thin-plate-spline refinement. The RoMa-family bars are visibly taller and separate more clearly from one another than in Figure 1, while the weak classical bars are essentially unchanged.
