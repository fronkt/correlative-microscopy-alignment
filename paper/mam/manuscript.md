# Image tiling does not solve field-of-view mismatch in correlative microscopy registration

**Frank Cai**

Purdue University, West Lafayette, IN 47907, USA

Correspondence: frankyc11223@gmail.com · ORCID 0009-0003-0041-1459

**Running head:** Image tiling and field-of-view mismatch

**Keywords:** correlative microscopy, image registration, feature matching, field of view, deep learning, benchmarking

---

## Abstract

Correlative microscopy combines images of one specimen from different instruments, such as an electron backscatter diffraction map and a scanning electron micrograph. Comparing them requires registration: locating every point of one image in the other. Registration is hard when the two images look different and when one covers far less of the specimen than the other. Image-matching networks trained on photographs are the strongest general tools for this step, and a natural way to extend them to a large difference in field of view is to cut the wider image into tiles, match each tile, and fit one transformation to all the matches. We test this on a public benchmark of 187 correlative image pairs and show, with worked examples, why it fails. The networks return the same large number of matches for every tile, including tiles that show none of the narrower image, and the correct matches are lost among them: on the tiled pairs, the median error rose from 68 to 538 pixels. A search that zooms in on a first estimate and checks each candidate avoids this failure and triples success when field of view is the only difficulty, but gains nothing on real pairs.

---

## 1. Introduction

Correlative microscopy links measurements of one specimen made on different instruments. A dislocation network seen in a transmission electron micrograph can be tied to the grain orientations measured by electron backscatter diffraction (EBSD); a strain map from digital image correlation in the scanning electron microscope (SEM) can be laid over the microstructure that produced it. Every such link starts with the same step. The two images have to be brought into one coordinate frame, so that a pixel in one can be located in the other. That step is called registration, and in correlative materials microscopy it is often the bottleneck (Durmaz et al., 2026b).

Figure 1 shows three pairs from the public benchmark used in this paper and the two things that make registration hard. The first is appearance. The two images of a pair are formed by different physics, so the same feature can be bright in one and dark in the other, or can appear as orientation colour in an EBSD map and as topographic contrast in an SEM image (Figure 1A, B). The second is field of view. One image often covers only a small part of the other: in Figure 1C the narrow image covers 4.9 % of the wide image's area, and in the most extreme pair of the benchmark it covers 1.85 %. Classical registration methods assume that the two images look alike and have similar scale. Feature matching with the scale-invariant feature transform (SIFT) (Lowe, 2004) and intensity-based registration by mutual information (Maes et al., 1997) both fail on most correlative pairs (Durmaz et al., 2026a), and we confirm this below.

Learned image matchers are the obvious alternative. An image matcher is a neural network that takes two images and returns correspondences: pairs of points, one in each image, that it judges to show the same place on the specimen. From enough correct correspondences, a transformation between the images can be fitted. LoFTR (Sun et al., 2021) and its faster successor (Wang et al., 2024) match without first detecting distinctive points, so they work in texture-poor images. RoMa (Edstedt et al., 2024) goes further and estimates a correspondence for every pixel, together with a certainty value for each. MatchAnything (He et al., 2025) retrains these networks on large synthetic datasets built to mimic different imaging modes, and Durmaz et al. (2026b) found its RoMa-based version the strongest of several such matchers across most series of the benchmark used here. None of these networks was trained on micrographs: they are used as released, which is called zero-shot use.

These matchers work at a fixed internal image size of several hundred pixels, so a large image is shrunk before matching. When the narrow image covers a few per cent of the wide one, its whole field shrinks to a small patch of the wide image and most of its detail is lost. The obvious remedy is to tile: cut the wide image into tiles about the size of the narrow image, match each tile against the narrow image, pool all the correspondences, and fit one transformation to them. The idea needs no training and treats the matcher as a black box, which makes it the first thing many users will try.

This paper tests that idea on all 187 pairs of the AmalgaMatch benchmark (Durmaz et al., 2026a) and shows why it fails. The reason turns out to be general: the matchers return the same number of correspondences for every tile, whether or not the tile contains any part of the narrow image, and the fitting step has no way to tell which tiles to believe. We then test a more careful design that zooms in on a first estimate and checks each candidate before accepting it. It avoids the failure and triples success when a small field of view is the only difficulty, but it gains nothing on the real pairs, where a small field of view tends to come with a large difference in appearance.

The paper is written for microscopists who want to use these tools on their own data. The work began with three stated hypotheses about tiling; Supplementary Section S6 reports their outcomes. Section 2 defines every technical term (Table 1) and walks through the registration of one real pair step by step (Figure 2). Section 3 shows what the methods achieve and, with images, how they fail (Figures 3 and 4). Section 4 turns the results into practical guidance.

## 2. Materials and Methods

### 2.1. The benchmark

AmalgaMatch (Durmaz et al., 2026a) is a public collection of 187 correlative image pairs from 19 specimen series, grouped into six kinds of correlative task: orientation mapping, serial sectioning, multiscale imaging, slip partitioning, fracture surfaces and dislocation characterisation. The materials include steels, nickel-based superalloys, cobalt-nickel alloys, tantalum, a niobium alloy, a refractory high-entropy alloy and a MAX-phase ceramic. The imaging modes include SEM with secondary-electron, backscattered-electron and in-lens detectors; EBSD maps of orientation, image quality, confidence, pattern sharpness and band contrast; SEM strain maps from digital image correlation; light optical images, height maps and extended-focus composites; scanning transmission electron microscopy (STEM) in bright field, dark field and annular dark field; and weak-beam dark-field transmission electron microscopy (TEM). Image sizes range from 238 to 13,333 pixels along the longer side.

Each pair comes with annotated points: positions that a person marked on the same feature in both images, a median of 20 per pair (range 8 to 129). These points are the ground truth. They are never shown to the methods and are used only to measure error.

We call the image that covers the larger area of the specimen the wide image and the other the narrow image. Every method in this paper locates the narrow image inside the wide one. Our data loader decides which image is the wide one from the pixel sizes recorded in the benchmark's image metadata.

### 2.2. Field of view

The field-of-view ratio of a pair is the area of the specimen shown in the narrow image divided by the area shown in the wide image. A ratio of 1 means the two images show the same area; 0.05 means the narrow image shows 5 % of what the wide one shows. We compute it from the annotated points rather than from the metadata. A straight-line (affine) mapping fitted to a pair's points converts areas in one image into areas in the other, and the ratio of the two image frames follows directly. For 13 pairs in three series the metadata pixel sizes disagree with the annotated points by factors between 2.9 and about 9,400 (Supplementary Section S2); for the rest the two agree within 25 %.

The ratio has a median of 0.82 over the benchmark. For analysis we group pairs into four field-of-view groups: below 0.05 (2 pairs), 0.05 to 0.25 (34), 0.25 to 0.5 (28) and 0.5 or more (123). Only 64 pairs fall below 0.5, and only 2 below 0.05, so the benchmark contains few examples of the severe mismatch that motivates this work. We return to this in Section 3.5.

### 2.3. Terms

Table 1 defines the technical terms used in the rest of the paper. Each term is also explained where it first matters.

### 2.4. How a pair is registered, step by step

Figure 2 follows one pair through the whole procedure: an SEM secondary-electron image of a spalled specimen (wide) and a backscattered-electron image of a small region of it (narrow).

**Step 1, matching.** The matcher receives both images and returns correspondences. For RoMa, our software shrinks each image so that its longer side is at most 832 pixels, and the network then works internally at 560 and 864 pixels. RoMa estimates a correspondence for every pixel with a certainty between 0 and 1. Its standard output step then sets every certainty above 0.05 to 1 and draws exactly 10,000 correspondences at random, favouring certain ones. It returned exactly 10,000 on every one of the 187 pairs, as did the MatchAnything version of RoMa. The other matchers return a variable number.

**Step 2, robust fitting.** Many correspondences are wrong, so the transformation is fitted with a robust estimator. The idea, due to Fischler and Bolles (1981), is to fit a candidate transformation to a small random subset of the correspondences, count how many of all the correspondences agree with it, repeat many times, and keep the candidate with the most agreement. Correspondences that agree to within a set distance, here 5.5 pixels, are called inliers; the rest are outliers. We use MAGSAC++ (Barath et al., 2020), a modern version of this idea that weighs agreement more smoothly than a hard cut-off. Two kinds of transformation are fitted. An affine transformation allows shift, rotation, scaling and shear (six numbers); a homography also allows perspective distortion (eight numbers). The one kept is chosen by a rule that rewards a small fitting residual and penalises the extra two numbers of the homography. In Figure 2A, robust fitting kept 2,032 of the 10,000 correspondences and rejected 7,968.

**Step 3, placing the narrow image.** The fitted transformation maps every point of the narrow image into the wide one. Drawing the narrow image's border through it shows where the method thinks the narrow image lies (Figure 2B).

**Step 4, measuring the error.** Each annotated point of the narrow image is mapped into the wide image by the fitted transformation, and its distance from the matching annotated point in the wide image is measured, in pixels of the wide image (Figure 2C). A pair's registration error is the mean of these distances.

**Step 5, scoring.** A pair counts as registered if its error is below 10 pixels. We also report 20 pixels, and for each method the median error over the pairs it produced a transformation for. For the pair in Figure 2 the error is 2.7 pixels, so it is registered.

The 10-pixel threshold has to be read against a floor set by the annotations themselves. The annotated points are placed by hand on features that do not always correspond exactly, and some specimens deform between images, so no single global transformation fits a pair's own annotated points perfectly. An affine mapping fitted by least squares to each pair's own points leaves a median mean residual of 10.3 pixels. A homography fitted the same way registers 93 of the 187 pairs within 10 pixels and 156 within 20. These are ceilings: no method that fits one transformation per pair can do better on this benchmark, and we report every method against them.

### 2.5. Matchers compared

Table 2 lists the methods. Two classical baselines use SIFT features, the second followed by mutual-information refinement. Four learned matchers are used zero-shot: LoFTR with its released outdoor weights; MatchAnything's version of efficient LoFTR (MatchAnything-ELoFTR); RoMa with its released outdoor weights; and MatchAnything's retrained weights for RoMa (MatchAnything-RoMa), which load into the same network as RoMa, tensor for tensor (603 of 603), so the two differ only in training.

### 2.6. Tiling by pooling

The first tiling scheme, which we call pooled tiling, follows the idea in Section 1 directly.

1. The tile size is set to the shorter side of the narrow image, in the narrow image's pixels.
2. The wide image is halved in size repeatedly while its pixels remain finer than the narrow image's, giving a stack of versions at decreasing resolution. Each version is cut into square tiles of that size, with neighbouring tiles overlapping by half a tile.
3. Every tile is matched against the whole narrow image.
4. Each tile's correspondences are mapped back into the coordinates of the original wide image, all of them are pooled, and one transformation is fitted to the pool as in Section 2.4.

Two facts about our implementation matter for reading the results, and we state them before the results rather than after. First, the tile size is measured in the narrow image's pixels. When the narrow image has finer pixels than the wide one, no reduced version is built and each tile covers more of the specimen than the narrow image does. When the wide image is smaller than one tile, the implementation filled the tile with mirror images of the wide image (Supplementary Figure S2). This happens on 57 of the 187 pairs. Those pairs test mirror padding, not tiling, so we report them separately. Second, the run with RoMa stopped at the 82nd of the 187 pairs because of a fault on the graphics processor, and every later pair failed with the same error message. The run was not repeated, so pooled tiling was evaluated on 81 pairs: 33 that were genuinely tiled (between 2 and 942 tiles, median 75) and 48 padded single tiles. Every comparison below uses the same pairs for both methods, and Supplementary Section S1 lists what was and was not evaluated.

### 2.7. The checked coarse-to-fine search

The second scheme never pools blindly. It builds candidate transformations one at a time and accepts a candidate only if an independent check prefers it to the current best, called the incumbent.

1. **Direct match.** Both whole images are matched and a transformation is fitted, exactly as in Section 2.4. This is the first incumbent.
2. **Tile search, only if support is weak.** If the direct fit kept fewer than 50 inliers, each tile is matched and fitted separately, and the best-checked tile's transformation replaces the incumbent if it checks better.
3. **Zoom.** The incumbent places the narrow image somewhere in the wide image. The wide image is cropped around that place with a 30 % margin, and the crop is matched against the narrow image. Because the crop is smaller, it is shrunk less before matching, and the matcher sees the region at higher resolution. The new transformation replaces the incumbent if it checks better.

The number of zoom steps differs between runs, and we state it wherever it matters. The run with RoMa on the 187 real pairs zoomed once. Our code was then changed to repeat the zoom up to three times, stopping when the check no longer improves, and every later run used that setting: MatchAnything-RoMa on the real pairs, both matchers on the field-of-view ladder, and the variants in Supplementary Section S4, which also compare one zoom with three directly.

The check is mutual information. The wide image is resampled into the narrow image's frame using the candidate transformation, and mutual information measures how well the grey levels of one predict the grey levels of the other over the region where they overlap. It is high when the images are aligned, even if their contrasts differ, and it uses only image content, never the annotated points. Because a candidate is accepted only when the check does not get worse, the search can never end with a lower check score than the direct match. That guarantee is about the check score, not about the registration error, which the check cannot see.

### 2.8. The field-of-view ladder

On real pairs, a small field of view and a large difference in appearance tend to occur together (Section 3.5). To see what field of view does on its own, we built a controlled series. For each matcher we take the pairs it registers within 20 pixels at full size. For each such pair we crop the narrow image, around the centre of its annotated points, so that its area becomes 0.5, 0.25, 0.1, 0.05 and 0.02 of the wide image's area (Figure 6A). Nothing else changes: the same two instruments, the same contrast, the same pixel sizes. All annotated points are kept, including those that fall outside the crop, so the error also measures how well the fitted transformation extends beyond the region the method could see. The candidate pairs form a fixed set of 63; of these, RoMa registers 38 within 20 pixels at full size and MatchAnything-RoMa registers 41. The crop sizes were computed from the metadata ratio, so for the 8 of the 63 pairs whose metadata is inconsistent (Section 2.2) the stated ratios are wrong; Section 3.4 reports the result with and without them.

### 2.9. Measuring how alike two images look

To put a number on appearance, we map the narrow image into the wide one using the annotated points, take the region where they overlap, reduce both to 64 grey levels, and compute normalised mutual information (NMI). It is 0 when the grey levels of one image say nothing about the other and 1 when one fully predicts the other. The measure uses no method's output.

### 2.10. Training on microscopy pairs

To test whether training on microscopy helps, we fine-tuned MatchAnything-RoMa on 131 of the 187 pairs, drawn from 12 of the 19 series, and kept 28 pairs aside as a test set. The part of the network that turns image features into correspondences was trained; the part that extracts the features was held fixed. One series, fracture surfaces of the niobium alloy C103 imaged by SEM and by light optical height mapping, had no training pairs at all, which makes it a test of what training does to a combination the network was not shown. Supplementary Section S5 gives the full protocol.

### 2.11. Refinement

The benchmark's protocol allows an optional last step: bending the fitted transformation to pass closer to its inliers with a thin-plate spline (Bookstein, 1989), a smooth interpolating warp. We report errors before this step throughout, because the step can only run when enough inliers survive, and so it runs on every pair for some methods and on none for others (Section 3.7). Errors after refinement are given in Supplementary Section S3.

### 2.12. Statistics

When two methods are compared on the same pairs, we use a paired bootstrap: the 187 pairs (or the relevant subset) are resampled with replacement 10,000 times, the difference between the methods is recomputed each time, and the middle 95 % of those differences is the 95 % confidence interval (CI). The *p*-value is two-sided: twice the smaller share of resampled differences on either side of zero, capped at 1. Intervals on single success rates are Wilson score intervals, which remain sensible near zero. The resampling uses a fixed seed, recorded in the released scripts.

### 2.13. Worked examples

The benchmark runs kept per-pair errors but not correspondences or fitted transformations. Figures 2, 3 and 4, which show registrations, therefore come from repeat runs of the same code on a desktop processor with a fixed random seed. Because the matcher draws its correspondences at random, a repeat run need not reproduce the benchmark error exactly. Each figure legend gives the repeat run's error. In all five repeat runs of a whole pair, the repeat run agrees with the benchmark run on whether the pair is registered (Supplementary Table S6).

### 2.14. Software, data and use of artificial-intelligence tools

All methods were run with the released weights named in Section 2.5 through our own open evaluation code (Data Availability). The large-language-model assistant Claude (Anthropic) was used to help write analysis and plotting code and to draft and edit the text. The author checked all code, reran every analysis from the released result files, and takes full responsibility for the content.

## 3. Results

### 3.1. Released matchers register about one pair in ten

Table 3 and Figure 5 show how many of the 187 pairs each method registers. The classical baselines reproduce the benchmark's own finding: SIFT registers 3 pairs within 10 pixels, and adding mutual-information refinement moves the median error from 903 to 824 pixels without registering a single additional pair. The learned matchers do better. RoMa registers 17 pairs within 10 pixels and 42 within 20; MatchAnything-RoMa registers 20 and 45. The difference between these two is not statistically significant: 3 more pairs within 10 pixels (+0.016, 95 % CI −0.011 to +0.043, *p* = 0.33), and the same median error to within a pixel (*p* = 0.86). Both are far below the ceiling of Section 2.4, 93 pairs within 10 pixels.

Figure 4A and 4B show what failure looks like. In Figure 4A the two images are STEM images of the same area from the same instrument, one recorded with a dark-field detector and one with an annular dark-field detector. The dislocations are visible in both, but with different contrast, and MatchAnything-RoMa places the narrow image 2,043 pixels from where it belongs. In Figure 4B the narrow image is a light optical micrograph that covers 1.9 % of a lower-magnification light optical image of the same specimen. At the wide image's pixel size, the narrow image's entire field is recorded in 239 by 195 pixels (middle panel), so the fine structure the narrow image shows is simply not present in the wide one. The matcher places it 623 pixels away.

These examples also show something about the matcher's output that matters for everything below. RoMa computes a certainty for every correspondence and counts any certainty above 0.05 as fully certain before drawing its 10,000 (Section 2.4). In both failures, not one pixel of the wide image cleared that cut-off. The matcher's own certainty was, in effect, saying that it had found nothing. It still returned 10,000 correspondences, drawn from those below the cut-off, and robust fitting still found 100 and 302 of them that agreed on a transformation, which was wrong.

The share of returned correspondences that clear the cut-off is informative across the benchmark too. One RoMa run of the checked search (the variant in Section 3.3 that discards correspondences below the cut-off) records how many of each match's correspondences cleared it, and on 149 pairs that run kept the direct match as its answer. All 15 of those pairs that were registered had at least 93 % above the cut-off (median 99 %). Of the 51 that missed by more than 100 pixels, 48 had less than 93 % (median 65 %), but 31 still had more than half. Across the 149 pairs, a lower share went with a larger error (Spearman's rank correlation −0.71). A low share is therefore a useful warning sign, while a high share does not guarantee success. The 93 % boundary was read off these same pairs, so it has not been tested on other data.

### 3.2. Pooling tiles makes registration worse, and why

Pooled tiling (Section 2.6) made registration worse (Table 4). On the 33 genuinely tiled pairs, the median error rose from 68 pixels for the whole image matched once to 538 pixels for the pooled tiles, and the error rose on 26 of the 33 pairs (Figure 3E). Of the 4 pairs the direct match registered within 20 pixels, pooling kept 1 and moved the others to errors of 25, 335 and 6,580 pixels. Pooling lowered the error on 7 pairs, all of which the direct match had missed by more than 100 pixels, and none of them came within 20 pixels. On the 48 padded pairs, the error rose on all 48 and the number registered within 10 pixels fell from 10 to 1; that failure belongs to the padding, not to tiling (Section 2.6).

Figure 3 shows the mechanism on one pair, an SEM backscattered-electron image and an EBSD pattern-sharpness map of the same sectioned specimen, which pooled tiling cut into 92 tiles. A tile that lies inside the narrow image's true outline returned 10,000 correspondences, 87 % of them with certainty above the 0.05 cut-off (Figure 3B). A tile that lies entirely outside the outline also returned 10,000 correspondences (Figure 3C). Only 11 % of those cleared the cut-off, so the matcher's certainty did register that something was wrong, but the output step returned the full 10,000 anyway, and pooled tiling, like most fitting procedures, does not use certainty. Every tile that does not contain the narrow image therefore adds 10,000 wrong correspondences to the pool.

The effect on robust fitting follows by counting. With 75 tiles, the median on the tiled pairs, a pool holds 750,000 correspondences, and the largest pool held 9,420,000 from 942 tiles. A tile outside the narrow image contributes only wrong correspondences, and a tile inside it contains only part of the narrow image, so many of its correspondences are wrong too: fitted on its own, the inside tile of Figure 3B misses by 744 pixels. The share of correspondences that robust fitting could keep fell from a median of 0.10 for the whole image to 0.0009 for the pooled tiles, more than a hundredfold (Figure 3D). Robust estimators tolerate a large share of outliers, but not an arbitrarily large one. Once correct correspondences are a small fraction of a percent of the pool, a random subset almost never consists of correct ones, and a wrong transformation that happens to agree with many wrong correspondences wins.

The argument does not depend on the tile size, the overlap or the particular matcher. It predicts the same failure for any scheme that pools the output of a matcher returning a fixed number of correspondences per call, because the number of wrong correspondences grows with the number of tiles that do not contain the narrow image. What would have to change is not the pyramid but the order of operations: decide which tile to believe before pooling anything.

One caution about scope. Pooled tiling was evaluated on 81 of the 187 pairs (Section 2.6), and the 106 it never reached include 6 pairs that RoMa registers directly, among them all three pairs of the MAX-phase TEM series. Those would be the most direct test of whether tiling destroys a working registration, and they are missing from this comparison.

### 3.3. Zooming with a check avoids the collapse but gains nothing on real pairs

The checked search (Section 2.7) never collapsed. With RoMa, zooming once, it produced a transformation for all 187 pairs and registered exactly as many within 10 pixels as the direct match did, 17 (difference 0.000, 95 % CI −0.016 to +0.016, *p* = 1.00). Within 20 pixels it registered 41 against 42 (*p* = 0.83), and the median error moved from 80.2 to 69.9 pixels, a change well inside its confidence interval (−10.3 pixels, 95 % CI −48.2 to +23.2, *p* = 0.37). With MatchAnything-RoMa, zooming up to three times, it registered 22 pairs against 20.

The search changed which pairs succeed without changing how many. It registered one pair in the 0.25 to 0.5 field-of-view group that the direct match missed, and lost one in the group of 0.5 and above. Pair by pair, it lowered the error on 87 pairs and raised it on 98. The check compares image content, not error, so it can prefer a transformation that is further from the annotated points.

The record of which step supplied each final answer shows where the search does its work. In the RoMa run, on 149 of the 187 pairs the direct match was kept; on 37 a zoom step improved on it; the tile search supplied the answer on one pair. The tile search runs only when the direct fit keeps fewer than 50 inliers, and in the separate direct runs every one of the 187 fits kept at least 50, including fits that missed by thousands of pixels. The matcher's habit of always returning 10,000 correspondences means robust fitting always finds some that agree, so a weak direct match never looks weak by that test.

Two variants did not help (Supplementary Section S4). Zooming up to three times instead of once registered 16 pairs within 10 pixels instead of 17, and 38 within 20 instead of 41 (*p* = 0.73 and 0.09). With three zooms in both, discarding the correspondences below the matcher's certainty cut-off before each fit registered 41 pairs within 20 pixels against 38 without the discard (*p* = 0.24). The discard removed little: a median of 8,581 of the 10,000 correspondences cleared the cut-off.

### 3.4. When field of view is the only difficulty, the checked search triples success

On the field-of-view ladder (Section 2.8), the scale problem can be seen on its own. Figure 6B follows MatchAnything-RoMa on the 41 pairs it registers within 20 pixels at full size. Used directly, it registers about half of them within 10 pixels at full size and at half size, fewer at a quarter, and then collapses: at a field-of-view ratio of 0.1 it registers 3 of 40. With the checked search, zooming up to three times, it registers 9 of 40 at that ratio, three times as many (0.075 to 0.225, difference +0.150, 95 % CI +0.050 to +0.275, *p* = 0.0028). Of those 9 successes, 7 came from a zoom step, 1 from the tile search followed by a zoom, and 1 from the direct match. The gain holds on the 22 of these pairs that were not used for fine-tuning (1 to 5, *p* = 0.023) and on the 34 whose metadata is consistent (3 to 9, *p* = 0.0028). At ratios of 0.05 and 0.02 every method fails. With RoMa the same comparison is not significant (2 to 3 of 36, *p* = 0.73).

So the scale mechanism works when scale is the only thing wrong. The part that works is the zoom: a first estimate, even a poor one, tells the search where to look, and matching a crop of the wide image recovers the resolution that shrinking the whole image threw away. Tiling contributed to one of the nine successes.

### 3.5. On real pairs, a small field of view comes with a large difference in appearance

Why, then, does the checked search gain nothing on the real pairs? Table 5 and Figure 7A break down success by field-of-view group. Below a ratio of 0.5 there are 64 pairs, and no method in Table 3 registers more than 3 of them. Nearly all successes on the benchmark come from pairs whose two images cover similar areas.

The pairs with small fields of view also tend to be the pairs whose images look least alike (Figure 7B). Across the 187 pairs, the logarithm of the field-of-view ratio and the NMI of Section 2.9 are correlated (Pearson *r* = +0.21, *p* = 0.005; Spearman's rank correlation +0.17, *p* = 0.023). The relationship is weak and uneven: the median NMI is 0.015, 0.002, 0.155 and 0.066 in the four groups from smallest to largest ratio, so the 0.25 to 0.5 group is the most alike of all. On this benchmark, then, a small field of view and a large appearance difference occur together often enough that a method which fixes only scale has little to act on, and there are too few successes below a ratio of 0.5 to separate the two causes. This is a finding about the benchmark rather than about the methods. A benchmark in which field of view varies while appearance is held fixed, as it is on our ladder, would answer the question directly.

### 3.6. Training on microscopy pairs helps what it is shown and harms what it is not

Fine-tuning MatchAnything-RoMa on 131 benchmark pairs (Section 2.10) cut the median error on the 16 TEM pairs of the test set from 294 to 61 pixels. It did so at a cost. The 4 test pairs from the C103 series, whose combination of SEM and light optical height mapping had no training pairs, were all registered within 20 pixels before fine-tuning and none after (Figure 4C shows one of them: 6 pixels before, 473 after). Over the whole test set the number registered within 20 pixels fell from 11 to 7 of 28. This is catastrophic forgetting (Kirkpatrick et al., 2017) in its narrow sense: the network lost an ability it had, on a combination it was never shown, while improving on what it was shown.

The result is not a single unlucky training run. Five further training runs, whose results survive, each registered between 25.0 % and 28.6 % of the test set within 20 pixels after refinement, against 39.3 % before fine-tuning. The cheapest remedy, a penalty that pulls the trained weights back toward the released ones (L2-SP; Li et al., 2018), gave the same range. Supplementary Section S5 gives every run and two weaknesses of the protocol that we disclose.

Figure 4C also shows that the matcher's certainty does not flag this failure. After fine-tuning, 66 % of the 10,000 correspondences it returned for that pair cleared the certainty cut-off, and the transformation was still wrong by 473 pixels.

### 3.7. Scoring after an optional refinement step can change the conclusions

The benchmark's protocol applies the thin-plate-spline refinement of Section 2.11 when a fit has enough inliers. That condition makes the refined error a different measurement for different methods. Refinement ran on all 187 pairs for every RoMa-based method, on 93 for LoFTR, on 70 for SIFT and on none for SIFT with mutual information, whose refined score is therefore its unrefined score. Scored after refinement, both of the comparisons that are null in Sections 3.1 and 3.3 become statistically significant: the checked search against the direct match (+0.021, *p* = 0.034) and MatchAnything-RoMa against RoMa (+0.032, *p* = 0.035). Refinement also turned 13 registrations into failures across the 16 method configurations in our result files. We therefore report unrefined errors throughout, and recommend that anyone reporting registration results with an optional post-processing step also report how often the step ran for each method, and check their main comparisons with it switched off. Supplementary Section S3 gives the full comparison.

## 4. Discussion

### 4.1. Practical guidance

For a microscopist registering correlative images today, the results suggest the following.

1. **Start with a released dense matcher, used directly.** MatchAnything-RoMa and RoMa were the strongest methods we tested. On pairs like these, expect to register about one pair in ten within 10 pixels, and check every result by overlaying the images.
2. **Do not cut the wide image into tiles and pool the matches.** It makes registration worse for a structural reason (Section 3.2), and no choice of tile size or overlap changes that reason.
3. **To bridge a large difference in field of view, zoom instead.** Match the whole images, crop the wide image around the first estimate, match again, and keep the new result only if an independent check, such as mutual information over the overlap, prefers it. This works when the first estimate is roughly in the right place (Section 3.4) and cannot rescue a pair whose images do not look alike.
4. **Look at the matcher's certainty.** On the 149 pairs where we could check it, every registered pair had at least 93 % of its correspondences above RoMa's certainty cut-off, and most badly failed pairs had fewer (Section 3.1). A low share is a warning sign worth checking for. A high share is not a guarantee: many failures had a high share, and the fine-tuned network in Figure 4C was confidently wrong.
5. **If you fine-tune, keep a held-out pair of every modality combination you care about.** Training on some combinations can break others (Section 3.6), and an average over a test set can hide it.
6. **When you report results, know the ceiling and score before optional post-processing.** Hand-annotated points do not fit a single transformation perfectly; on this benchmark a transformation fitted to the annotations themselves registers only 93 of 187 pairs within 10 pixels.

### 4.2. Why pooled tiling fails in general

The mechanism does not depend on the details of our implementation. A dense matcher of this kind is built to produce a correspondence field, and its standard output step draws a fixed number of correspondences from that field whatever the images contain. Its certainty map does carry information about whether a match exists, as Figures 3 and 4 show, but that information is discarded when the correspondences are handed to a fitting step that treats them all alike. Pooling many such outputs multiplies the wrong correspondences by the number of tiles that do not contain the narrow image, and robust fitting has a limit to how many outliers it can reject. Any scheme that pools before it decides which part of the wide image to trust inherits this.

The same property explains why the checked search's tile stage rarely ran: a direct match never looks weak by its inlier count when it always returns 10,000 correspondences (Section 3.3). Using the certainty is not a simple fix either. Discarding correspondences below the matcher's cut-off before fitting did not help (Section 3.3), because on many failed pairs most correspondences clear the cut-off too (Section 3.1). What the pooled design lacks is a decision about which tile to trust, and a per-correspondence threshold does not supply it.

### 4.3. Limitations

**One benchmark.** Every measurement is on AmalgaMatch. The argument of Section 4.2 should hold for any matcher with a fixed output count and any pooling scheme, but we have not tested that.

**The pooled-tiling run is incomplete.** It covers 81 of 187 pairs and misses 6 pairs that RoMa registers directly (Section 3.2). Rerunning the remaining 106 pairs needs 4,303 further tile matches on a graphics processor. The implementation also sized tiles in the narrow image's pixels and padded small wide images with mirror copies (Section 2.6); sizing tiles in the wide image's pixels would test tiling on the padded pairs as well.

**Metadata.** For 13 pairs the metadata pixel sizes disagree with the annotated points, so the loader may have treated the narrower image as the wide one, and the tiling methods used a wrong scale for those pairs. The field-of-view groups use the ratio implied by the annotated points, and the ladder result holds without the affected pairs.

**Unequal ladder sets.** Each matcher's ladder uses the pairs that matcher registers at full size (38 for RoMa, 41 for MatchAnything-RoMa), so the ladder compares the checked search with direct matching within each matcher, not the matchers with each other.

**The check is not the error.** The checked search never lowers its own mutual-information score, but it raised the registration error on 98 of 187 pairs.

**Not everything regenerates from the released files.** The per-run results of three fine-tuning runs were not retained (Supplementary Section S5). Everything else in this paper regenerates from the released per-pair result files and scripts.

## 5. Conclusions

Cutting the wide image into tiles, matching each tile and pooling the matches is a natural, training-free answer to a large field-of-view mismatch, and it does not work. The matchers return the same number of correspondences for every tile whether or not the tile contains the narrow image, so the correct correspondences are swamped and robust fitting fails. A search that matches the whole images first, then zooms in on that estimate and accepts a new answer only when an independent check prefers it, avoids the failure and triples success when field of view is the only difficulty. On the 187 real pairs of the benchmark it gains nothing: there, a small field of view tends to come with a large difference in appearance, and the benchmark has too few registered pairs below a field-of-view ratio of 0.5 to separate the two. The released matchers register about one pair in ten within 10 pixels, against a ceiling of one pair in two set by the annotations themselves, and the gap is the problem the field still has to solve.

## Supplementary Material

Supplementary Material is available online and contains: S1, the pooled-tiling run in detail; S2, field-of-view metadata compared with the annotated points; S3, results after thin-plate-spline refinement; S4, variants of the checked search; S5, the fine-tuning protocol and every training run; S6, the project's original hypotheses and their outcomes; and S7, a comparison of the worked-example repeat runs with the benchmark runs.

## Acknowledgments

The author thanks the authors of the AmalgaMatch benchmark for releasing the dataset under an open licence. The micrographs in Figures 1 to 4 and 6 and in Supplementary Figure S2 are from that dataset (Durmaz et al., 2026a), used under the Creative Commons Attribution 4.0 licence.

## Competing Interests

The author declares no competing interests.

## Author Contributions

**Frank Cai:** Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing — original draft, Writing — review and editing, Visualization.

## Data Availability

The AmalgaMatch dataset is publicly available from the Fraunhofer Fordatis repository (doi:10.24406/fordatis/436). The analysis code, evaluation software, both tiling schemes, the field-of-view ladder, the fine-tuning trainer, every analysis script and the per-pair result files are openly available at https://github.com/fronkt/correlative-microscopy-alignment and archived at Zenodo (doi:10.5281/zenodo.20819649). Every number in this paper is recomputed from the result files by `scripts/mam_rewrite_numbers.py`, and every figure is drawn by `scripts/plot_mam_rewrite.py`; the worked-example repeat runs are regenerated, with the same random seed, by `scripts/mam_examples_run.py`. The exceptions are the per-run results of three fine-tuning runs, which were not retained. Fine-tuned model weights (445 MB) are available from the author on reasonable request.

## References

Barath, D., Noskova, J., Ivashechkin, M. & Matas, J. (2020). MAGSAC++, a fast, reliable and accurate robust estimator. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 1304–1312.

Bookstein, F.L. (1989). Principal warps: Thin-plate splines and the decomposition of deformations. *IEEE Trans. Pattern Anal. Mach. Intell.* **11**, 567–585.

Durmaz, A.R., Lamb, J.D., Echlin, M.P. & Pollock, T.M. (2026a). AmalgaMatch: A benchmark dataset for cross-modal image matching in correlative materials microscopy. *Fordatis, Fraunhofer Research Data Repository*.

Durmaz, A.R., Lamb, J.D., Echlin, M.P. & Pollock, T.M. (2026b). Foundation models for multimodal image data fusion in materials science. *Front. Mater.* **13**.

Edstedt, J., Sun, Q., Bökman, G., Wadenbäck, M. & Felsberg, M. (2024). RoMa: Robust dense feature matching. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 19790–19800.

Fischler, M.A. & Bolles, R.C. (1981). Random sample consensus: A paradigm for model fitting with applications to image analysis and automated cartography. *Commun. ACM* **24**, 381–395.

He, X., Yu, H., Peng, S., Dong, D., Tan, D., Zhou, X., Bao, H. & Shen, Z. (2025). MatchAnything: Universal cross-modality image matching with large-scale pre-training. *arXiv preprint* arXiv:2501.07556.

Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A.A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., Hassabis, D., Clopath, C., Kumaran, D. & Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks. *Proc. Natl. Acad. Sci. U. S. A.* **114**, 3521–3526.

Li, X., Grandvalet, Y. & Davoine, F. (2018). Explicit inductive bias for transfer learning with convolutional networks. In *Proceedings of the 35th International Conference on Machine Learning (ICML)*, vol. 80, pp. 2825–2834.

Lowe, D.G. (2004). Distinctive image features from scale-invariant keypoints. *Int. J. Comput. Vis.* **60**, 91–110.

Maes, F., Collignon, A., Vandermeulen, D., Marchal, G. & Suetens, P. (1997). Multimodality image registration by maximization of mutual information. *IEEE Trans. Med. Imaging* **16**, 187–198.

Sun, J., Shen, Z., Wang, Y., Bao, H. & Zhou, X. (2021). LoFTR: Detector-free local feature matching with transformers. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 8922–8931.

Wang, Y., He, X., Peng, S., Tan, D. & Zhou, X. (2024). Efficient LoFTR: Semi-dense local feature matching with sparse-like speed. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.

## Tables

**Table 1.** Terms used in this paper, in the order they first matter.

| Term | Meaning here |
|----------------------|------------------------------------------------------------|
| Registration | Finding where every point of one image lies in the other, so that the two can be overlaid. |
| Wide image, narrow image | The image of a pair that shows more of the specimen, and the one that shows less. Methods locate the narrow image inside the wide one. |
| Field-of-view ratio | Area of specimen shown in the narrow image divided by the area shown in the wide image; 1 means the same area. |
| Annotated points (ground truth) | Points marked by hand on the same features in both images. Used only to measure error. |
| Image matcher | A neural network that takes two images and returns correspondences. |
| Correspondence | A pair of points, one in each image, that a matcher judges to show the same place. |
| Dense matcher | A matcher that estimates a correspondence for every pixel (RoMa and its MatchAnything version). |
| Certainty | A value between 0 and 1 that RoMa attaches to each correspondence. Its output step treats anything above 0.05 as fully certain and then draws 10,000 correspondences. |
| Zero-shot | Used with the released weights, without training on micrographs. |
| Fine-tuning | Further training of a released network on pairs from the benchmark. |
| Transformation | The mathematical map from narrow-image coordinates to wide-image coordinates. |
| Affine transformation | A transformation allowing shift, rotation, scaling and shear (six numbers). |
| Homography | A transformation that also allows perspective distortion (eight numbers). |
| Robust fitting | Fitting a transformation while ignoring correspondences that disagree with it; here MAGSAC++. |
| Inlier, outlier | A correspondence that agrees with the fitted transformation to within 5.5 pixels, and one that does not. |
| Registration error | Mean distance, in wide-image pixels, between where the fitted transformation puts each annotated point and where it was annotated. |
| Registered | Registration error below 10 pixels (20 pixels where stated). |
| Pooled tiling | Cutting the wide image into tiles, matching each against the narrow image, and fitting one transformation to all correspondences together. |
| Checked search | Matching the whole images, then optionally searching tiles and zooming in, accepting each new result only if mutual information prefers it. |
| Zoom step | Cropping the wide image around the current estimate and matching the crop, which the matcher then sees at higher resolution. |
| Mutual information | How well the grey levels of one image predict those of the other; high when the images are aligned even if their contrast differs. |
| Normalised mutual information (NMI) | Mutual information scaled to lie between 0 (unrelated) and 1 (one image fully predicts the other); used here to measure how alike a pair looks. |
| Field-of-view ladder | A controlled series in which the narrow image is cropped to smaller and smaller areas while everything else stays the same. |
| Thin-plate-spline refinement | An optional smooth warp applied after fitting to bring the transformation closer to its inliers. |
| Paired bootstrap, confidence interval, *p*-value | Resampling the pairs to estimate how much a difference between two methods could vary by chance (Section 2.12). |

**Table 2.** Methods compared. All learned matchers are used with their released weights.

| Method | Kind | Trained on | Output |
|--------------------|------------------------------|------------------|----------------|
| SIFT | Classical feature matching | Not trained | Variable number of correspondences |
| SIFT + mutual information | SIFT, then intensity-based refinement | Not trained | One transformation |
| LoFTR | Learned, matches without detecting points first | Outdoor photographs | Variable number |
| MatchAnything-ELoFTR | Efficient LoFTR, retrained | Synthetic cross-modal images | Variable number |
| RoMa | Learned, dense | Outdoor photographs | Always 10,000 |
| MatchAnything-RoMa | RoMa, retrained | Synthetic cross-modal images | Always 10,000 |
| + checked search | Wrapper around RoMa or MatchAnything-RoMa (Section 2.7) | Not trained | As the matcher |
| + pooled tiling | Wrapper around RoMa (Section 2.6) | Not trained | 10,000 per tile |

**Table 3.** Registration of all 187 pairs, errors before refinement. "Fitted" counts the pairs for which the method returned a transformation at all; the others count as failures. Median errors are over the fitted pairs. The last row is the ceiling of Section 2.4: a homography fitted to each pair's own annotated points. The checked search zoomed once with RoMa and up to three times with MatchAnything-RoMa (Section 2.7). Pooled tiling is not in this table because it was evaluated on 81 pairs; see Table 4.

| Method | Registered within 10 pixels | Registered within 20 pixels | Median error (pixels) | Fitted |
|----------------------------------|------------|------------|------------|--------|
| SIFT | 3 | 4 | 903.1 | 169 |
| SIFT + mutual information | 3 | 4 | 823.6 | 169 |
| LoFTR | 12 | 17 | 270.1 | 183 |
| MatchAnything-ELoFTR | 2 | 6 | 510.1 | 177 |
| RoMa | 17 | 42 | 80.2 | 187 |
| RoMa + checked search | 17 | 41 | 69.9 | 187 |
| MatchAnything-RoMa | 20 | 45 | 81.0 | 187 |
| MatchAnything-RoMa + checked search | 22 | 43 | 77.6 | 187 |
| Ceiling: homography fitted to the annotated points | 93 | 156 | – | – |

**Table 4.** Pooled tiling compared with matching the whole image once, with RoMa, on the same pairs; each cell gives the whole-image value, then the pooled value. "Share kept" is the median fraction of correspondences that robust fitting kept as inliers. The other 106 of the 187 pairs were not evaluated, because the run stopped (Section 2.6).

| | Genuinely tiled | Padded into one tile |
|--------------------------------------------|--------------------|--------------------|
| Pairs | 33 | 48 |
| Tiles per pair | 2 to 942 (median 75) | 1 |
| Median error (pixels) | 67.6 → 537.7 | 660.0 → 6156.3 |
| Registered within 20 pixels | 4 → 1 | 11 → 3 |
| Registered within 10 pixels | 1 → 0 | 10 → 1 |
| Share kept by robust fitting | 0.10 → 0.0009 | 0.090 → 0.0065 |
| Pairs on which pooling raised the error | 26 of 33 | 48 of 48 |

**Table 5.** Pairs registered within 10 pixels, by field-of-view group, errors before refinement. The groups use the ratio implied by the annotated points (Section 2.2). NMI is the median normalised mutual information of the group's pairs (Section 2.9).

| | Below 0.05 | 0.05 to 0.25 | 0.25 to 0.5 | 0.5 and above |
|----------------------------------|----------|-----------|-----------|-----------|
| Pairs | 2 | 34 | 28 | 123 |
| Median NMI | 0.015 | 0.002 | 0.155 | 0.066 |
| SIFT | 0 | 0 | 1 | 2 |
| SIFT + mutual information | 0 | 0 | 1 | 2 |
| LoFTR | 0 | 0 | 1 | 11 |
| MatchAnything-ELoFTR | 0 | 0 | 0 | 2 |
| RoMa | 1 | 0 | 1 | 15 |
| RoMa + checked search | 1 | 0 | 2 | 14 |
| MatchAnything-RoMa | 0 | 0 | 1 | 19 |
| MatchAnything-RoMa + checked search | 0 | 0 | 2 | 20 |

## Figure Legends

**Figure 1.** The registration problem in three pairs from the AmalgaMatch benchmark. Top row: the wide image, with the true outline of the narrow image (yellow) and the annotated points (yellow dots). Bottom row: the narrow image with the same annotated points. **A:** an EBSD pattern-sharpness map and an SEM backscattered-electron image of the same area of a cobalt-nickel alloy; the two images cover nearly the same area but record different physical signals. **B:** a light optical image of a bainitic steel and an EBSD orientation map of a region covering a quarter of it. **C:** an SEM secondary-electron image of a spalled specimen and a backscattered-electron image of a region covering 4.9 % of it. Micrographs: Durmaz et al. (2026a), CC BY 4.0.

*Alt text:* Three columns of paired micrographs. In each column, the upper image is the larger field of view with a yellow outline showing where the lower image belongs, and yellow dots marking hand-annotated matching points. The left pair covers the same area in two different contrasts; the middle pair is a colour optical image and a colourful orientation map of a quarter of it; the right pair is a low-magnification electron image with a small outlined rectangle and a high-magnification image of that rectangle.

**Figure 2.** Registering one pair step by step, with RoMa (the pair of Figure 1C). **A:** Steps 1 and 2. The matcher proposed 10,000 correspondences; 120 are drawn, each as a dot of the same colour in both images. Robust fitting kept the 2,032 that agree on one transformation (green) and rejected 7,968 (vermillion). **B:** Step 3. The fitted transformation places the narrow image (dashed vermillion outline) over its true position (yellow). **C:** Step 4. At each annotated point (yellow dot), an arrow shows where the fitted transformation puts the corresponding point of the narrow image, drawn five times longer than the true miss so that it is visible. **D:** Step 5. The misses at the 19 annotated points; their mean, 2.7 pixels, is the pair's registration error, below the 10-pixel threshold. This is a repeat run (Section 2.13); the benchmark run gave 3.3 pixels.

*Alt text:* Four panels. The first shows a low-magnification electron image and a high-magnification image with matching coloured dots; green dots cluster in the small region where the images overlap and red-orange dots are scattered elsewhere. The second shows a dashed outline lying on top of a solid yellow outline. The third shows short arrows at annotated points. The fourth is a histogram of per-point errors, all below 10 pixels, with the mean marked at 2.7 pixels.

**Figure 3.** Why pooling the matches of many tiles fails. **A:** An SEM backscattered-electron image of a sectioned cobalt-nickel alloy, cut by pooled tiling into 92 tiles (the 77 full-resolution tiles are drawn in white), with the true outline of the narrow image, an EBSD pattern-sharpness map, in yellow. Contrast adjusted for display. **B:** The tile outlined in green, which lies inside the true outline. RoMa returned 10,000 correspondences; their positions are drawn in blue where the matcher's certainty was above its 0.05 cut-off (87 %) and in black where it was below. **C:** The tile outlined in vermillion, which contains none of the narrow image. RoMa again returned 10,000 correspondences, only 11 % of them above the cut-off. **D:** On all 33 genuinely tiled pairs, the fraction of correspondences that robust fitting could keep, for the whole image matched once (blue) and for the pooled tiles (purple); lines join the same pair. **E:** Registration error of the same 33 pairs, whole image against pooled tiles; points above the dashed line got worse. Panels B and C are repeat runs (Section 2.13); panels D and E are from the benchmark runs.

*Alt text:* Five panels. The first shows an electron image divided by a white grid of square tiles, with a green square inside a yellow outline and an orange square outside it. The next two show those two tiles covered with dots: the inside tile has large blue regions, the outside tile is covered mostly in black dots with few blue ones. A dot plot on a logarithmic scale shows the fraction of kept correspondences falling steeply from the whole-image match to the pooled tiles for almost every pair. A scatter plot on logarithmic axes shows most pairs above the diagonal, meaning larger errors after pooling.

**Figure 4.** Examples of failure. In each row, the left panel shows the wide image with the true outline of the narrow image (yellow) and where the matcher placed it (dashed vermillion), and the right panel shows the narrow image. **A:** Appearance. Two STEM images of the same area of a refractory high-entropy alloy, recorded with a dark-field and an annular dark-field detector; the dislocations appear with different contrast. The middle panel shows the region the narrow image covers, as the wide image records it. MatchAnything-RoMa misses by 2,043 pixels (benchmark run: 508 pixels). **B:** Field of view. A light optical image of a cobalt-nickel alloy and a higher-magnification light optical image covering 1.9 % of its area. The middle panel shows that the wide image records the narrow image's entire field in 239 by 195 pixels. MatchAnything-RoMa misses by 623 pixels (benchmark run: 630). **C:** Forgetting. An SEM secondary-electron image and a light optical extended-focus image of the same C103 fracture surface, a combination absent from the fine-tuning data. MatchAnything-RoMa registers it (6 pixels; benchmark run 6.2); after fine-tuning on other benchmark pairs, the same network misses by 473 pixels (benchmark run: 387). All three rows are repeat runs (Section 2.13). Micrographs: Durmaz et al. (2026a), CC BY 4.0.

*Alt text:* Three rows of micrographs with outlines. In the first row, a dashed outline cuts diagonally across a transmission electron image while the solid yellow outline frames the whole image. In the second row, a small yellow rectangle marks the true location in a low-magnification optical image and a dashed diamond sits in the wrong place; a heavily pixelated enlargement shows how little detail the wide image holds there. In the third row, the dashed outline matches the yellow one before fine-tuning and is badly skewed after it.

**Figure 5.** Number of the 187 pairs registered within 10 pixels (**A**) and 20 pixels (**B**), errors before refinement. Blue: matchers used directly. Orange: the same matchers inside the checked search. Error bars are 95 % Wilson intervals. The dashed line is the ceiling set by the annotations: a homography fitted to each pair's own annotated points.

*Alt text:* Two horizontal bar charts with the same eight methods. Classical methods register 2 to 12 pairs; the RoMa-based methods register 17 to 22 pairs within 10 pixels and 41 to 45 within 20 pixels. Dashed lines at 93 and 156 mark the best any single transformation could achieve.

**Figure 6.** The field-of-view ladder. **A:** For one pair, the crops of the narrow image at area ratios of 0.5, 0.25, 0.1, 0.05 and 0.02, drawn where they lie in the wide image; the outermost yellow outline is the uncropped narrow image. **B:** MatchAnything-RoMa on the 41 pairs it registers within 20 pixels before cropping: the fraction registered within 10 pixels as the narrow image is cropped, used directly (blue) and inside the checked search (orange). Error bars are 95 % Wilson intervals. At a ratio of 0.1 the checked search registers 9 of 40 pairs against 3 of 40 (*p* = 0.0028).

*Alt text:* Left, an electron image with five nested yellow rectangles labelled 0.5 down to 0.02. Right, a line chart: both curves start near 0.5, fall as the crop gets smaller, and reach zero at 0.05 and 0.02; between 0.25 and 0.1 the orange curve stays above the blue one, reaching about three times its value at 0.1.

**Figure 7.** Why the benchmark cannot separate field of view from appearance. **A:** Fraction of pairs registered within 10 pixels in each field-of-view group, for RoMa (circles) and MatchAnything-RoMa (squares), used directly (blue) and inside the checked search (orange), with 95 % Wilson intervals; the number of pairs in each group is given under its label. **B:** Normalised mutual information of each pair against its field-of-view ratio (logarithmic axis); dotted lines mark the group boundaries. Pairs with less area in common tend to look less alike, but the trend is weak.

*Alt text:* Left, a dot chart of success by field-of-view group: near zero in the two middle groups, around 0.1 to 0.15 in the largest group, and with a very wide interval in the smallest group, which holds only two pairs. Right, a scatter plot of 187 pairs; most pairs sit at ratios near 1, a few at small ratios, and similarity values are low overall with a slight upward trend toward larger ratios.
