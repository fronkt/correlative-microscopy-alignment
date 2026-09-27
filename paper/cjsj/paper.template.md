% Which Automatic Image Registrations Can a Microscopist Trust? A Label-Free Test on a Correlative Microscopy Benchmark
% Frank Cai
% Homestead High School, Fort Wayne, Indiana, USA

Abstract—Correlative microscopy combines images of the same specimen taken with different instruments, which first requires registering the images: finding the transformation that maps one onto the other. Foundation image matchers can do this automatically, but on the AmalgaMatch benchmark they succeed on only about a quarter of image pairs, and they give no warning when they fail. I tested whether scores that need no ground truth can tell which registrations to trust. Each of {{n_pairs}} image pairs was registered {{n_voters}} ways, and three scores were compared: the fraction of proposed correspondences that survive the robust fit, the agreement between independent registrations, and their combination. {{ABSTRACT_RESULTS}} The hypotheses, statistics and pass criteria were published before any experiment was run.

# Introduction

Correlative microscopy places images of the same region of a specimen, taken with different instruments, on one coordinate grid, so that every position carries several kinds of measurement at once, for example crystal orientation from electron backscatter diffraction (EBSD) and surface contrast from a scanning electron microscope (SEM) [1]. The first step is registration: finding the transformation that maps the narrower image into the wider one. Doing this by hand is slow, and foundation image matchers trained on millions of photographs, such as RoMa [3] and its cross-modally retrained version MatchAnything-RoMa [4], promise to automate it. On AmalgaMatch, a public benchmark of {{n_pairs}} correlative image pairs from materials science [1], [2], these matchers register only about one pair in four to within 20 pixels [5].

Failure alone would be manageable if the software reported it. It does not. A dense matcher returns 10,000 correspondences for every pair, whether or not the two images show the same place, and robust fitting always finds some that agree [5]. A researcher who runs automatic registration on a new batch of images therefore cannot tell which results to keep.

Durmaz et al. [2] found that the fraction of correspondences surviving the robust fit separated good from bad registrations on 48 pairs from four subsets of the benchmark, and proposed, without testing it, using this fraction to pick the best registration and to send doubtful ones to a human. This paper tests that proposal on the whole benchmark. **H1:** the retained fraction predicts success on all pairs, and a cut-off chosen on the kinds of pairs Durmaz et al. studied still works on the other kinds. **H2:** agreement among several independent registrations of the same pair predicts success better. **H3:** choosing one registration per pair with these scores registers more pairs than the best single method. The hypotheses, the statistics and their pass criteria were committed to a public repository before any experiment was run [6].

# Methods

## Data
AmalgaMatch [1] contains {{n_pairs}} image pairs from {{n_scenes}} scenes, {{n_subclasses}} material and instrument combinations, and six task groups (for example serial sectioning, fracture surfaces and dislocation imaging in the transmission electron microscope). Every pair has hand-annotated corresponding points, which are used only to score results, never to produce them. The data are released under a CC BY 4.0 licence.

## Candidate registrations
Each pair was registered {{n_voters}} ways, which form the voting pool. Five matchers were applied to the whole images: the classical SIFT [7], LoFTR [8], RoMa [3], and the MatchAnything versions of RoMa and of Efficient LoFTR [4]. RoMa and MatchAnything-RoMa were also run inside a coarse-to-fine search that zooms into the wide image [5]. Finally, RoMa and MatchAnything-RoMa were run on four label-free transformations of the images: the narrow image with inverted contrast, the narrow image with its brightness histogram matched to the wide image, both images after contrast-limited adaptive histogram equalization, and both images replaced by their gradient magnitude. As a control, RoMa and MatchAnything-RoMa were rerun with five further random seeds; these ten runs never vote. Every set of correspondences was fitted with MAGSAC++ [9] at a 5.5-pixel threshold to an affine transformation or a homography, following the benchmark protocol [1].

## Label-free scores
Three scores were computed without any annotated points. **S1**, the retained fraction [2], is the number of correspondences kept by the robust fit divided by the number proposed. **S2**, agreement, compares a candidate with the other members of the voting pool: for each other candidate, the two transformations are applied to a 5 × 5 grid of points spanning the narrow image and the mean distance between the two mapped grids is taken; S2 is minus the median of these distances, so a candidate that lands where most others land scores high. **S3** averages the percentile ranks of S1 and S2.

## Evaluation
A registration's error is the mean distance, in pixels of the wide image, between where it maps the annotated points and where they truly lie, with no refinement step. A registration succeeds if this error is at most 20 pixels. The best any single global transformation can do is measured by fitting a homography to each pair's own annotated points. How well a score separates successes from failures is measured by the area under the receiver operating characteristic curve (AUROC), the probability that a randomly chosen success scores higher than a randomly chosen failure; 0.5 is chance and 1 is perfect. Because pairs from the same scene share images, 95 % confidence intervals come from resampling whole scenes with replacement 10,000 times. Paired differences in success are tested with the exact McNemar test [10]. For the transfer test, a cut-off on S1 was chosen by Youden's index [11] on the two task groups Durmaz et al. studied (same-slice and serial sectioning) and applied unchanged to the other four. For triage, pairs are ranked by score and the highest-scoring ones accepted.

{{RESULTS_AND_DISCUSSION}}

# Acknowledgment

The benchmark images and annotations are the work of Durmaz et al. [1] and are used under CC BY 4.0. The author used Claude (Anthropic), an AI assistant, to search the literature, to help design the experiment, to write the analysis and plotting code, and to draft and edit the text; the author chose the research question, reviewed the design, code and results, revised the manuscript, and takes responsibility for its content. Code and results: github.com/fronkt/correlative-microscopy-alignment, branch cjsj-tta.

# References

1. A. R. Durmaz, J. D. Lamb, K. Zaripova, M. Vailhe, J. T. Pürstl, J. Schulte, M. Ackermann, M. P. Echlin, and T. M. Pollock, "A correlative microscopy dataset for multimodal data fusion and image matching in materials science," *Scientific Data*, 2026, doi:10.1038/s41597-026-07961-2. Dataset: doi:10.24406/fordatis/436.
2. A. R. Durmaz, J. D. Lamb, M. P. Echlin, and T. M. Pollock, "Foundation models for multimodal image data fusion in materials science," *Frontiers in Materials*, vol. 13, art. 1815017, 2026, doi:10.3389/fmats.2026.1815017.
3. J. Edstedt, Q. Sun, G. Bökman, M. Wadenbäck, and M. Felsberg, "RoMa: Robust dense feature matching," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2024, pp. 19790–19800.
4. X. He, H. Yu, S. Peng, D. Dong, D. Tan, X. Zhou, H. Bao, and Z. Shen, "MatchAnything: Universal cross-modality image matching with large-scale pre-training," arXiv:2501.07556, 2025.
5. F. Cai, "For foundation-model registration in correlative microscopy, cross-modal appearance matters more than field of view," Research Square preprint, 2026, doi:10.21203/rs.3.rs-10311464/v1.
6. F. Cai, pre-registration of this study, commit 60c04bf, github.com/fronkt/correlative-microscopy-alignment (branch cjsj-tta), 27 Sep. 2026.
7. D. G. Lowe, "Distinctive image features from scale-invariant keypoints," *Int. J. Computer Vision*, vol. 60, pp. 91–110, 2004.
8. J. Sun, Z. Shen, Y. Wang, H. Bao, and X. Zhou, "LoFTR: Detector-free local feature matching with transformers," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2021, pp. 8922–8931.
9. D. Barath, J. Noskova, M. Ivashechkin, and J. Matas, "MAGSAC++, a fast, reliable and accurate robust estimator," in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2020, pp. 1304–1312.
10. Q. McNemar, "Note on the sampling error of the difference between correlated proportions or percentages," *Psychometrika*, vol. 12, pp. 153–157, 1947.
11. W. J. Youden, "Index for rating diagnostic tests," *Cancer*, vol. 3, pp. 32–35, 1950.
