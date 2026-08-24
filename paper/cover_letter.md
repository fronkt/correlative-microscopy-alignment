> **SUPERSEDED — historical record of the Scientific Reports submission.**
>
> This is the cover letter as submitted to *Scientific Reports* (submission
> 46c1e084-d0e4-4fb1-b217-a2b79135181f), which was **rejected** on 2026-08-24.
> It is kept unchanged as the record of what was sent.
>
> Do not reuse it as-is. It names the pre-revision title, "For foundation-model
> registration in correlative microscopy, cross-modal appearance matters more than
> field of view", which the round-6 revision retired because a directly measured
> appearance axis does not support it (see `REVISIONS.md`, round 6). The current
> title is "Pyramidal wrappers break non-abstaining dense matchers: a diagnostic
> study of foundation-model registration in correlative microscopy".
>
> A new cover letter is required and cannot be written until the target venue is
> chosen. It should lead with the non-abstention mechanism and the controlled FOV
> ladder, and should disclose the prior review history.

# Cover letter — *Scientific Reports*

Frank Cai
Purdue University, West Lafayette, IN, USA
frankyc11223@gmail.com · ORCID 0009-0003-0041-1459

[Date]

To the Editors, *Scientific Reports*

Dear Editors,

I am pleased to submit my manuscript, **"For foundation-model registration in
correlative microscopy, cross-modal appearance matters more than field of view,"**
for consideration as an Article in *Scientific Reports*.

Correlative microscopy requires images of one specimen, acquired by different
instruments (SEM, EBSD, TEM, optical), to be brought into a common coordinate frame.
Registration is the bottleneck, because the modalities share little appearance and
their fields of view can differ by orders of magnitude. A natural and increasingly
popular hope is that pretrained foundation matchers, optionally wrapped in a
scale-aware patching layer, can solve this without retraining. This study tests that
hope end-to-end on the full AmalgaMatch benchmark (187 pairs, 19 material subsets) and
reports what actually works, including the interventions that do not.

The contributions are methodological and diagnostic rather than a single headline
score:

1. **A mechanistic account of why naive pyramidal wrappers break dense matchers.**
   Because dense matchers never abstain, every tile returns thousands of confident
   correspondences that flood robust estimation (inlier fraction 0.114 to 0.005;
   median error 76 to 1794 px). This is a structural property of the matcher class,
   and it predicts the failure for any pool-then-fit pyramid. A redesigned,
   verified coarse-to-fine wrapper recovers a small but significant, regression-free
   gain.

2. **A controlled FOV-ladder protocol** that decouples scale from appearance by
   cropping base-matchable pairs, isolating the variable that the real distribution
   confounds. With scale isolated, the wrapper triples success at 10% FOV
   (p = 0.0014), validating the mechanism the real benchmark cannot.

3. **Evidence that cross-modal appearance, not FOV, is the dominant uncontrolled
   constraint on this benchmark**, reached by elimination: the only off-the-shelf
   intervention with a significant gain was a change of backbone weights
   (MatchAnything-RoMa), and domain fine-tuning produced the largest single error
   reduction observed (~5x on in-distribution TEM).

4. **An eight-seed robustness analysis of decoder-only fine-tuning** that overturns an
   earlier single-seed narrative and isolates breadth of modality coverage, not the
   optimiser or a weight anchor, as the binding constraint on forgetting.

I believe the work fits *Scientific Reports* well. It is a technically sound,
reproducible study whose value does not depend on a positive headline: it reports a
rejected hypothesis alongside the controlled experiment that explains why, and it
offers a concrete, actionable recommendation for correlative-microscopy pipelines. The
emphasis on honest negative and null results, with statistical power and reproducibility
limitations disclosed rather than hidden, is exactly the kind of contribution the
journal's soundness-based criterion is designed to support, and it should be useful to
the broad imaging, microscopy, and machine-learning readership.

All code, the evaluation harness, and the result-regeneration scripts are openly
available on GitHub and archived at Zenodo (doi:10.5281/zenodo.20819649). The
AmalgaMatch dataset is public (doi:10.24406/fordatis/436). Every table and figure
regenerates from the released result files.

This manuscript is original, has not been published previously, and is not under
consideration elsewhere. There are no competing interests, and the work received no
specific funding. As sole author I am responsible for all aspects of the study.

Thank you for considering this submission. I look forward to the reviewers' comments.

Sincerely,

Frank Cai
