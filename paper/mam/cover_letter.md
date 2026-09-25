# Cover letter

Dear Dr. Michael,

Please consider the enclosed manuscript, **"Image tiling does not solve field-of-view mismatch in correlative microscopy registration,"** as a Regular Article in *Microscopy and Microanalysis*.

This is a rewritten version of manuscript MAM-26-246, which you declined on 11 September 2026. Your comment was that the manuscript "may contain important new information for the microscopy community" but was "extremely difficult to read," and you suggested that it "be written in a more tutorial way so that terms and methods are better described and examples of the failures are shown." I agreed with that assessment. The earlier text had been condensed from a version written for a machine-learning venue, and it assumed a reader who already knew the methods. I have rewritten the paper from the beginning for a microscopist reader, and I want to be open that this is the same study.

**Terms.** Every technical term is now defined in plain language where it first matters, and Table 1 collects them in one place. Abbreviations are kept to the standard microscopy ones.

**Methods.** Section 2.4 walks through the registration of one real pair step by step (Figure 2): what the matching network returns, how robust fitting keeps some correspondences and rejects others, how the fitted transformation places the narrow image, and how the error is measured. Both tiling schemes are described as numbered steps, and a new paragraph explains the ceiling that the hand-annotated points themselves set on any method.

**Examples of the failures.** The earlier version reported failures only as rates. The rewrite shows them as images of real benchmark pairs. Figure 1 shows the problem itself. Figure 3 shows the mechanism of the main negative result on one pair: a tile containing none of the narrow image still yields 10,000 correspondences, which are pooled with the rest. Figure 4 shows three kinds of failure: a pair that looks different in the two images, a pair with a very small field of view, and a pair that a fine-tuned network forgets how to register. Section 4.1 then turns the results into practical guidance for someone registering their own images.

**Corrections.** While preparing these examples I rechecked every number against the per-pair result files, and found three errors in the earlier version that I want to disclose.

1. The earlier text said that pooled tiling produced "hard failures" on 106 of the 187 pairs. In fact, the computation for those 106 pairs stopped because of a fault on the graphics processor, so the method was never evaluated on them. The rewrite reports pooled tiling on the 81 pairs where it did run, compared with direct matching on the same pairs. It also separates the 33 pairs that were genuinely tiled from 48 on which the implementation padded the image instead of tiling it (Section 2.6, Table 4, Supplementary Section S1).
2. The field-of-view groups had been computed from the benchmark's pixel-size metadata, although the text described them as derived from the annotated points. For 13 pairs the metadata disagrees with the annotated points. The groups are now computed from the annotated points, and the disagreement is documented (Section 2.2, Supplementary Section S2).
3. One comparison of the coarse-to-fine search mixed two settings, and the rewrite states which setting each run used (Section 2.7, Supplementary Section S4).

None of these changes the conclusions. On the genuinely tiled pairs, pooled tiling raised the median error from 68 to 538 pixels. The checked coarse-to-fine search still gains nothing on the real pairs and still triples success when field of view is the only difficulty. Every number in the rewrite is now recomputed from the released result files by a script, and a separate check confirms that the text matches it.

This work has not been published and is not under consideration elsewhere. The author declares no competing interests. The code, the per-pair result files and every analysis script are openly available, and the benchmark itself is public. The use of an artificial-intelligence assistant is described in Section 2.14.

Thank you for your comment on the earlier version, and for considering this one.

Sincerely,

**Frank Cai**
Purdue University, West Lafayette, IN 47907, USA
frankyc11223@gmail.com · ORCID 0009-0003-0041-1459
