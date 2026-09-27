# Cover letter

Dear Editors,

Please consider the enclosed manuscript, **"Image tiling does not solve field-of-view mismatch in correlative microscopy registration,"** as a Regular Article in *Microscopy and Microanalysis*.

Correlative workflows increasingly depend on registering images from instruments that share almost no visual appearance and can differ in field of view by more than an order of magnitude. Pretrained dense image matchers are now the strongest available tool for that step, and the obvious way to extend them to a large field-of-view gap is to tile the wider image, match each tile, pool the correspondences and fit one transform. It needs no training and treats the matcher as a black box. We expected it to work; it was the central hypothesis of the project.

It does not work, and the manuscript's main contribution is the reason. Dense matchers never decline to match. A tile that does not overlap the target at all still returns its full complement of confident correspondences, so pooling floods robust estimation: across the 187 pairs of the public AmalgaMatch benchmark the median inlier fraction falls from 0.114 to 0.005, and median error rises from 80 to 2708 pixels. The argument makes no reference to tile size, overlap or backbone, so it predicts failure for any pool-then-fit tiling scheme built on a matcher of this class. We think this is worth reporting precisely because the approach is so natural — it is the first thing a practitioner will try, and the failure is silent, producing a confident transform rather than an error.

We then report what does and does not repair it. A verified coarse-to-fine wrapper, which admits a candidate transform only when a mutual-information verifier beats the incumbent, removes the collapse entirely but buys nothing in aggregate on this benchmark. On a controlled ladder that crops real pairs to shrinking field of view with appearance, modality and pixel size held fixed, the same wrapper triples success at 10 % field of view. The benchmark simply never presents scale as an isolated failure mode: we measure the appearance axis directly and show that its low-field-of-view pairs are also its most appearance-divergent, and that below an area ratio of 0.5 the benchmark carries too little leverage to rank the two constraints against each other. We state that as a finding about the benchmark, and we withdraw an earlier claim of ours that was not entitled to it.

One further result may interest the journal's methodological readership beyond this particular application. Our pipeline offers an optional thin-plate-spline refinement stage, and refinement coverage varies from 0/187 to 187/187 across the configurations we compare. Scoring the identical experiments after that stage converts both of our null aggregate results into significant ones. A refined table is therefore not one metric but two, interleaved by configuration. We headline the unrefined error because it scores every configuration identically, we report both throughout, and we note that the change removes statistical significance from two of our own results rather than creating it. The recommendation we draw is cheap and general: report post-processing coverage per configuration, and check headline contrasts with the optional stage disabled.

The manuscript is a diagnostic study, and several of its central results are negative. We have tried to report them at the strength the evidence supports and no higher, including in the three places where it cost us a claim we previously made.

The evaluation harness, wrapper implementations, ladder construction, fine-tuning trainer and every analysis script are openly released, and all tables and figures regenerate from the released per-pair result files. The two exceptions are stated explicitly in the Data Availability statement.

This work is original, is not under consideration elsewhere, and has not been published previously. The author declares no conflict of interest and has no financial interest in any of the software evaluated.

Thank you for your consideration.

Sincerely,

**Frank Cai**
Purdue University, West Lafayette, IN 47907, USA
frankyc11223@gmail.com · ORCID 0009-0003-0041-1459
