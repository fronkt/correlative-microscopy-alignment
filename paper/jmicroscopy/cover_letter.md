Frank Cai
Purdue University, West Lafayette, IN, USA
frankyc11223@gmail.com · ORCID 0009-0003-0041-1459

4 October 2026

The Editors
*Journal of Microscopy*
Royal Microscopical Society

Dear Editors,

I submit the manuscript "Burned-in data bars cause confident false registrations in correlative microscopy, and the retained-inlier fraction flags failures on new materials data" for consideration as an Original Article.

**What the paper shows.** Pretrained image matchers can now register correlative micrographs automatically, but they report a result for every pair whether or not it is correct. The paper makes two contributions to checking such results without ground truth.

First, it identifies a practical failure mode: an instrument data bar burned into the same rows of both images. The matchers align the bar instead of the specimen, return a transform close to the identity, and do so with high confidence. In a pre-registered intervention on 67 such pairs, cropping the bar from both images removed all 110 near-identity locks, raised the number of correctly registered pairs from 6 to 16, and cut confident false accepts from 42 to 14. A sham crop of overlay-free pairs changed results on only 8.8% of runs.

Second, it tests a label-free quality score, the fraction of point matches that survive robust fitting, with a cut-off fixed in advance on 55 pairs from three independent materials data sources. The score separated correct from incorrect registrations (AUROC 0.983). The abstract and limitations state plainly that 41 of these pairs are near-ceiling same-modality BSE pairs, and that optical-to-electron pairs were excluded by a pre-registered ground-truth check. Two further pre-registered hypotheses, on choosing among candidate registrations, were not supported and are reported as null results. One deviation from the pre-registered wording is reported in its own section.

I believe the work suits the *Journal of Microscopy* because its main practical message is aimed at microscopists rather than computer-vision specialists: remove burned-in annotation before automatic registration, and check confident results on any new kind of image pair. The pre-registrations, code and per-run results are public.

**Related submission, disclosed in full.** *Microscopy and Microanalysis*, manuscript MAM-26-277 (under review; editor assigned 2 October 2026). "Image tiling does not solve field-of-view mismatch in correlative microscopy registration." That paper uses the same benchmark and some of the same matchers, but asks a different question: how differences in field of view and image tiling affect registration accuracy. It does not examine label-free quality scores, burned-in overlays or any of the data in Arm 2 of this manuscript, and the text of the two manuscripts does not overlap. The present paper notes that benchmark success rates on overlay-bearing pairs, including those in MAM-26-277, partly reflect the overlay.

This manuscript has not been published and is not under consideration at any other journal.

**Other declarations.** I am the sole author. There is no funding and no conflict of interest. All data are public: AmalgaMatch, the DefDAP HR-DIC/EBSD records and refodat.86 under CC BY 4.0, and NIST AM Bench 2022 under the NIST open-data licence. The use of AI tools is disclosed in the manuscript.

Thank you for considering this manuscript.

Sincerely,
Frank Cai
