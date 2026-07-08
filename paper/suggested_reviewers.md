# Suggested reviewers — *Scientific Reports*

The manuscript sits at the intersection of (a) learned dense feature matching /
image registration and (b) correlative materials microscopy. A balanced panel
should cover both. Candidates below are drawn from the work the paper builds on and
positions against. None are collaborators or recent co-authors of the author (sole
independent author, no conflicts).

## Tier 1 — strongest fit

1. **Johan Edstedt** — Linköping University, Sweden.
   First author of RoMa (the primary backbone evaluated). Expert in dense feature
   matching and robust correspondence. Best placed to assess the matcher-side claims
   and the non-abstain mechanism.
   *Field: dense matching / computer vision.*

2. **Xiaowei Zhou** — Zhejiang University, China.
   Senior author of LoFTR, Efficient LoFTR, and MatchAnything — the families the paper
   compares and fine-tunes. Authoritative on detector-free matching and cross-modal
   pretraining.
   *Field: feature matching / cross-modal pretraining.*

3. **Ali Riza Durmaz** — Fraunhofer IWM / University of Freiburg, Germany.
   Lead author of AmalgaMatch and the companion foundation-model evaluation that this
   paper extends. Deepest knowledge of the benchmark, GT construction, and the
   correlative-microscopy use case. (Note: as the dataset creator, may be viewed as a
   close party; suggest including but defer to editor.)
   *Field: ML for materials microscopy.*

## Tier 2 — methodology and robust estimation

4. **Daniel Barath** — ETH Zurich, Switzerland.
   Author of MAGSAC++, the robust estimator at the core of the pipeline. Ideal for the
   robust-estimation and inlier-flooding argument.
   *Field: robust geometric estimation.*

5. **Paul-Edouard Sarlin** — (SuperGlue / LightGlue lineage), industry/academia.
   Expert on learned matching and on matchers that *can* abstain — directly relevant to
   the paper's central distinction between sparse and dense matchers.
   *Field: learned feature matching.*

## Tier 3 — correlative microscopy domain

6. **McLean P. Echlin** — University of California, Santa Barbara, USA.
   Correlative/serial-sectioning microscopy and multimodal materials characterization.
   Brings the practitioner perspective on whether the recommendations are deployable.
   *Field: correlative materials microscopy.*

7. **Marc De Graef** — Carnegie Mellon University, USA.
   EBSD, diffraction imaging, and quantitative microscopy. Strong domain check on the
   modality-coverage and EBSD/TEM claims.
   *Field: quantitative electron microscopy.*

## Recommended selection

A typical 3–4 reviewer request: **Edstedt** + **Zhou** (matching side) and
**Echlin** or **De Graef** (microscopy side), with **Barath** as a methodology
alternate. Suggest **Durmaz** only if the editor is comfortable including the
benchmark's creator.

## Opposed reviewers

None. (Sole independent author; no competitive or conflicting parties to exclude.)
