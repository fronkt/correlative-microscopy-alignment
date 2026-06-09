# Project Context: Multi-Scale Multimodal Alignment for Correlative Materials Microscopy

## Problem Statement
Correlative materials microscopy spatially associates datasets from disparate
modalities (SEM, EBSD, AFM, TEM) on the same specimen to expose physical
phenomena (dislocations, slip partitioning, multi-phase boundary dynamics).
Registration is hard because the modalities share little mutual information,
use different contrast mechanisms, and exhibit field-of-view (FOV) mismatches
as low as 2%.

## Why Foundational Matchers Fall Short
MatchAnything, RoMa, and ELoFTR were trained on macroscale natural images.
Out-of-domain materials microstructures plus extreme scale gaps make zero-shot
application unreliable, particularly at low FOV ratios.

## Proposed Approach (one sentence)
Wrap a pre-trained dense matcher (RoMa / ELoFTR) in a scale-aware pyramidal
patching layer that crops the wide-FOV image into target-resolution tiles,
runs per-tile dense matching against the narrow-FOV image, and aggregates
correspondences through RANSAC to fit a single global homography / affine.

## Hypothesis (testable)
Pyramidal patching + RANSAC aggregation lifts registration accuracy by >35%
over zero-shot foundational baselines on AmalgaMatch image pairs with FOV
ratios down to 2%, while holding mean keypoint error below 1.5 px.

## Benchmark
AmalgaMatch — 187 image pairs across 6 correlative microscopy challenge
groups and 19 material subclasses.

## Key Variables
| Symbol | Meaning | Bound / Target |
|--------|---------|----------------|
| I_s    | Wide-FOV source image | 1024x1024 px, metals/alloys/ceramics |
| I_t    | Narrow-FOV target image | FOV ratio 2%–20% of I_s |
| H      | Homography (or affine A) | 3x3 (or 2x3) mapping I_t -> I_s |
| P_match| Match precision @ 5 px | > 85% |
| mu_err | Mean Euclidean keypoint error | < 1.5 px |

## Control Groups
- A: Zero-shot ELoFTR / RoMa / MatchAnything on raw pairs.
- B: Classical MMI + SIFT intensity-based registration.
- Exp: Adaptive Pyramidal Flow Matcher (this project).

## Pipeline Stages
1. **Pyramidal Scale Extraction.** Use the supplied physical-scale metadata
   to build a pyramid of I_s whose levels match the nominal physical
   resolution of I_t. Sliding window with 50% overlap.
2. **Feature Extraction + Matching.** Push each tile through RoMa / ELoFTR
   alongside I_t to get a dense correspondence flow.
3. **Consensus Transform Estimation.** RANSAC across pooled correspondences,
   solve

       min_H  sum_i || x_i^(s) - (H x_i^(t)) / (h_3^T x_i^(t)) ||_2^2

4. **Validation + Sensitivity.** Sweep FOV ratio 50% -> 2% on AmalgaMatch;
   locate the breakdown point for each backbone.

## Success Criteria (must all hit)
- P_match > 85% at 5 px threshold on AmalgaMatch test split.
- mu_err  < 1.5 px on AmalgaMatch test split.
- >= 35% relative improvement over the best zero-shot baseline at FOV <= 5%.
- Documented breakdown curve vs. FOV ratio for every backbone.

## Open Risks
- AmalgaMatch licensing / availability of GT correspondences.
- GPU memory blow-up at finest pyramid level with dense matchers (RoMa is
  heavy). Mitigation: tile-batched inference, mixed precision.
- Modality pairs where mutual information is near zero (e.g., AFM topo vs.
  EBSD IPF) may still defeat the matcher; flag as separate failure mode
  rather than averaging into headline metrics.
