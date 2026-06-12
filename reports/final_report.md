# Multi-Scale Alignment of Correlative Materials Microscopy with Foundational Dense Matchers — Final Report

*Project: correlative-microscopy-alignment. Data: AmalgaMatch (Durmaz et
al., DOI 10.24406/fordatis/436), 187 pairs, 19 subsets. All numbers
regenerable from `results/baselines_A.csv` via `scripts/compare_v2.py`,
`scripts/bootstrap_ci.py`, `scripts/h3_family_readout.py`; figures via
`scripts/plot_baselines.py` (written to `reports/figs/baselines/`).*

## 1. Summary

We asked whether a scale-aware pyramidal patching wrapper around
pretrained dense matchers (RoMa, ELoFTR-family, MatchAnything) can lift
cross-modal microscopy registration, especially at severe field-of-view
(FOV) mismatch. The answer, after a full controls-plus-ablations pass
over all 187 AmalgaMatch pairs:

- **H1 (pyramid gives >=35% gain at FOV<=5%): REJECTED.** The naive
  pyramid catastrophically degrades dense matchers; a redesigned
  verified coarse-to-fine wrapper (pyramid v2) recovers a small,
  significant gain (+2 SR@10 points, p=0.017) but nowhere near the
  aspiration, and FOV<=5% remains at zero for every config.
- **H2 (RoMa beats ELoFTR-family at low FOV): SUPPORTED.** RoMa-family
  configs dominate MatchAnything-ELoFTR everywhere (SR@10 0.10-0.13 vs
  0.01-0.02), including every low-FOV stratum.
- **H3 (affine sufficient vs homography): MOSTLY SUPPORTED.** The
  BIC-style selector picks affine on 69% of well-registered pairs;
  homography still wins the remaining 31%, so automatic selection (not
  a hard affine restriction) is the right protocol.
- **The largest single lever was none of the above: swapping in
  cross-modal-trained weights (MatchAnything-RoMa) gave the project's
  only significant headline gain over the zero-shot bar** (SR@10
  0.10 -> 0.13, +0.032, 95% CI [+0.005, +0.064], p=0.018).

The structural conclusion: **on AmalgaMatch, cross-modal appearance —
not scale mismatch — is the binding constraint.** The pyramid solves the
problem these pairs mostly don't have, and cannot touch the one they do.

## 2. Protocol

Per pair: match, robust-fit (MAGSAC++, 5.5 px reprojection threshold,
affine-vs-homography by BIC), thin-plate-spline refinement on inliers,
then mean Euclidean distance (ED) on GT correspondences projected into
source coordinates. Aggregates are success rates SR@{5,10,20} px over
pairs and median ED. Significance: paired bootstrap, B=10000, 95% CIs.
Note the original plan's mu_err < 1.5 px gate was retired early: the GT
itself fits a global affine only to ~10.3 px median residual, so
threshold metrics are floored well above the plan's assumption.
FOV strata use GT-implied area ratio; "severe" = 0.05-0.25 (n=33);
<0.05 has only n=4 pairs under any definition.

## 3. Headline results

| config | med ED (px) | SR@5 | SR@10 | SR@20 |
|---|---:|---:|---:|---:|
| SIFT (Control A) | 908 | 0.01 | 0.02 | 0.02 |
| SIFT + MMI (Control B) | 824 | — | ~0.02 | — |
| LoFTR | 270 | 0.03 | 0.06 | 0.10 |
| MatchAnything-ELoFTR | 510 | 0.01 | 0.01 | 0.02 |
| RoMa zero-shot | 76.3 | 0.05 | 0.10 | 0.23 |
| RoMa + pyramid v1 | 1794 | 0.00 | 0.01 | 0.02 |
| RoMa + pyramid v2 | 74.0 | 0.05 | 0.12 | 0.25 |
| MA-RoMa | 84.0 | 0.04 | 0.13 | 0.24 |
| **MA-RoMa + pyramid v2** | **72.7** | 0.05 | **0.13** | 0.24 |

(Figures: `sr_bars.png` for the full set, `group_heatmap.png` per task
group, `fov_curves.png` for the FOV breakdown.)

Classical methods reproduce the AmalgaMatch paper's finding (SIFT
succeeds on ~3/187); mutual-information refinement (Control B) moves
median ED but flips no pair to success.

## 4. The pyramid story (Aims 2-4)

**v1 failure and root cause.** Tiling + pooled correspondences destroyed
RoMa (med ED 76 -> 1794, 106/187 hard failures). Dense matchers never
abstain: every tile returns ~10k confident matches regardless of
content, so pooling floods MAGSAC++ (inlier fraction 0.114 -> 0.005).
This is a property of the matcher class, not a tuning issue.

**v2 redesign.** Verified coarse-to-fine: direct match first, then
candidate stages (tile search on weak support, zoom refinement), each
candidate accepted only if an MI-on-overlap verifier improves on the
incumbent. RoMa SR@10 0.10 -> 0.12 (+0.021, CI [+0.005, +0.043],
p=0.017), zero pairs lost, and the only nonzero severe-stratum results
of the project (0.00 -> 0.03).

**Knobs (4.1c).** Iterated zoom: not significant vs direct, borderline
worse than single zoom — chaining zooms multiplies verifier error.
Certainty gating (0.5): significantly worse than plain v2 (SR@20
-0.037, CI [-0.070, -0.011]) — discarding low-certainty matches starves
RANSAC exactly where everything is low-certainty. Plain v2 is final.

**Wrapper limits.** On the stronger MA-RoMa backbone the wrapper's
headline contribution vanishes (SR@10 flat) and the no-regression
property breaks (2 gained / 2 lost, all 7-13 px threshold-straddlers).
The verifier is good enough to rescue a weak backbone's coarse
failures, not to discriminate near-threshold transforms.

## 5. The backbone lever

MatchAnything-RoMa's released checkpoint is key-for-key compatible with
the roma_outdoor architecture (603/603 tensors, all retrained), so it
drops into the same wrapper. Its gain (+0.032 SR@10 vs RoMa, p=0.018)
sits entirely in the >=0.5 FOV stratum (0.13 -> 0.18): cross-modal
training fixes appearance on pairs that were nearly matchable, with an
all-or-nothing profile (SR@5 slightly worse, med ED higher when it
misses). Severe-FOV pairs — small, low-texture, modality-divergent —
remain at zero for every configuration tested.

## 6. Limitations and what we would do next

1. **Severe FOV is unsolved.** The honest framing: no zero-shot or
   wrapped config registers FOV<=5% pairs; n=4 also makes the stratum
   nearly unmeasurable. Materials-domain fine-tuning (synthetic
   cross-modal augmentation on SEM/EBSD/TEM data) is the only credible
   path; it was out of scope for this zero-shot study.
2. **Verifier ceiling.** MI-on-overlap cannot rank transforms within
   ~10 px of each other; a learned verifier or GT-free residual proxy
   would let v2's stage machinery (and iterated zoom) pay off.
3. **Single-run sweeps.** GPU nondeterminism is below threshold noise,
   but the 4 threshold-straddling v2 swaps suggest reporting multi-seed
   variance for any future near-threshold claims.
4. **Protocol floor.** The ~10.3 px GT affine residual floors all
   px-threshold metrics; sub-5 px claims on AmalgaMatch should be
   treated as measuring the GT, not the method.
