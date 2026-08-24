# The cross-modal appearance axis, measured

**Status:** committed measurement, revision cycle (`sci-rep-revision`).
**Reproduce:** `python scripts/appearance_axis.py --recompute` (~3.5 min, CPU only, no GPU).
**Artifacts:** `scripts/appearance_axis.py`, `results/appearance_nmi.csv` (187 rows),
this file.

## Why this exists

Reviewer 2's central objection is asymmetric evidence: the manuscript isolates
field-of-view (FOV) overlap with a controlled ladder, but infers "cross-modal
appearance divergence" by elimination — whatever the FOV ladder does not
explain gets attributed to appearance. An axis inferred by residual is not an
axis. This report replaces the inference with a measurement, and then reports
what the measurement says, including where it contradicts the manuscript.

## What is measured

For each of the 187 AmalgaMatch pairs, the ground-truth correspondences give a
global affine `source -> target`; the target is warped back into the source
frame; and **normalised mutual information** is computed over the pixels valid
in both frames:

    NMI = MI(a, b) / min(H(a), H(b))

with a 32-bin joint histogram (`cma.metrics.mutual_information`, the same
estimator the classical control pipeline already uses), the overlap crop's long
side capped at 384 px, and the mask re-thresholded after resampling so no
interpolated border pixel enters the histogram. High NMI means the two
modalities encode statistically similar intensity structure over the same
physical region; low NMI means they disagree about what the same material looks
like. Every estimator choice, and why each one is pinned, is in the module
docstring of `scripts/appearance_axis.py`.

**One implementation caveat is deliberate and documented.** The marginal
entropies use min-max scaling before binning while `mutual_information`
internally uses 1–99th percentile clipping, so NMI here is not guaranteed to
lie in [0, 1]. This is the estimator whose numbers were pre-verified, so it is
the one committed. The internally consistent variant (both marginals taken from
the same joint histogram as the MI) is available as
`--variant marginal-entropy` and gives essentially the same answer:
r = +0.210 (p = 0.0040) against r = +0.215 (p = 0.0031), median NMI 0.0434
against 0.0555. The conclusions below do not depend on the choice.

**The estimator is fragile in one specific way.** If non-finite pixels reach
`np.histogram2d` they fall outside every bin, silently shrink the effective
sample, and can flip the sign of the FOV/NMI correlation reported below. The
script therefore removes non-finite pixels from the overlap mask before any
histogram is formed, again after resampling (`INTER_AREA` propagates NaN), and
**aborts rather than writing a partially-NaN CSV** if fewer than 187/187 pairs
resolve to a finite NMI. `cv2.setRNGSeed(0)` pins the RANSAC draw so repeat runs
are byte-identical — verified by running `--recompute` twice from a clean
process and diffing `results/appearance_nmi.csv` (identical, 187/187 pairs,
~210 s per run).

### Distribution and face validity

n = 187 pairs over 35 scenes. NMI: min 0.0014, q25 0.0112, **median 0.0555**,
q75 0.1448, max 0.5706, mean 0.0896.

The ordering by task family is the one a microscopist would predict, which is
the main sanity check available:

| task group | n | median NMI |
|---|---|---|
| SlipPartitioning (SEM-DIC vs EBSD) | 31 | 0.0020 |
| Multiscale | 13 | 0.0148 |
| SameSlice | 26 | 0.0237 |
| DislocationCharacterization (TEM) | 69 | 0.0631 |
| FractureSurfaces | 6 | 0.0861 |
| SerialSectioning | 42 | 0.1607 |

SEM-DIC against EBSD — strain field against crystal orientation, two genuinely
different physical observables — sits at the floor. Adjacent-slice
BSE/EBSD serial sectioning sits at the top. The same material can appear at both
ends depending on which modalities are paired (`CoNi-AM67`: 0.008 in its
Multiscale OM–SEM subset, 0.205 in its SEM–EBSD serial-sectioning subset), which
is evidence the metric tracks the modality pairing rather than the specimen.

## 1. The confound, as a number

The manuscript asserts that low FOV overlap and appearance divergence are
entangled on this benchmark. They are:

- **Pearson r[log10(GT area ratio), NMI] = +0.215, p = 0.0031, n = 187.**
- Spearman rho = +0.367, p < 0.0001.
- Median NMI at area ratio < 0.25: **0.0022** (n = 37).
  At area ratio >= 0.50: **0.0725** (n = 126).
  Mann–Whitney U = 218.0, **p < 0.0001** (two-sided).

Narrow-FOV pairs on this benchmark are also the pairs whose modalities look
least alike, by a factor of ~33 in median NMI. **FOV and appearance are not
separable observationally on AmalgaMatch.** Any claim that one matters more than
the other, made by comparing subsets of these 187 pairs, is confounded.

Two controls:

- **It is not a sample-size artefact.** MI estimators are biased at small
  sample sizes and narrow-FOV pairs have smaller overlaps, so this had to be
  checked. r[n_overlap_px, NMI] = −0.018 (p = 0.81) — overlap size does not
  predict NMI — while r[n_overlap_px, log10 area ratio] = +0.319. The partial
  correlation controlling for overlap size is **+0.233 (p = 0.0013)**, slightly
  *stronger* than the raw one. 34/187 pairs fall below the 384 px cap.
- **It survives scene aggregation, weakly.** Collapsing to one median NMI and
  one median area ratio per scene (n = 35, no pair counted twice): Mann–Whitney
  on the same two bins gives U = 11.0, **p = 0.0037**, medians 0.0211 vs 0.1107;
  the Pearson correlation drops to non-significance, r = +0.227, **p = 0.19**.
  With 35 scenes there is not enough independent data to establish the linear
  trend; the bin-level separation is what holds up.

### The trend is monotone only in the rank sense

Using the FOV bins from `scripts/plot_baselines.py:91`:

| area-ratio bin | n | median NMI |
|---|---|---|
| [0.00, 0.05) | 4 | 0.0196 |
| [0.05, 0.25) | 33 | 0.0021 |
| [0.25, 0.50) | 24 | 0.1648 |
| [0.50, 10.0) | 126 | 0.0725 |

This is **not** monotone. The n = 4 extreme-multiscale bin is not the
lowest-NMI bin, and the 0.25–0.50 bin is the highest. The correlation is carried
by "everything below 0.25 is low-NMI", not by a smooth gradient, which is why
Spearman (+0.367) is markedly larger than Pearson (+0.215). The linear
correlation should not be reported without this caveat.

## 2. The 2x2

Median split on both axes: log10(area ratio) at −0.0919 (area ratio 0.809), NMI
at 0.0555. High-FOV n = 94 / low-FOV n = 93; high-NMI n = 94 / low-NMI n = 93.
SR@10px uses the paper's convention (TPS-refined error, falling back to raw
`mu_ed` when the TPS field is blank). Only `roma/direct` and `ma_roma/direct`
are tabulated: `ma_roma_ft` was fine-tuned on 131 of these 187 pairs
(`results/split.json`) and cannot appear in any aggregate over all 187 without
being restricted to held-out pairs, which would leave too few pairs per cell to
mean anything.

**roma/direct — SR@10 overall 0.0963 (18/187)**

| | low NMI | high NMI | row |
|---|---|---|---|
| **high FOV** | 6/43 = 0.140 | 7/51 = 0.137 | 13/94 = 0.138 |
| **low FOV** | 4/50 = 0.080 | 1/43 = 0.023 | 5/93 = 0.054 |
| **col** | 10/93 = 0.108 | 8/94 = 0.085 | |

**ma_roma/direct — SR@10 overall 0.1283 (24/187)**

| | low NMI | high NMI | row |
|---|---|---|---|
| **high FOV** | 6/43 = 0.140 | 11/51 = 0.216 | 17/94 = 0.181 |
| **low FOV** | 5/50 = 0.100 | 2/43 = 0.047 | 7/93 = 0.075 |
| **col** | 11/93 = 0.118 | 13/94 = 0.138 | |

### Significance testing, clustered on scene

Pairs within a scene share a source image and are not independent, so the
resampling unit is the **scene** (`pair_id` with the trailing `#k` stripped;
35 scenes for 187 pairs), not the pair. B = 10,000, percentile 95% CI on the
difference, two-sided p, seed 0 — the same bootstrap protocol used elsewhere in
the paper, with the cluster substituted for the pair.

| effect | roma/direct | ma_roma/direct |
|---|---|---|
| FOV, marginal (high − low) | **+0.085** [−0.047, +0.396] p = 0.208 | **+0.106** [−0.051, +0.502] p = 0.190 |
| appearance, marginal (high − low) | **−0.022** [−0.195, +0.110] p = 0.726 | **+0.020** [−0.156, +0.199] p = 0.788 |
| FOV effect, within low NMI | +0.060 [−0.235, +0.466] p = 0.628 | +0.040 [−0.275, +0.448] p = 0.776 |
| FOV effect, within high NMI | +0.114 [+0.000, +0.579] p = 0.052 | +0.169 [+0.002, +0.762] p = 0.049 |
| appearance effect, within low FOV | −0.057 [−0.312, +0.051] p = 0.412 | −0.054 [−0.335, +0.075] p = 0.440 |
| appearance effect, within high FOV | −0.002 [−0.343, +0.391] p = 0.902 | +0.076 [−0.262, +0.594] p = 0.527 |

## 3. What this shows, and what it does not

**The FOV effect is directionally consistent; the appearance effect is not.**
All four FOV contrasts (two backbones x two NMI strata) are positive:
+0.060, +0.114, +0.040, +0.169. The four appearance contrasts are −0.057,
−0.002, −0.054, +0.076 — near zero and sign-inconsistent for RoMa, and
sign-flipping across FOV strata for MA-RoMa. The RoMa marginal appearance effect
is *negative*: pairs whose modalities look more alike register slightly fewer
successes, which is not a real effect but is certainly not support for the
opposite one.

**This does not support the manuscript's title claim that appearance divergence
matters more than FOV.** On this operationalisation, on this benchmark, the
opposite ordering is the one the data weakly favours: FOV moves SR@10 in a
consistent direction across every cell, and NMI does not. The title claim should
be withdrawn or restated as a hypothesis, not defended with this table.

**But no effect here is statistically established.** Once clustered on scene,
*nothing* reaches p < 0.05 except two borderline within-stratum FOV contrasts
(p = 0.052, p = 0.049), and those CIs run to +0.58 and +0.76 — they are not
usable estimates. The marginal FOV effect itself is p = 0.21 / p = 0.19. The
reason is visible in the data: for roma/direct, successes live in only 10 of 35
scenes and the top three scenes hold 8 of the 18; for ma_roma/direct, 15 of 35
scenes and 8 of 24. Resampling scenes resamples nearly the entire signal.
**AmalgaMatch has 35 independent scenes, and at a ~10% success rate that is not
enough to resolve a 5–10 point SR@10 difference between observational subsets.**
This is a statement about the benchmark's power, not about the effects being
absent — and it is the honest reason the manuscript's FOV ladder (a within-pair
controlled manipulation, where each pair is its own control) carries the FOV
argument and this table cannot.

**This is one operationalisation of "appearance", not the concept.** NMI over a
global-affine-aligned overlap captures *statistical dependence between intensity
distributions*. It does not capture:

- **Structural/topological divergence** — whether the same features are
  *present*. Two images can have high pixel-level MI while the grain boundaries
  a matcher keys on exist in only one of them.
- **Anything a global affine cannot align.** Non-rigid distortion, out-of-plane
  tilt, and serial-section material change all get charged to appearance here.
- **Spatial arrangement.** A joint histogram is permutation-invariant over
  pixels; scrambling one image identically in both frames leaves NMI unchanged
  while destroying every match.
- **Scale-dependent texture.** The 384 px cap is a deliberate decision (without
  it NMI would partly measure image size), but it also means fine-scale texture
  divergence — plausibly what actually breaks a dense matcher — is downsampled
  away before measurement.
- **What the backbone's features see.** NMI is a raw-pixel statistic. A
  foundation-model matcher operates on learned features whose invariances may
  make a low-NMI pair easy or a high-NMI pair hard.

A negative result for NMI is therefore **not** a negative result for "appearance
divergence". It is a negative result for *this* measure, which is the strongest
claim the data supports and the one the revision should make.

## 4. What to do with this in the revision

1. Report the confound (r = +0.215, p = 0.0031; median NMI 0.0022 vs 0.0725,
   p < 0.0001) as the *reason* the two axes cannot be separated observationally
   on AmalgaMatch. This directly answers R2 with a number where the manuscript
   currently has an assertion.
2. Retire the "appearance matters more than FOV" framing. The measured
   appearance axis does not support it.
3. Keep the FOV ladder as the controlled result and state explicitly that the
   observational 2x2 is underpowered at 35 scenes — do not present the 2x2 as
   corroboration of the ladder.
4. If a genuinely controlled appearance experiment is wanted, it has to
   *manipulate* appearance at fixed geometry (the existing
   `results/fov_sweep_{invert,gamma,edge,smooth}.csv` sweeps are the closest
   thing already in the repo) rather than stratify on a measured covariate.
