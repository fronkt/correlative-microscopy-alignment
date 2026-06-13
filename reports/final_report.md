# Multi-Scale Alignment of Correlative Materials Microscopy with Foundational Dense Matchers — Final Report

*Project: correlative-microscopy-alignment. Data: AmalgaMatch (Durmaz et
al., DOI 10.24406/fordatis/436), 187 pairs, 19 subsets. Zero-shot/wrapper
numbers regenerable from `results/baselines_A.csv` via
`scripts/compare_v2.py`, `scripts/bootstrap_ci.py`,
`scripts/h3_family_readout.py`; the FOV ladder (incl. fine-tuned backbone)
from `results/fov_ladder.csv` via `scripts/plot_fov_ladder.py` and
`scripts/fov_ladder_bootstrap.py`; the fine-tuning experiment (section 7)
via `scripts/finetune_ma_roma.py` + `scripts/ft_test_analysis.py`
(split `results/split.json`, log `results/finetune_log.csv`). Figures in
`reports/figs/baselines/`.*

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
- **The largest single lever among off-the-shelf options was none of the
  above: swapping in cross-modal-trained weights (MatchAnything-RoMa)
  gave the project's only significant zero-shot-bar headline gain** (SR@10
  0.10 -> 0.13, +0.032, 95% CI [+0.005, +0.064], p=0.018).
- **Materials-domain fine-tuning confirms appearance is the lever, but
  131 pairs trade coverage for depth (section 7).** Decoder-only
  fine-tuning of MA-RoMa cut median error on in-distribution TEM pairs
  5.2x (321 -> 62 px, the largest movement any intervention produced) yet
  *regressed* SR@20 overall (0.393 -> 0.250, CI [-0.286, -0.036]) by
  catastrophically forgetting an unseen modality combo. The direction is
  right; the data budget is not.

The structural conclusion: **on AmalgaMatch, cross-modal appearance —
not scale mismatch — is the binding constraint.** The pyramid solves the
problem these pairs mostly don't have, and cannot touch the one they do.
The FOV-ladder experiment (section 6) proves both halves of that
sentence: with appearance held fixed and FOV swept on real pairs, the
same wrapper triples success at 10% FOV (0.07 -> 0.23, p=0.0014) — the
mechanism is sound; the real distribution just never isolates scale.

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

## 6. The FOV ladder: decoupling scale from appearance (Aim 3)

The real dataset cannot answer Aim 3 ("failure FOV per backbone"):
low-FOV pairs are simultaneously the most appearance-divergent, and only
4 pairs sit below ratio 0.05. We therefore cropped the target of every
base-matchable pair (direct mu_ed < 20 px; 36-40 pairs/backbone) to
absolute area ratios {0.5, 0.25, 0.1, 0.05, 0.02} — appearance, modality
gap and pixel sizes fixed, FOV swept. All GT is kept for evaluation
(out-of-crop points test the global transform's extrapolation);
transform-based mu_ed only. (`results/fov_ladder.csv`,
`reports/figs/baselines/fov_ladder.png`.)

Findings:

- **Direct failure FOV sits between 0.25 and 0.1** for both RoMa and
  MA-RoMa: SR@10 holds near base levels through 0.5, bends at 0.25,
  collapses at 0.1. Hard floor at 0.02 for every config.
- **With scale isolated, the pyramid delivers exactly what it was
  designed for: at FOV 0.1, MA-RoMa + pyramid v2 holds SR@10 0.23 vs
  0.07 direct** — +0.150, 95% CI [+0.050, +0.275], p=0.0014, a >3x
  relative gain in precisely the regime H1 targeted.
- This resolves the project's central tension: **the wrapper mechanism
  is sound for scale; the real distribution simply never isolates scale
  as the failure mode.** H1-as-stated remains rejected on real pairs,
  but the usable-FOV envelope extends from ~0.25 (direct) to ~0.1
  (wrapped, strong backbone).

## 7. Materials-domain fine-tuning: the appearance lever, tested

Section 6 left appearance as the binding constraint and named
materials-domain fine-tuning as the credible path. We tested it directly.
A scene-level 131/28/28 split (within-scene pairs share images, so a
pair-level split would leak; `results/split.json`) gives a held-out test
set; **headline numbers are the 28 test pairs only, val drives selection,
test is touched once.** Fine-tuning is decoder-only — the VGG+DINOv2
encoder is frozen in eval mode (its BatchNorm stats must not drift),
~100M decoder params trained on densified sparse GT (sparse correspondences
splined into a dense A->B warp, supervised inside the GT-support region),
with FOV-crop + photometric augmentation. AdamW 2e-5, 1500 steps on a
5090 (~90 min incl. validation); selection by median val ED picked step
900 (val med 17.55 vs 21.69 zero-shot). (`scripts/finetune_ma_roma.py`,
`src/cma/train/`; checkpoint local-only, 445 MB.)

**Test result (direct, TPS-refined ED, paired bootstrap B=10k):**

| method | SR@5 | SR@10 | SR@20 | med ED |
|---|---:|---:|---:|---:|
| MA-RoMa zero-shot | 0.000 | 0.214 | 0.393 | 83.5 |
| MA-RoMa fine-tuned | 0.000 | 0.250 | 0.250 | 51.6 |

SR@10 +0.036 (n.s.); **SR@20 -0.143, CI [-0.286, -0.036] — significantly
worse**; med ED -31.8 px (n.s.). The aggregate hides two opposite effects
that the per-subclass cut separates:

- **In-distribution, fine-tuning works, hard.** The 16 TEM dislocation
  test pairs (50 TEM pairs in train) improved median ED 320.8 -> 61.8 px,
  a 5.2x cut and the single largest metric movement in the project — but
  mostly from ~320 to ~60 px, still above the 20 px bar, so SR barely
  registers it. EBSD subclasses held.
- **Off-distribution, it forgets catastrophically.** C103 SEM-SE<->LOM-
  height has zero train pairs (the subclass exists only in val/test);
  zero-shot MA-RoMa solved all 4 test pairs (7-16 px), the fine-tuned
  decoder broke 3 outright (109/375/953 px). These four pairs *are* the
  entire SR@20 regression. With 2 C103 pairs in val, a median selection
  metric could not see the damage.

**FOV ladder on the fine-tuned backbone** (same 63-pair testbed, via
`run_fov_ladder.py --restrict-pairs-csv`; its direct rows would otherwise
expand the eligible set to 120) sharpens the picture:

- Fine-tuning **helps the moderate-FOV regime**: at rung 0.25 the
  fine-tuned model holds SR@10 0.37-0.39 (median 13-14 px) vs MA-RoMa's
  0.28-0.30 (15-22 px) — best there.
- The **pyramid's scale lift is backbone-agnostic but smaller** on the
  fine-tuned model: at the severe 0.1 rung the wrapper adds SR@10 +0.078
  (CI [+0.020, +0.157], p=0.0154, n=51) — significant, the mechanism is
  real on a second backbone — but roughly half the +0.150 (p=0.0014) it
  gives plain MA-RoMa. Fine-tuning and the wrapper address overlapping
  scale/appearance failures, so they don't fully stack.
- Below 0.05, every config still collapses to zero.

**Verdict.** Appearance is attackable with domain data — but 131 pairs
over 12 subclasses buy depth on trained modalities at the cost of
forgetting untrained ones. A deployable fine-tune needs forgetting
mitigation (replay of zero-shot/natural-image batches, LoRA-style
low-rank updates, or per-modality experts) and, above all, broader
modality coverage. Fine-tuning and the pyramid wrapper are complementary
but partly redundant: the former extends the matchable-appearance
envelope, the latter the scale envelope.

## 8. Limitations and what we would do next

1. **Severe FOV on real pairs remains unsolved, and appearance is why.**
   The wrapper that triples success at controlled 10% FOV moves real
   severe-FOV pairs barely at all, because those pairs fail on appearance
   first. Section 7 shows materials-domain fine-tuning *can* attack
   appearance (5.2x on in-distribution TEM) but, at this data budget,
   forgets untrained modality combos. The next concrete step is a
   forgetting-robust fine-tune (replay / LoRA / per-modality experts) on
   broader modality coverage, evaluated on the same held-out test split
   and FOV ladder used here.
2. **Verifier ceiling.** MI-on-overlap cannot rank transforms within
   ~10 px of each other; a learned verifier or GT-free residual proxy
   would let v2's stage machinery (and iterated zoom) pay off.
3. **Single-run sweeps.** GPU nondeterminism is below threshold noise,
   but the 4 threshold-straddling v2 swaps suggest reporting multi-seed
   variance for any future near-threshold claims.
4. **Protocol floor.** The ~10.3 px GT affine residual floors all
   px-threshold metrics; sub-5 px claims on AmalgaMatch should be
   treated as measuring the GT, not the method.
