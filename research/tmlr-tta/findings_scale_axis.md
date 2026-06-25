# Findings — Scale-axis pivot (2026-06-25, box 192.165.134.28)

Status: **PAUSED for re-plan.** Scale-axis TTA is weak/entangled; the two-axis
thesis needs reconsideration. Raw data: `results/scale_pivot_real.csv`
(guarded 15-pair run), `results/scale_pivot.csv` (synthetic, saturated).

## What ran
1. **Synthetic FOV ladder** — *no signal*: all methods SR=1.0 at every FOV
   (RoMa solves same-modality natural warps; ledger insight #1). Wrong testbed.
2. **Real AmalgaMatch FOV ladder, n=6 (one subclass), raw TTA** — unstable:
   tta_scale sometimes helps hugely (41 vs 1521), often collapses (12712 px).
   Single-instance adaptation collapse = the pre-flagged risk, observed.
3. **Real ladder, n=15 diverse, gentler hyperparams + inlier-ratio guard:**

   | rung | direct | pyramid_v2 | tta_scale | tta_guarded |
   |---|---|---|---|---|
   | 0.25 | 0.00/0.14 \| 50 | 0.00/0.36 \| 53 | 0.07/0.21 \| 38 | 0.07/0.36 \| 38 |
   | 0.10 | 0.00/0.00 \|1079 | 0.00/0.00 \| 539 | 0.00/0.00 \| 578 | 0.00/0.00 \| 378 |
   | 0.05 | 0.00/0.00 \|1316 | 0.00/0.00 \|1107 | 0.00/0.00 \| 944 | 0.00/0.00 \|1081 |

## Verdicts
- **The inlier-ratio guard is INVALID.** RANSAC inlier ratio measures internal
  match consistency, not GT correctness; keep/revert decisions were near-random
  (reverted clear wins like 21<49; kept clear losses like 2930>1678). The
  claimed "tta_guarded ≥ pyramid by construction" guarantee is false.
- **Scale-TTA benefit is small and regime-limited.** Real but minor gain only at
  mild crop (rung 0.25: SR@10 0→0.07, median 53→38, genuine per-pair wins). At
  severe crop (0.1/0.05) everything fails on appearance — a SCALE signal cannot
  fix APPEARANCE. The real scale axis is *entangled* with appearance on
  AmalgaMatch, so it is a poor clean testbed for an isolated scale claim.

## Implications / open options
- **Appearance axis is the untested, genuinely-novel half** (label-free TTA vs
  forgetting-prone supervised ft on ANHIR cross-stain / 3MOS optical–SAR). It is
  where the real contribution lives and has not been run.
- One unused lever for scale: an **overlap mutual-information guard**
  (`cma.metrics.mutual_information`, modality-robust) instead of inlier ratio —
  may track correctness where inlier ratio failed. Untried.
- Likely re-plan: **demote scale to a minor ablation; make the paper the
  appearance-axis story.** Decide next session.

## Method status (unchanged, still valid)
`cma.tta` forward is box-validated (smoke OK: loss ↓, 120 norm-affine params,
stateless reset). The machinery works; the *scale signal's value* is the issue.
