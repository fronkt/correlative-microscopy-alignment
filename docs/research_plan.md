# Research Plan

## 1. Goal
Demonstrate that a scale-aware, patch-pyramid wrapper around pre-trained
dense matchers (RoMa, ELoFTR, MatchAnything) registers correlative
materials microscopy image pairs significantly better than zero-shot
application, especially at low FOV ratios (<= 5%).

## 2. Specific Aims
- **Aim 1.** Reproduce zero-shot baselines (Control A) and classical
  MMI+SIFT (Control B) on AmalgaMatch, recording P_match @ 5 px,
  mu_err, success rate, and runtime.
- **Aim 2.** Implement the Adaptive Pyramidal Flow Matcher and integrate
  RoMa / ELoFTR / MatchAnything as interchangeable backbones.
- **Aim 3.** Quantify gains at multiple FOV ratios (50%, 25%, 10%, 5%, 2%)
  and isolate the failure FOV per backbone.
- **Aim 4.** Sensitivity / ablation study across pyramid depth, tile
  overlap, RANSAC threshold, and transform family (homography vs. affine).

## 3. Hypotheses
- **H1.** Pyramidal patching gives >= 35% relative reduction in mu_err
  at FOV <= 5% versus the best zero-shot baseline.
- **H2.** RoMa (dense, coarse-to-fine) beats ELoFTR at very low FOV
  because its dense flow tolerates partial overlap better.
- **H3.** Affine (6-DoF) is sufficient for the AmalgaMatch test set;
  full homography offers no statistically significant gain (paired
  bootstrap on per-pair mu_err).

## 4. Data
- **Primary:** AmalgaMatch (187 pairs, 6 groups, 19 material subclasses).
- **Splits:** Stratified 70/15/15 train (for hyperparameter selection only;
  no backbone fine-tuning in v1) / val / test, stratified by group +
  material subclass.
- **Metadata required:** per-image physical pixel size (nm/px); absent
  values are estimated from EXIF or per-modality priors and flagged.

## 5. Methodology

### 5.1 Pyramidal Scale Extraction
- Compute scale ratio r = pix_size(I_t) / pix_size(I_s).
- Pyramid level L_k: downsample I_s by factor 2^k until tile_size matches
  I_t. Choose tile size = I_t spatial dims (square-padded).
- Sliding window with 50% overlap; record (tile_id, top-left coord,
  scale level) for back-projection.

### 5.2 Backbone Matching
- For each tile T_k,j, run matcher(T_k,j, I_t) to get dense flow F.
- Convert F to keypoint pairs, then back-project (tile -> I_s) coordinates.
- Pool correspondences across all tiles, attach confidence + scale tags.

### 5.3 Consensus Transform Estimation
- Robust estimator: MAGSAC++ (RANSAC variant), inlier threshold tuned on
  val split.
- Fit both 6-DoF affine and 8-DoF homography; pick by AIC-like criterion
  on inlier residuals.

### 5.4 Evaluation Metrics
- **P_match @ k px** for k in {1, 3, 5, 10}.
- **mu_err** (mean) and **med_err** (median) Euclidean error on GT
  keypoints.
- **Success rate**: fraction of pairs with mu_err < 5 px.
- **Runtime per pair**, peak GPU memory.

### 5.5 Statistical Protocol
- Paired bootstrap (n=10000) on per-pair errors for method-vs-method
  significance.
- Report 95% CIs alongside point estimates.
- Per-group breakdown to expose modality-specific failures.

## 6. Risk Mitigation
| Risk | Mitigation |
|------|------------|
| AmalgaMatch unavailable / partial GT | Negotiate access early; fall back to MatchAnything's released benchmark for partial validation |
| RoMa OOM at finest pyramid level | Batched tile inference, fp16, gradient checkpointing off (no training) |
| Modality pairs with ~0 mutual info | Report separately; do not contaminate aggregate metrics |
| Scale metadata missing | Document the estimator; sensitivity-test +/-20% scale error |

## 7. Deliverables
- Reproducible codebase (Python, PyTorch) with single-command eval.
- Numerical results table (Control A, Control B, Experimental) per group.
- FOV breakdown plots per backbone.
- Short technical report (8-12 pages) with figures.

## 8. Timeline (rough, 10 weeks)
- W1: Repo scaffold, env, AmalgaMatch access.
- W2: Baselines (Control A + B) reproduced.
- W3-4: Pyramid extractor + matcher wrapper.
- W5: RANSAC aggregation + metric harness.
- W6: Full eval, first numbers.
- W7-8: Ablations + sensitivity sweep.
- W9: Writeup.
- W10: Buffer / failure-mode deep dive.
