# Stage 2 — Implementation Plan (TTA module + experiments)

Branch `tmlr-tta`. Build on existing `cma/` (RoMa decoder, `forward_decoder_only`,
`_l2sp_penalty`). Local torch is CPU-only (2.11+cpu) → unit-test logic locally, run
experiments on a vast.ai RTX 5090.

## Decoder contract (verified from romatch source)
`model.decoder(f_q, f_s)` → `corresps[scale]` for scale ∈ {16,8,4,2,1}, each a dict with
`flow` (B,2,H,W normalized [-1,1] grid coords) and `certainty` (B,1,H,W logits). Extra
keys (`gm_cls`, `delta_flow`) appear only in `model.training`. Internal cross-scale
detach (`self.detach`) — gradients still reach each scale's norm-affine params.

## Module: `src/cma/tta/`
- `losses.py` — label-free objectives (CPU-unit-tested on synthetic tensors):
  - `multi_scale_consistency_loss(corresps,…)` — **SCALE signal**: Charbonnier disagreement
    between each scale's `flow` (upsampled) and the finest `flow`, certainty-weighted.
  - `cycle_consistency_loss(flow_ab, flow_ba,…)` — **APPEARANCE signal**: forward-backward
    composition error via `grid_sample`, occlusion-masked, certainty-weighted (UnFlow-style).
  - `coral_loss(feat, ref_mean, ref_cov)` — **APPEARANCE signal**: CORAL distance of decoder
    feature stats to a precomputed source reference (mean/cov stored once).
- `adapt.py` — `collect_norm_affine_params(module)` (CPU-unit-tested) + `tta_adapt(pair,…)`:
  build im_A/im_B batch (no GT) via the dataset transform, forward decoder, compute the
  axis-weighted loss + L2-SP anchor (`_l2sp_penalty`), step AdamW on norm-affine params for
  K steps, return the adapted model for the standard `RoMaMatcher` path. *(forward wiring
  validated on the box.)*

## Axis routing
Pick the objective by where the target lands on the severity metrics: high FOV-shift →
scale signal; high Fréchet-DINOv2 appearance distance → cycle+CORAL. The 2-D surface tests
whether each signal lights up its own axis (SQ-X).

## Experiment matrix (5090)
3 domains (AmalgaMatch / ANHIR / 3MOS-subsampled) × backbones {RoMa, ELoFTR, MA-RoMa(ceiling)}
× ladder {vanilla, pyramid-only [Pivot-S], DMP, TENT/BN, supervised-ft±L2SP [Pivot-A], TTA
(scale/appear/both), AdaBN} × multi-seed. Metrics: success_rate, med_err, P_match@k, mu_err
+ source-retention. Paired bootstrap CIs.

## Gates
1. CPU unit tests green (this increment). 2. Box smoke: one pair, loss decreases, match()
runs on adapted model. 3. Pivot-S check (TTA > pyramid-only) before full matrix. 4. Multi-seed.

## Done-when
2-D severity surface + baseline ladder reproduced multi-seed; hands to academic-paper (write).
