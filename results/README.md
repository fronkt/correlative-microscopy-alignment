# Results

Summary of synthetic sweeps run so far. Real AmalgaMatch numbers will land
once the dataset is on disk.

## Setup

- Hardware: RTX 5090, torch 2.11+cu128 (vast.ai box).
- Data: synthetic textured pairs (`cma.data.synthesize_pair` /
  `synthesize_cross_modal_pair`), 1024 -> 256, 5 deg rotation, sigma=0.005
  noise, 8 pairs per FOV ratio.
- Methods:
  - `classical_sift`: SIFT directly on (I_s, I_t) + MAGSAC++ (Control B).
  - `pyramid_sift`: SIFT inside the proposed scale-pyramid wrapper.
  - `pyramid_loftr`: kornia LoFTR (pretrained outdoor) inside the same
    pyramid wrapper. This is the foundational backbone we have wired up
    so far; RoMa / ELoFTR / MatchAnything still need vendoring.

## Same-modality sweep (`fov_sweep_gpu.csv`)

mean μ_err (px) — all methods sub-pixel, P@5 = 1.00 everywhere:

| FOV  | classical_sift | pyramid_sift | pyramid_loftr |
|-----:|---------------:|-------------:|--------------:|
| 0.50 | 0.65           | 0.81         | 1.27          |
| 0.25 | 0.36           | 0.52         | 0.73          |
| 0.10 | 0.10           | 0.10         | 0.17          |
| 0.05 | 0.05           | 0.05         | 0.09          |
| 0.02 | 0.16           | 0.16         | 0.24          |

Runtime per pair: classical 0.22-0.24 s (CPU), pyramid_sift 1.06-1.65 s
(CPU), pyramid_loftr 0.47-0.62 s (GPU).

**Takeaway:** on in-distribution textured pairs SIFT already saturates;
neither the pyramid wrapper nor the foundational backbone provides any
accuracy gain, and they cost 2x-7x more wall-time.

## Cross-modal sweeps (the test that decides the hypothesis)

Cross-modal modes applied to the target only (geometry/keypoints
preserved):

- `invert`  : full intensity inversion (e.g. SE vs BSE polarity flip)
- `gamma`   : random gamma in [0.3, 3.0]
- `smooth`  : heavy gaussian blur (sigma 2-4)
- `edge`    : gradient-magnitude image only (EBSD boundary-like)
- `stack`   : two random transforms composed

mean μ_err (px), all FOV averaged (lower is better):

| mode   | classical_sift | pyramid_sift | pyramid_loftr |
|--------|---------------:|-------------:|--------------:|
| (none) | 0.27           | 0.31         | 0.50          |
| gamma  | sub-px*        | sub-px*      | sub-px*       |
| smooth | 0.37           | mostly sub-px (1 outlier) | catastrophic at FOV ≤ 0.05 |
| invert | ~2.6k          | ~1.0k        | ~730          |
| edge   | ~5.6k          | ~2.7k        | ~430          |
| stack  | ~1.3k          | ~1.6k        | ~580          |

`*` see per-mode CSVs for exact numbers — gamma was the only fully
recoverable cross-modal mode.

**Interpretation.**

1. **Mild contrast changes (gamma, smooth) are tractable.** SIFT keypoint
   gradients survive smooth contrast warps; the foundational matcher is
   not needed. This explains why classical SIFT is still competitive on
   many same-microscope, different-setting pairs.

2. **Hard cross-modality (invert, edge, stack) breaks every method we
   have wired so far** — including LoFTR. Pretrained LoFTR was trained on
   natural-image pairs and is not contrast-inversion-invariant. Its dense
   transformer correspondences degrade about as badly as SIFT's
   handcrafted DoG keypoints.

3. **The pyramid wrapper does not rescue a non-cross-modal-trained
   backbone.** This sharpens the research-plan hypothesis H1: the
   pyramid's contribution is *scale handling*; the contribution we need
   for AmalgaMatch's hardest groups is *modality handling*, which comes
   from the backbone, not the pyramid. So a real test of H1 requires a
   backbone that is itself cross-modal-trained — i.e. MatchAnything, or
   a microscopy-fine-tuned LoFTR/RoMa.

## What this means for the project plan

- The pyramid + RANSAC machinery is sound on the matchable subset.
- The headline AmalgaMatch comparison must use **MatchAnything** as the
  primary foundational backbone (and RoMa as a secondary, since it shows
  some cross-modal robustness in the literature). The current LoFTR
  result is the "what naive foundational matching looks like" baseline.

## MatchAnything vendoring + verification (2026-06-04)

Vendored via HF transformers — `zju-community/matchanything_eloftr` loads
in 2.7s on RTX 5090 (Apache-2.0). No upstream repo clone needed.

Wrapper verified correct:
- self-match on 256x256: 9991 dense pairs, near-zero error
- MatchAnything model-card example (US Capitol pair): 714 matches with
  median confidence 0.42 / 95th 0.88

But the synthetic harness (both layered-noise and natural-image source)
is **not a valid benchmark for MatchAnything**. Numbers from this run:

| FOV  | classical_sift | pyramid_loftr | pyramid_matchanything | classical_matchanything |
|-----:|---------------:|--------------:|----------------------:|------------------------:|
| 0.50 | 0.76 px        | 1.34 px       | 32 px                 | 147 px                  |
| 0.25 | 0.39 px        | 0.73 px       | 34 px                 | 135 px                  |
| 0.10 | 0.22 px        | 0.28 px       | 9.4 px                | 107 px                  |
| 0.05 | 0.18 px        | 0.18 px       | 19 px                 | NaN (RANSAC failed)     |
| 0.02 | 1033 px*       | 0.40 px       | 14 px                 | NaN                     |

(*classical_sift's FOV=0.02 entry is dominated by one bad outlier; median is 0.50 px.)

MatchAnything's pretraining priors are tuned for multi-view scene pairs
(3D structure, viewpoint changes, real cross-modality) — not single-image
homographic warps of generic textures. **Don't draw H1 conclusions from
this table.** The real comparison runs the moment AmalgaMatch is on disk.

## AmalgaMatch on disk + loader rewrite (2026-06-09)

The real release (Fordatis DOI 10.24406/fordatis/436) is extracted under
`data/AmalgaMatch/`: 19 subset directories, 187 registration pairs total —
exactly matching the paper's counts. `cma/data/amalgamatch.py` was rewritten
for the real layout (per-subset `eval_indexs/*.npz` pickled dicts carrying
`image_paths` / `image_metadata` / `pair_infos` / `gt_2D_matches`).

Conventions validated against the data itself:

- GT columns are `[x_i, y_i, x_j, y_j]` in `pair_infos` index order
  (confirmed by per-image resolution bounds on all 19 subclasses).
- The loader orients each pair so `source` = larger physical FOV, swapping
  GT columns when needed; the GT-implied affine scale then agrees with the
  pixel-size ratio (median deviation 2.8% over all 187 pairs).

**Important GT-quality finding** (`scripts/check_gt_consistency.py`): the
hand-annotated GT correspondences fit a *global affine* only to a median
residual of **10.3 px** (p90 31 px, max 58 px on Ta-AM-Spalled). Unless the
deformation is genuinely non-affine (serial sectioning, distortion) and a
model captures it, **μ_err < 1.5 px against this GT is not achievable with
affine/homography fits** — the success-gate metric needs rethinking
(per-point matching metrics, or comparing against the GT-fit residual floor
as the oracle).

First real end-to-end run (`scripts/run_real_pair.py`, SE/BSE same-slice
pair, SIFT backbone): pipeline runs clean (15 tiles, 732 correspondences,
5 s) but registration fails (μ_err ≈ 8300 px) — consistent with the
synthetic finding that cross-detector contrast kills SIFT. Control A
zero-shot baselines (Phase 1.3) are now unblocked.

## Backbone wiring state (2026-06-04)

| backbone        | status                          | weights                  |
|-----------------|---------------------------------|--------------------------|
| SIFT            | done                            | n/a (cv2.SIFT_create)    |
| LoFTR (kornia)  | done                            | auto-download outdoor    |
| MatchAnything   | done (HF transformers)          | zju-community/matchanything_eloftr |
| RoMa            | done (Parskatt/romatch PyPI)    | auto-download roma_outdoor (425 MB) + DINOv2 (1.1 GB) |
| ELoFTR (HF)     | shell — equivalent loaded via MatchAnything backbone |                          |
