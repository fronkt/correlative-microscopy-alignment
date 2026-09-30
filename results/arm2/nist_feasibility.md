# NIST AM Bench 2022 serial sectioning as a cross-modal GT benchmark: feasibility

Date 2026-09-29. Records: doi:10.18434/mds2-2767 (IN718), mds2-2765 (IN625). Public HTTPS only
(data.nist.gov RMM API + /od/ds/ links). No registration or matching was run on any cross-modal pair.
Data: `C:\Users\frank\Documents\materials-bench\nist\` (1.1 GB used of the 10 GB budget; RMM JSONs and
READMEs in `meta\`, diagnostics in `fits\`). Code: `scripts/nist_*.py`. Fit table: `results/arm2/nist_gt_fits.json`.

## Verdict

| Route | Verdict |
|---|---|
| (a) Recover the raw->registered transform from files the authors published | **NO-GO** |
| (b) Derive GT by same-modality raw<->registered homography fit, chained through the registered frame | **GO for IN718 BSE1/BSE2/optical pairs and IN625 BSE2-vs-BSE1; NO-GO for EBSD pairs and IN625 optical pairs** (registered-frame relation not recoverable), still inherits the authors' MI errors |
| Hand check of the automatic GT on ~20 pairs | **GO, mandatory** (only independent check of the registered-frame assumptions) |

## (a) Is the published transform recoverable? No.

Evidence (READMEs, RMM component lists of all 8479 + 4646 files, DREAM.3D pipeline JSONs):
- Only warped/registered images are published, not transforms. No coefficient/matrix/deformation file exists
  for the raw->registered step. The only non-image file in the registered folders is
  `MI_Final_Coefficient.txt`, IN718 BSE2 only: 512 lines `section,value`, sections 13-524, values -1.32 to
  -0.20 (median -1.03). It is a per-section final mutual-information cost, not a transform.
- The 3D-reconstruction pipeline JSONs (`3 - Optical_stack`, `4 - Find_Optical_to_Build`, `6/7/8 - Move_*_to_Build`)
  start from the already registered images (`OM_7_Aligned_Cropped_*`) and only stack, crop, resample and apply a
  rigid build-coordinate translation (ICP on the optical surface). They do not contain the raw->registered warp.
- README: optical registration is rigid only ("no non-linear warping"); BSE1/BSE2/EBSD "warped and registered" to
  optical/BSE1 with an automatic MI pipeline (not in the publication). The warp is nonrigid.
- IN625 has no MI file at all.

## (b) Same-modality GT fit (SIFT+RANSAC seed -> dense DIS flow -> homography on a dense grid)

Method (`scripts/nist_fit_homography.py`): fit raw -> registered image of the SAME modality on 4 IN718 sections
(100, 250, 400 plus IN625 200 as a check). Dense flow gives the recovered warp field; a homography is least-squares
fitted to ~a few thousand grid correspondences (90% trimmed). Residuals in registered-frame pixels
(0.854 um BSE IN718, 0.345 um optical, 1 um EBSD, 0.683 um BSE IN625). Sizes 2-4 MP images: 1 px = 0.05% of width.

| section/modality | reg shape | SIFT inl. | NCC after H | homography resid median / p90 / max (px) | quadratic resid median / p90 | median flow after SIFT-H |
|---|---|---|---|---|---|---|
| 718/s100 bse1 | 2022x1980 | 7070 | 0.834 | 1.90 / 3.96 / 8.6 | 0.52 / 1.00 | 1.8 |
| 718/s250 bse1 | 2022x1980 | 6634 | 0.876 | 0.87 / 1.51 / 3.4 | 0.29 / 0.47 | 0.9 |
| 718/s400 bse1 | 2022x1980 | 6115 | 0.818 | 1.18 / 2.43 / 6.5 | 0.52 / 1.13 | 1.2 |
| 718/s100 bse2 | 2022x1980 | 3263 | 0.899 | 0.54 / 1.10 / 2.8 | 0.25 / 0.46 | 0.5 |
| 718/s250 bse2 | 2022x1980 | 1889 | 0.867 | 0.46 / 0.86 / 1.8 | 0.36 / 0.55 | 0.5 |
| 718/s400 bse2 | 2022x1980 | 1954 | 0.843 | 0.56 / 1.22 / 3.1 | 0.24 / 0.51 | 0.6 |
| 718/s100 optical | 5008x4904 | 2063 | 1.000 | 0.35 / 1.75 / 9.6 | 0.36 / 1.75 | 0.5 |
| 718/s250 optical | 5008x4904 | 4489 | 1.000 | 0.07 / 0.13 / 5.3 | 0.07 / 0.13 | 0.4 |
| 718/s400 optical | 5008x4904 | 4761 | 1.000 | 0.06 / 0.12 / 2.0 | 0.06 / 0.12 | 0.4 |
| 718/s100 ebsd (BC image) | 900x900 | 9680 | 0.813 | 0.57 / 1.35 / 4.7 | 0.39 / 0.81 | 0.7 |
| 718/s250 ebsd | 900x900 | 7978 | 0.809 | 0.79 / 1.84 / 7.6 | 0.61 / 1.33 | 0.9 |
| 718/s400 ebsd | 900x900 | 7185 | 0.767 | 0.94 / 2.15 / 10.2 | 0.83 / 1.80 | 1.0 |
| 625/s200 bse1 | 2122x2122 | 6933 | 0.804 | 1.47 / 3.20 / 7.9 | 0.35 / 0.75 | 1.4 |
| 625/s200 bse2 | 2122x2122 | 3768 | 0.916 | 0.63 / 1.26 / 2.6 | 0.29 / 0.54 | 0.7 |
| 625/s200 optical | 5900x5900 | 7600 | 1.000 | 0.07 / 0.26 / 7.6 | 0.07 / 0.26 | 0.1 |

Findings:
- Optical registration is rigid (NCC 0.9997-0.99995, residual at the ~0.07 px noise floor, matching the README).
  Occasional larger max values are flow outliers at edges/scratches; s100 optical p90 1.75 px is such a case.
- BSE1 (and to a lesser degree BSE2, EBSD) warps are genuinely nonrigid but mild: a global homography leaves a
  median 0.9-1.9 px, p90 1.5-4.0 px, max 3-9 px (0.05-0.2% of image width); a quadratic model removes ~70% of it.
  A homography is adequate as GT for pass/fail thresholds of ~10 px or more; below ~5 px the non-homographic part of
  the GT (BSE1 side) is comparable to the threshold, so report GT-noise-inflated thresholds or restrict to >=5 px.
- Quality caveats: (i) NCC of only 0.77-0.92 for BSE/EBSD shows the registered copies were resampled/smoothed, not that
  the fit is bad; (ii) the flow field has its own ~0.3 px noise floor; (iii) the fit only measures the authors' warp,
  and the authors' own MI registration errors (raw modality -> optical) stay in the GT and cannot be quantified from
  the files. The BSE1_reg -> optical relation is only known to the extent that the MI registration was accurate.
- **Chain assumption for cross-modal GT**: GT(left raw -> right raw) = inv(H_right) S H_left, with S the relation between
  the registered frames. For IN718 BSE1_reg and BSE2_reg share the 2022x1980 grid (S = identity), and BSE1_reg covers the
  same physical extent as OM_reg (2022 x 0.854 = 1727 um vs 5008 x 0.345 = 1728 um, size ratio 2.4767 both axes), so S is a
  pure scaling. Visual sanity check (raw optical vs BSE1 warped by the chain, IN718 s100): sample outline, notch and
  corners coincide at display scale (`fits\718_s100_bse1-om_gt_overlay.png`); this does not verify sub-10-px accuracy.
- **Not recoverable**: (1) EBSD_reg (900x900 at 1 um) has no stated offset relative to BSE1_reg; the build-coordinate
  xdmf origins are after crop + section-to-section handling, so the offset is not derivable without the 18 GB `.h5ebsd`/
  `.dream3d` stacks (and even then uncertain). (2) IN625: BSE1_reg is 2122 px x 0.683 um = 1449 um but Optical_Aligned is
  5900 px x 0.345 = 2035 um, i.e. different extents with unknown offset, so BSE-vs-optical GT is not chainable. IN625
  BSE2_reg and BSE1_reg share the 2122x2122 grid, so BSE2-vs-BSE1 is chainable.
- Raw image sizes are much larger than 2-4 MP for optical (IN718 6673x6896 = 46 MP, 92 MB; IN625 5500x5500). Optical must
  be downsampled ~4x before the benchmark; apply the same scaling matrix to the GT.

## (c) Pre-declared sparse design

- Sections: every 25th section (multiples of 25) inside the registered range: IN718 13-524 -> 25, 50, ..., 500 (20
  sections); IN625 96-605 -> 100, 125, ..., 600 (21 sections). Adjacent picks 25 sections apart (>= 20 required; ~53 um of depth at
  ~2.12 um/section; adjacent sections are near-duplicates, so also cluster by specimen in the statistics). If a
  picked section lacks a file (component lists show 512-516 of ~513), use the nearest available section toward the lower
  number, declared in advance here.
- Pairs (left -> right, GT-chainable): IN718 x 3: BSE1-vs-optical, BSE2-vs-BSE1 (2.9x FOV change, 607 um vs 1750 um),
  BSE2-vs-optical (~3.8x FOV change; optical FOV 2300 um). IN625 x 1: BSE2-vs-BSE1. **Total 20x3 + 21 = 81 pairs with
  automatic GT.** Hand-annotation-only extras (not recommended for the primary test): IN625 BSE-vs-optical (2x21) and
  EBSD-vs-BSE (20 + 21 = 41 per pairing choice, 82 for two EBSD pairings): 42-124 more pairs with no GT chain.
- Download (raw + registered needed to build GT; registered can be deleted after the fit):
  IN718 per section raw BSE1 8 + BSE2 6 + optical 92 + registered 16 + 16 + 49 = 187 MB -> 20 x 187 = **3.7 GB**;
  IN625 (BSE only) 8 + 9 + 18 + 18 = 53 MB -> 21 x 53 = **1.1 GB**. Total **~4.9 GB** (already have 3 IN718 + 1 IN625 sections,
  1.1 GB on disk). Within the 10 GB budget; no more downloads until asked.
- GPU: scan estimate 150 pairs = 8-10 GPU-h (0.055-0.067 GPU-h/pair at 2-4 MP) -> **81 pairs ~ 4.5-5.5 GPU-h** for one method
  set; the 46 MP optical downsample and GT fitting are CPU (~1 min per section on this laptop).
- Hand check: 20 pairs x ~3 min = 1 h; sample the 20 stratified across alloy, pair type and depth (>=1 per pair type
  per section tercile); the tool prints the error under the GT immediately. Decision rule to pre-register: if the median
  hand-vs-GT error over checked pairs exceeds the smallest benchmark success threshold / 2, treat GT as too weak for that
  pair type.

## Files

`scripts/nist_list_components.py`, `nist_fetch.py`, `nist_info.py`, `nist_fit_homography.py`, `nist_gt_chain.py`,
`nist_overlay.py`; `tools/handcheck/{handcheck.py,README.md,demo_pairs.json,demo_pairs.csv}`; `tests/test_handcheck.py`
(6 tests pass, GUI path exercised with Agg and simulated click events).

## Build (2026-09-30) -- STATUS: PARTIAL, blocked by NIST file-server outage

Approved: build the 81 pairs (IN718 sections 25..500 step 25 x 3 pair types; IN625 sections 100..600 step 25 x BSE2-vs-BSE1), 8 GB budget.

Done and tested:
- `scripts/nist_build_pool.py` (writes pool pairs via `arm2_common.finalize_pair`, component "NIST", group "SerialSection",
  subclass e.g. `IN718-BSE1-vs-OM`), `scripts/nist_loader_dryrun.py`, `scripts/nist_handcheck_list.py`; `nist_fetch.py` now retries with backoff.
- Rules (also in the script docstring and in every record): optical raw downsampled by a fixed factor 4 (INTER_AREA; BSE native); grey values =
  percentile stretch (5, 99.5) inside the specimen mask, to uint16 (optical texture spans <2% of the 16-bit range, pores clip);
  specimen mask = Otsu on a 600-px thumbnail, largest component, holes filled (BSE2 = whole image, its FOV lies inside the specimen);
  GT points = fixed 5x5 grid over the target mask bbox shrunk by 15% per side, mapped target->source through the composed GT homography
  (points are never moved; counts inside mask / inside source image are stored per record); source = larger FOV image;
  cluster = `NIST:<alloy>:blk<section//100>` (alloy x block of 100 sections; IN718 blocks 0-5 -> up to 6 clusters, IN625 blocks 1-6 -> up to 6).
- Records also carry: section, alloy, modality_pair, block, image sizes, GT-fit residual stats (same-modality dense-homography/quadratic residuals,
  NCC, SIFT inliers) for both modalities, the target->source GT homography in pool-PNG pixels.
- Built and loaded so far (IN718 sections 100, 250, 400 x 3 pairs + IN625 section 200) = 10 pairs; dry-run loader: 10/10 load through
  `MaterialsPoolLoader` (the loader `arm2_run_candidates.py` swaps in; no `materials_pool.py` change was needed), 0 flagged, all 25 GT points inside both images,
  GT->homography refit residual 1e-4 px, scale_ratio (tgt px / src px) IN718 BSE1/OM 0.619, BSE2/BSE1 0.404, BSE2/OM 0.250, IN625 BSE2/BSE1 0.500.
  Frozen overlay detector (run by `arm2_assemble.py`, columns overlay_src/overlay_tgt): 0 hits on all 20 NIST images (none expected: raw images, no banners).
  Visual check of the s100 BSE1-vs-OM thumbnail (`results/arm2/thumbs/NIST_718_s100_bse1-om_0.jpg`): GT grid sits on the specimen in both images;
  optical grain contrast is very low, only pores/edges are visible.
- Handcheck: `python tools\handcheck\handcheck.py --pairs <json>` starts on the Agg backend (verified on the demo list; 6 pytest tests pass). The 20-pair list
  could not be drawn yet because it needs the full 81 pairs.

Blocked: from ~22:00 data.nist.gov file links (/od/ds/...) return HTTP 524 / time out for every file (RMM API still answers 200). Downloaded so far 2.05 GB
(complete sections: IN718 25, 50, 100, 250, 400; IN625 100-225, 275 partial). A background job cannot outlive an agent turn, so resume from the main session:

```powershell
# 1. download (retries indefinitely, resumes at file level) + same-modality fits, in the background
& 'C:\Program Files\Git\bin\sh.exe' C:/Users/frank/Documents/materials-bench/nist/runall.sh
# 2. when nist\ALLDONE exists: build, assemble (manifest + overlay detector), dry-run, hand-check list
cd C:\Users\frank\Documents\cma-triage-ext
python scripts\nist_build_pool.py
python scripts\arm2_assemble.py
python scripts\nist_loader_dryrun.py
python scripts\nist_handcheck_list.py
python tools\handcheck\handcheck.py --pairs tools\handcheck\nist_handcheck_pairs.json --out C:\Users\frank\Documents\materials-bench\nist\handcheck_clicks.csv
```
(Fit outputs land in `results/arm2/nist_gt_fits_a.json` and `_d.json`; `nist_build_pool.py` merges all `nist_gt_fits*.json`.)
Expected final size ~4.9 GB total (within the 8 GB budget); 81 pairs; up to 12 clusters (6 IN718 blocks + 6 IN625 blocks).
