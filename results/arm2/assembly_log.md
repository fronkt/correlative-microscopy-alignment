# Arm-2 materials pool: assembly log (2026-09-29)

Rules kept: no matcher/registration algorithm was run on any image (only least-squares affine/homography fits to
GT points, and the frozen overlay *detector*); no browser, no proxy, no keyed API; public HTTP only (Zenodo,
Mendeley Data public API via curl, refodat public REST API, DataCite). Data in `C:\Users\frank\Documents\materials-bench\`
(outside any repo); code = new files only (`scripts/arm2_*.py`, `src/cma/materials_pool.py`). Nothing committed.
Python packages installed for HTTP range reads / 7z: `remotezip`, `py7zr` (PyPI; not data hosts, logged for honesty).

Result (REVISED after coordinator follow-up): **14 pairs usable** = A 11 + D(refodat.86) 3; Component C EXCLUDED (see below; the earlier 37 figure is superseded); B BLOCKED; D(refodat.90) NOT ASSEMBLED. 13 clusters (A 11 region units, C 1, D 1). Cluster = region; 4 A regions share the 21218524 specimen and 2 share 14245480, so a dataset-level
cluster (`cluster` prefix = Zenodo id) gives 6 A clusters / 8 in total; both levels are available in the manifest.

## Download sizes (reported; total > 1 GB was reached without a pause, because no interactive channel existed)
Component A about 1.9 GB (range reads of DIC txt + EBSD crc/cpr + notebook + patterns; 14532401 plain files 0.47 GB;
21218524 0.68 GB; others 0.17-0.23 GB each), C 40 MB, D refodat.86 ~0.29 GB (222 MB QGIS 7z + ctf/jpg), refodat.90
0.75 GB (Full-alginment.7z, listed but not used). No zip > 2 GB was downloaded whole. Ånes: only 1-2 KB CSV/txt members and
zip central directories / one npy header were read.

## Component A: DefDAP HR-DIC vs EBSD (all CC BY 4.0) -> 11 pairs, 11 region clusters (6 dataset-level)
Pair definition (analogue of AmalgaMatch SlipPartitioning: EBSD IQ/CI/PRIAS vs SE / exx strain maps): EBSD band contrast
(8-bit, 1-99 pct stretch) vs DIC max-shear strain map (16-bit, clip at p99.5, DIC-grid resolution, cropped exactly as
in the authors' notebook). Max shear computed DefDAP-style from the DaVis displacements (small-strain form, NaN->0).
Homologous points copied from the notebooks; DIC points are in the CROPPED grid coordinates (verified: point extremes
equal the cropped map size, e.g. Ti64 bulk corner points (2157,1672) = cropped grid (2157,1672)), so strain-map pixel =
notebook DIC coordinate with no windowSize factor. The pattern-image variant (needs windowSize) was not built: the strain
map is the realistic cross-modal pair and the pattern image is absent for several records (HEA) -> logged omission.
Step choice (points are step-independent): last/highest available step of each notebook (8383311 SetA_003; 21218524
irr_003/nirr_003/irr_002/nirr_002; Ti64 pore B00002 and bulk B00035 = the maps the notebook plots; CS1 B00012; Ni Integrated
B00011 = notebook Step 11; HEA 2.25 % files, assuming notebook "B00003" is the third step: consistent with cropped
sizes, and the GT does not depend on it). Notebook EBSD pre-rotations replicated (21218524 TD: rotate 180; Ni 13755208:
rot90 clockwise); Ni mask (mask<115 -> 0) applied. Source/target: source = larger physical FOV (EBSD in all 11 pairs).
Checks: thumbnails for all 11 inspected with the Read tool (results/arm2/thumbs/*): points fall on matching grain
junctions / pore / hydride features. HEA CG/FG assignment (notebook HEA2 = CG, HEA3 = FG) fixed by EBSD map extents and DIC crop bounds.
Exclusions/notes: Ti64 bulk: the notebook's 4 corner anchor points (one EBSD x = -20, outside the map) EXCLUDED -> 6 real points
(authors used piecewiseAffine); 8383311 keeps its 23 points (two near-duplicate corner-anchor pairs (0,0)/(5,6) kept as given);
GT models: piecewiseAffine (8383311, Ti64 bulk), affine (Ti64 pore, CS1, Ni, HEA), DefDAP default (21218524). Not opened: 10478594,
Manchester AZ31/Ni shot-peened records, Maj/Sorhaug/Paysan (not on the pre-declared list).
16633511 Case Study 1 only (Case Studies 2, 3 are 8.8 / 14.1 GB zips, not on the list).
Overlay detector (see below): 2 hits, both visually FALSE POSITIVES (Ti64 bulk EBSD "top:55", CS1 strain map "bottom:106"; no banner present).

## Component B: Anes Acta Mater 2023 (Zenodo 7383087, CC BY 4.0) -> BLOCKED (0 pairs)
Archive read by range: 12 ROIs (0s 3, 175c 3, 300c 3, 325c 3), control-point CSVs with 48-113 points each were read and
have valid extents (src <= ~1225, dst <= ~5100). But the two images they refer to are NOT shipped: the CSV name
`..._cropped2_fused_cropped_cropped` denotes ImageJ-stitched BSE tiles followed by two crops, and an EBSD image.
Evidence: (i) 300c.zip contains no BSE images at all (only a link to Zenodo 6470216, not on the list); 0s/175c/325c hold only
raw overlapping 4500x tiles (4-6 per ROI) that need ImageJ stitching; (ii) `bse_labels_filled_filtered.npy` is an int32 particle map of
shape e.g. (3924, 5239) vs CSV dst max (3530, 3497): a larger frame whose crop offsets are unknown; (iii) EBSD `.ang` grid for 300c/1 is
900 x 921 while CSV src x reaches 914 > 900, so the CSV frame is not the shipped grid (crop/transposition/rotation unknown).
Reconstructing the frames would need the authors' (GitHub, GPL-3.0, not an allowed host) scripts or an image-similarity search
(a registration; forbidden here). Mitigation: contact author (H. W. Anes) for the cropped fused images. Also EXCLUDED as planned:
GitHub dataset (a) hakonanes/correlated-grains-particles-workflow (GPL-3.0; also not an allowed host).

## Component C: Li & Shaffer (Mendeley srscfwnrwt v1, data_licence CC BY 4.0, Mendeley copy) -> 23 pairs, 1 cluster
Files: `Correlative analysis example - 10kX/slicebyslice/{OM-NN,SEM-NN}.tif` + `landmarks1-NN.csv` (BigWarp; 4-7 active points).
Python `requests` gets HTTP 403 from Mendeley, plain `curl` works (same public URLs).
Coordinate handling (inferred, documented in scripts/arm2_build_C.py): fixed points = SEM px (1024x718); moving points are ~32x smaller
than the OM tiles, so OM px = x_moving * 32 (fitted moving*32 -> fixed affine gives scale 1.11-1.21, rotation ~7 deg, matching the
visible tilt of the SEM tiles; a +0.5 half-pixel variant left all points ~15 px down-right of blob centres in the overlay check and was dropped).
Source = SEM (fixed), target = OM (moving; already a smooth upsampled concentration map); physical pixel sizes not stated, so
`scale_ratio` comes from the GT affine (1.12-1.2). Caveats: (1) one region, serial slices -> ONE cluster; (2) slices 20 and 21 share an
identical landmark file and identical OM pixels (near-duplicate pair); (3) the landmark-conversion factor is an inference (evidence above),
not a documented fact; (4) GT quality is the weakest of the pool (affine LOO 11-80 px = 1-6 % of the diagonal on 4-7 points): 20 of 23
flagged > 20 px. Only slice-by-slice OM/SEM 10kX example has landmarks (other subsets have none).

## Component D: Weimar refodat (CC BY 4.0 per DataCite)
The landing pages are PoW-gated (302 -> /pow-challenge); not solved or circumvented. The repository's documented public REST API
(`/api/v2/objects/<id>/derivates[/<der>/contents[/<file>]]`, HTTP 200, Range supported, no challenge) lists and serves the files.
* refodat.86 (BSE vs EBSD of 7-day alite): 3 pairs, 1 cluster (one specimen, Sites 3/4/5). GT = QGIS georeferencer GCP files
  (`*.tif.points`, 8/11/10 points) in QGIS_BSE-EBSD-alignment.7z; source = BSE mosaic window cut around the EBSD footprint (footprint
  from GT affine, +5 % margin; the shipped georeferenced BSE crop is 15072 x 19168 px), target = EBSD band-contrast tif that the points
  reference. mapY/sourceY = -row (QGIS); pixel-edge coordinates converted to pixel centres (-0.5). The shipped BSE crop is a ~1.7x
  downsample of the 112.4 nm mosaic (derived 179-190 nm/px). Very hard cross-modal pairs (18x pixel-size ratio removed by cropping: ratio ~10.6-11.2).
  Site 4 reference EBSD image is the "Reanalyzed" band contrast. Transform type used by the authors is not stored in the files.
* refodat.90 (ITZ mortar, LM/CM/BSE/EDX/XRF): NOT ASSEMBLED (not blocked by access: 14 files listed, `Full-alginment.7z` 750 MB
  downloaded and listed). It holds one `.points` file per modality mapping that modality's pixels to a COMMON map frame (crop, fullmap_SiK,
  full_phase_map, polished_surface_ref, sdr_map*, NI1/NI2, BSE `Layer-Tile Set (2) (stitched)`); there are no pairwise homologous
  point pairs. A pair (X, BSE) would have to be synthesised from two fitted georeferences (GT = model, not hand points; LOO would be
  ~0 by construction) and the modality-to-frame transform types live in the QGIS project. Left out of the pool rather than mix
  synthetic-from-model GT with hand GT; the files are in raw/refodat/90 if the team wants a labelled synthetic-GT extension.

## GT quality (results/arm2/gt_quality.csv; residuals in source px, target->source, LOO over hand points)
* A: affine LOO mean 0.9-3 px for the six dense pairs (CS1, Zr4 x4, 8383311), 4.3-10.6 px for Ni / HEA FG / HEA CG / Ti64 pore, **27.1 px for Ti64 bulk
  (flag; authors used piecewiseAffine and only 6 points survive)**. All A pairs have >= 5 points.
* C: 20 of 23 flagged (>20 px), mean LOO 11-80 px, 4-7 points; no pair < 4 points; homography LOO undefined (n-1 < 4) for 5 pairs.
* D86: site 3 30.1 px (flag), site 4 16.7 px, site 5 28.8 px (flag) on 1000-7000-px images (0.3-0.5 % of the diagonal), reflecting real distortion
  and hand-picking on a 2 um EBSD grid vs 180 nm BSE (affine LOO 30 px in BSE px is only ~3 EBSD px).
* Flags: 23 pairs exceed the 20-px affine-LOO gate (C 20, D86 2 [sites 3, 5], A 1 [Ti64 bulk]); 0 pairs have < 4 points. The gate is in absolute px, so
  image sizes are not comparable (D86 30 px = 0.46 % of the diagonal; C 20-80 px = 1.6-6.4 %): use `aff_loo_mean_pct_diag` in the CSV for a size-normalised gate.
  Exact list in manifest_summary.json.

## Overlay detector (src/cma/overlay.py, present; frozen)
Run as detector only (`detect_overlay(read_gray_u8(path))`) on all 74 images: 2 hits, both false positives after inspection (above).
Manifest columns `overlay_src`, `overlay_tgt` carry the raw detector output; the Arm-1 crop rule was NOT applied to the pool.

## Loader / runner
`src/cma/materials_pool.py::MaterialsPoolLoader` reads `materials-bench/manifest.csv`; same interface as AmalgaMatchLoader
(`iter(groups, subclasses)`, `load_pair`, `records`, `__len__`); pair_id `"<scene>#0"`; source = larger FOV; `scale_ratio` = target px / source px
(GT-affine scale where pixel sizes unknown). `scripts/arm2_run_candidates.py` swaps the loader into `run_triage_candidates.py`
(not run: outcomes must stay unseen). Records also carry `cluster` (specimen/region independence unit) and `component`.
Minor: `group` labels used: SlipPartitioning (A), OM-SEM-SerialSection (C), Multiscale (D); they are new (not among the 6 AmalgaMatch groups).


## REVISION: Component C EXCLUDED (0 pairs; records moved to materials-bench/records_excluded_C, evidence thumbs in results/arm2/thumbs_excluded_C)
Coordinator viewed slice 05: OM-side points did not land on the SEM features. Re-derivation from metadata:
* Record's "File format and instructions.pdf": OM images 155 um (372 px); 3DOM.tif is 85x65 px, 2.398 px/um, 0.322 um slices; Avizo .hx: correlated
  OM and SEM stacks share the box 27.8972 x 19.5526 um = SEM 1024x718 at 27.24 nm/px. The 4-7 landmarks have moving coordinates 5-27 (calibrated um, the
  OM tiles carry `unit=um`) and fixed coordinates 130-957 (SEM pixels). The documented um->SEM-px factor is 36.71 (1/0.027244), not the 32 I inferred.
* But the shipped OM-NN.tif (1024x718, tiff resolution tag 1.0, no origin/offset) is a resampled crop whose physical origin/extent relative to the
  BigWarp OM frame is not stored anywhere in the record (no BigWarp project/xml, no TrakEM2 file for this example), so the OM pixel frame cannot be derived
  from metadata. Zoomed crops with the documented 36.71 factor (slice 05, 4 landmarks: none on a feature in the OM) still failed; I did not tune it.
* The strict gate cannot be met by any conversion: affine LOO residual in source px is invariant to the moving-coordinate scale/offset, and it is 11-80 px
  (1-6 % of the diagonal; only 3/23 slices below 20 px, none below the 1 % = 12.5 px gate except possibly slice 06 at 0.88 %) because the landmarks were placed
  for a nonrigid BigWarp warp. => Component C excluded (reason: OM frame not recoverable + GT not affine-consistent at the 1 % gate). Slices 20/21 had an
  identical landmark file and OM pixels; moot after exclusion (would have kept 20 only).
* Earlier statements in the C section above (23 pairs, x32 conversion "evidence") are retracted: the scale ~1.15 in the affine fit was the 36.71/32 ratio.

## D refodat.86 overlay review (sites 3 and 5, flagged >20 px LOO)
Viewed both thumbnails. Site 5: BSE pore-edge points 0, 2, 7, 8 sit at or beside dark pores that have a dark counterpart in the EBSD band-contrast map at the same
relative positions; other points on particle/paste boundaries. Site 3: points on grey particles/pores with matching EBSD blobs (1, 2) but the 2 um EBSD vs 180 nm BSE
makes exact identification only ~1-3 EBSD px reliable. LOO 29-30 px in BSE px = ~3 EBSD px = 0.3-0.5 % of the diagonal: consistent with hand-picking noise, keep.
