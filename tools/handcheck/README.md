# handcheck: hand-validate automatic GT transforms

Shows two images side by side (downsampled to <= 1200 px for display); you click the same
feature in each. Clicks are stored in FULL-RESOLUTION pixels (pixel-centre convention, (0,0) =
centre of the top-left pixel). After the 6th point of a pair the tool prints the error of your
clicks under the automatic GT homography (right-image full-res px), so you see immediately whether
the GT holds. The GT prediction is never drawn on the images (avoids anchoring).

## Setup (Windows PowerShell)

```powershell
cd C:\Users\frank\Documents\cma-triage-ext
python -m pip install numpy matplotlib opencv-python tifffile   # already present on this laptop
```

## Demo (2 downloaded NIST IN718 sections, 6 pairs: sections 100 and 250)

```powershell
python tools\handcheck\handcheck.py --demo
# clicks -> handcheck_demo_clicks.csv (in the current directory); skips -> handcheck_demo_clicks.csv.skips.csv
```

Delete `handcheck_demo_clicks.csv*` to redo the demo from scratch. Loading the raw optical image
(46 MP) takes a few seconds per pair.

## Real run on a pair list

Pair list = CSV with header `pair_id,left_path,right_path,gt_h`; `gt_h` is a JSON list of 9 numbers
(row-major 3x3 H mapping LEFT full-res px to RIGHT full-res px; may be empty = no error report).
`scripts\nist_gt_chain.py --out-csv` writes one:

```powershell
python scripts\nist_gt_chain.py --alloy 718 --sections 100 250 --out-csv pairs.csv
python tools\handcheck\handcheck.py --pairs pairs.csv --out handcheck_clicks.csv
```

Re-run the same command to resume: pairs already saved or skipped are not shown again.
Options: `--n-points 6` (default), `--max-disp 1200`.

## NIST hand-check list (20 pairs, stratified, seed 20260930)

Created by `python scripts\nist_handcheck_list.py` once all 81 NIST pool pairs exist (strata: IN718 BSE1-vs-OM 6,
BSE2-vs-BSE1 5, BSE2-vs-OM 5; IN625 BSE2-vs-BSE1 4; spread over depth terciles). Left = small-FOV target, right =
large-FOV source, both are the pool PNGs, coordinates are pool-PNG pixels. Run:

```powershell
cd C:\Users\frank\Documents\cma-triage-ext
python tools\handcheck\handcheck.py --pairs tools\handcheck\nist_handcheck_pairs.json --out C:\Users\frank\Documents\materials-bench\nist\handcheck_clicks.csv
```

(`--pairs` accepts the `.json` list or a `.csv`; there is also `nist_handcheck_pairs.csv`.) Pick features that are
visible in BOTH modalities (large pores, inclusions, sample edge corners, scratches); the optical images have very low
grain contrast, so expect to skip some pairs (`s`).

## Workflow per pair

1. Click a feature in the LEFT image (cyan cross), then the same feature in the RIGHT image
   (red cross, numbered). Spread the 6 points over the field of view; use sharp, unambiguous
   features (grain-boundary triple points, pores, inclusions), not edges.
2. Keys: `u` undo last click (also un-finishes a completed pair), `s` skip (recorded instantly as
   "no confident shared features"; nothing to type), `n` next pair (only after all points are recorded),
   `q` save and quit.
3. Zoom/pan with the matplotlib toolbar; clicks are ignored while a toolbar tool is active, so
   press the magnifier/arrow button again to turn it off before clicking.

## Output

`--out` CSV: `pair_id,k,x_left,y_left,x_right,y_right,timestamp` (rewritten atomically after each pair).
Skips: `<out>.skips.csv` with `pair_id,reason,timestamp`.

## Tests

```powershell
python -m pytest tests\test_handcheck.py -q
```
