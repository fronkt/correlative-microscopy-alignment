"""Component D1: refodat.86 (Kleiner, Bauhaus-Univ. Weimar; CC BY 4.0) BSE mosaic vs EBSD band contrast.

Source of GT: QGIS georeferencer GCP files ('*.tif.points': mapX,mapY,sourceX,sourceY,...) shipped in
QGIS_BSE-EBSD-alignment.7z, obtained through the repository's public REST API (/api/v2/objects/.../contents),
i.e. without touching the PoW-gated landing page.
  mapX/mapY  = coordinates in 'Layer-Tile Set (stitched)_crop_georef.tif' (pixel-scale 1, origin 0; mapY = -row)
  sourceX/Y  = coordinates in 'Band Contrast N Map Data M.tif' (QGIS: sourceY = -row)
Both are pixel-edge (area) coordinates; we subtract 0.5 to get array-index (pixel-centre) coordinates.
The BSE crop is 15072 x 19168 px; we cut a window = bounding box of the EBSD footprint mapped into the BSE with the
GT affine (fit to GT points only) + 5 % margin, so the pair has manageable size. The window origin is subtracted
from the BSE GT coordinates. No matching algorithm is run.
"""
from __future__ import annotations

import csv
import glob
import sys
from pathlib import Path

import cv2
import numpy as np
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm2_common import BENCH, finalize_pair, sha256, to_u8  # noqa: E402

Q = BENCH / "raw/refodat/86/qgis"
SITES = {3: ("Site 3", "Band Contrast 6 Map Data 8"), 4: ("Site 4", "Band Contrast 9 Reanalyzed Map Data 12"),
         5: ("Site 5", "Band Contrast 10 Map Data 13")}
CTF = {3: "C3S 7d EBSD Specimen 1 Site 3 Map Data 8.ctf", 4: "C3S 7d EBSD Specimen 1 Site 4 Map Data 12.ctf",
       5: "C3S 7d EBSD Specimen 1 Site 5 Map Data 13.ctf"}
BSE_PX_NM_NOMINAL = 112.4  # DataCite description (full-resolution stitched mosaic)


def ctf_step_um(p: Path) -> float:
    for line in open(p, encoding="latin-1"):
        if line.startswith("XStep"):
            return float(line.split()[1])
    raise ValueError(p)


def main() -> None:
    bse_path = Q / "Layer-Tile Set (stitched)_crop_georef.tif"
    bse = tifffile.imread(bse_path)
    H, W = bse.shape
    bse_sha = sha256(bse_path)
    for site, (folder, stem) in SITES.items():
        pts = Q / "EBSD" / folder / f"{stem}.tif.points"
        rows = list(csv.reader(open(pts, encoding="utf-8")))[2:]
        rows = [r for r in rows if r and r[4] == "1"]
        a = np.array([[float(v) for v in r[:4]] for r in rows])
        bse_xy = np.column_stack([a[:, 0] - 0.5, -a[:, 1] - 0.5])
        eb_xy = np.column_stack([a[:, 2] - 0.5, -a[:, 3] - 0.5])
        eb_path = Q / "EBSD" / folder / f"{stem}.tif"
        eb = cv2.imdecode(np.frombuffer(eb_path.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
        eh, ew = eb.shape
        A = np.hstack([eb_xy, np.ones((len(eb_xy), 1))])
        X, *_ = np.linalg.lstsq(A, bse_xy, rcond=None)
        corners = np.array([[0, 0], [ew, 0], [ew, eh], [0, eh]], float)
        cb = np.hstack([corners, np.ones((4, 1))]) @ X
        mx = 0.05 * (cb[:, 0].max() - cb[:, 0].min())
        my = 0.05 * (cb[:, 1].max() - cb[:, 1].min())
        x0 = int(max(0, np.floor(cb[:, 0].min() - mx)))
        x1 = int(min(W, np.ceil(cb[:, 0].max() + mx)))
        y0 = int(max(0, np.floor(cb[:, 1].min() - my)))
        y1 = int(min(H, np.ceil(cb[:, 1].max() + my)))
        crop = np.ascontiguousarray(bse[y0:y1, x0:x1])
        bxy = bse_xy - np.array([x0, y0])
        scale = float(np.sqrt(abs(np.linalg.det(X[:2]))))
        step_um = ctf_step_um(Q / "EBSD" / folder / CTF[site])
        px_bse_nm = step_um * 1e3 / scale
        px_eb_nm = step_um * 1e3
        print(f"site {site}: n={len(a)} EBSD {eb.shape} BSE crop {crop.shape} window x[{x0},{x1}) y[{y0},{y1}) "
              f"affine scale {scale:.2f} -> BSE px {px_bse_nm:.1f} nm (nominal {BSE_PX_NM_NOMINAL})")
        finalize_pair(
            pair_id=f"refodat86_site{site}#0", component="D", cluster="refodat.86:alite-7d-specimen1", group="Multiscale",
            subclass="C3S-alite-BSE-vs-EBSD-BC", img_a=crop, img_b=to_u8(eb.astype(np.float64)), xy_a=bxy, xy_b=eb_xy,
            name_a="C3S_7d BSE stitched (crop_georef window)", name_b=f"{stem}.tif", px_a_nm=px_bse_nm, px_b_nm=px_eb_nm,
            mod_a="SEM-BSE mosaic window (8-bit as shipped)", mod_b="EBSD band contrast (uint16 -> 8-bit, 1-99 pct stretch)",
            licence="CC BY 4.0 (DataCite rightsList)", source_url="https://doi.org/10.71758/refodat.86",
            files=dict(archive_members=f"QGIS_BSE-EBSD-alignment.7z: Layer-Tile Set (stitched)_crop_georef.tif; "
                                       f"EBSD/{folder}/{stem}.tif(.points)",
                       raw_sha256={"bse_crop_georef": bse_sha, f"{stem}.tif": sha256(eb_path), pts.name: sha256(pts)},
                       bse_window_xyxy=[x0, y0, x1, y1]),
            notes="GT = QGIS georeferencer GCPs (transform type not stored in files); half-pixel (-0.5) applied; "
                  "BSE window cut around EBSD footprint; BSE pixel size derived from EBSD step / GT affine scale "
                  f"(nominal full-res mosaic {BSE_PX_NM_NOMINAL} nm)", fov_a=crop.shape[1] * px_bse_nm,
            fov_b=eb.shape[1] * px_eb_nm)


if __name__ == "__main__":
    main()
