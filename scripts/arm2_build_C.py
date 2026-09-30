"""Component C: Li & Shaffer (Imperial) OM vs SEM slice pairs, Mendeley Data srscfwnrwt v1 (CC BY 4.0).

Files: raw/mendeley/slicebyslice/{OM-NN.tif, SEM-NN.tif, landmarks1-NN.csv}. Landmarks are BigWarp CSVs
(name, active, x_moving, y_moving, x_fixed, y_fixed). Fixed = SEM (1024x718 px, pixel coordinates).
Moving coordinates are ~32x smaller than the 1024-px OM tiles (max ~27 vs 1024): the OM was displayed by
BigWarp on a 32x-coarser grid (1024/32 x 718/32 = 32 x 22.4). Evidence for the factor 32: a least-squares
similarity/affine fit of moving*32 -> fixed gives scale ~1.13-1.21 and rotation ~7 deg, which matches the
visible SEM tilt of the SEM tiles (a factor of 1 or 1024/27 would give an implausible scale of 30 or 0.9-1.0
with a large shift). Conversion used: x_OMpx = x_moving * 32 (no half-pixel term: the +0.5 variant left all points ~15 px down-right of blob centres in the overlay check); the
half-pixel choice moves points by < 16 OM px, below the landmark noise (LOO residual ~ 8 px).
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import numpy as np
from skimage import io as skio

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm2_common import BENCH, finalize_pair, sha256  # noqa: E402

D = BENCH / "raw" / "mendeley" / "slicebyslice"
K = 32.0


def landmark_file(n: int) -> Path:
    c = [p for p in D.glob("landmarks1-*.csv") if re.fullmatch(rf"landmarks1-{n}\.+csv", p.name)]
    assert len(c) == 1, (n, c)
    return c[0]


def main() -> None:
    for n in range(1, 24):
        lm = landmark_file(n)
        rows = [r for r in csv.reader(open(lm, encoding="utf-8")) if r]
        a = np.array([[float(v) for v in r[2:6]] for r in rows if r[1].strip().lower() == "true"])
        om_xy = a[:, :2] * K
        sem_xy = a[:, 2:]
        om = skio.imread(D / f"OM-{n:02d}.tif")
        sem = skio.imread(D / f"SEM-{n:02d}.tif")
        finalize_pair(
            pair_id=f"matpool_C_slice{n:02d}#0", component="C", cluster="mendeley-srscfwnrwt:graphite-epoxy-10kX",
            group="OM-SEM-SerialSection", subclass="ExfoliatedGraphite-epoxy-OMvsSEM",
            img_a=sem, img_b=om, xy_a=sem_xy, xy_b=om_xy, name_a=f"SEM-{n:02d}.tif", name_b=f"OM-{n:02d}.tif",
            px_a_nm=None, px_b_nm=None, mod_a="SEM (8-bit, as shipped; pre-rotated tile)",
            mod_b="OM optical concentration map (8-bit, as shipped, upsampled)", licence="CC BY 4.0 (Mendeley data_licence)",
            source_url="https://doi.org/10.17632/srscfwnrwt.1",
            files=dict(archive_members=f"Correlative analysis example - 10kX/slicebyslice/{{SEM-{n:02d}.tif,OM-{n:02d}.tif,{lm.name}}}",
                       raw_sha256={p.name: sha256(p) for p in (D / f"OM-{n:02d}.tif", D / f"SEM-{n:02d}.tif", lm)}),
            notes=f"BigWarp landmarks, moving(OM) x32 conversion (see module docstring); source=SEM(fixed), target=OM(moving); "
                  f"pixel sizes not stated in the record -> scale_ratio taken from the GT affine", fov_a=None, fov_b=None)
        print("slice", n, len(a))


if __name__ == "__main__":
    main()
