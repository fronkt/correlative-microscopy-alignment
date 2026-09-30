"""Probe Anes Acta 2023 (Zenodo 7383087): list ROIs, fetch control-point CSVs (tiny), report extents."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm2_fetch import BENCH, fetch_members, zenodo_url  # noqa: E402
from remotezip import RemoteZip  # noqa: E402

REC = "7383087"
OUT = BENCH / "raw" / REC


def main() -> None:
    for cond in ("0s", "175c", "300c", "325c"):
        with RemoteZip(zenodo_url(REC, f"{cond}.zip")) as z:
            names = [i.filename for i in z.infolist()]
        csvs = [n for n in names if n.endswith(".csv")]
        for n in csvs:
            roi = n.split("/")[1]
            p = fetch_members(REC, f"{cond}.zip", [n], OUT / cond / roi)[0]
            a = np.array([[float(v) for v in r] for r in list(csv.reader(open(p)))[1:]])
            print(cond, roi, "n=", len(a), "src max", a[:, :2].max(0), "min", a[:, :2].min(0),
                  "dst max", a[:, 2:].max(0), "min", a[:, 2:].min(0))
        tiles = [n for n in names if "/bse/" in n and n.endswith(".tif")]
        print(cond, "tiles", len(tiles), "rois", sorted({n.split('/')[1] for n in names if n.count('/') > 1}))


if __name__ == "__main__":
    main()
