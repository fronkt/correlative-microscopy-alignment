"""Print shape/dtype/range of downloaded NIST section images and CTF headers (no processing)."""
import glob
import os
import sys

import numpy as np
import tifffile

ROOT = r"C:\Users\frank\Documents\materials-bench\nist"


def main():
    pat = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "*", "s*", "*")
    for p in sorted(glob.glob(pat)):
        if p.endswith(".tif"):
            with tifffile.TiffFile(p) as t:
                pg = t.pages[0]
                print(os.path.relpath(p, ROOT), pg.shape, pg.dtype, "tiles" if pg.is_tiled else "strips",
                      "res", pg.tags.get("XResolution").value if "XResolution" in pg.tags else None)
        elif p.endswith(".ctf"):
            with open(p, errors="replace") as f:
                hdr = [next(f).strip() for _ in range(15)]
            print(os.path.relpath(p, ROOT), [h for h in hdr if h.split("\t")[0] in ("XCells", "YCells", "XStep", "YStep", "AcqE1")])


if __name__ == "__main__":
    main()
