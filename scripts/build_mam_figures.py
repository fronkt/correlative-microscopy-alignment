# -*- coding: utf-8 -*-
"""Check the Microscopy & Microanalysis figure package, and flatten its PNGs.

    python scripts/plot_mam_rewrite.py     # draws every figure into paper/mam/figures
    python scripts/build_mam_figures.py    # then checks them

Until the 2026-09 rewrite this script COPIED figures from paper/figs into the
package. The rewrite's figures are drawn straight into paper/mam/figures by
scripts/plot_mam_rewrite.py, so copying would now overwrite them with the four
figures of the rejected submission (MAM-26-246). This version only checks.

The journal wants each figure as an individual file, vector with embedded fonts for
charts and diagrams, and at least 300 dpi for colour half-tones (600-900 dpi for line
art if raster). The PDF is the submission copy; the PNG is for the manuscript file and
is flattened here to RGB, since some submission systems reject an alpha channel.
Figures 1-4 and S2 embed micrographs, so their PDFs legitimately contain rasters.
"""
import pathlib
import re
import sys

from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "paper/mam/figures"

FIGURES = [
    ("1", "The registration problem in three pairs", True),
    ("2", "Registering one pair step by step", True),
    ("3", "Why pooling the matches of many tiles fails", True),
    ("4", "Examples of failure", True),
    ("5", "Pairs registered by each method, against the annotation ceiling", False),
    ("6", "The field-of-view ladder", True),
    ("7", "Field of view against appearance", False),
    ("S1", "Metadata against annotated field-of-view ratio", False),
    ("S2", "Mirror padding in the pooled-tiling implementation", True),
    ("S3", "Registered pairs before and after refinement", False),
]
MIN_DPI = 300


def main() -> int:
    problems, notes = [], []
    expected = {f"Figure{k}.{ext}" for k, _, _ in FIGURES for ext in ("pdf", "png")}
    for extra in sorted(p.name for p in OUT.glob("Figure*") if p.name not in expected):
        problems.append(f"{extra}: file in the package with no entry here (stale figure?)")

    for key, desc, has_micrographs in FIGURES:
        pdf, png = OUT / f"Figure{key}.pdf", OUT / f"Figure{key}.png"
        if not pdf.exists() or not png.exists():
            problems.append(f"Figure {key}: missing {'PDF' if not pdf.exists() else 'PNG'}")
            continue
        raw = pdf.read_bytes()
        if b"/Type3" in raw:
            problems.append(f"Figure {key}: Type 3 fonts in the PDF")
        if not re.search(rb"/BaseFont", raw):
            problems.append(f"Figure {key}: no embedded fonts; text may be flattened to paths")
        has_raster = b"/Subtype /Image" in raw or b"/Subtype/Image" in raw
        if has_raster and not has_micrographs:
            notes.append(f"Figure {key}: PDF embeds a raster although it has no micrograph")

        im = Image.open(png)
        dpi = round(im.info.get("dpi", (0, 0))[0])
        if im.mode != "RGB":
            rgba = im.convert("RGBA")
            flat = Image.new("RGB", rgba.size, (255, 255, 255))
            flat.paste(rgba, mask=rgba.split()[-1])
            flat.save(png, dpi=(dpi, dpi))
            im = flat
        if dpi < MIN_DPI - 1:
            problems.append(f"Figure {key}: PNG is {dpi} dpi, below {MIN_DPI}")
        width_mm = im.size[0] / max(dpi, 1) * 25.4
        print(f"Figure{key:<3} {im.size[0]:5d}x{im.size[1]:<5d} {dpi:3d} dpi {width_mm:6.1f} mm wide  {desc}")

    for n in notes:
        print("  note: " + n)
    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for p in problems:
            print("  " + p)
        return 1
    print(f"\n{len(FIGURES)} figures checked in {OUT.relative_to(ROOT)}: vector PDF with embedded fonts, "
          f"RGB PNG at >= {MIN_DPI} dpi")
    return 0


if __name__ == "__main__":
    sys.exit(main())
