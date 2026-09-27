# -*- coding: utf-8 -*-
"""Check the Microscopy & Microanalysis figure package.

    python scripts/plot_mam_rewrite.py     # draws every panel into paper/mam/figures/panels
    python scripts/build_mam_figures.py    # then checks them

Until the 2026-09 rewrite this script COPIED figures from paper/figs into the package; it
has only checked since. On 2026-09-27 the MSA office unsubmitted MAM-26-277 because the
composite figures were ~1600 pixels wide, multi-panel and carried caption text. Its terms,
which this script enforces for Figures 1-7:

- each image of a figure is its own file (panels/Figure<N><letter>.tif), no multi-panel files;
- at least 2550 pixels per line (8.5 in at 300 dpi), at least 300 dpi;
- RGB, no alpha channel.

"No legend inside the file" cannot be checked by a script; look at the layout sheets
(Figure<N>_layout.jpg). The supplementary figures S1-S3 go into supplementary.pdf, not the
upload, and keep their PDF + PNG checks.
"""
import pathlib
import re
import sys

from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "paper/mam/figures"
PANELS = OUT / "panels"
sys.path.insert(0, str(ROOT / "scripts"))
from plot_mam_rewrite import LAYOUT, MIN_PX  # noqa: E402  (the single source of the panel list)

SUPPLEMENTARY = [
    ("S1", "Metadata against annotated field-of-view ratio", False),
    ("S2", "Mirror padding in the pooled-tiling implementation", True),
    ("S3", "Registered pairs before and after refinement", False),
]
MIN_DPI = 300


def main() -> int:
    problems = []
    expected = {f"Figure{n}{c}.tif" for n, rows in LAYOUT.items() for c in "".join(rows)}
    present = {p.name for p in PANELS.glob("*")} if PANELS.exists() else set()
    problems += [f"panels/{x}: missing" for x in sorted(expected - present)]
    problems += [f"panels/{x}: not in LAYOUT (stale panel?)" for x in sorted(present - expected)]
    allowed = ({f"Figure{n}_layout.jpg" for n in LAYOUT}
               | {f"Figure{k}.{e}" for k, _, _ in SUPPLEMENTARY for e in ("pdf", "png")})
    for p in sorted(OUT.glob("Figure*")):
        if p.name not in allowed:
            problems.append(f"{p.name}: a file in the package that is not a panel, a layout sheet or a "
                            "supplementary figure (a composite of the unsubmitted package?)")
    if (OUT / "tif").exists():
        problems.append("figures/tif/: the composite TIFFs the office rejected are still on disk")

    for name in sorted(expected & present, key=lambda s: (int(re.search(r"\d+", s)[0]), s)):
        im = Image.open(PANELS / name)
        dpi = round(im.info.get("dpi", (0, 0))[0])
        comp = im.info.get("compression", "raw")
        bad = []
        if im.size[0] < MIN_PX:
            bad.append(f"{im.size[0]} pixels wide, below {MIN_PX}")
        if dpi < MIN_DPI:
            bad.append(f"{dpi} dpi, below {MIN_DPI}")
        if im.mode != "RGB":
            bad.append(f"mode {im.mode}, not RGB")
        problems += [f"panels/{name}: {b}" for b in bad]
        print(f"{name:<14} {im.size[0]:5d} x {im.size[1]:<5d} {dpi:4d} dpi  {comp:<10} {'OK' if not bad else 'FAIL'}")
    n_layout = sum((OUT / f"Figure{n}_layout.jpg").exists() for n in LAYOUT)
    if n_layout != len(LAYOUT):
        problems.append(f"{len(LAYOUT) - n_layout} layout sheet(s) missing")

    for key, desc, has_micrographs in SUPPLEMENTARY:
        pdf, png = OUT / f"Figure{key}.pdf", OUT / f"Figure{key}.png"
        if not pdf.exists() or not png.exists():
            problems.append(f"Figure {key}: missing {'PDF' if not pdf.exists() else 'PNG'}")
            continue
        raw = pdf.read_bytes()
        if b"/Type3" in raw:
            problems.append(f"Figure {key}: Type 3 fonts in the PDF")
        if not re.search(rb"/BaseFont", raw):
            problems.append(f"Figure {key}: no embedded fonts; text may be flattened to paths")
        dpi = round(Image.open(png).info.get("dpi", (0, 0))[0])
        if dpi < MIN_DPI - 1:
            problems.append(f"Figure {key}: PNG is {dpi} dpi, below {MIN_DPI}")

    if problems:
        print(f"\n{len(problems)} PROBLEM(S):")
        for p in problems:
            print("  " + p)
        return 1
    print(f"\n{len(expected)} panel files for {len(LAYOUT)} figures: each >= {MIN_PX} pixels wide, >= {MIN_DPI} dpi, "
          f"RGB; {n_layout} layout sheets; {len(SUPPLEMENTARY)} supplementary figures checked")
    return 0


if __name__ == "__main__":
    sys.exit(main())
