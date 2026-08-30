# -*- coding: utf-8 -*-
"""Assemble the Microscopy & Microanalysis figure package.

    python scripts/build_mam_figures.py

The journal wants each figure as an individual file, every panel of a multi-panel
figure in ONE file, and -- for charts and diagrams, which is all four of ours --
vector with embedded fonts. Its own words: "Vector graphics (maps, charts,
diagrams) should be saved as .eps or .svg files", and .pdf is in the accepted
list. So the PDF is the submission copy for every figure.

The PNG is built too, because it is what gets pasted into the manuscript file and
what a human actually looks at, but it is not the thing uploaded. That matters
here: M&M's raster minimums are 300 dpi only for colour half-tones, and
600-900 dpi for line art of the kind these all are. Shipping the 300 dpi PNGs
would have been below the journal's own floor; the PDFs make the question moot.

Figures 1 and 3 use the *_raw variants, i.e. the unrefined parametric error,
which is the paper's primary metric; Figure 4 deliberately uses the refined
variant, because its whole point is the contrast with Figure 1.
"""
import pathlib
import re
import shutil
import sys

from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "paper/figs"
OUT = ROOT / "paper/mam/figures"

# The method schematic was cut on 2026-08-30 (author's call). Methods 2.8 and 2.9
# already state the wrapper's control flow and the ladder construction in prose,
# and the schematic never illustrated the paper's actual mechanism. Its generator
# is still in paper/schematics/ if it is ever wanted back.
FIGURES = [
    (1, "sr_bars_raw",      "Success rates, unrefined error (primary metric)"),
    (2, "fov_ladder",       "Controlled field-of-view ladder"),
    (3, "fov_curves_raw",   "Success at 10 px by field-of-view stratum"),
    (4, "sr_bars",          "Success rates, TPS-refined error (contrast to Fig. 1)"),
]

MIN_DPI = 300           # M&M's colour half-tone floor; line art wants 600-900
LINE_ART_DPI = 600


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    problems, notes = [], []

    for num, stem, desc in FIGURES:
        png, pdf = SRC / (stem + ".png"), SRC / (stem + ".pdf")

        if not pdf.exists():
            problems.append("Figure %d: no vector source (%s.pdf); the journal "
                            "asks for vector for charts and diagrams" % (num, stem))
        else:
            shutil.copyfile(pdf, OUT / ("Figure%d.pdf" % num))
            raw = (OUT / ("Figure%d.pdf" % num)).read_bytes()
            if b"/Type3" in raw:
                problems.append("Figure %d: Type 3 fonts in the PDF" % num)
            if b"/Subtype /Image" in raw or b"/Subtype/Image" in raw:
                notes.append("Figure %d: PDF embeds a raster (expected only for "
                             "a heat map)" % num)
            if not re.search(rb"/BaseFont", raw):
                problems.append("Figure %d: no embedded fonts in the PDF; text "
                                "may have been flattened to paths" % num)

        if not png.exists():
            problems.append("Figure %d: source missing (%s.png)" % (num, stem))
            continue

        im = Image.open(png)
        dpi = im.info.get("dpi", (0, 0))[0]
        # Keep whatever the source was rendered at, provided it clears the
        # minimum. Force-stamping 300 on a figure rendered higher would tell
        # Word it is larger than it is, by exactly that ratio.
        out_dpi = max(MIN_DPI, round(dpi))

        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            flat = Image.new("RGB", im.size, (255, 255, 255))
            flat.paste(im, mask=im.split()[-1])
            im = flat
        elif im.mode != "RGB":
            im = im.convert("RGB")

        im.save(OUT / ("Figure%d.png" % num), dpi=(out_dpi, out_dpi))

        if dpi < MIN_DPI - 1:
            problems.append("Figure %d: PNG is %.0f dpi, below the %d dpi minimum"
                            % (num, dpi, MIN_DPI))
        elif dpi < LINE_ART_DPI - 1:
            notes.append("Figure %d: PNG is %.0f dpi, below M&M's %d dpi line-art "
                         "tier -- fine, since the PDF is what is uploaded"
                         % (num, dpi, LINE_ART_DPI))

        # Printed width. M&M states no maximum, so this is reported, not gated;
        # what it governs is how far production has to scale the figure down, and
        # therefore how small the smallest type ends up.
        width_mm = im.size[0] / out_dpi * 25.4
        print("Figure%d  %5dx%-5d  %3d dpi  %5.1f mm wide  %s  <- %s"
              % (num, im.size[0], im.size[1], out_dpi, width_mm,
                 "pdf+png" if pdf.exists() else "png only", stem))
        print("         %s" % desc)

    for n in notes:
        print("  note: " + n)
    if problems:
        print("\n%d PROBLEM(S):" % len(problems))
        for p in problems:
            print("  " + p)
        return 1
    print("\n%d figures written to %s" % (len(FIGURES), OUT.relative_to(ROOT)))
    print("PDF is the submission copy; PNG is for the manuscript file")
    return 0


if __name__ == "__main__":
    sys.exit(main())
