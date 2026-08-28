# -*- coding: utf-8 -*-
"""Assemble the Microscopy & Microanalysis figure package.

    python scripts/build_mam_figures.py

The journal wants each figure uploaded as an individual file, every panel of a
multi-panel figure in ONE file, and a minimum of 300 dpi for colour. Source
plots are already generated at 300 dpi by scripts/plot_baselines.py and
scripts/plot_fov_ladder.py; this script selects the right variants, flattens
RGBA onto white (transparency prints as black in some RIPs), and re-stamps the
dpi so it survives the copy.

Figure order follows the manuscript. Note that Figures 2 and 4 use the *_raw
variants, i.e. the unrefined parametric error, which is the paper's primary
metric; Figure 5 deliberately uses the refined variant, because its whole point
is the contrast with Figure 2.
"""
import pathlib
import sys

from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "paper/figs"
OUT = ROOT / "paper/mam/figures"

FIGURES = [
    (1, "method_schematic.png", "Method schematic (panels A, B)"),
    (2, "sr_bars_raw.png",      "Success rates, unrefined error (primary metric)"),
    (3, "fov_ladder.png",       "Controlled field-of-view ladder"),
    (4, "fov_curves_raw.png",   "Success at 10 px by field-of-view stratum"),
    (5, "sr_bars.png",          "Success rates, TPS-refined error (contrast to Fig. 2)"),
]

MIN_DPI = 300


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    problems = []

    for num, name, desc in FIGURES:
        src = SRC / name
        if not src.exists():
            problems.append("Figure %d: source missing (%s)" % (num, name))
            continue

        im = Image.open(src)
        dpi = im.info.get("dpi", (0, 0))[0]

        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            flat = Image.new("RGB", im.size, (255, 255, 255))
            flat.paste(im, mask=im.split()[-1])
            im = flat
        elif im.mode != "RGB":
            im = im.convert("RGB")

        dest = OUT / ("Figure%d.png" % num)
        im.save(dest, dpi=(MIN_DPI, MIN_DPI))

        if dpi < MIN_DPI - 1:
            problems.append("Figure %d: source was %.0f dpi, below the %d dpi minimum"
                            % (num, dpi, MIN_DPI))

        # Legibility at the journal's single-column width of 84 mm.
        width_mm = im.size[0] / MIN_DPI * 25.4
        print("Figure%d.png  %5dx%-5d  %.0f dpi  %5.0f mm wide at 300 dpi  <- %s"
              % (num, im.size[0], im.size[1], MIN_DPI, width_mm, name))
        print("             %s" % desc)

    if problems:
        print("\n%d PROBLEM(S):" % len(problems))
        for p in problems:
            print("  " + p)
        return 1
    print("\n%d figures written to %s" % (len(FIGURES), OUT.relative_to(ROOT)))
    print("all at %d dpi, RGB, one file per figure" % MIN_DPI)
    return 0


if __name__ == "__main__":
    sys.exit(main())
