"""Prove an exported schematic is publication-legal, without opening it.

    python verify.py fig2_schematic fig4_schematic \
        --expect "TiC core" --expect "fig4_schematic:Transmission"

A bare --expect must appear in every figure; prefix it with "<stem>:" to require it in
just one.

For each <stem>, checks <stem>.pdf and <stem>.svg for the things that get a figure
bounced or that silently ship wrong: Type 3 fonts, embedded rasters in what is supposed
to be line art, text flattened to paths in the SVG so co-authors cannot edit it, and
label strings that must be present (spelling fixes especially -- a typo you corrected in
the code is worth asserting in the output).

Exit status is non-zero if anything fails, so it can gate a build.
"""
import argparse
import os
import re
import sys


def check(stem, expect):
    ok = True
    pdf_path, svg_path = f"{stem}.pdf", f"{stem}.svg"
    print(f"== {stem}")

    if not os.path.exists(pdf_path):
        print("   MISSING", pdf_path)
        return False
    raw = open(pdf_path, "rb").read()

    t3 = raw.count(b"/Type3")
    imgs = raw.count(b"/Subtype /Image") + raw.count(b"/Subtype/Image")
    fonts = sorted(set(m.decode() for m in
                       re.findall(rb"/BaseFont\s*/([A-Za-z0-9+\-]+)", raw)))
    print(f"   pdf  Type3={t3}  rasters={imgs}  size={len(raw)}")
    print(f"   pdf  fonts={fonts}")
    if t3:
        print("   FAIL Type 3 fonts present -- set pdf.fonttype=42")
        ok = False
    if imgs:
        print("   FAIL embedded raster in vector line art")
        ok = False
    if not fonts:
        print("   FAIL no embedded fonts -- text may have been flattened to paths")
        ok = False
    # Two typefaces inside one figure is the mathtext-fallback bug. Compare families,
    # not PostScript names: one Arial ships as both "ArialMT" and "Arial-BoldMT".
    families = set(re.sub(r"(MT|PS)$", "", f.split("+")[-1].split("-")[0])
                   for f in fonts)
    if len(families) > 1:
        print(f"   WARN more than one font family embedded: {sorted(families)}"
              " -- check mathtext.fontset")

    if not os.path.exists(svg_path):
        print("   MISSING", svg_path)
        return False
    svg = open(svg_path, encoding="utf-8").read()
    n_img, n_txt = svg.count("<image"), svg.count("<text")
    print(f"   svg  <image>={n_img}  <text>={n_txt}  size={len(svg)}")
    if n_img:
        print("   FAIL raster embedded in SVG")
        ok = False
    if n_txt == 0:
        print("   FAIL no <text> elements -- set svg.fonttype='none'")
        ok = False
    for s in expect:
        target, _, text = s.rpartition(":") if ":" in s else ("", "", s)
        if target and target != stem:
            continue
        hit = text in svg
        print(f"   svg  {text!r}: {'ok' if hit else 'MISSING'}")
        ok &= hit
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stems", nargs="+", help="file stems, without extension")
    ap.add_argument("--expect", action="append", default=[],
                    help="string that must appear as real text in every SVG; "
                         "prefix with '<stem>:' to require it in one figure only")
    a = ap.parse_args()
    ok = all(check(s, a.expect) for s in a.stems)
    print("\nPASS" if ok else "\nFAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
