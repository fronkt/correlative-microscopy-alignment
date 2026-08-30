"""Crop and enlarge part of a rendered figure, so it can actually be inspected.

The working loop for a schematic is render -> LOOK AT IT -> fix -> render. At full
figure width almost every real defect is invisible: a hollow sphere, two outlines
grazing, a leader landing on the wrong object, a label overlapping geometry by a
hair. All of them are obvious at 4x on the panel that contains them.

    python crop.py fig.png --grid              # overlay a labelled 0..1 grid, to aim with
    python crop.py fig.png 0.5 0.0 1.0 0.5     # crop that region, enlarged, to fig_crop.png

Fractions are l t r b with the origin at the TOP-LEFT, which is how you read the
image. Pillow ships with matplotlib, so this needs nothing new.
"""
import argparse
import os

from PIL import Image, ImageDraw


def grid(path, out=None, n=10):
    """Write a copy with an n x n labelled grid, to choose crop fractions from."""
    im = Image.open(path).convert("RGB")
    d = ImageDraw.Draw(im)
    w, h = im.size
    for i in range(1, n):
        x, y = w * i / n, h * i / n
        d.line([(x, 0), (x, h)], fill=(255, 0, 0), width=max(1, w // 1200))
        d.line([(0, y), (w, y)], fill=(255, 0, 0), width=max(1, h // 1200))
        d.text((x + 3, 3), f"{i / n:.1f}", fill=(255, 0, 0))
        d.text((3, y + 3), f"{i / n:.1f}", fill=(255, 0, 0))
    out = out or f"{os.path.splitext(path)[0]}_grid.png"
    im.save(out)
    return out, im.size


def crop(path, box, out=None, min_width=1400, scale=None):
    """Crop `box` = (l, t, r, b) as fractions and upscale it for inspection."""
    im = Image.open(path)
    w, h = im.size
    l, t, r, b = box
    px = (int(l * w), int(t * h), int(r * w), int(b * h))
    if px[2] - px[0] < 2 or px[3] - px[1] < 2:
        raise SystemExit(f"crop box is empty: {box}")
    c = im.crop(px)
    k = scale if scale else max(1.0, min_width / c.width)
    if k > 1.0:
        c = c.resize((int(c.width * k), int(c.height * k)), Image.LANCZOS)
    out = out or f"{os.path.splitext(path)[0]}_crop.png"
    c.save(out)
    return out, c.size


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("image")
    ap.add_argument("box", nargs="*", type=float, metavar="L T R B",
                    help="crop fractions, origin top-left")
    ap.add_argument("--grid", action="store_true", help="write a labelled grid instead")
    ap.add_argument("-o", "--out")
    ap.add_argument("--scale", type=float, help="fixed magnification")
    ap.add_argument("--min-width", type=int, default=1400)
    a = ap.parse_args()
    if a.grid:
        out, size = grid(a.image, a.out)
    else:
        if len(a.box) != 4:
            ap.error("give four fractions L T R B, or --grid")
        out, size = crop(a.image, a.box, a.out, a.min_width, a.scale)
    print(f"{out}  {size[0]}x{size[1]}")


if __name__ == "__main__":
    main()
