"""Build the CJSJ figures file from the journal's own PowerPoint template.

Keeps the template's title slide (retitled), then one TITLE_AND_BODY slide per figure: the picture scaled
into the template's picture area (1.48-7.66 in x, 0.12-4.35 in y) and the caption in the body placeholder,
formatted like the template's example ("Figure n.<tab>caption"). The template's two example slides are
removed. Captions come from paper/cjsj/captions.md: blocks of "Figure n | image path" then caption text.
"""

from __future__ import annotations

import re
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.util import Emu, Pt

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "paper/cjsj/templates/CJSJ-Figures-Template.pptx"
CAPTIONS = ROOT / "paper/cjsj/captions.md"
OUT = ROOT / "paper/cjsj/Cai_CJSJ_2026_Figures.pptx"
AREA = (Emu(311700), Emu(122975), Emu(8520600), Emu(4226875))  # left, top, width, height for pictures


def parse(path: Path) -> list[tuple[str, Path, str]]:
    items, cur = [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(Figure \d+)\s*\|\s*(\S+)\s*$", line.strip())
        if m:
            if cur:
                items.append(cur)
            cur = [m.group(1), ROOT / m.group(2), ""]
        elif cur is not None:
            cur[2] = (cur[2] + " " + line.strip()).strip()
    if cur:
        items.append(cur)
    return [tuple(i) for i in items]


def main() -> None:
    prs = Presentation(str(TEMPLATE))
    title = prs.slides[0]
    title.shapes[0].text_frame.text = "Which Automatic Image Registrations Can a Microscopist Trust?"
    title.shapes[1].text_frame.text = "Figures. Frank Cai, 2026-2027"
    layout = prs.slides[1].slide_layout
    # drop the two example slides
    sldIdLst = prs.slides._sldIdLst
    for sldId in list(sldIdLst)[1:]:
        prs.part.drop_rel(sldId.rId)
        sldIdLst.remove(sldId)
    for name, img, cap in parse(CAPTIONS):
        s = prs.slides.add_slide(layout)
        body = [ph for ph in s.placeholders if ph.placeholder_format.idx != 0] or list(s.placeholders)
        for ph in list(s.placeholders):
            if ph not in body[:1]:
                ph._element.getparent().remove(ph._element)
        tf = body[0].text_frame
        body[0].left, body[0].top, body[0].width, body[0].height = Emu(311700), Emu(4380000), Emu(8520600), Emu(700000)
        tf.text = f"{name}.\t{cap}"
        tf.word_wrap = True
        for p in tf.paragraphs:
            for r in p.runs:
                r.font.size = Pt(8)
                r.font.name = "Times New Roman"
        w, h = Image.open(img).size
        L, T, W, H = AREA
        scale = min(W / w, H / h)
        pw, ph_ = int(w * scale), int(h * scale)
        s.shapes.add_picture(str(img), L + (W - pw) // 2, T + (H - ph_) // 2, pw, ph_)
    prs.save(str(OUT))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
