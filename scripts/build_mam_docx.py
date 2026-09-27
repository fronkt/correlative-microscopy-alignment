# -*- coding: utf-8 -*-
"""Build the Microscopy & Microanalysis submission DOCX from the markdown source.

    python scripts/build_mam_docx.py

Applies the journal's stated manuscript format: 12 pt, double-spaced throughout
(title page, abstract, text, references, tables and figure legends), ~1 inch
margins. Continuous line numbers are added as a reviewing courtesy.

Note the theme-font trap: Word resolves <w:rFonts w:asciiTheme=...> ahead of an
explicit w:ascii, so the theme attributes must be stripped or the font silently
reverts. See reference_journal_submission_skill in the operator's notes.
"""
import pathlib
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "paper/mam/manuscript.md"
OUT = ROOT / "paper/mam/manuscript.docx"
TMP = ROOT / "paper/mam/_raw.docx"

TWIPS_INCH = 1440
LINE_DOUBLE = "480"   # 240 twips = single; 480 = double
PT12 = "24"           # half-points
LINE_NUMBERS = True


def build():
    # --resource-path lets a source embed images relative to its own folder
    # (the supplement does; the manuscript does not, the journal wants figures
    # as separate uploads).
    subprocess.run(
        ["pandoc", str(SRC), "-o", str(TMP), "--standalone",
         "--resource-path", str(SRC.parent)],
        check=True, cwd=str(ROOT),
    )


def to_pdf(docx: pathlib.Path) -> pathlib.Path:
    """Export through Word itself, so the PDF is exactly what Word lays out."""
    pdf = docx.with_suffix(".pdf")
    ps = (
        "$w = New-Object -ComObject Word.Application; $w.Visible = $false; "
        f"$d = $w.Documents.Open('{docx}', $false, $true); "
        f"$d.SaveAs2('{pdf}', 17); $d.Close($false); $w.Quit()"
    )
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True)
    print("exported %s" % pdf.relative_to(ROOT))
    return pdf


def patch_styles(xml: str) -> str:
    # Strip theme fonts everywhere; they win over explicit w:ascii in Word.
    xml = re.sub(r'\s+w:(ascii|hAnsi|eastAsia|cs)Theme="[^"]*"', "", xml)

    # Force Normal to 12 pt Times New Roman, double-spaced, no paragraph gap.
    def fix_normal(m):
        block = m.group(0)
        block = re.sub(r"<w:rFonts[^/>]*/>", "", block)
        block = re.sub(r"<w:sz\s+w:val=\"\d+\"\s*/>", "", block)
        block = re.sub(r"<w:szCs\s+w:val=\"\d+\"\s*/>", "", block)
        block = re.sub(r"<w:spacing[^/>]*/>", "", block)
        rpr = ('<w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" '
               'w:cs="Times New Roman"/><w:sz w:val="%s"/><w:szCs w:val="%s"/></w:rPr>'
               % (PT12, PT12))
        ppr = ('<w:pPr><w:spacing w:before="0" w:after="0" w:line="%s" '
               'w:lineRule="auto"/></w:pPr>' % LINE_DOUBLE)
        block = block.replace("</w:style>", ppr + rpr + "</w:style>")
        return block

    xml = re.sub(r'<w:style [^>]*w:styleId="Normal".*?</w:style>', fix_normal, xml, flags=re.S)

    # Double-space every other style too, so headings/captions match the body.
    xml = re.sub(r'<w:spacing((?:(?!w:line=)[^/>])*)/>',
                 r'<w:spacing\1 w:line="%s" w:lineRule="auto"/>' % LINE_DOUBLE, xml)
    return xml


def patch_document(xml: str) -> str:
    xml = re.sub(r'\s+w:(ascii|hAnsi|eastAsia|cs)Theme="[^"]*"', "", xml)

    # 1 inch margins + continuous line numbering. Pandoc emits no <w:pgSz>/
    # <w:pgMar> at all, so these must be inserted, not substituted -- and
    # CT_SectPr is a sequence: footnotePr, endnotePr, type, pgSz, pgMar,
    # paperSrc, pgBorders, lnNumType, ... Order it wrong and Word repairs
    # the file, silently discarding the section properties.
    page = ('<w:pgSz w:w="12240" w:h="15840"/>'
            '<w:pgMar w:top="%d" w:right="%d" w:bottom="%d" w:left="%d" '
            'w:header="720" w:footer="720" w:gutter="0"/>'
            % (TWIPS_INCH, TWIPS_INCH, TWIPS_INCH, TWIPS_INCH))

    if "<w:pgMar" in xml:
        xml = re.sub(r"<w:pgSz[^/>]*/>", "", xml)
        xml = re.sub(r"<w:pgMar[^/>]*/>", page, xml)
    elif "</w:footnotePr>" in xml:
        xml = xml.replace("</w:footnotePr>", "</w:footnotePr>" + page, 1)
    else:
        xml = xml.replace("<w:sectPr>", "<w:sectPr>" + page, 1)

    if LINE_NUMBERS and "<w:lnNumType" not in xml:
        xml = xml.replace(
            "</w:sectPr>",
            '<w:lnNumType w:countBy="1" w:restart="continuous"/></w:sectPr>')
    return xml


def repack():
    with zipfile.ZipFile(TMP) as zin:
        items = zin.infolist()
        data = {i.filename: zin.read(i.filename) for i in items}

    data["word/styles.xml"] = patch_styles(
        data["word/styles.xml"].decode("utf-8")).encode("utf-8")
    data["word/document.xml"] = patch_document(
        data["word/document.xml"].decode("utf-8")).encode("utf-8")

    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zout:
        for i in items:
            zout.writestr(i, data[i.filename])
    TMP.unlink()


def audit():
    """Re-open the built file and assert the format actually took."""
    problems = []
    with zipfile.ZipFile(OUT) as z:
        styles = z.read("word/styles.xml").decode("utf-8")
        docxml = z.read("word/document.xml").decode("utf-8")

    if "Theme=" in styles or "Theme=" in docxml:
        problems.append("theme fonts survived; Word will override the font")
    if 'w:line="%s"' % LINE_DOUBLE not in styles:
        problems.append("double spacing not applied to styles")
    if 'w:sz w:val="%s"' % PT12 not in styles:
        problems.append("12 pt not applied to Normal")
    if 'w:top="%d"' % TWIPS_INCH not in docxml:
        problems.append("1 inch margins not applied")
    if LINE_NUMBERS and "<w:lnNumType" not in docxml:
        problems.append("line numbering not applied")

    size_kb = OUT.stat().st_size / 1024
    print("built %s (%.0f KB)" % (OUT.relative_to(ROOT), size_kb))
    if problems:
        for p in problems:
            print("  FAIL: " + p)
        return 1
    spacing = "double-spaced" if LINE_DOUBLE == "480" else "single-spaced"
    numbers = "line numbers" if LINE_NUMBERS else "no line numbers"
    print(f"  format audit: 12 pt / {spacing} / 1 in margins / {numbers} / no theme fonts")
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", nargs="?", default=str(SRC), help="markdown source (default: the manuscript)")
    ap.add_argument("out", nargs="?", default=None, help="DOCX to write (default: next to the source)")
    ap.add_argument("--single", action="store_true", help="single spacing (supplement, letters)")
    ap.add_argument("--no-line-numbers", action="store_true")
    ap.add_argument("--pdf", action="store_true", help="also export a PDF through Word")
    args = ap.parse_args()
    SRC = pathlib.Path(args.src).resolve()
    OUT = pathlib.Path(args.out).resolve() if args.out else SRC.with_suffix(".docx")
    TMP = OUT.with_name("_raw_" + OUT.name)
    if args.single:
        LINE_DOUBLE = "240"
    LINE_NUMBERS = not args.no_line_numbers
    if not shutil.which("pandoc"):
        sys.exit("pandoc not found on PATH")
    build()
    repack()
    rc = audit()
    if args.pdf and rc == 0:
        to_pdf(OUT)
    sys.exit(rc)
