"""Render reports/abstract.docx from the abstract text.

Keeps the .docx regenerable and in sync with abstract.md / abstract.txt.
Usage: python scripts/build_abstract_docx.py
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

TITLE = ("Multi-Scale Alignment of Correlative Materials Microscopy with "
         "Foundational Dense Matchers")

ABSTRACT = (
    "Correlative materials microscopy pairs images of the same specimen "
    "across modalities (SEM, EBSD, TEM, optical) that share little visual "
    "appearance and often differ in field of view (FOV) by more than an "
    "order of magnitude, defeating classical feature-based registration. We "
    "asked whether a scale-aware pyramidal patching wrapper around pretrained "
    "dense matchers (RoMa, ELoFTR-family, MatchAnything) could lift "
    "cross-modal registration on AmalgaMatch (Durmaz et al.; 187 pairs, 19 "
    "subsets), particularly at severe FOV mismatch. Across a full "
    "controls-and-ablations pass we find: (1) a naive tiling pyramid "
    "catastrophically degrades dense matchers, because they never abstain — "
    "every tile returns thousands of confident matches that flood robust "
    "estimation (median error 76 → 1794 px); (2) a redesigned verified "
    "coarse-to-fine wrapper recovers a small but significant gain (SR@10 0.10 "
    "→ 0.12, p = 0.017) without losing any pair, yet leaves success at FOV "
    "≤ 5% at zero. The largest off-the-shelf lever was instead the backbone: "
    "swapping in cross-modal-trained MatchAnything-RoMa weights gave the only "
    "significant zero-shot-bar headline gain (SR@10 +0.032, p = 0.018), "
    "entirely among high-FOV pairs. This isolates the binding constraint: on "
    "AmalgaMatch, cross-modal appearance — not scale — is what fails. A "
    "controlled FOV-ladder experiment, which crops real base-matchable pairs "
    "to sweep FOV with appearance fixed, confirms both halves: the same "
    "wrapper triples success at 10% FOV (SR@10 0.07 → 0.23, p = 0.0014), so "
    "the scale mechanism is sound — the real distribution simply never "
    "isolates scale as the failure mode. Finally, we test the appearance "
    "lever directly: decoder-only fine-tuning of MatchAnything-RoMa on 131 "
    "held-out-split training pairs cuts median error on in-distribution TEM "
    "pairs 5.2× (321 → 62 px, the largest single movement in the study), "
    "but regresses overall SR@20 (0.393 → 0.250) by catastrophically "
    "forgetting a modality combination absent from training. Appearance is "
    "therefore attackable with domain data, but a 131-pair budget trades "
    "modality coverage for in-domain depth. We conclude that deployable "
    "cross-modal microscopy registration needs a forgetting-robust domain "
    "fine-tune over broad modality coverage, with the pyramid wrapper as a "
    "complementary — and provably effective — scale layer on top."
)

VERDICTS = [
    ("H1 (pyramid ≥ 35% gain at FOV ≤ 5%): ", "rejected."),
    ("H2 (RoMa beats ELoFTR-family at low FOV): ", "supported."),
    ("H3 (affine sufficient vs. homography): ",
     "mostly supported (affine selected on 69% of well-registered pairs)."),
]

PROVENANCE = ("Full methods, tables, and figures: reports/final_report.md. "
              "All numbers regenerable from results/ via the scripts "
              "referenced therein.")


def main() -> None:
    doc = Document()
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run(TITLE)
    run.bold = True
    run.font.size = Pt(14)

    h = doc.add_paragraph()
    h.add_run("Abstract").bold = True

    body = doc.add_paragraph(ABSTRACT)
    body.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    hv = doc.add_paragraph()
    hv.add_run("Hypothesis verdicts").bold = True
    for prefix, verdict in VERDICTS:
        p = doc.add_paragraph(style="List Bullet")
        p.add_run(prefix)
        p.add_run(verdict).bold = True

    prov = doc.add_paragraph()
    prov.add_run(PROVENANCE).italic = True

    out = Path("reports/abstract.docx")
    doc.save(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
