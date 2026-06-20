# Manuscript — Correlative Microscopy Registration

Manuscript for the `correlative-microscopy-alignment` project, tracked in this
repo under `paper/`. The project **code** lives in the repository root; this
folder holds the **paper** (LaTeX source, bibliography, figures, Markdown render).

**Target journal:** *Scientific Reports* (Nature Portfolio).
Single English abstract, Results→Discussion→Methods order, numbered references.

## Contents

| File | Purpose |
|------|---------|
| `main.tex` | Manuscript in Scientific Reports LaTeX (`wlscirep.cls`). |
| `references.bib` | 17 references, all verified against primary sources. |
| `paper.md` | Human-readable full render (tables + figures inline). |
| `paper.docx` | Word version (figures embedded), built from `paper.md` via `pandoc paper.md -o paper.docx --resource-path=.`. |
| `make_schematic.py` | Regenerates Fig. 1 (method schematic). |
| `figs/` | `method_schematic.{png,pdf}` (generated) + 4 result figures copied from `reports/figs/baselines/`. |

## Build the PDF

There is no LaTeX engine on the authoring machine. To compile:

1. Open the official **Scientific Reports** template on Overleaf (it bundles
   `wlscirep.cls` and `naturemag.bst`).
2. Replace its `main.tex` and `references.bib` with the files here; upload the
   `figs/` folder.
3. Compile (pdfLaTeX → BibTeX → pdfLaTeX ×2). BibTeX renumbers references by
   order of first appearance automatically.

Locally, once a TeX distribution is installed:

```
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Regenerate the schematic if needed:

```
python make_schematic.py
```

## Figures

- **Fig. 1** `method_schematic.png` — (a) verified coarse-to-fine wrapper, (b) FOV-ladder protocol. Generated.
- **Fig. 2** `sr_bars.png` — success rates, all configs.
- **Fig. 3** `group_heatmap.png` — per-task-group success.
- **Fig. 4** `fov_curves.png` — SR vs FOV stratum (native pairs).
- **Fig. 5** `fov_ladder.png` — controlled FOV ladder.

All numerical values trace to `../results/` in the public code repo and are
regenerable via the scripts named in `../reports/final_report.md`.

## Provenance

Every reference was checked against a primary source (publisher page, arXiv,
or proceedings) before inclusion — no citation is unverified. Numbers in the
tables and abstract are copied from `../reports/final_report.md`.
