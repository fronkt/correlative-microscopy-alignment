# Journal of Microscopy draft: working notes (2026-10-04)

Files in this folder:
- `manuscript.md`: full first draft.
- `cover_letter.md`
- `make_figures.py`, which writes `figures/Fig{1,2,3}_*.pdf` (vector) and `.png` (600 dpi).

Figure 1 reads the AmalgaMatch images from `../correlative-microscopy-alignment/data/AmalgaMatch` (not in this worktree). Without them, it falls back to the committed smoke thumbnail. The script asserts that every plotted headline number equals `arm1_summary.json` / `arm2_summary.json`, including recomputing the Arm 2 AUROC of 0.983 from `candidates.csv`.

## 1. Journal guidelines: what was verified

Wiley's author-guidelines page (https://onlinelibrary.wiley.com/page/journal/13652818/homepage/forauthors.html) and ScholarOne (mc.manuscriptcentral.com/jmi) returned HTTP 403 to plain fetches. They were not bypassed (no browser, no proxy). "Partly verified" below means the fact came from a search-engine snippet of the official page and has not been read on the page itself.

| Item | Finding | Status | Source |
|---|---|---|---|
| Article types | Full papers (Original Articles), hot-topic fast-track communications, reviews (reviews: contact the office first). Hot Topic / rapid papers are about 3,000 words with at most 10 references. | Partly verified | https://www.rms.org.uk/library/journal-of-microscopy.html (fetched); snippet of the Wiley page |
| Article type chosen | **Original Article**. The Hot Topic type is a short explainer, not an empirical study, and caps references at 10 (we have 32). | Judgement | — |
| Length limit for Original Article | Not found | NOT verified | — |
| Abstract limit / structure | Not found. Recent papers (jmi.70039, jmi.70021, via Europe PMC) have single unstructured abstracts of about 100–150 words. Our abstract is 252 words. | NOT verified | Europe PMC PMC12523988, PMC12637018 |
| Lay description | Required: a non-technical summary of at most 400 words, alongside the abstract. Placement not confirmed. Ours is about 250 words. | Partly verified | snippet of the Wiley page |
| Keywords | Up to 6. We give 6. | Partly verified | snippet; jmi.70039 shows 4 |
| Reference style | Numbered in order of citation (superscripts in text); APA-like list with the journal name in full and a bare DOI at the end. Applied in the draft. | Verified from published papers | PMC12523988, PMC12637018 reference lists |
| Figure specs | Not found. Figures were made at 170 mm width, vector PDF plus 600-dpi PNG; check the required formats (TIFF/EPS?) on the page. | NOT verified | — |
| Data availability statement | Wiley requires one in all journals; J Microsc's policy tier (expects vs mandates) is unknown. A statement is included. | Partly verified | https://authorservices.wiley.com/author-resources/Journal-Authors/open-access/data-sharing-citation/faqs.html (search result) |
| Preprints | Articles previously posted as preprints are considered; the submitted version may be posted at any time. | Partly verified | snippet of the Wiley page |
| Open access | Hybrid journal. The snippet gives the APC as US$5,000 / £3,300 / €4,150, with CC BY, CC BY-NC and CC BY-NC-ND licences offered. | Hybrid verified; APC partly verified | RMS page; fundedaccess snippet |
| Purdue OA agreement | Under the BTAA–Wiley Open Publishing Agreement, corresponding authors at Purdue West Lafayette can publish OA in Wiley hybrid journals with no APC. Acceptance and licence must fall by 31 Dec 2027. Whether J Microsc is on the list, and whether Frank counts as eligible, is not confirmed. | Partly verified | https://btaa.org/library/open-scholarship/agreements/wiley-open-access-agreement ; https://blogs.lib.purdue.edu/news/2026/04/27/btaa-finalizes-2026-2027-open-access-publishing-agreement-with-wiley/ ; https://guides.lib.purdue.edu/c.php?g=1115699&p=8875585 |
| AI disclosure | Wiley's general policy: AI cannot be an author, and use must be disclosed. No J Microsc-specific text was retrieved. | NOT verified | — |
| Cover letter, title page, COI, running title | Not found. A running head, COI and funding statement were included anyway. | NOT verified | — |

**Frank: open the Wiley author-guidelines page in your own browser** and confirm the unverified rows, especially the abstract limit, the lay-description placement, the figure formats and the AI policy wording.

## 2. Word count (computed from manuscript.md)

- Title: 25 words. It may be too long; a shorter option is "Burned-in data bars and label-free triage of automatic correlative-microscopy registration: a pre-registered follow-up".
- Abstract: 252 words.
- Lay description: about 250 words.
- Main text, Introduction to Conclusions: about 4,310 words, including Table 1 (107 words).
- The request was 3–4k words. The easiest cuts are in Methods 2.3/2.4 (about 1,650 words): move the crop-row details, the component descriptions and the controls to Supporting Information.
- 32 references, 3 figures, 1 table.

## 3. Numbers checked against the files

Every number in the manuscript was taken from the following files:
- `results/arm1/arm1_report.md` and `arm1_summary.json`
- `results/arm2/arm2_report.md`, `arm2_summary.json` and `gate_result.json`
- `prereg/*.md`
- `tasks/todo.md`

A few numbers were recomputed by script from `arm1.csv` and `arm2/candidates.csv`: the per-configuration locks of 35/35/20/20; 10 of 35 locked runs succeeding after cropping; the identities of the 3 accepted Arm 2 failures; and the one success below the cut-off.

**No number in the task prompt disagreed with the files.** Notes:
- The CJSJ paper text rounds the cut-off to 0.171; the files and this draft use 0.1711.
- The prompt says "STEM banners". The benchmark labels these images TEM. The data bar (dwell, collection angle, DF4 detector) shows that they are STEM, and the draft says "STEM (labelled TEM in the benchmark)".
- Correction made while drafting: the accepted DefDAP failure in Arm 2 is the Ti-6Al-4V **pore** pair (349 px), not the GT-flagged bulk pair.

## 4. Open questions for Frank

1. **Title.** Is the long title acceptable? The short option is in Section 2 above.
2. **H-A3 deviation adjudication.** The prereg body says "fresh uncropped rows for all other held-out pairs". Addendum A (ae57397, committed before any Arm 1 run) says stored CJSJ rows for pairs that were not re-run. 42 of 119 held-out pairs used stored rows. The draft reports this as a deviation from the original wording (Section 2.6) with a `[FRANK: adjudicate]` tag. Pick one:
   - (a) Keep it as a deviation.
   - (b) Call it a pre-run clarification by Addendum A. That is defensible, because the addendum predates the runs, but the analysis script itself labelled it a deviation.
   Either way the McNemar result cannot change.
3. **Affiliation.** The CJSJ paper used Homestead High School. This draft uses Purdue, as instructed. Make sure the J Microsc affiliation and the CJSJ disclosure are consistent with each other.
4. **Repository public?** The draft cites `github.com/fronkt/correlative-microscopy-alignment`, branch `triage-ext`, as public pre-registration evidence. Confirm that the repo is public; the timestamps only count if it is. Consider a Zenodo release with a DOI.
5. **Funding line.** It says no funding, with GPU time paid personally (about $0.80 total: Arm 1 $0.50, Arm 2 about $0.30). Keep or remove the dollar figure.
6. **Suggested reviewers.** None are proposed yet. Decide about Durmaz, whose datasets and score are tested here.
7. **Self-citation of the zoom search.** `pyramid_v2` is cited to MAM-26-277 (ref 7), where it is called "checked search". Confirm that this is the right attribution; the CJSJ paper credited the Research Square preprint (10.21203/rs.3.rs-10311464/v1) instead.
8. **Earlier reference errors.** The reference check found wrong page numbers in the CJSJ and M&M lists:
   - LoFTR is 8918–8927, not 8922–8931.
   - MAGSAC++ is 1301–1309, not 1304–1312.
   Both are corrected here. Consider fixing them in MAM-26-277 at revision.
9. **Figure 1 licence.** It reproduces an AmalgaMatch image (CC BY 4.0) with attribution in the legend. Confirm you are happy with that.

## 5. Reference verification status

Most references were verified through the Crossref API, DataCite, the Zenodo API or doi.org on 2026-10-04 by a research subagent. The remaining flags are marked inline in the reference list:
- Ref 1 (Sci Data 2026): online, but Crossref has no volume or article number yet. Check the diacritic in "Pürstl".
- Ref 18 (Holm 1979): there is no Crossref DOI. The JSTOR stable number 4615733 is not confirmed.
- Ref 15 (Zuiderveld 1994): the DOI resolves, but the editor name (Heckbert) is not confirmed.
- Refs 23–25 and 29–30: creator lists are truncated with "et al.". Complete them from the Zenodo and NIST records.
- Ref 27 (DefDAP software): confirm the title string and creators on Zenodo (version DOI 10.5281/zenodo.10160238, concept DOI 10.5281/zenodo.3688096).
- Ref 28 (refodat.86): **[CITATION PENDING – Kleiner]**. DataCite gives Kleiner, F. (2026), "BSE and EBSD measurements of 7d hydrated alite", refodat, doi:10.71758/refodat.86, CC BY 4.0. DataCite lists the publisher as refodat (Thuringia), not Bauhaus-Universität Weimar.
- Refs 6 and 7 are Frank's own works, cited as "submitted" and "under review"; they need no DOI.

## 6. Pre-submission checklist

- [ ] **Kleiner citation.** Get refodat.86's preferred citation from F. Kleiner and replace ref 28 and the `[CITATION PENDING – Kleiner]` tag.
- [ ] **H-A3 deviation.** Adjudicate (question 2) and remove the `[FRANK: adjudicate]` tag in Section 2.6.
- [ ] **AI statement.** Write `[FRANK WRITES: AI-use statement]` in the manuscript, and fix the matching sentence in the cover letter.
- [ ] **Purdue Wiley OA eligibility.** Check with Purdue Libraries (guides.lib.purdue.edu, Wiley hybrid page) that J Microsc is covered by the BTAA–Wiley agreement and that you qualify as a Purdue corresponding author. If so, choose OA (CC BY) at acceptance.
- [ ] **arXiv preprint decision.** Preprints appear to be allowed (partly verified). Decide whether to post before submission. If you post, add the identifier to the cover letter and check that CJSJ's policy allows a preprint of the follow-up.
- [ ] Confirm the unverified guideline rows in Section 1 on the live Wiley page: abstract limit, lay-description placement, figure formats, AI wording.
- [ ] Complete the flagged reference metadata (Section 5).
- [ ] Make the repository public, or archive it on Zenodo, and update Data availability.
- [ ] Convert to .docx (journal-submission skill) once the text is final, and export the figures in the required format.
- [ ] Decide on the title and the funding sentence; add suggested reviewers if wanted.
- [ ] Final check that no CJSJ 187-pair result is presented as new. The intro uses only the cut-off, the 42 false accepts and the fact that S1 separated successes; all are attributed to ref 6.
