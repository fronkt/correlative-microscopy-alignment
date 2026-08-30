# Lessons (project-internal)

## Two agent sessions on one vast box will fight over disk

A second Claude session (symmc-flow) shared the 16G box at 142.171.48.138
and our 9.3G AmalgaMatch extraction was deleted mid-sweep (z3 died at
153/187 pairs; c50 never started). The other session had installed a 7.5G
system-python torch stack; combined footprints exceeded the disk, and the
dataset was the biggest evictable thing. Rules: (1) on any "new" box, run
`ls -lat /root` and `df -h /` FIRST and look for foreign project dirs;
(2) before a multi-GB download, confirm headroom for BOTH tenants;
(3) treat sudden FileNotFoundError mid-sweep on a shared box as eviction,
not as a code bug — check `ls data/` before debugging the loader.

## PowerShell strips inner double quotes from ssh remote commands

`ssh host 'nohup bash -c "cmd1 && cmd2" &'` from PowerShell delivers the
remote string WITHOUT the inner double quotes, so `bash -c` takes only the
first word as its script and the rest runs as separate shell syntax — cmd1
silently becomes a no-op (this cost us a missing MA-pyramid pass; it also
broke an earlier python -c). Rules: (1) one command per ssh call needs no
inner quoting — prefer that; (2) anything compound goes in a script file,
scp'd over, `sed -i 's/\r$//'`, then `bash file`; (3) never nest double
quotes inside an ssh arg from PowerShell.

## pkill -f from an ssh one-liner kills its own shell

The remote `bash -c` cmdline contains the pattern text, so
`pkill -f run_baselines_A` matches it and drops the connection (exit 255)
even when the target was also killed. Split the pattern in the source
string (`pkill -f "run_base""lines_A"`) and treat exit 255 after a pkill
as expected.

## Runner hangs at exit after GPU backbones

The sweep process reliably wedges after writing its last CSV row (teardown
of CUDA/transformers threads), surviving indefinitely. Don't wait for a
clean exit: treat "all expected rows present" as done, then pkill.

## Unanchored .gitignore `data/` silently untracked src/cma/data/

Same trap as the tar lesson below, git flavor: a bare `data/` in
.gitignore matches every directory named `data` at any depth, so
`src/cma/data/` was never committed and the public repo shipped without
the entire data module for two commits. Anchor project-artifact ignores
to the root (`/data/`) and, after any .gitignore edit, verify with
`git status --ignored --short src` that nothing under src/ is ignored.

## Windows MAX_PATH bites real datasets

Several AmalgaMatch file paths exceed 260 chars; `Path.exists()` /
`cv2.imread` silently fail with LongPathsEnabled=0. Loader-level fix:
`\\?\`-prefixed absolute paths for all IO + `cv2.imdecode` on bytes
instead of `cv2.imread`. Don't require users to flip the registry.

## Tar exclude pattern matches by basename

`tar --exclude=data` matches **every** path component named `data`, not
just a top-level `./data/`. This silently killed `src/cma/data/` during
the first transport to the GPU box. If you need a top-level-only
exclusion use `--exclude='./data'` (with the leading `./`).

## PowerShell -> SSH -> bash -> python quoting

Inline Python via `ssh ... "python -c '...'"` from PowerShell is
quoting hell. PowerShell, bash, and Python all want their own quoting
rules and the dollar sign means different things to each. **Default to
writing the script as a `scripts/*.py` file, scp it once, then run by
path.** Reserve inline `-c` for one-liners with no quotes or vars.

## Same-modality sweeps don't test the research hypothesis

`synthesize_pair` builds same-modality textured pairs. SIFT already
saturates on those, so any "pyramid wrapper helps" claim measured on
those pairs is meaningless. The hypothesis is about cross-modal contrast
mismatch — `synthesize_cross_modal_pair` is the minimum viable proxy
until AmalgaMatch lands. **Don't waste GPU on same-modality sweeps
unless you're sanity-checking a new backbone.**

## kornia LoFTR is the "naive foundational baseline," not the hero

kornia ships pretrained outdoor/indoor LoFTR weights — convenient for
plumbing tests, but those weights were trained on natural-image pairs
and are NOT robust to contrast inversion or modality flips. The cross-
modal sweep numbers (errors in 100s of px) confirm this. **Don't pitch
kornia-LoFTR as "the foundational backbone for AmalgaMatch."** The
backbone that earns its keep is MatchAnything (designed cross-modal) or
a microscopy-fine-tuned model.

## MatchAnything via transformers — wrapper validated, but cannot be benchmarked on this project's synthetic harness

Vendoring path: `pip install transformers torchvision` + 
`AutoModelForKeypointMatching.from_pretrained("zju-community/matchanything_eloftr")`. 
No upstream zju3dv repo clone needed. Apache-2.0 weights.

Wrapper correctness verified two ways:
- self-match on 256x256 image: 9991 dense pairs at sub-pixel accuracy
- model-card example pair (US Capitol): 714 matches, confidence
  median 0.42 / 95th 0.88, keypoints span both images correctly

But on this project's `synthesize_pair` outputs — both layered noise AND
`skimage.data.astronaut` — MatchAnything produces ~3% inlier rate and
10s-100s of px RANSAC error, while LoFTR gets sub-pixel on the same
inputs. MatchAnything's training priors are tuned for multi-view scene
pairs (3D structure, viewpoint changes, real cross-modality), not
single-image homographic warps of generic textures.

**Therefore: do not benchmark MatchAnything on `synthesize_pair`.** The
H1 test for MatchAnything requires either AmalgaMatch or a multi-view
natural-image dataset (MegaDepth / ScanNet pairs). The wrapper is
production-quality and will work the moment real correlative pairs land.

## Numbers written into a manuscript must be re-derived from source (2026-08-24)

The Scientific Reports revision surfaced three failure modes worth keeping.

**1. A metric choice can manufacture significance.** Accuracy was reported after
thin-plate-spline refinement. Both native-pair headlines were significant on that
metric and null on raw matcher error (p = 0.034 -> 1.00 and 0.035 -> 0.33), because
refinement converts 13 otherwise-successful fits into failures, and its coverage is
non-uniform across configurations (1.000 for the dense RoMa family, 0.000 for
Control B). *Therefore: whenever a pipeline has a post-processing stage, report the
headline under both with and without it before believing either.*

**2. The declared statistical protocol was not the one implemented.** Methods said
"two-sided bootstrap probabilities"; every p-value was a one-sided tail mass. The
code was honest (`bootstrap_ci.py` printed "p(one-sided)"), but nobody reconciled the
two for months. One conclusion flipped when corrected. *Therefore: grep the Methods
claims against the code that produces them, as a checklist, before submitting.*

**3. Agent-supplied numbers are not verified numbers.** During this revision an agent
supplied 2x2 contrast p-values that were written into the Discussion. Re-running the
committed script showed two of them wrong (0.040 -> 0.052, 0.045 -> 0.049), and one
crossed 0.05. A follow-up forensic audit of *every* number in the manuscript then found
17 more errors, including a fabricated causal mechanism, a sentence quoting the wrong
backbone's numbers, and two claims that contradicted each other across sections.
*Therefore: no number goes into a deliverable until it has been recomputed from the
source data in this session. Delegation is fine for finding numbers; it is not
sufficient for publishing them.*

**Corollary that bit twice:** a results CSV is not append-only-safe. Adding
`ma_roma_ft` rows to `baselines_A.csv` silently (a) would have put train-contaminated
configurations into regenerated figures, and (b) broke the H3 readout so the published
69 % affine figure no longer regenerated. *Therefore: any script that aggregates over
"all backbones" needs an explicit exclusion list for models trained on the benchmark.*

## Changing the primary metric invalidates prose, not just tables (2026-08-24)

Moving the correlative-microscopy paper from the TPS-refined error to the
unrefined error for the TMLR submission required recomputing the tables, which
was obvious, and recomputing **every sentence that quotes a number**, which was
not. Five claims survived the table rewrite and were still wrong: an ablation
whose significance existed only under the old metric, a per-pair narrative
("two gained, two lost") that had a different shape under the new one, a
comparison that reversed sign, and two statements about a match cap.

*Therefore: when the primary metric changes, treat every numeric claim in the
prose as unverified, including ones that were correct in the previous version.
The table is the easy part.*

**What caught them:** an explicit list of carried-over claims, checked one at a
time against the source data, rather than a read-through. A read-through would
have passed all five, because each was internally plausible and each had been
true under the old metric.

**Corollary worth keeping:** the verification is now a script
(`scripts/verify_tmlr_draft.py`) that asserts 32 specific values appear, that 12
retired phrasings do not, and that two deliberately-retained withdrawn claims
still sit inside their retraction. A phrase-level ban list produced two false
positives on the first run -- both were retractions naming the old claim -- which
is itself the signal that the ban list needed to encode *context*, not just
presence.

---

## A schematic's proportions are claims, and claims get asserted (2026-08-30)

Fig. 1's field-of-view ladder drew its rungs at `sqrt(r / 0.5)` of a base size —
relative to the 0.5 rung — while the caption said the ratios were relative to the
source. The 0.25 rung therefore sat at half the outer square's area, and the
figure was right only because the outer square happened to also be 0.5. The
source field, the actual denominator, was never drawn at all.

Nobody catches that by reading the code; it reads fine. It is caught by writing
the identity down as an assertion next to the constant:

```python
_sides = [SRC_SIDE_IN * np.sqrt(r) for r in FOV_RATIOS]
for _r, _s in zip(FOV_RATIOS, _sides):
    assert abs((_s / SRC_SIDE_IN) ** 2 - _r) < 1e-12
```

**Rule:** every proportion a schematic asserts gets a named constant, its source
in a comment, and an assertion that the drawn geometry produces it. If a caption
states a ratio, the thing it is a ratio *of* has to be on the page.

## Alt text drifts from the figure, silently

The Fig. 1 alt text described "a wide-field micrograph with successively smaller
crop boxes drawn on it". No micrograph was ever in that panel — it was always an
abstract nest of rectangles. The alt text was written from the *intent* and never
re-checked against the render. The word-count gate, the citation gate and the
section-order gate all passed over it, because none of them look at the figure.

**Rule:** re-read alt text against the rendered PNG, not against the legend. It
is prose about an image, so it is the one part of a manuscript that no text gate
can check.

## A house style that lives in one figure is not a house style

Fig. 1 set `pdf.fonttype=42` via the skill's palette. `plot_baselines.py` and
`plot_fov_ladder.py` did not, so every vector export of Figs 2-5 carried Type 3
fonts — the single most common reason a publisher bounces artwork. It went
unnoticed for the whole project because those scripts only ever wrote PNGs, where
the setting has no effect; it surfaced the moment PDFs were needed.

**Rule:** publisher rcParams belong to every figure script in the repo, not to
the one that happens to import the shared palette. And a compliance check has to
run over the whole figure package, not the figure being worked on.

## verify.py silently drops any --expect containing a colon

`--expect "verifier: mutual"` is parsed as stem `verifier`, text ` mutual`, and
since no figure is named `verifier` the check is skipped — reported as neither
pass nor fail, just absent from the output. The PASS looked complete. Only
counting the printed lines against the number of `--expect` flags revealed it.

**Rule:** when a checker reports per-item results, count them. A gate that can
silently skip an item is a gate that can pass an unchecked figure.
