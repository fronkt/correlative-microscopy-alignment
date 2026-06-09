# Lessons (project-internal)

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
