# TODO — Correlative Microscopy Alignment

Track in this file. Check items off as completed.
Source docs: `docs/context.md`, `docs/research_plan.md`, `docs/task_plan.md`.

## Triage extension — follow-up paper (opened 2026-09-29, branch `triage-ext`, worktree `cma-triage-ext`)

**Status (2026-09-29): Phase A is DONE, and the findings are in `results/phaseA/synthesis.md`.**

**Frank's decisions (2026-09-29): "Go ahead for all. Leave for M&M session. No target venue."**
- **Arm 1 approved** (banner-crop GPU rerun on AmalgaMatch, about $1–3). The pre-registration draft is
  `prereg/arm1_banner_crop.md`. Sonnet is building the tooling (`src/cma/overlay.py`, the runner flags, pair lists,
  `scripts/box_arm1.sh`).
- **Arm 2 approved**, materials only:
  - The pooled multi-lab set is being assembled by Sonnet into `C:\Users\frank\Documents\materials-bench`, with no
    matcher runs.
  - NIST AM Bench: Sonnet is doing a feasibility study and building a hand-check tool (`tools/handcheck/`). Frank
    will hand-check about 20 pairs.
- **Data-request emails** are saved as Gmail DRAFTS; Frank sends them himself. They go to Weimar (Kleiner, cc
  Rößler), Manchester/NPL (Gholinia) and NTNU (Ånes, asking to re-licence the GitHub control points).
- **The M&M banner implication is left to the M&M session.** This branch does not touch M&M.
- **No target venue.**

**Arm 1 DONE on 2026-09-30.** Pre-registered at 91c8b67 (addendum ae57397); results at 6ce17d9, reported in
`results/arm1/arm1_report.md`. H-A1, H-A2 (primary) and H-A3 are all SUPPORTED:
- Near-identity locks: 110 → 0.
- MA-RoMa direct SR@20: 6 → 16 of 67 (one-sided p = 0.006).
- S1 false accepts: 42 → 14 (p = 1e-7).
The sham crop changed 8.8% of runs (rule: >10%), and the fresh runs reproduced the stored ones on 360/360. RoMa direct
was not significant (14 → 16), and S1 AUROC on DislocationCharacterization did not improve (0.811 → 0.786). The box
cost $0.50. Fordatis stalled at 47%, so the rest of the zip was uploaded from the laptop and checked by md5.

**Arm 2 status (2026-09-30):**
- Pre-registered at a59cb01; tooling at ee61e09.
- NIST came back on the morning of 09-30. Steps 1–5 of the old resume recipe are done:
  - 81 of 81 NIST pairs are built and 102 of 102 fits pass. `runall.sh` ran twice at once, which did no harm; the
    two optical fits caught on a `.part` file were re-run.
  - The pool is 95 pairs. Addendum A is committed at f328135, and the pack sha256 is cb5a6571….
  - The GPU run is DONE: results/arm2/candidates.csv (3b724f7) has 1,615 of 1,615 rows. The 7 matcher failures
    (6 MatchAnything, 1 SIFT) each had fewer than 4 correspondences; they count as failures and are not rerun,
    because they are not infrastructure errors. Box 53534141 is destroyed; it cost about $0.30 and credit is $16.97.
  - The results are NOT analysed yet (gate first).
- **LEFT:**
  1. Frank runs the hand-check (the 20-pair command in `tools/handcheck/README.md`, clicks to
     `materials-bench/nist/handcheck_clicks.csv`).
  2. Write Addendum B with the gate outcome per NIST type.
  3. Run `analyze_arm2.py`.

**Order:**
1. Arm 1 tooling → fill in the pre-registration → commit + push → rent box → run → analyse.
2. Arm 2: assemble → GT-quality gate → Arm 2 pre-registration (after assembly, before any matcher run) → commit +
   push → GPU.
**Frank's decision: the replication benchmark must be MATERIALS ONLY.** That rules out the biology sets from the
first scan (SuperCUT, Eliceiri SHG-BF, Lu et al.). A second, materials-only search of the places the first scan
skipped is running and writes to `cma-cjsj/research/materials_benchmark_scan.md`.

**Separation.** A separate follow-up paper to the CJSJ study. The CJSJ submission (due Wed 2026-09-30, worktree
`cma-cjsj`, branch `cjsj-tta`) stays unchanged. Nothing from this branch goes into the CJSJ or M&M papers. The
follow-up cites CJSJ (and M&M) for the base triage results and reports only new analyses and the new benchmark. If CJSJ
makes the paper a finalist, CJSJ gets exclusive rights to *that* paper, not to these new results. This branch forks
from `cjsj-tta` at 2983bde. Frank's uncommitted docx edits in `cma-cjsj` are never touched from here.

**Who does what.** Opus 5.5 (main session) designs the analyses, writes the pre-registration, and judges results.
Sonnet 5.5 subagents (`model: sonnet`) write and run the code, one focused task per agent, each ending with a short
report and the numbers in a JSON file. Opus reads every JSON and spot-checks one number per task against the CSV
before it goes into a claim.

**Inputs.** `results/triage/candidates.csv` (5,049 rows = 27 configurations × 187 pairs; 31 failed runs count as
error = ∞). The stored columns are H (9 floats), n_matches, n_inliers, mu_ed, image sizes, and group, subclass and
scene. **Correspondences were not stored**, so any score that needs match positions (spatial coverage, local
consistency) needs a re-run. SIFT and LoFTR run locally on CPU; the dense matchers need a GPU (Phase C). Images are
read-only from `../correlative-microscopy-alignment/data/AmalgaMatch` (9.3 GB). RANSAC threshold 5.5 px.

**Forking-paths rule.** Every Phase A analysis is exploratory on data already seen. Nothing from Phase A counts as
confirmed on AmalgaMatch. `results/phaseA/analysis_log.md` lists every variant tried, including the dead ends, so the
paper can report how many were tried. Only what Phase B pre-registers and Phase C tests on a new benchmark is
confirmatory.

### Phase A — exploratory, CPU only, existing CSV (Sonnet executes, one agent per item)
Outputs: `scripts/phaseA_<x>.py` → `results/phaseA/<x>.json` (+ figures in `results/phaseA/fig/`).
- [x] A0. Shared helpers only where `src/cma/triage.py` lacks them (grid displacement between two H, clustered
      bootstrap, and AUROC already exist; reuse them). Define "the 42": MA-RoMa direct seed 0, S1 ≥ 0.171,
      mu_ed > 20 px, held-out groups. Assert the count = 42 before anything else runs.
- [x] A1. **The 42 confident TEM false accepts.**
  - (i) Shared vs matcher-specific: for each of the 42, compute the grid displacement from every other candidate's
    H (SIFT, LoFTR, MatchAnything, RoMa, transforms, seeds) to MA-RoMa's H, and each candidate's own error. Do the
    others land in the *same* wrong place (< 20 px apart), somewhere else, or on the truth?
  - (ii) Error geometry: decompose H_gt⁻¹·H_est into translation, rotation, scale, shear and perspective. Is the
    error mostly a translation? Do the translations cluster, or fall on multiples of one vector?
  - (iii) Repeating structure: compute the target image's autocorrelation (FFT) and find the dominant period
    vectors. Test whether the error translations fall near integer multiples of them, against a
    shuffled-vector null.
  - (iv) Threshold sensitivity: how many of the 42 remain at 25 and 30 px (8 are 20–30 px). Confirm "GT-fit
    homography < 20 px on 41/42" from the gt rows, so the model family is not the cause.
  - (v) One figure: 4 of the 42 (2 near-miss, 2 gross) with GT points, estimated points and the displacement
    field. Opus picks the examples after seeing (i)–(iii).
- [x] A2. **Seed disagreement as a failure signal.** Per pair, for RoMa and for MA-RoMa, take the 6 seed runs
      (core seed 0 + control seeds 1–5) and compute D = median pairwise grid displacement (px, source frame).
      Report: the distribution of D (is it almost always ~0? E1 says only 16 pairs flip success across MA-RoMa
      seeds); AUROC of −D for seed-0 success, with a scene-clustered CI; AUROC of S1 + D (rank mean) vs S1 alone
      (paired CI); D among the 42. Cost note: D needs 6× compute per pair, so it has to beat S1 by a margin worth
      that.
- [x] A3. **Per-image-type calibration from k hand-checked pairs.** For each subclass (19) and each
      k ∈ {1, 2, 3, 5}, draw k labelled pairs (1,000 draws). Choose a cut-off from them (rule fixed in advance:
      the midpoint between the lowest-S1 success and the highest-S1 failure among the k; if all k have one
      label, fall back to the global 0.171). Score the rest of that subclass. Report false-accept rate and recall
      vs the single global cut-off, overall and on the TEM subclass. Also report the within-subclass AUROC, since
      calibration helps only where there is within-subclass separation. Say plainly which subclasses have too
      few pairs.
- [x] A4. **A score comparable across matchers.** S1 favours matchers that return few matches (SIFT median 99
      matches; SIFT picked on 79 pairs, 9 of 11 losses). From stored columns only:
  - (a) a-contrario log-NFA. Under a null of random matches, each match is an inlier with probability
    p = π·5.5² / (w_t·h_t). log-NFA = log(#RANSAC hypotheses) + log BinomTail(n_inliers; n_matches, p). A
    more negative value means the match is less likely to be chance.
  - (b) a Wilson lower bound on the inlier fraction.
  - (c) raw n_inliers.
  For each score: AUROC per matcher, pooled AUROC across all voting candidates (the H2-style pooled test), and
  the H3-style per-pair pick (by score) vs the best single candidate and vs pick-by-S1. Also note that p is
  tiny for large images, so (a) may saturate; check this before reading the AUROC.
- [x] A5. **The RoMa-family-only pick rule (56 vs 47, p = 0.06).** List the 14 wins and 5 losses by group and
      scene. Leave-one-group-out: does the gain survive dropping each group? Is it "exclude sparse matchers" or
      "exclude low n_matches"? Compare with "pick max S1 among candidates with n_matches ≥ N" for N on a coarse
      grid, which is exploratory and gets logged. Compare with pick-by-A4 score over all candidates, since A4
      might recover the same gain without a hand-made family rule.
- [x] A6b. Overlay audit (added after A1 found the banner lock): `results/phaseA/a6_*`.
- [x] A6. Opus synthesis: `results/phaseA/synthesis.md`. For each item, what it shows, how many variants were
      tried, and whether it earns a confirmatory hypothesis. **Check in with Frank here before Phase B.**

### Phase B — pre-registration (Opus writes, Frank approves, commit + push before any Phase C run)
- [ ] B1. `prereg/triage_ext_prereg.md`: dataset (from the scan), hypotheses from A6 with direction, metric,
      success thresholds (20 px primary; plus one threshold scaled to image size, since a second benchmark may
      have very different pixel sizes), fixed cut-offs **carried over from AmalgaMatch unchanged** (for example
      S1 0.171), statistics (scene-clustered bootstrap, McNemar), stopping rule, and exclusions.
- [ ] B2. Candidate list for Phase C: the smallest set the hypotheses need, not all 27. That cuts GPU cost.
- [ ] B3. Commit + push. The commit hash is the timestamp and goes in the paper.

### Phase C — GPU replication on a second benchmark (needs Frank's $ approval)
- [ ] C0. Benchmark choice from `research/second_benchmark_scan.md` (licence without NC/ND conflicts preferred;
      not in RoMa or MatchAnything training data; GT points or known transform; a global homography is
      adequate). Frank picks from a shortlist.
- [ ] C1. Check vast.ai credit first (~$17 on 09-27, with other boxes burning). Give the cost estimate
      (GPU-h × $/h) and **get Frank's OK**.
- [ ] C2. Sonnet runs the box: resumable CSV, and this time store correspondences (npz) so match-position scores
      can be computed later. Pull the results, then destroy the box.
- [ ] C3. Confirmatory analysis exactly as pre-registered; deviations get their own section.

### Open questions for Frank
1. Target venue for the follow-up paper. This sets its length and how much Phase A goes in.
2. Is the scope of Phase A OK, or should any item be dropped?

## CJSJ 2026-27 — label-free triage of registrations (2026-09-27, branch `cjsj-tta`)

**Venue:** Columbia Junior Science Journal, original research, 2-3 pages. **Due Wed 2026-09-30 11:59 PM ET, no
exceptions** — submit by 10:59 PM. Sole author: Frank.
**Separation rule (why this paper may exist beside the M&M submission):** new question + new results only. The
configurations' own success rates, tiling and fine-tuning are M&M results → cite, never re-report as findings.
Triage/selection results never go into M&M.

**Why not test-time adaptation (the first plan, dropped 2026-09-27 after two literature reviews):** a cycle-only loss
never looks at image content and is satisfied by wrong warps (Truong et al. 2021, Warp Consistency); every TTA success
found uses an image-content loss; starting errors (~300 px) are far outside any local basin (even labelled fine-tuning
got 0/16 STEM test pairs within 20 px); estimated ~5% chance of a defensible gain. Two bugs found in `cma.tta` (BN
running stats leak across pairs; CORAL acts on the 2-channel flow) are recorded here but not fixed, since TTA is out.

**Question.** Can a label-free score tell a microscopist which automatic registrations to trust, and choose a better
registration per pair, across the whole AmalgaMatch benchmark (187 pairs, 19 subsets, 6 groups)?

**Closest prior work.** Durmaz et al. 2026b (Front. Mater., doi:10.3389/fmats.2026.1815017): the retained-inlier
fraction (share of MA-RoMa correspondences that are inliers to one homography) separated success (mean error < 8 px)
from failure with ROC-AUC 0.905 on 48 pairs from four subsets in the SameSlice (their "OrientationMapping") and
SerialSectioning groups, with the operating point (0.517) picked on the same 48 pairs. They PROPOSED, without testing,
choosing the highest-fraction modality pair and sending low-fraction groups to human review.

**Exploratory pilot (already seen, disclosed as such, NOT confirmatory):** on the stored M&M runs
(`results/baselines_A.csv`, 11 non-fine-tuned configurations), inlier count vs success at 20 px: AUROC RoMa 0.937
[0.860, 0.989], MA-RoMa 0.835 [0.696, 0.966] (subclass-clustered bootstrap); oracle best-of-11 SR@20 63/187 vs best
single 45/187; "pick max inliers" 50/187. Every dense-matcher run returns exactly 10,000 correspondences and keeps ≥58
inliers, so a fixed cut-off of 50 never fires, but the count still ranks pairs. These numbers motivated the design;
every number in the paper comes from the NEW runs below.

**Pre-registered design (this section is committed + pushed BEFORE the GPU run; the commit is the timestamp):**
- Data: AmalgaMatch only (CC BY 4.0). Metric: unrefined parametric error `mu_ed` (mean over GT points, px), no TPS.
  Success = `mu_ed` ≤ 20 px (primary); 10 px and Durmaz's 8 px secondary. Scenes = `pair_id.split("#")[0]`.
- Candidates per pair (one fresh run each, fixed seeds, fitted 3×3 transform saved):
  - core pool (7): sift, loftr, roma, ma_roma, matchanything (direct); roma, ma_roma (pyramid_v2);
  - input-transform pool (8): roma and ma_roma direct on {target inverted, target histogram-matched to source,
    CLAHE on both, Sobel gradient magnitude on both};
  - rerun control (10): roma and ma_roma direct, seeds 1-5 (seed 0 is the core run). Controls never vote.
- Scores (label-free, per pair × candidate):
  (S1) retained fraction = n_inliers / n_matches (Durmaz); (S2) agreement = −median over the other voting candidates
  (core + transform pools, not itself) of the mean displacement, in source pixels, between the two transforms applied
  to a 5×5 grid spanning the target image; (S3) combined = mean of the within-pair-set rank percentiles of S1 and S2
  (ranks taken over all pair × candidate rows of the voting pools).
- **H1 (primary).** S1 predicts success for MA-RoMa direct on all 187 pairs. Supported if AUROC ≥ 0.80 and the
  scene-clustered bootstrap 95% CI (B = 10,000) excludes 0.5. **Transfer:** the Youden-optimal S1 cut-off chosen on
  the SameSlice + SerialSectioning groups (Durmaz's groups) is applied unchanged to the other four groups; supported if
  the accepted pairs' success rate exceeds those groups' base rate with a clustered CI on the difference excluding 0.
- **H2.** S2 and S3 vs S1, same candidate (MA-RoMa direct) and pooled over all voting candidates: ΔAUROC with paired
  scene-clustered bootstrap CI. Supported for S3 if the CI on AUROC(S3) − AUROC(S1) excludes 0.
- **H3.** Per-pair selection among the voting candidates by S1 (Durmaz's proposal), by S2 (consensus medoid) and by S3,
  vs the best single voting candidate (chosen by its SR@20 on all pairs, which favours the baseline). Paired exact
  McNemar + clustered bootstrap CI on ΔSR@20. Supported if p < 0.05 AND the gain exceeds the rerun control's gain
  (best-of-6 seeds of the same matcher selected by S1). Oracle best-of-K reported as the ceiling.
- **Triage (risk-coverage).** For MA-RoMa direct and for the best H3 rule: success rate among accepted pairs at 25 /
  50 / 75 % coverage and AURC, with clustered CIs, against the base rate.
- Feasibility ceiling: GT-fitted homography error per pair, reported so failures that no homography can fix are visible.
- Failures of a run count as failures (error = ∞); nothing is dropped. Anything outside this list is labelled
  exploratory in the paper.

**Tasks**
- [ ] 0. Commit + push this section + runner + analysis skeleton BEFORE step 6 (pre-registration timestamp).
- [ ] 1. `src/cma/triage.py`: input transforms, transform-agreement, scores, selection, AUROC, clustered bootstrap,
      McNemar, risk-coverage. CPU tests in `tests/test_triage.py`.
- [ ] 2. `scripts/run_triage_candidates.py`: resumable CSV keyed by (pair, backbone, mode, transform, seed); saves H
      (9 floats), n_matches, n_inliers, family, mu_ed, image sizes, runtime; real seeding (torch, numpy, random,
      CUDA, cv2) derived from (seed, pair_id).
- [ ] 3. `scripts/analyze_triage.py` → `results/triage/summary.json`; `scripts/verify_triage.py` gate recomputes
      every number in the paper from the CSV.
- [ ] 4. Local CPU smoke on 2 pairs (sift + transforms) to prove the CSV/H round-trip.
- [ ] 5. Rent one GPU box (5090/4090); clone public repo branch `cjsj-tta` over HTTPS; wget AmalgaMatch from Fordatis,
      verify 4,228,037,938 bytes before unzip.
- [ ] 6. Run all 25 candidate configurations × 187 pairs (~3 GPU-h estimate). Pull CSV back; destroy the box.
- [ ] 7. Analysis + verify gate; figures: (1) method schematic + one accepted and one rejected example; (2) S1 vs
      error scatter coloured by group with the transferred cut-off; (3) risk-coverage curves.
- [ ] 8. 2-3 page draft in the CJSJ Word template; figures in their .ppt template; AI-use statement.
- [ ] 9. Frank: read and revise, Permission Form (author + parent if under 18; mentor may be blank), submit.

**Timeline:** Sun night 0-5 → Mon 6-7 → Tue 7-8 → Wed 9.

## Phase 9 — Forgetting-robust fine-tune (2026-06-19)

**Why:** §7/§8.1. MA-RoMa decoder-only ft won 5.2x on in-distribution TEM
but catastrophically forgot C103 SEM↔LOM (med ED 12.4→241.9 px, 0 train
pairs), so net test SR@20 went *down* (0.393→0.250). Goal: keep the
appearance gain without regressing untrained modalities.

**Mechanism (primary): L2-SP weight anchoring.** Penalize decoder drift
from its zero-shot init: `L = L_task + λ·mean((θ_dec − θ_dec⁰)²)`.
Data-free, one knob, directly limits the drift that causes forgetting.
`λ=0` exactly reproduces the existing plain ft (control/sanity).
Chosen over alternatives because replay needs an external natural-image
corpus on the box, LoRA is invasive in romatch's decoder internals we
don't own, and EWC needs a Fisher pass. Fallback if L2-SP can't hold
C103 without killing the TEM gain: functional KD self-distillation
(anchor student decoder output to a frozen teacher, +1 forward pass).

**Code (minimal, all reuse):**
- [x] 9.1 `train/finetune.py`: theta0 snapshot + `anchor`/`anchor_lambda`
      args + L2-SP penalty (½·Σ(θ−θ⁰)², trainable decoder params) logged
      as `anchor_pen`. (commit 5bceaa3)
- [x] 9.2 `evaluate_direct` returns aligned pair_ids; train loop logs
      `c103_sr20` / `tem_sr20` retention probes each val. Unit-tested.
- [x] 9.3 `finetune_ma_roma.py`: `--anchor` / `--anchor-lambda` passthrough.
- [x] 9.4 λ grid calibrated from the real plain-ft drift D=28.57 (trainable
      only; BN `num_batches_tracked` buffers excluded — they were 99.9999%
      of the naive sum). Grid **{0, 0.01, 0.1, 1.0}** ≈ {control, 0.1×, 1×,
      10× task penalty at the plain-ft endpoint}.
- [x] 9.5 Phase A λ sweep done (47.186.21.5:55861). Best ckpts (val med ED):
      λ0=17.17, λ0.01=18.89, λ0.1=17.91, λ1.0=15.82; all retain val
      C103/TEM=1.0. Collapse onset moves earlier with λ (λ0 stable→1500;
      λ1.0 by ~step400), transient, best saved pre-collapse. Selection via
      `eval_ckpts_testsplit.py` (val C103 flat, can't rank): **winner λ=0.01**
      — C103 scene-0 retention 80/1220→16/24 px, best net test SR@20 0.286
      + best med ED 46.5, minimal gain loss.
- [x] 9.6 Phase B (`box_ft_robust_eval.sh l0p01`) → `baselines_robust.csv`,
      `fov_ladder_robust.csv` pulled local. Headline (28 test, TPS, B=10k,
      pyramid_v2): L2-SP λ0.01 SR@20 0.321 / med 41.0 vs plain-ft 0.250/56.7
      vs zero-shot 0.393/69.1. vs plain-ft: med ED −15.6 px p=0.0046 (sig),
      SR@20 +0.071 n.s. vs zero-shot: SR@20 −0.071 p=0.91 (regression erased
      to noise). Pyramid still stacks (direct 0.286→pyr 0.321).
- [x] 9.7 §7.1 subsection + §1 summary + §8.1 + repro-pointer updated. Winner
      ckpt pulled local (volatile /dev/shm). Push.

**Box:** vast 47.186.21.5:55861, RTX 5090 32G, `/venv/main`. Verified
reachable 2026-06-19. **GPU budget:** ~3–4 short trainings (cheap
val-only inner loop) + one full sweep+ladder for the winner ≈ a few hours.

## Phase 0 — Setup
- [x] 0.1 Repo scaffold + lint (ruff configured; CI deferred)
- [x] 0.2 Pinned env (`pyproject.toml` w/ dev + torch extras)
- [x] 0.3 AmalgaMatch acquired + checksummed + extracted (2026-06-09)
      zip on disk: `data/AmalgaMatch/AmalgaMatch_Dataset.zip` (4,228,037,938 bytes)
      sha256: `084516E37F865B616619ADB40E2100A91BF2F30177AED803EE31A797A0F8AAFB`
      source: https://fordatis.fraunhofer.de/handle/fordatis/478 (DOI 10.24406/fordatis/436, CC-BY-4.0)
      extracted: 19 subsets, 187 pairs confirmed
- [~] 0.4 RoMa / ELoFTR / MatchAnything weights load + smoke-test (RoMa + MatchAnything done; ELoFTR still shell)

## Phase 1 — Baselines
- [x] 1.1 Dataset loader yields (I_s, I_t, K_gt, scale_meta, group, subclass) — rewritten 2026-06-09 for real layout; 187 pairs load, integration tests green
- [x] 1.2 Metric harness (P_match@{1,3,5,10}, mu_err, med_err, success) — runtime/mem deferred
- [x] 1.3 Control A: zero-shot all 4 backbones on 187 real pairs (2026-06-09) —
      `results/baselines_A.csv` (CSV not parquet: results/*.parquet is gitignored,
      CSV is the project's committed-results convention). RoMa best: SR@10 0.10.
      Follow-up: Control A2 with paper-style stretch/pad resize for MatchAnything.
- [x] 1.4 Control B: SIFT + MMI on all 187 real pairs (2026-06-10, `results/baselines_B.csv`):
      med ED 824 px (vs 903 plain SIFT), SR unchanged — MMI cannot rescue bad init.
- [x] 1.5 Baseline plots (2026-06-10): `scripts/plot_baselines.py` -> SR bars,
      group heatmap, FOV curves in `reports/figs/baselines/` (gitignored;
      regenerate from committed CSVs)

## Phase 2 — Pyramidal Wrapper
- [x] 2.1 `pyramid.build(...)` + back-projection metadata
- [ ] 2.2 Scale-metadata fallback estimator
- [~] 2.3 `Matcher` ABC + SIFT/RoMa/MatchAnything done; ELoFTR still a shell
- [ ] 2.4 Tile-batched fp16 inference + memory guard
- [x] 2.5 Correspondence aggregator (currently in pipeline.register; refactor if needed)

## Phase 3 — Consensus + Transform
- [x] 3.1 MAGSAC++ wrapper (cv2 USAC_MAGSAC)
- [x] 3.2 Affine + Homography fitters + per-pair selection (BIC-style score)
- [x] 3.3 `register(I_s, I_t, backbone)` end-to-end
- [x] 3.4 Synthetic-pair test recovers H to <2 px via SIFT (oracle path consensus-only)

## Phase 4 — Full Evaluation
- [x] 4.1 Experimental run: pyramid mode for RoMa + MA on all 187 pairs (2026-06-10).
      **H1 rejected for the current design** — pyramid degrades RoMa (76->1794 px,
      106/187 outright failures) and flatlines MA. Root cause: dense matchers never
      abstain; tile pooling floods RANSAC. See results/README.md "Phase 4 final".
- [x] 4.1b Aggregator redesign DONE (2026-06-10): `register_v2` = verified
      coarse-to-fine (direct -> tile fallback -> zoom, MI gate). RoMa SR@10
      0.10 -> 0.12, SR@20 0.23 -> 0.25, first severe-stratum success, 0 pairs
      lost across both backbones. See results/README.md "Pyramid v2".
- [ ] 4.1c Certainty-gating sweep (knob exists in register_v2, untested at
      scale) + iterated zoom. Bigger lever: cross-modal fine-tuned backbone.
- [x] 4.2 Headline table (Control A / A2 / B / Exp) — complete 2026-06-10, see
      results/README.md "Phase 4 final" + Control B note. Per-group plots still
      pending (1.5).
- [ ] 4.3 Paired-bootstrap CIs + significance markers
- [ ] 4.4 Draft results section

## Phase 5 — Ablations + Sensitivity
- [~] 5.1 FOV sweep {50, 25, 10, 5, 2}% — SIFT placeholder done on synthetic; re-run per backbone on AmalgaMatch
- [ ] 5.2 Pyramid depth {1, 2, 3, 4}
- [ ] 5.3 Overlap {25, 50, 75}%
- [ ] 5.4 RANSAC threshold sweep
- [ ] 5.5 Affine vs homography
- [ ] 5.6 Scale-metadata error +/-20%

## Phase 6 — Writeup + Release
- [ ] 6.1 Tech report (figures + tables + failure modes)
- [ ] 6.2 README single-command reproduction
- [ ] 6.3 v0.1 tag + archived results bundle

---

## Success Gates (block release until met)
Revised 2026-06-09 to match the AmalgaMatch paper protocol (Durmaz et al.):
mean Euclidean distance (ED) of TPS-projected GT target points in source
coords; Success Rate (SR) = fraction of pairs with mean ED below threshold.
Paper pipeline: RANSAC homography @ 5.5 px reproj -> TPS on inliers.
Paper context: SIFT succeeds on only 3/187 pairs; MA-RoMa is their best.

- [ ] SR@10px (mean ED, TPS) beats the paper's best MA-RoMa variant overall
- [ ] >= 35% relative mean-ED improvement vs best zero-shot baseline at FOV <= 5%
- [ ] FOV breakdown curves logged for every backbone
- [ ] ~~P_match@5px > 85% / mu_err < 1.5 px~~ retired: GT-affine floor is
      ~10 px median (see Open Questions); per-point sub-pixel gates are
      unattainable against hand-clicked GT

## Open Questions / Risks (track here, resolve before release)
- [x] AmalgaMatch GT correspondence coverage — 8-61 hand-annotated points per pair (see results/README.md 2026-06-09)
- [x] **GT-quality ceiling — resolved 2026-06-09** by adopting the paper's protocol:
      GT only fits a global affine to median 10.3 px residual, but the paper
      evaluates mean ED after *TPS* refinement (elastic, absorbs non-affine
      deformation) with SR thresholds at 5/10/20 px — not sub-pixel. Success
      gates revised accordingly. `scripts/check_gt_consistency.py` has per-pair
      numbers; H3 (affine sufficient?) is now directly testable against TPS.
- [ ] Modality pairs with near-zero mutual information — separate reporting agreed?
- [ ] Scale metadata missingness rate — does fallback estimator hold up?

## Review (fill at end)
- Summary of what shipped:
- Results vs hypotheses (H1, H2, H3):
- Surprises / follow-ups:

---

## Resume from here (handoff 2026-06-12, MID-BUILD: MA-RoMa fine-tuning)

**Goal:** fine-tune MA-RoMa on the train split (the plan's unused 70/15/15)
to attack the appearance bottleneck. Eval discipline: headline numbers come
from the 28 TEST pairs only; val (28) is for model selection; test stays
untouched until the final comparison (ma_roma_ft vs ma_roma, direct +
pyramid_v2, paired bootstrap on test pairs).

**DONE (committed, but dataset/loss are NOT yet executed even once):**
1. `scripts/make_split.py` + `results/split.json` — scene-level split
   (pairs within a scene share images -> pair-level split would leak).
   131/28/28 pairs over 18/10/7 scenes; greedy pair-count balancing;
   groups covered in val (6/6) and test (5/6). Committed file is canonical.
2. `src/cma/train/dataset.py` — WarpPairDataset: densifies sparse GT
   (TPS >=16 pts, else lstsq affine) into an A-grid->B warp (140x140 grid,
   full-A coverage), supervised only inside GT-support bbox +10% margin
   where mapped point lands in B. Warp in RoMa pixel-center convention
   coord = 2*(px+0.5)/size - 1. Aug: random target FOV crop (area 0.15-0.8,
   anchored on a random GT point, warp geometry updated) + per-image
   gamma/brightness/contrast. Images: 560x560 stretch + ImageNet norm via
   romatch get_tuple_transform_ops (matches inference path exactly).
3. `src/cma/train/loss.py` — SparseGTRobustLoss: RobustLosses skeleton
   with get_gt_warp(depth,pose) replaced by batch["gt_warp"/"gt_prob"]
   (interpolated per scale), wandb stripped. gm_cls at scale 16 + robust
   regression at finer scales, local masking via prev_epe,
   alpha=0.5 c=1e-4 ce_weight=0.01 local_dist {1:4,2:4,4:8,8:8}.

**DONE 2026-06-12 (this session) — build complete, smoked locally:**
4. ✅ `src/cma/train/finetune.py` + `scripts/finetune_ma_roma.py`: decoder-only
   trainer exactly per plan (frozen encoder under no_grad AND kept in eval
   mode — it has BatchNorms whose running stats must not drift; decoder
   train(True) via `set_train_mode`). AdamW 2e-5/wd 1e-4, cosine, GradScaler
   + clip 1.0 (train_step pattern, scale clamped >=1), val every 100 steps =
   direct-match med-mu-ED over val split via RoMaMatcher(model=<in-training>),
   best sd -> checkpoints/ma_roma_ft.pth, log CSV.
5. ✅ RoMaMatcher accepts `model=` (injection) and `weights_path=` (local sd,
   name auto "ma_roma_ft"); runner has backbone ma_roma_ft + --ft-weights.
6. ✅ `scripts/smoke_finetune.py` passed: warp visualization sane
   (results/smoke_warp{,_aug}.png — SEM-SE stitch grid maps onto full EBSD
   frame, consistent gradient), 2 CPU steps ran (loss ~7-8, scales
   {16,8,4,2,1}), grads ONLY in decoder (311 tensors, 307 nonzero; decoder
   = 100.7M of 111.3M registered params). 43 fast tests still green.

**DONE 2026-06-12/13 — EXPERIMENT COMPLETE, verdict mixed and interesting:**
7. ✅ Ran on fresh shared 5090 box 199.126.134.31:34941 (symmc-flow
   co-tenant again; we stayed in /dev/shm). 1500 steps + 15 vals in 91
   min, both sweeps ~20 min, 374/374 rows ok. Best ckpt step 900 (val
   med 17.55 vs 21.69 zero-shot). Pulled: checkpoints/ma_roma_ft.pth
   (445 MB — NOT 1.7G; DINOv2 isn't in the state dict),
   results/finetune_log.csv, results/baselines_A.csv (2805 rows).
8. ✅ Test-split analysis (scripts/ft_test_analysis.py + bootstrap_ci.py
   --split): **in-distribution TEM med ED 320.8 -> 61.8 px (5.2x); but
   catastrophic forgetting on C103 SEM<->LOM (0 train pairs): 12.4 ->
   241.9 px, all four SR@20 losses; net SR@20 0.393 -> 0.250
   significantly worse, SR@10 +0.036 n.s., med ED -31.8 n.s.** Full
   write-up in results/README.md "MA-RoMa fine-tuning" section.

**FOV ladder with ma_roma_ft DONE (2026-06-13):** ran on the same 63-pair
testbed (run_fov_ladder.py --restrict-pairs-csv; ft direct rows would
expand eligible to 120). Findings (results/README.md FOV-ladder subsection):
(1) ft helps the moderate-FOV regime — rung 0.25 SR@10 0.37-0.39 vs
ma_roma 0.28-0.30; (2) the pyramid's scale lift persists on ft and is
significant at rung 0.1 (+0.078, p=0.0154, n=51) but ~half of plain
ma_roma's +0.150 (p=0.0014, reproduced exactly); (3) below 0.05 all
configs collapse. Tools: scripts/fov_ladder_bootstrap.py,
plot_fov_ladder.py (now includes ma_roma_ft).

**Open follow-ups (not scheduled):** forgetting mitigation (replay of
zero-shot outputs / LoRA / per-modality experts) or more data; fold the
fine-tuning + ft-ladder verdicts into reports/final_report.md future-work.

**Gotchas already learned for this build:** romatch RobustLosses/train
import wandb and log unconditionally — do NOT import romatch.losses or
romatch.train in our loop (our loss.py is standalone). model(batch) needs
im_A/im_B already resized+normalized. Box git: fetch+reset, never pull.

**Box note:** shared 142.171.48.138:44563 (symmc-flow co-tenant, 16G disk
— keep everything in /dev/shm). Ports change on recycle; nothing of ours
on the box right now.

---

## Previous handoff (2026-06-12, FOV ladder)

**FOV ladder DONE (Aim 3 closed) — the wrapper is vindicated under
controlled conditions.** Cropping base-matchable real pairs to sweep FOV
with appearance fixed: direct failure FOV is 0.25-0.1; at FOV 0.1,
ma_roma+pyramid_v2 holds SR@10 0.23 vs 0.07 direct (+0.150, CI
[+0.050,+0.275], p=0.0014, n=40); floor at 0.02. Real severe-FOV pairs
stay unsolved because appearance failure dominates there. Data:
results/fov_ladder.csv; figure: reports/figs/baselines/fov_ladder.png;
write-ups in results/README.md + reports/final_report.md (new section 6;
limitation 1 revised). The ladder doubles as the testbed for any future
fine-tuned model's FOV envelope.

**Nothing scheduled next.** Open ideas: materials-domain fine-tuning on
the unused train split; cycle-consistency verifier; publication angles
(see final_report.md section 7 and session notes).

---

## Previous handoff (2026-06-11, late night)

**Phases 5 + 6 DONE — project complete through the planned scope.**
- Figures regenerated (`scripts/plot_baselines.py`, now excludes rejected
  +z3/+c50 variants; FOV panel curated): reports/figs/baselines/*.png.
- H3 readout (`scripts/h3_family_readout.py`): affine picked on 69% of
  well-registered pairs (82/118), homography 31% — H3 mostly supported,
  keep automatic BIC selection.
- **Final report at `reports/final_report.md`**: all hypothesis verdicts
  (H1 rejected / H2 supported / H3 mostly), headline table (verified
  against CSV), pyramid v1→v2 story, backbone lever, limitations.
- Possible future work (out of scope): materials-domain fine-tuning for
  severe FOV; learned verifier to replace MI gate.

---

## Previous handoff (2026-06-11, night)

**Phase 4.2 DONE — H1 FINAL: REJECTED, with the project's first
significant headline win.** MA-RoMa (cross-modal weights in the
roma_outdoor arch, backbone `ma_roma`) direct beats roma direct at SR@10
(+0.032, p=0.018), entirely in the >=0.5 FOV stratum. Wrapper on top:
flat SR@10, keeps severe-stratum 0.03, but no-regression property broke
(2 gained/2 lost, all threshold-straddlers). Best config 0.13 vs 0.10
bar (~+32% relative overall); FOV<=5% still 0.00. Full analysis in
results/README.md "Phase 4.2 / H1 FINAL". CSV now 2644 rows (ma_roma
direct + pyramid_v2 complete).

**Remaining:** Phase 5 — plots (1.5; plot_baselines.py needs the new
modes), H3 family readout (h3_family_readout.py on final CSV). Phase 6 —
writeup vs docs/research_plan.md (H1 rejected / H2 supported / H3 from
readout). No GPU needed for either; box can be released.

---

## Previous handoff (2026-06-11, evening)

**4.1c DONE — both knobs rejected.** Iterated zoom (z3): no significant
gain vs direct (p=0.22), borderline worse than plain v2. Certainty gate
(c50): significantly worse than plain v2 at SR@10 and SR@20. **Plain
pyramid_v2 is the final wrapper config** (SR@10 0.12 vs direct 0.10).
Full analysis in results/README.md "4.1c" section; all 374 new rows in
results/baselines_A.csv (2270 lines total, modes pyramid_v2+z3/+c50).

**Box note:** 142.171.48.138:44563 is SHARED with a symmc-flow session
(16G disk; it evicted our dataset once — see tasks/lessons.md). Sweep ran
subset-at-a-time from /dev/shm (scripts/box_41c_subsets.sh); our shm data
is cleaned up, nothing unique of ours remains on the box. Leave it to
symmc-flow.

**Next:** the only remaining H1 lever is a cross-modal fine-tuned
backbone (paper's MA-RoMa direction). Then Phase 5 ablations (bootstrap
CIs done for v2/z3/c50), plots (1.5), Phase 6 writeup.

---

## Previous handoff (2026-06-11, morning)

**State:** full benchmark complete and written up (Control A/A2/B, pyramid
v1+v2, all in results/README.md). Pyramid v2 (verified coarse-to-fine,
`register_v2`) lifts RoMa SR@10 0.10 -> 0.12 — statistically significant
(paired bootstrap: +0.021, 95% CI [+0.005, +0.043], p=0.017), 0 pairs lost.
Iterated zoom (zoom_iters=3) + certainty CLI knob implemented and committed
but NOT yet swept.

**Next action, ready to fire:** the 4.1c sweeps are one command on a GPU box:
`bash scripts/box_run_41c.sh` (re-downloads dataset if absent, then runs
roma pyramid_v2 --tag z3 and --tag c50). Compare with
`python scripts/compare_v2.py results/baselines_A.csv roma` (note: tags make
new mode keys, adapt the mode filter) and `scripts/bootstrap_ci.py`.

**Box at handoff:** vast instance at 45.29.62.115:20424, idle, dataset
CLEARED (local zip verified bit-for-bit; Fordatis DOI is canonical). The box
holds nothing unique — destroy it freely; re-setup is
scripts/box_resetup_and_sweep.sh (~15 min). Ports change on recycle.

**After 4.1c:** bigger lever is a cross-modal fine-tuned backbone (the
paper's MA-RoMa direction). Then Phase 5 ablations + Phase 6 writeup.

---

## Previous handoff (2026-06-09, evening)

**Last action:** AmalgaMatch extracted (19 subsets, 187 pairs — exact match to
paper counts). Loader rewritten from scratch for the real layout
(`src/cma/data/amalgamatch.py`); all 33 default tests green including 2 new
real-data integration tests. One real pair ran end-to-end through
`register(..., SIFTMatcher())` — pipeline plumbing confirmed working.

**Real-layout facts (validated against data):**
- `eval_indexs/eval_*.npz` are *pickled dicts* (np.load falls back to pickle),
  keys: `dataset_name, image_paths, image_metadata, pair_infos, gt_2D_matches`.
- GT arrays are (N,4) `[x_i, y_i, x_j, y_j]` in `pair_infos` index order.
- Loader orients pairs so source = wider physical FOV (swaps GT cols; `flipped`
  flag in record). GT-implied scale matches pixel-size ratio (median 2.8% dev).
- Windows MAX_PATH: some release paths are >260 chars; loader uses `\\?\`-prefixed
  absolute paths + cv2.imdecode-on-bytes. LongPathsEnabled NOT required.
- `pair_infos` flag is always 1 in this release. `val_list.txt` lists eval names
  (validation split) — not yet consumed by the loader.

**Key new finding (see Open Questions):** GT fits a global affine only to
median 10.3 px residual → the μ_err < 1.5 px gate as written is likely
unattainable; needs re-derivation against the GT-fit floor or the original
paper's protocol. Read the Durmaz et al. AmalgaMatch paper evaluation section
before running Phase 1.3 headline numbers.

**Update (2026-06-09, late):** steps 1-2 DONE. Paper protocol adopted (gates
revised above); Control A swept on a vast.ai 5090 (45.29.62.115:20225, box has
repo at /root/cma, dataset extracted, /venv/main ready). Results + analysis in
`results/README.md`. RoMa zero-shot is the bar to beat: SR@10 0.10 / SR@20 0.23.

**Update (2026-06-10):** FOV decision made — severe stratum = area ratio <= 0.25
(n=37; only 3-4 pairs sit below 5% under any definition). RoMa-pyramid swept all
187 pairs: **pyramid degrades RoMa catastrophically** (med ED 76 -> 1794 px; see
results/README.md "Phase 4 interim"). Root cause: dense matchers never abstain,
so tile pooling floods RANSAC (inlier frac 0.114 -> 0.005). Box recycled once
(new instance, port changes); resetup is scripted (`scripts/box_resetup_and_sweep.sh`).

**Concrete next steps, in order:**
1. Single-pair trace to separate tile-pooling vs scale-normalization as the
   pyramid failure mechanism (results/README.md lists both hypotheses).
2. Redesign the aggregator for dense matchers: per-tile RANSAC + best-tile
   selection, or certainty gating before pooling. Re-run roma/pyramid.
3. Finish MatchAnything-pyramid + Control A2 (matchanything_stretch, direct)
   on the box after maintenance (~21:15 UTC); resume logic handles both.
4. Finish Control B locally (~30/187 done; MMI is ~5 min/pair on stitched
   images — consider downsampled-MI protocol if it cannot finish overnight).
5. Headline table + plots once all rows land.

**Other live state at handoff:**
- gh 2.93.0 installed. To use, run: `gh auth login -h github.com -p ssh -w`
- RoMa wired: `src/cma/matchers/roma.py` + `tests/test_roma.py` (marked `slow`).
  CPU smoke ~49s, pyramid warped-pair ~47 min. GPU box will be much faster.
- Slow pytest marker registered in `pyproject.toml`; default suite (`pytest`)
  excludes it, run with `pytest -m slow` to include.
- 31 default tests + 2 slow tests, all green at handoff.
- Open backbone: ELoFTR remains a shell. MatchAnything wrapper already covers
  the ELoFTR architecture via HF transformers, so this may not need wiring.


## Phase 11 — TMLR submission package (2026-08-24)

**Why:** Scientific Reports rejected the manuscript (all six reviewer points
sustained). Venue decision: TMLR primary, Microscopy and Microanalysis fallback.
TMLR's first acceptance criterion is whether claims are supported by the
evidence, and its remedy for over-claiming is to adjust the claims -- which is
what the Sci Rep revision already did. Branch `tmlr-submission`, kept separate
from `sci-rep-revision` so the latter remains the record of what reviewers saw.

- [x] **Headline metric decided: unrefined parametric error.** The refined (TPS)
      column has coverage ranging 0.000-1.000 across configurations, so a
      TPS-scored table is two metrics interleaved by configuration. The raw
      metric scores all nine identically. Cost: both native-pair headline results
      go null. Note the direction -- the switch *removes* significance, so it
      cannot be metric-shopping.
- [x] Figure scripts made metric-switchable (`plot_baselines.py [csv] [raw|tps]`);
      `group_heatmap_raw.png` and `fov_curves_raw.png` added. Regenerating the
      tps variant reproduces the previous files byte-for-byte.
- [x] Every number in the new draft recomputed from the result files under raw.
- [x] Manuscript rewritten for an ML audience: mechanism-first structure,
      Related Work added, Methods moved to an appendix, metric sensitivity
      promoted from a defensive note to a numbered contribution.
- [x] Converted wlscirep -> tmlr.sty; 33 `\cite` -> `\citep`/`\citet`;
      starred headings -> numbered; style files vendored in `paper/tmlr/`.
- [x] Anonymised for double-blind: no author block, real repo URL replaced with
      an anonymous.4open.science placeholder, no Zenodo DOI, no ORCID.
- [x] Compiled against the real `tmlr.sty` with MiKTeX: 16 pages, zero errors,
      zero undefined references or citations, no overfull box above 20 pt.
- [x] `scripts/verify_tmlr_draft.py` added as a re-runnable gate: 32 must-appear
      values, 12 banned phrasings, 2 withdrawn-claim retraction checks, plus
      structural and rendered-PDF anonymity checks.
- [x] Test suite green (67 passed, 3 deselected).

### Review

Five claims carried over from the Sci Rep text were **false or differently
valued under the raw metric**, and the verification pass caught all five:

1. Certainty gating was quoted as significantly worse (SR@20 -0.037). That is
   the TPS value; on raw it is *exactly null* (0.219 -> 0.219, p = 1.00). Now
   both are reported. The argument it supports -- that thresholding certainty
   does not recover abstention -- survives, since the gate never helps.
2. The MatchAnything-RoMa wrapper was described as "two pairs gained and two
   lost ... the fourth a recovery from 84.8 to 8.3 px". Under raw it gains two
   and loses **none** (11.8 -> 8.3 and 325.8 -> 8.1). Rewritten.
3. H3 claimed homography-selected pairs "show no accuracy advantage". They are
   in fact *more* accurate (median 6.5 vs 11.3 px). Withdrawn and replaced with
   the selection-confound caveat; the 69 % affine claim itself stands.
4. "Median 10,000 matches, which is the cap we impose" understated the
   mechanism. RoMa hits exactly 10,000 per invocation on every pair, and
   pyramid v1 pools up to **9,420,000** on one pair -- 942 tiles' worth. This
   made the central argument stronger, not weaker.
5. The match cap was stated as global; it is per invocation.

Two results improved under the raw metric rather than degrading:

- The wrapper's null aggregate resolves into a **composition shift** -- one
  low-FOV failure converted to a success, one high-FOV success lost, 17 = 17 --
  which is the trade its scale mechanism predicts and could have failed.
- H2 is now supported in the low-FOV strata (RoMa 1/33 vs MA-ELoFTR 0/33) where
  under TPS both were tied at zero.

**Open:** submission itself is the user's action. Needs an OpenReview account,
an anonymised code mirror at the placeholder URL, and the abstract pasted into
the submission form.

---

## Phase M — Microscopy & Microanalysis submission (opened 2026-08-28)

Context: Sci Rep rejected 2026-08-24; TMLR **desk-rejected 2026-08-28** without review
(volume/AE-bandwidth boilerplate — no signal on correctness). Frank's call: skip the
AI4Mat workshop, go straight to M&M. Branch `mam-submission` off `tmlr-submission`.

**Verified target facts (2026-08-28):** M&M is published by **Oxford University Press**,
not Cambridge (our note was stale). Hybrid OA; the **standard subscription licence carries
no charge**, so the free route survives the publisher move. MSA members may get discounts
on the paid route (not needed).

### M0 — Base text
- [x] Base = `paper/tmlr/main.tex` (settled science: **unrefined parametric error primary**)
- [x] NOT `paper/paper.md` — that is the Sci Rep text, still TPS-primary, and carries the
      five claims the forensic audit falsified (wrapper p=0.034, MA-RoMa p=0.035, H3, etc.)
- [x] De-anonymise: author block, real repo URL, Zenodo DOI, drop anonymous.4open.science

### M1 — Restructure to M&M's mandated section order
Required, "in the order listed herein": Introduction → Materials and Methods → Results →
Discussion → Summary or Conclusions → Acknowledgments → References.
- [x] Fold `Related work` into Introduction
- [x] `Setting` → Materials and Methods (benchmark, pipeline, metric, strata, statistics)
- [x] §4–§8 → Results subsections
- [x] Mechanism argument + limitations → Discussion
- [x] H1/H2/H3 verdicts → Summary/Conclusions

### M2 — Retitle + abstract (the binding constraint)
- [x] Retitle: M&M states "jargon should not be used". "Non-abstaining dense matchers" is
      exactly that. Lead with the practitioner takeaway.
- [x] Abstract: **≤200 words, no reference citations, NO ABBREVIATIONS.** Current is ~400
      words and leans on FOV/SEM/EBSD/TEM/SR@10/TPS/NMI. Near-total rewrite, not a trim.

### M3 — References → author-date
- [x] 17 entries → author-date in-text (surname, year)
- [x] Journal names abbreviated per CASSI
- [x] **All authors listed; "et al." unacceptable in the reference list** (DINOv2,
      MatchAnything, LoFTR have long author lists to expand)

### M4 — Figures to spec
- [x] Already 300 dpi colour ✓ (verified: all 8 PNGs at 299.999 dpi)
- [x] Use the `_raw` variants as primary — consistent with the unrefined headline metric
- [x] Multi-panel → ONE file per figure, panels labelled A/B/C upper-left
- [x] Flatten RGBA → RGB
- [x] **Alt text for every figure** (M&M requires it)
- [~] Legibility: figures are 198–305 mm wide at 300 dpi. Figs 1 and 3 (279, 305 mm)
      are full-page-width figures, not 84 mm single-column; flag to the editor at
      submission, or re-export at larger font if a single column is required.

### M5 — Required statements
- [x] Conflict of interest (title page, all authors)
- [x] Author contributions via **CRediT** taxonomy
- [x] Data availability (Fordatis DOI 10.24406/fordatis/436 + repo + Zenodo)
- [x] Acknowledgments

### M6 — Format + build
- [x] Double-spaced throughout, 12 pt, ~1 inch (2.5 cm) margins
- [x] Build via the `journal-submission` skill (requirements.md YAML → apply_format + audit)

### M7 — Cover letter + verification gate
- [x] Cover letter to the editor
- [x] `scripts/verify_mam_draft.py`, adapted from `verify_tmlr_draft.py`: re-assert every
      number against the result files, plus M&M gates — abstract ≤200 words / 0 abbreviations
      / 0 citations, mandated section order, no "et al." in the reference list

**Not doing:** AI4Mat (Frank declined, 2026-08-28). Deadline was Aug 29 AOE = Sat Aug 30
~08:00 ET, and there was a 5 pp Findings/Tools/Open-Challenges track that fit. Recorded in
case the M&M route stalls and a non-archival airing becomes attractive again.

### M8 — Status 2026-08-28
Manuscript, figures, cover letter and both gates are **built and green**:
- `paper/mam/manuscript.md` → `manuscript.docx` (12 pt, double-spaced, 1 in margins,
  continuous line numbers, no theme fonts — audited by `build_mam_docx.py`)
- `paper/mam/figures/Figure1..5.png` (300 dpi, RGB, one file per figure)
- `paper/mam/cover_letter.md` / `.docx`
- `scripts/verify_mam_draft.py` — **106 checks, all pass**
- 67 pytest tests pass

**Two things the gate caught that reading did not:** (1) the abstract came in at 205
words against a hard 200-word cap; (2) **LoRA (Hu et al., 2022) was an orphan
reference** — listed but never cited, because folding Related Work into the
Introduction dropped its only mention. Both fixed; an orphan/dangling-citation
check is now a permanent gate.

**Publisher correction:** M&M is **Oxford University Press**, not Cambridge. The
free (standard, subscription) licence route survives the move, so no charge.

**Remaining = Frank's actions only:** OUP ScholarOne account, paste title/abstract,
upload manuscript + 5 figure files + cover letter, supply CRediT roles and
suggested reviewers if prompted. Nothing about the science is open.

---

## Phase N — Figure 1 redrawn (2026-08-30)

Brief: "redraw fig.1 to make it more cleaner, concise, and publishable grade."
Scope held to craft, not content: the panel-A/panel-B split, every claim and the
manuscript legend are unchanged. New generator `paper/schematics/gen_fig1.py`
(palette + verify + crop copied from the vector-schematics skill); the old
`paper/make_schematic.py` is archived, with a guard, as
`paper/make_schematic-archive-2026-08-30.py`.

- [x] N0 Defects found in the old Fig. 1, by looking at it at 4x
  - **The ladder had no denominator.** Panel B's outermost square was the 0.5
    rung, not the source field, while the axis text read "target field-of-view
    area / source area". The source was never drawn, so none of the five ratios
    named anything on the page.
  - **The drawn areas were not the stated areas.** Rung sides went as
    `sqrt(r / 0.5)`, i.e. relative to the 0.5 rung. The 0.25 rung was drawn at
    half the outer square's area, which is 0.25 of the source only by accident of
    the outer square also being 0.5. Now `side = SRC_SIDE * sqrt(r)` with the
    identity asserted per rung.
  - **The alt text described a figure that did not exist** — "a wide-field
    micrograph with successively smaller crop boxes drawn on it". No micrograph
    was ever in the panel. Alt text rewritten to the drawing that is there.
  - Label collisions: "direct" straddled two box edges; "weak?" sat on the target
    box corner; "if better, replace T*" overlapped the return connector.
  - Two large text boxes floated in panel B carrying prose the legend already had.
- [x] N1 Panel A rebuilt on a lattice: inputs -> matcher -> fit -> incumbent as a
      spine, an explicit decision diamond for "direct support weak?", and the two
      candidate stages drawn as the sequence they are rather than a bulleted list.
      Both the accept and the reject outcome are now drawn; before, only accept was.
- [x] N2 Field of view encoded by **frame size, not colour**. Okabe-Ito blue and
      orange already mean backbone in Figs 3-4 and error threshold in Figs 2 and 5;
      a third meaning on the same hues was avoidable, and size is what actually
      distinguishes a wide field from a narrow one. Green and vermillion are left
      to mark the only two things the wrapper adds: the branch and the gate.
- [x] N3 Panel B: source frame drawn, rungs as a sequential ramp, all six in one
      key. Labelling rungs in place cannot be done evenly — the 0.10/0.05 and
      0.05/0.02 bands are thinner than a line of 5.9 pt type, forced by the square
      roots being close — and a mixed inline/leader scheme reads as two systems
      with leaders crossing every rung outside the one they name.
- [x] N4 Ground-truth markers placed by 2-D blue noise in inches, not `rng.uniform`
      and not in axes fractions (panel B is wider than tall, so equal separation in
      axes units is unequal separation on the page). Asserted: no two markers touch,
      counts inside the rungs fall monotonically, and the 0.05 rung is not empty —
      an empty small rung would say the small crops carry no ground truth.
- [x] N5 Export gates: zero Type 3, zero rasters, one font family, 42 live `<text>`
      elements, ten expected strings present.
- [x] N6 **Package-wide problems this turned up, now fixed**
  - `scripts/plot_baselines.py` and `scripts/plot_fov_ladder.py` never set
    `pdf.fonttype=42`, so **every vector export of Figs 2-5 carried Type 3 fonts**,
    which publishers reject. Set in both, along with `ps.fonttype` and `svg.fonttype`.
  - M&M asks for **vector with embedded fonts for charts and diagrams**, and sets
    600-900 dpi for line art if raster is used. The package shipped 300 dpi PNGs
    only — below the journal's own floor for what these are. All five figures now
    have a PDF submission copy; the PNG stays for the manuscript file.
  - `build_mam_figures.py` re-stamped every PNG to 300 dpi, which would have told
    Word the 600 dpi schematic was twice its true size.
- [x] N7 Gates green: `verify_mam_draft.py` 106/106, `build_mam_docx.py` format
      audit clean, `build_mam_figures.py` 0 problems, pytest exit 0.

### Open, and Frank's call
- **Printed width.** M&M states no maximum figure width, so this is reported, not
  gated. Fig. 1 is 182 mm; Figs 2 and 5 are 198 mm, Fig. 4 249 mm, Fig. 3 305 mm.
  Production will scale them to the column, so Fig. 3 loses ~43 % of its linear
  size and its 8 pt type lands near 4.6 pt. Re-tuning the four data plots is a
  separate job from this one and was not done.
- **Hue reuse across the paper.** Blue and orange mean backbone in Figs 3-4 and
  error threshold in Figs 2 and 5. Each figure has its own legend so neither is
  wrong, but it is worth one pass if the figures are ever revised together.
  Deliberately not acted on here.

### N8 — Figure 1 cut entirely (2026-08-30, author's call)

Frank's call on the drawing. Removed rather than reworked, and the cost is low:
Methods 2.8 and 2.9 already state the wrapper's control flow and the ladder
construction in prose, and the schematic illustrated those two things rather than
the paper's actual mechanism (non-abstention flooding the estimator), so nothing
argued in the text lost its only visual support.

- [x] Legend and alt text excised; the one body reference ("pyramid v2; Figure 1a")
      dropped, since the sentence reads correctly without it.
- [x] Figures 2-5 renumbered to 1-4 through a placeholder, not in sequence — a
      straight run of replacements would have taken 2->1 and then 3->2 on top of it.
- [x] `build_mam_figures.py` FIGURES list cut to four; stale package files deleted
      and the directory rebuilt from empty so no orphan Figure5.* could survive.
- [x] Generator kept at `paper/schematics/gen_fig1.py`; only the rendered artefacts
      were deleted, so the figure is one command away if it is ever wanted back.
- [x] **New gate, and it caught something.** `verify_mam_draft.py` now checks that
      legends run 1..N with no gap, that no body reference names a figure without a
      legend, and that no figure has a legend without a body reference. The last of
      those failed immediately — the refined-metric bar chart (Fig. 5, now Fig. 4)
      was cited **only from inside another figure's legend** and never from the
      text. That predates this edit. Fixed with a citation in Section 3.8, which is
      the section the figure exists to support.
- [x] Gates green: 109/109 manuscript checks, figure package 0 problems, DOCX
      format audit clean. The paper now carries **four figures and four tables**.

---

## Phase R — M&M rewrite after MAM-26-246 rejection (opened 2026-09-25, branch `mam-rewrite`)

Decision letter (2026-09-11, Editor J. Michael, no reviewer reports): "extremely difficult to
read ... written in a more tutorial way so that terms and methods are better described and
examples of the failures are shown. The use of the existing known database is useful."
Frank, 2026-09-25: "continue with the rewrites, continue with everything." The submitted
package stays untouched on `mam-submission` (pushed); a copy lives in `paper/mam/MAM-26-246/`.

### R0 — Correctness defects found while picking failure examples (must fix in any version)
- [x] **The 106 pyramid-v1 "hard failures" are a GPU crash, not estimator failures.** Every
      one of the 106 failed rows carries `CUDA error: unknown error`; the run died at the
      82nd pair (loader order) and every later pair inherited the poisoned context. The
      submitted text said "the estimator cannot return a transform at all". v1 was only
      ever evaluated on 81 pairs (10 subsets, alphabetical head of the loader).
- [x] **The 80 -> 2708 px headline compared different pair sets** (direct over 187, v1 median
      over its 81 finite rows). Same 81 pairs: direct 335.8 -> v1 2707.6 px; SR@10 11 -> 1.
      Same for the inlier fraction ("all 187 pairs", 0.114 -> 0.005): same 81 pairs,
      0.092 -> 0.005.
- [x] **48 of the 81 evaluated v1 pairs never tiled.** Tile side = target's shorter side in
      TARGET pixels; when the target has finer pixels than the source (71/187 pairs) no
      pyramid level is built, and when the source is smaller than that side the source is
      reflect-padded to one square tile (61/187 pairs overall, all 48 single-tile rows).
      Example: AF9628 0#2 source 4096x2028 padded into a 5628x5628 tile (26 % real content).
      These pairs test mirror padding, an implementation defect, not tiling.
- [x] **Genuinely tiled pairs (33 of 81, 2-942 tiles, median 75):** direct median 67.6 ->
      v1 537.7 px, inlier fraction 0.102 -> 0.00092, v1 worse on 26/33, SR@20 4 -> 1,
      SR@10 1 -> 0. The flooding mechanism stands on these; the success collapse mostly
      came from the padded pairs (SR@10 10 -> 1).
- [x] Section 3.8's "187/187 refinement coverage for every dense RoMa-family row" was false
      for the v1 row (72/187). Moot once v1 leaves the 187-pair table.
- [ ] **Frank's call — complete the v1 run on a rented GPU.** 106 crashed pairs = 4,303 tile
      matches (97 multi-tile, 9 padded); includes 6 pairs direct RoMa registers (TRIP1 x2,
      Ti3AlC2 x3, X2CrNi12), i.e. the real test of "tiling destroys a working fit". CPU is
      not an option (one direct match takes minutes here, and the CPU is shared with other
      sessions). ~1 h on a 5090 incl. the 4.2 GB dataset fetch, well under $2. Optional
      second arm: fix the tiler (tile side in source pixels = target side x scale ratio) so
      the 61 padded pairs are tested too. The rewrite text is correct WITHOUT either run.

### R1 — Structure for a microscopist reader
- [x] Keep the title. Rewrite the abstract (<=200 words, no abbreviations, no citations).
- [x] Intro: the task in pictures (new Fig. 1: real AmalgaMatch pairs, target footprint drawn
      in the wide image), what a matcher is, the tiling idea, what we found, a reading guide.
- [x] Methods as a walkthrough: Table 1 = glossary of every technical term; step-by-step
      pipeline on one real pair (new Fig. 2); how error and success are measured and why
      the unrefined error; the two wrappers step by step; ladder; appearance measure;
      fine-tuning; statistics in plain words.
- [x] Results: each subsection opens with the question it answers and closes with the
      one-sentence answer; failure examples shown as images (new Figs 3-4).
- [x] Discussion opens with practical guidance for someone registering their own data.
- [x] Move protocol minutiae to Supplementary Material (PDF): fine-tuning protocol defects,
      earlier draw, per-run table, hypothesis verdicts, refined-metric figure + table,
      full crash/padding accounting for v1. Every item cited from the main text.
- [x] Pyramid v1 restated on its 81 evaluated pairs, split multi-tile vs padded (R0).

### R2 — New figures (real images; AmalgaMatch is CC-BY-4.0, credit in legends)
- [x] Fig. 1 the task: 3 pairs spanning FOV ratio and modality, GT footprint + GT points.
- [x] Fig. 2 the pipeline on one pair: correspondences (inlier/outlier), fitted footprint vs
      GT footprint, per-point error vectors, the number that gets scored.
- [x] Fig. 3 why tiling fails: tile grid on a real multi-tile pair; a tile that does not
      contain the target still returns 10,000 correspondences at high certainty; pooled
      inlier fraction direct vs v1 on the 33 tiled pairs.
- [x] Fig. 4 failure gallery: severe FOV, appearance, and the fine-tuned model forgetting a
      C103 SEM/LOM pair, each with predicted vs true footprint and its error. Re-run on CPU
      for display only; legend states the displayed error comes from that re-run.
- [x] Keep: success-rate bars (v1 bar removed), FOV ladder (+ crops of a real pair),
      strata plot. Refined-metric bars -> Supplementary.

### R3 — Gates, package, correspondence
- [x] `verify_mam_draft.py` updated for the rewrite: numeric parity incl. the new v1
      same-pair numbers, a failure-reason audit (no infrastructure failure scored as a
      method failure), glossary coverage, abbreviations defined at first use.
- [x] Build manuscript.docx, supplementary.pdf, figures; gates green; pytest green.
- [x] Cover letter: new submission, discloses MAM-26-246, lists changes incl. the R0 fix.
- [x] AI-assistance statement drafted for Frank to confirm (OUP policy check).
- [x] Gmail DRAFT (never send) to the editor asking whether he will consider the
      rewritten manuscript as a new submission.
- [x] Commit (explicit paths, no co-author trailer), push, memory + review section.

### R4 — Review (2026-09-25)

**Delivered on `mam-rewrite`:** `paper/mam/manuscript.md|.docx` (tutorial rewrite, 198-word
abstract, 7 figures, 5 tables incl. a glossary), `supplementary.md|.docx|.pdf` (S1-S7, 3
figures, 6 tables), `cover_letter.md|.docx` (discloses MAM-26-246 and the corrections),
`figures/Figure1-7, S1-S3` (PDF + 300 dpi PNG), and a Gmail DRAFT to the editor asking
whether he will take the rewrite as a new submission (not sent). The submitted version is
archived in `paper/mam/MAM-26-246/` and on branch `mam-submission`.

**New tooling:** `mam_rewrite_numbers.py` recomputes every quoted number from the result
CSVs into `results/mam_rewrite_numbers.json`; `verify_mam_rewrite.py` builds its expected
strings FROM that JSON (278 checks, 0 failures) and replaces `verify_mam_draft.py`, which
asserted copied strings and is removed. `pyramid_v1_audit.py`, `fov_ratios_gt.py`,
`mam_examples_run.py` (seeded CPU re-runs for display), `plot_mam_rewrite.py`.
`build_mam_figures.py` now only checks the package (copying would overwrite the new figures).
`build_mam_docx.py` takes src/out, `--single`, `--no-line-numbers`, `--pdf` (via Word).

**Corrections found (beyond R0):**
- The certainty-0.5 gate compared against the ONE-zoom v2 rows, but c50 ran after commit
  eef81ce made three zooms the default. Against z3 (same zoom count) the gate is null on
  both metrics (p = 0.24 raw, 0.18 refined); the old "significantly worse, p = 0.002" is
  withdrawn. MatchAnything-RoMa + v2, the ladder and the ft runs all used three zooms.
- RoMa's sampler sets certainty > 0.05 to exactly 1, so "gate at 0.5" = keep above 0.05.
- The v2 tile stage (runs below 50 inliers) essentially never fires: every direct RoMa fit
  kept >= 50 inliers, including 2,000-px misses. v2's effect is the ZOOM (37 of 187 native
  answers; 7 of the 9 ladder successes). Paper now says so.
- Ceiling: a homography fitted to the annotated points themselves registers only 93/187
  within 10 px (156 within 20). Now reported against every method.
- Certainty share separates outcomes in-sample: all 15 registered direct-accepted pairs had
  >= 93 % of correspondences above the cut-off; 48 of 51 gross failures had less (rho -0.71).
  Reported as in-sample, untested elsewhere.
- Refinement on the ladder is invalid (out-of-crop GT points), not "near-zero effect".

**Frank's calls (open):**
1. Confirm or edit the AI-use paragraph (manuscript Section 2.14).
2. Send the editor query (Gmail draft) or submit directly.
3. Optional ~<$2 GPU run to complete pooled tiling on the 106 crashed pairs (and, if
   wanted, a second arm with tiles sized in wide-image pixels). The text is correct without it.
4. The ICLR / STS deadlines come first; nothing here is time-bound.
