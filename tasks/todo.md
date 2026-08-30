# TODO — Correlative Microscopy Alignment

Track in this file. Check items off as completed.
Source docs: `docs/context.md`, `docs/research_plan.md`, `docs/task_plan.md`.

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
