# Arm 1 report: does cropping burned-in overlays remove the near-identity lock?

Source CSV: `C:\Users\frank\Documents\cma-triage-ext\results\arm1\arm1.csv`. Analysis follows `prereg/arm1_banner_crop.md` exactly; the interpretive choices are listed at the end.

## Row counts
- Rows in file: 708; unique runs after dropping repeated attempts: 708; expected 708: **OK**.
- Repeated attempts of the same run (last one used): 0.
- Unique runs by job (pair list x crop), observed / expected: overlay|none 268/268; overlay|overlay 268/268; ti3alc2|none 12/12; sham|none 80/80; sham|sham 80/80
- Rows with a commit hash: 91c8b6725ea5b3fd117257056f9c9236e5a4ce1c.

## Failed runs
0 rows have status other than ok (0 of them with CUDA/OOM/timeout-type messages = infrastructure). By the prereg every failed run counts as a failure, and an infrastructure failure should have been rerun once.

## H-A1: the lock goes away (67 overlay pairs x 4 configurations)
- Near-identity locks: uncropped 110, cropped 0 (of 268 paired runs).
- Discordant pairs: lock only uncropped 110, lock only cropped 0.
- Exact McNemar: one-sided p = 7.704e-34 (two-sided 1.541e-33).
- Criterion: one-sided exact McNemar p < 0.05 and near-identity locks fall when cropped. **Supported: YES.**

## H-A2: success rises (primary: MA-RoMa direct, 67 overlay pairs)
- SR@20 uncropped 6/67, cropped 16/67; success only when cropped 12, only when uncropped 2.
- Exact McNemar: one-sided p = 0.00647 (two-sided 0.01294). Criterion: p < 0.05 and more successes.
- **Supported: YES.** Reported as: supported.

Secondary configurations (same test; Holm correction across the 3, on the one-sided p):

| config | successes_uncropped | successes_cropped | only_cropped | only_uncropped | p_one_sided | p_holm | supported |
|---|---|---|---|---|---|---|---|
| roma\|direct | 14 | 16 | 5 | 3 | 0.363 | 0.727 | NO |
| roma\|pyramid_v2 | 14 | 16 | 5 | 3 | 0.363 | 0.727 | NO |
| ma_roma\|pyramid_v2 | 6 | 16 | 12 | 2 | 0.00647 | 0.0194 | YES |

## H-A3: triage improves (S1 cut-off fixed at 0.1711, MA-RoMa direct, 4 held-out groups)
- Held-out pairs: 119. False accepts, all-uncropped arm 42; cropped-overlay arm 14.
- Discordant: false accept only uncropped 30, only in cropped arm 2.
- Exact McNemar: one-sided p = 1.232e-07 (two-sided 2.463e-07). Criterion: p < 0.05 and fewer false accepts. **Supported: YES.**
- Overlay pairs alone: false accepts 42 uncropped vs 14 cropped; other held-out pairs (identical rows in both arms): 0.
- Reported, not tested: S1 AUROC on DislocationCharacterization (n = 69): uncropped 0.811, cropped 0.786, difference -0.025, paired-bootstrap 95% CI [-0.151, 0.106].

> **DEVIATION NOTE for the human to adjudicate (not decided silently).** The prereg says H-A3 uses "the fresh uncropped rows for all other held-out pairs", but Arm 1 only reran the 67 overlay pairs, the 3 Ti3AlC2 pairs and 20 sham pairs.
> For the other held-out pairs I used the STORED CJSJ MA-RoMa direct rows (core pool, seed 0): 42 of 119 held-out pairs (fresh uncropped rows were available for 77); of the stored ones 0 are overlay pairs (should be 0).
> Because the cropped and uncropped arms use the SAME row for every non-overlay pair, these pairs are concordant and cannot affect the McNemar test; they do affect the absolute false-accept counts and the DislocationCharacterization AUROC. The reproduction check below shows how far stored and fresh rows agree.

## Controls and validity checks
### Sham crop (20 pairs x 4 configurations)
- Paired runs 80; successes uncropped 7, sham-cropped 4; success only uncropped 5, only sham 2; two-sided McNemar p = 0.4531.
- Runs whose success changed: 7 = 8.8% (rule: more than 10% means the crop itself matters). **Crop itself matters: NO.**

### Ti3AlC2 (3 TEM pairs without a banner, uncropped)
- 12 runs; successes in fresh rerun 12, in stored CJSJ rows 12; success in both 12. Environment reproduces on this control (fresh success = stored success on every run): **YES**.

| pair_id | cfg | have_fresh | fresh_success | fresh_mu | stored_success | stored_mu |
|---|---|---|---|---|---|---|
| eval_Ti3AlC2-MAX-Phase_TEM_DislocationCharacterization_0#0 | roma\|direct | True | True | 2.04 | True | 2.04 |
| eval_Ti3AlC2-MAX-Phase_TEM_DislocationCharacterization_0#0 | ma_roma\|direct | True | True | 2.04 | True | 2.04 |
| eval_Ti3AlC2-MAX-Phase_TEM_DislocationCharacterization_0#0 | roma\|pyramid_v2 | True | True | 2.04 | True | 2.04 |
| eval_Ti3AlC2-MAX-Phase_TEM_DislocationCharacterization_0#0 | ma_roma\|pyramid_v2 | True | True | 2.04 | True | 2.04 |
| eval_Ti3AlC2-MAX-Phase_TEM_DislocationCharacterization_0#1 | roma\|direct | True | True | 1.78 | True | 1.78 |
| eval_Ti3AlC2-MAX-Phase_TEM_DislocationCharacterization_0#1 | ma_roma\|direct | True | True | 1.82 | True | 1.82 |
| eval_Ti3AlC2-MAX-Phase_TEM_DislocationCharacterization_0#1 | roma\|pyramid_v2 | True | True | 1.78 | True | 1.78 |
| eval_Ti3AlC2-MAX-Phase_TEM_DislocationCharacterization_0#1 | ma_roma\|pyramid_v2 | True | True | 1.82 | True | 1.82 |
| eval_Ti3AlC2-MAX-Phase_TEM_DislocationCharacterization_0#2 | roma\|direct | True | True | 1.28 | True | 1.28 |
| eval_Ti3AlC2-MAX-Phase_TEM_DislocationCharacterization_0#2 | ma_roma\|direct | True | True | 1.26 | True | 1.26 |
| eval_Ti3AlC2-MAX-Phase_TEM_DislocationCharacterization_0#2 | roma\|pyramid_v2 | True | True | 1.28 | True | 1.28 |
| eval_Ti3AlC2-MAX-Phase_TEM_DislocationCharacterization_0#2 | ma_roma\|pyramid_v2 | True | True | 1.26 | True | 1.26 |

### Reproduction (fresh uncropped vs stored CJSJ mu_ed)
- Matched within 1 px on 360/360 = 100.0% of runs (needs at least 95%). **Reproduces: YES.** Comparisons stay fresh-vs-fresh either way.
- Per configuration: ma_roma|direct 90/90; ma_roma|pyramid_v2 90/90; roma|direct 90/90; roma|pyramid_v2 90/90
- Per pair set: overlay 268/268; sham 80/80; ti3alc2 12/12

### The confound, stated in advance (verbatim from the prereg)
> In AmalgaMatch, overlay, identical image size and TEM modality coincide perfectly. Arm 1 can show that removing the band changes the outcome; it cannot separate the band from anything else about these TEM images.

### Clustering
A pair-level p-value treats the pairs as independent, which they are not (67 pairs from 4 scenes and about 5 GT-offset configurations); the tests are pair-level as pre-registered, and the per-scene and per-offset tables must be read alongside them.

## Per-scene breakdown (67 overlay pairs; locks pooled over the 4 configurations)

| group | pairs | runs | locks_uncropped | locks_cropped | maroma_direct_success_uncropped | maroma_direct_success_cropped | all_success_uncropped | all_success_cropped |
|---|---|---|---|---|---|---|---|---|
| eval_C103_SEM-SE_LOM-DC-Height_FractureSurfaces_2 | 1 | 4 | 0 | 0 | 1 | 1 | 2 | 2 |
| eval_MoTaTiZrHf-HEA-DDRX-1100C_TEM_DislocationCharacterization_0 | 32 | 128 | 34 | 0 | 4 | 13 | 36 | 56 |
| eval_MoTaTiZrHf-HEA-DDRX-900C_TEM_DislocationCharacterization_0 | 16 | 64 | 26 | 0 | 0 | 0 | 0 | 0 |
| eval_MoTaTiZrHf-HEA-DDRX-900C_TEM_largeFOV-DislocationCharacterization_0 | 18 | 72 | 50 | 0 | 1 | 2 | 2 | 6 |

## Per offset-configuration breakdown (scene + GT translation rounded to the nearest 20 px)

| group | pairs | runs | locks_uncropped | locks_cropped | maroma_direct_success_uncropped | maroma_direct_success_cropped | all_success_uncropped | all_success_cropped |
|---|---|---|---|---|---|---|---|---|
| eval_C103_SEM-SE_LOM-DC-Height_FractureSurfaces_2 (160,-960) | 1 | 4 | 0 | 0 | 1 | 1 | 2 | 2 |
| eval_MoTaTiZrHf-HEA-DDRX-1100C_TEM_DislocationCharacterization_0 (-20,-60) | 16 | 64 | 10 | 0 | 2 | 4 | 12 | 22 |
| eval_MoTaTiZrHf-HEA-DDRX-1100C_TEM_DislocationCharacterization_0 (-40,-160) | 16 | 64 | 24 | 0 | 2 | 9 | 24 | 34 |
| eval_MoTaTiZrHf-HEA-DDRX-900C_TEM_DislocationCharacterization_0 (60,20) | 16 | 64 | 26 | 0 | 0 | 0 | 0 | 0 |
| eval_MoTaTiZrHf-HEA-DDRX-900C_TEM_largeFOV-DislocationCharacterization_0 (120,500) | 9 | 36 | 26 | 0 | 1 | 2 | 2 | 6 |
| eval_MoTaTiZrHf-HEA-DDRX-900C_TEM_largeFOV-DislocationCharacterization_0 (200,380) | 9 | 36 | 24 | 0 | 0 | 0 | 0 | 0 |

## Interpretation notes and choices the prereg left open
- One-sided vs two-sided: the prereg calls the hypotheses one-sided and directional (alpha 0.05) but its support rule says 'p < 0.05 and locks fall / more successes'. The primary p here is the ONE-SIDED exact McNemar (binomial tail in the predicted direction); the two-sided exact p (cma.triage.mcnemar_exact) is reported beside it. A hypothesis is supported only if the one-sided p < 0.05 AND the direction is right; where the two-sided p would give a different verdict this is flagged in the table.
- Lock thresholds: 'within 20 px of identity' is implemented strictly (< 20), and 'more than 40 px' as > 40, matching the frozen A6 audit (NEAR_ID_PX / GT_DISP_MIN). The 5x5 grid uses the ORIGINAL target size (from the stored CJSJ row), because for cropped runs the Arm-1 row reports H in original coordinates but its h_t/w_t may describe the cropped image.
- Failed / missing runs: no H means not a success and not a lock. A row the CSV lacks entirely is treated the same way in the paired tables, but the row-count check fails loudly (see 'Row counts').
- Duplicate keys (an infrastructure rerun): the LAST row per (pair, backbone, mode, crop) is used, matching 'a second failure counts as a failure'. Number of duplicates is reported.
- Sham '>10%' rule: computed as (runs whose success indicator differs between uncropped and sham-cropped, in either direction) / (paired sham runs) > 0.10, i.e. more than 8 of 80. The sham McNemar p is two-sided (no predicted direction).
- Reproduction: a fresh-vs-stored pair 'matches' if both mu_ed are finite and differ by <= 1 px, or both runs failed. Denominator = every uncropped Arm-1 run (overlay-none, Ti3AlC2, sham-none).
- Ti3AlC2 'succeed in the CJSJ run': judged from the stored CJSJ rows for the same configuration; the fresh rerun is compared to that.
- Offset configuration: GT translation = H_gt(target centre) - target centre (source px), each component rounded to the nearest 20 px, grouped per scene; the found configurations are listed (the prereg says 'about 5').
- H-A3 S1 = n_inliers / n_matches (as in the triage study; -inf for failed runs); false accept = S1 >= 0.1711 and mu_ed > 20. The AUROC bootstrap is a PAIR-level paired bootstrap (B = 10000, seed 0), because the prereg says 'paired bootstrap' and there are too few scenes for a cluster bootstrap; its CI is therefore optimistic.
- Holm correction: applied to the ONE-SIDED p-values of the 3 secondary H-A2 configurations.

(Everything above the interpretation notes is a pre-registered quantity; the breakdown tables are required descriptive reporting, not tests.)
