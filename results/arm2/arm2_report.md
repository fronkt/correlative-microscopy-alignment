# Arm 2 report: does label-free triage replicate on independent materials data?

Source CSV: `C:\Users\frank\Documents\cma-triage-ext\results\arm2\candidates.csv`; manifest `C:\Users\frank\Documents\materials-bench\manifest.csv`. Analysis follows `prereg/arm2_materials_replication.md`; constants carried over unchanged from AmalgaMatch: S1 cut-off 0.1711, best single = ma_roma|pyramid_v2|none. Interpretive choices are listed at the end.

## Pool and row counts
- Manifest pairs: 95 (P 14, NIST 81 of up to 81; nominal total 95; missing vs nominal 0).
- Rows in file 1615; unique keys 1615; repeated attempts (last used) 0; stray rows (control pool or unknown pair) 0; expected 1615 = 95 x 17: **OK**.
- Failed rows 7 (0 with CUDA/OOM/timeout-type messages = infrastructure, rerun once per the prereg). Every failed run counts as a failure.

| count | infrastructure | error |
|---|---|---|
| 2 | no | only 0 correspondences |
| 2 | no | only 1 correspondences |
| 2 | no | only 2 correspondences |
| 1 | no | only 3 correspondences |

## NIST GT gate
Hand-check file `C:\Users\frank\Documents\materials-bench\nist\handcheck_clicks.csv`: 17 pairs clicked; skipped 3 (NIST_718_s025_bse2-om: no confident shared features; NIST_718_s100_bse1-om: no confident shared features; NIST_718_s150_bse1-om: no confident shared features). Rule: per pair type, the pooled median of the distance between the GT-mapped click and the click, in target px, must be <= 10.

| pair_type | pairs_clicked | points | median_target_px | median_source_px | max_target_px | verdict | reason |
|---|---|---|---|---|---|---|---|
| IN625-BSE2-vs-BSE1 | 4 | 24 | 8.01 | 4.01 | 31 | PASS | median <= 10 px |
| IN718-BSE1-vs-OM | 4 | 24 | 12.5 | 7.77 | 24.9 | EXCLUDED | median > 10 px |
| IN718-BSE2-vs-BSE1 | 5 | 30 | 9.77 | 4.88 | 135 | PASS | median <= 10 px |
| IN718-BSE2-vs-OM | 4 | 24 | 25.4 | 7.9 | 1.81e+03 | EXCLUDED | median > 10 px |
- Pair types in the primary set: IN625-BSE2-vs-BSE1, IN718-BSE2-vs-BSE1; excluded: IN718-BSE1-vs-OM, IN718-BSE2-vs-OM.
- Primary set: 55 pairs = P 14 + N 41; clusters 24.

## Pre-registered hypotheses, primary set

### Threshold 20 px (PRIMARY) (n = 55 pairs)
- **H2-1 S1 predicts MA-RoMa direct success (primary): SUPPORTED.** 46 successes / 9 failures; AUROC 0.983; cluster-bootstrap 95% CI [0.950, 1.000] (5th percentile 0.957; dropped replicates 1).
- **H2-2 the cut-off transfers: SUPPORTED.** S1 >= 0.1711: 48 accepted of 55; base rate 0.836; accepted success rate 0.938; difference +0.101, CI [0.028, 0.208] (5th percentile 0.033; dropped replicates 0).
- **H2-3 R beats pick-by-S1: NOT SUPPORTED.** R 50 vs 49; success only under R 1, only under the comparator 0; exact McNemar one-sided p = 0.5 (two-sided 1); descriptive cluster-bootstrap CI of the SR difference [0.000, 0.067].
- **H2-4 R beats best single MA-RoMa pyramid_v2 (secondary, low power stated in advance): NOT SUPPORTED.** R 50 vs 47; success only under R 3, only under the comparator 0; exact McNemar one-sided p = 0.125 (two-sided 0.25); descriptive cluster-bootstrap CI of the SR difference [0.000, 0.130].
- SR counts: R 50, pick-by-S1 49, best single 47, MA-RoMa direct 46, oracle best-of-15 51 (of 55).

### Threshold 1% of the source diagonal (secondary, reported for every hypothesis) (n = 55 pairs)
- **H2-1 S1 predicts MA-RoMa direct success (primary): UNTESTABLE.** 48 successes / 7 failures; AUROC 0.991; no CI computed (untestable rule).
- **H2-2 the cut-off transfers: SUPPORTED.** S1 >= 0.1711: 48 accepted of 55; base rate 0.873; accepted success rate 0.979; difference +0.106, CI [0.029, 0.217] (5th percentile 0.034; dropped replicates 0).
- **H2-3 R beats pick-by-S1: NOT SUPPORTED.** R 51 vs 51; success only under R 0, only under the comparator 0; exact McNemar one-sided p = 1 (two-sided 1); descriptive cluster-bootstrap CI of the SR difference [0.000, 0.000].
- **H2-4 R beats best single MA-RoMa pyramid_v2 (secondary, low power stated in advance): NOT SUPPORTED.** R 51 vs 49; success only under R 2, only under the comparator 0; exact McNemar one-sided p = 0.25 (two-sided 0.5); descriptive cluster-bootstrap CI of the SR difference [0.000, 0.096].
- SR counts: R 51, pick-by-S1 51, best single 49, MA-RoMa direct 48, oracle best-of-15 51 (of 55).

## Descriptive: P alone and N alone (point estimates only, no tests)

| set | threshold | n | R | pick_by_S1 | best_single | ma_roma_direct | oracle_best_of_15 | auroc_S1_ma_roma_direct | n_accepted_at_cut | accepted_success | p_R_vs_S1_one_sided | p_R_vs_best_one_sided |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| P alone | 20 px | 14 | 9 | 8 | 7 | 7 | 10 | 0.959 | 10 | 7 | 0.5 | 0.25 |
| P alone | 1% diag | 14 | 10 | 10 | 9 | 9 | 10 | 1 | 10 | 9 | 1 | 0.5 |
| N alone (gate-passing types) | 20 px | 41 | 41 | 41 | 40 | 39 | 41 | 1 | 38 | 38 | 1 | 0.5 |
| N alone (gate-passing types) | 1% diag | 41 | 41 | 41 | 40 | 39 | 41 | 1 | 38 | 38 | 1 | 0.5 |

## Per pair type / component (primary set, 20 px)

| group | n | clusters | ma_roma_direct | R | pick_by_S1 | best_single | oracle | gt_homography |
|---|---|---|---|---|---|---|---|---|
| IN625-BSE2-vs-BSE1 | 21 | 6 | 21 | 21 | 21 | 21 | 21 | 21 |
| IN718-BSE2-vs-BSE1 | 20 | 6 | 18 | 20 | 20 | 19 | 20 | 20 |
| P:A | 11 | 11 | 6 | 8 | 7 | 6 | 8 | 11 |
| P:D | 3 | 1 | 1 | 1 | 1 | 1 | 2 | 3 |

## Ceiling and per-candidate success (primary set)
- GT-fit homography SR@20: 55 / 55; GT-fit affine SR@20: 55 / 55.

| candidate | sr20 | sr_1pct |
|---|---|---|
| sift\|direct\|none | 41 | 41 |
| loftr\|direct\|none | 42 | 43 |
| roma\|direct\|none | 40 | 42 |
| ma_roma\|direct\|none | 46 | 48 |
| matchanything\|direct\|none | 9 | 12 |
| roma\|pyramid_v2\|none | 42 | 44 |
| ma_roma\|pyramid_v2\|none | 47 | 49 |
| roma\|direct\|invert | 33 | 33 |
| roma\|direct\|histmatch | 41 | 42 |
| roma\|direct\|clahe | 41 | 43 |
| roma\|direct\|gradmag | 35 | 35 |
| ma_roma\|direct\|invert | 40 | 41 |
| ma_roma\|direct\|histmatch | 45 | 47 |
| ma_roma\|direct\|clahe | 45 | 47 |
| ma_roma\|direct\|gradmag | 41 | 43 |

## Overlay pairs (manifest overlay_src / overlay_tgt not 'none'), analysed as-is and reported separately
[{'pair_id': 'NIST_625_s200_bse2-bse1#0', 'ma_roma_direct_success': True}, {'pair_id': 'NIST_718_s300_bse2-bse1#0', 'ma_roma_direct_success': True}, {'pair_id': 'defdap_14245480_Ti64_bulk#0', 'ma_roma_direct_success': True}, {'pair_id': 'defdap_16633511_Ti_CS1#0', 'ma_roma_direct_success': False}]

## Interpretation notes and choices the prereg left open
- Candidate order for R: the prereg lists 'Core: SIFT, LoFTR, RoMa, MA-RoMa and MatchAnything direct, plus RoMa and MA-RoMa pyramid_v2; Transform pool: RoMa and MA-RoMa direct on {invert, histmatch, CLAHE, gradmag}'. That listing order is used for every tie (R and pick-by-S1): sift, loftr, roma, ma_roma, matchanything, roma pyramid_v2, ma_roma pyramid_v2, roma x (invert, histmatch, clahe, gradmag), ma_roma x (same). R = argmax n_inliers over the 15 (a failed run has n_inliers = -inf; if all 15 fail, R fails). n_inliers of a pyramid_v2 row is the final-transform inlier count as written by run_triage_candidates.py.
- S1 = n_inliers / n_matches per row (as in the CJSJ / Arm 1 code); a failed run or one with zero matches has S1 = -inf, is never accepted, and counts as a failure. Pick-by-S1 (H2-3) = argmax S1 over the same 15, ties to the first listed.
- One-sided vs two-sided: the McNemar p in H2-3 / H2-4 is the ONE-SIDED exact binomial tail P(X >= n_pos), X ~ Bin(n_pos + n_neg, 1/2), in the predicted direction (R better). 'Supported' needs p < 0.05 AND R having more successes. The two-sided exact p (cma.triage.mcnemar_exact) is printed beside it. H2-4 has no stated support rule beyond the test; the same rule is applied and it is labelled secondary.
- Bootstrap CIs are the 95% PERCENTILE interval (2.5th and 97.5th percentiles, cma.triage.ci95) over B = 10,000 cluster resamples, seed 0, resampling whole manifest `cluster` values within the analysed set. H2-1 needs AUROC >= 0.80 and the 2.5th percentile > 0.5; H2-2 needs the 2.5th percentile of (accepted success rate - base rate) > 0. The 5th percentile (the one-sided alpha = 0.05 bound) is reported beside it and would give the same verdict unless flagged. Replicates in which a statistic is undefined (one class missing, nobody accepted) are dropped and their number is reported.
- H2-1 untestable rule: counted on MA-RoMa direct success in the analysed set AFTER the GT gate (primary set), separately for each threshold. Untestable is reported as UNTESTABLE, neither supported nor failed, and no CI is computed.
- H2-2: accepted = S1 >= 0.1711 (inclusive). The statistic is the success rate among accepted MA-RoMa direct pairs minus the base success rate of MA-RoMa direct over all pairs in the set (the base rate includes the accepted pairs). If no pair is accepted the hypothesis is reported NOT SUPPORTED (no accepted pairs).
- Thresholds: success is mu_ed <= 20 px (inclusive), in SOURCE pixels, exactly as in the runner. The secondary threshold is mu_ed <= 0.01 * sqrt(h_s^2 + w_s^2), with the source size from the manifest (same frame as mu_ed). Every hypothesis is computed under both; the 20 px verdict is the primary one. Pair sets, S1 and R do not depend on the threshold; only the success labels do.
- Failed / missing runs: status != ok, a non-finite mu_ed, or a row absent from the CSV counts as a failure (mu_ed = inf, S1 = -inf, n_inliers = -inf). Row-count check: expected (manifest pairs) x 17; a mismatch stamps the report NOT the pre-registered analysis and exits non-zero unless --allow-partial. Repeated attempts of one key (an infrastructure rerun): the LAST row is used and the number of repeats is reported.
- Set definitions: P = every non-NIST manifest pair (components A and D, 14 pairs, 12 clusters nominally); N = NIST pairs of the pair types that pass the gate. Primary set = P union N(passing). 'P alone' / 'N alone' are descriptive point estimates (no bootstrap, no significance claims). Pair types are the manifest `subclass` (IN718-BSE1-vs-OM, IN718-BSE2-vs-BSE1, IN718-BSE2-vs-OM, IN625-BSE2-vs-BSE1). Pairs of a failing type are excluded from every analysis, including 'N alone'; only the gate table shows them.
- Missing NIST pairs (download failures) are simply absent from the manifest; the report gives the manifest count against the nominal 95 (14 + 81) and the number of NIST pairs.
- GT gate arithmetic: the prereg says the distance between the GT-mapped click and the click, 'in target px of the pool images'. The hand-check tool records left = target, right = source (pool PNG pixels). The GT-mapped position of the SOURCE click is taken through the inverse GT homography into the target frame and compared with the target click, so the distance is in target pixels (which are finer than source pixels, so this is the stricter reading). The GT homography is the least-squares fit (cv2.findHomography, method 0) to the pool's 25 GT points target -> source, which were generated from the chained GT homography, so it reproduces it to the CSV rounding. The median is taken over ALL clicked points of a pair type pooled (not a median of per-pair medians); pass = median <= 10 px inclusive. Skipped pairs contribute no points. A type with no clicked points cannot be verified and is EXCLUDED (reported as such). Source-px medians are shown too.
- The gate only reads Frank's click file; an absent file means the primary analysis is refused. --no-gate-dry-run skips the gate (all NIST types kept), writes arm2_DRYRUN_* files and stamps them; if a click file exists it is still ignored in a dry run.
- Overlay pairs: the prereg says any shared-overlay pair found later is 'analysed as it is and reported separately'. Manifest pairs whose overlay_src / overlay_tgt is not 'none' are listed in a separate descriptive table and stay in the primary set.
- H2-3 / H2-4 use the pair-level exact McNemar as pre-registered; pairs within a cluster are not independent, so a cluster bootstrap CI on the success-rate difference is printed beside each as a DESCRIPTIVE sensitivity check (not a decision rule).

(Everything before the interpretation notes is a pre-registered quantity or required descriptive reporting; nothing here is a claim beyond the pre-registered tests.)
