# A2 log: seed disagreement D (exploratory, forking-paths list)

Script: scripts/phaseA_a2_seed_disagreement.py (run with PYTHONPATH=src). Outputs: a2_seed_disagreement.json, a2_per_pair.csv.
All numbers are exploratory on data already seen; nothing here is confirmatory.

Setup checks. roma and ma_roma each have exactly 6 direct/none runs per pair (seed 0 core + seeds 1-5 control), 187 pairs, no duplicates,
0 failed runs in any of the 12 seed columns (so all failed-run rules give identical numbers). Grid = 5x5 over the target image (h_t,w_t),
source px, mean over grid points per transform_distance; D = median over the 15 seed pairs. Success = mu_ed <= 20. Bootstrap: clusters = scene
(scene column; 187 pairs), B = 10,000, seed 0; rank percentiles for the combined score computed once on the full 187 (as in analyze_triage.py).
Seed-0 successes: roma 38, ma_roma 44 (of 187).

Flips (any seed differs in success across the 6 seeds): ma_roma always 35 / never 136 / flip 16 (reproduces E1's 16). roma always 31 / never 140 / flip 16.

## Variants tried (every one; AUROC = seed-0 success; delta = AUROC(combined) - AUROC(S1), paired scene-cluster bootstrap)
S1 alone: roma 0.944 [0.866, 0.995]; ma_roma 0.854 [0.724, 0.976].

| variant | backbone | AUROC(-D) [CI] | AUROC(combined) | delta [CI] |
|---|---|---|---|---|
| D_strict (+inf if any seed failed; PRIMARY) | roma | 0.917 | 0.950 | +0.006 [-0.015, 0.028] |
| D_strict | ma_roma | 0.870 | 0.876 | +0.022 [-0.003, 0.051] |
| D_ignore (drop failed seeds) | both | identical to D_strict (no failed runs) | | |
| D_raw (median with failed entries = inf) | both | identical to D_strict | | |
| D_no_seed0 (seeds 1-5 only, 10 pairs; no leakage of the labelled run) | roma | 0.907 | 0.945 | +0.001 [-0.023, 0.021] |
| D_no_seed0 | ma_roma | 0.859 | 0.871 | +0.017 [-0.008, 0.043] |
| D2 = seeds 0,1 only (one extra run) | roma | 0.892 | 0.946 | +0.002 [-0.018, 0.028] |
| D2 | ma_roma | 0.836 | 0.859 | +0.005 [-0.021, 0.029] |

-D CIs for the primary variant are in the JSON (auroc_minusD). Every other 2-seed pair (a,c) is in the JSON (point estimates only): AUROC(-D) roma 0.866-0.919,
ma_roma 0.813-0.888; combined roma 0.928-0.952, ma_roma 0.850-0.894. A free-choice best pair (3-4, ma_roma 0.894) is a max over 15 and NOT a result.
Dead ends / non-results: no variant's paired CI excludes 0; D never beats S1 alone on AUROC for either backbone.

## Distribution of D (D_strict, px)
roma: median 25.0 overall; seed-0 success median 2.3 (q90 11.9, max 17.8; D<1 in 34%, <5 in 68%, <20 in 100%); seed-0 failure median 45.7 (D<1 1%, <5 8%, <20 34%).
ma_roma: overall median 17.5; success median 2.8 (D<1 30%, <5 70%, <20 93%, max 68.7); failure median 35.7 (D<1 3%, <5 13%, <20 43%).
So D is NOT almost always ~0: only 10% (ma_roma) / 7% (roma) of pairs have D<1 px; the 16 flip pairs are only those that cross the 20 px line, while most pairs
have seeds that wander by several px to hundreds of px. Failures with D<20 are common (34-43%), i.e. consistent-but-wrong runs exist.

## Within-group AUROC of -D (D_strict) vs S1 (point [95% CI], groups with both classes)
ma_roma: Dislocation (n=69, 8 succ) D 0.818 vs S1 0.811; SameSlice (26, 16) D 0.863 vs S1 0.856; SerialSectioning (42, 9) D 0.946 [0.849, 0.995] vs S1 0.916; Multiscale (13, 5) 1.0 vs 1.0.
roma: Dislocation (69, 17) D 0.760 vs S1 0.869; SameSlice (26, 12) D 0.976 vs 0.976; SerialSectioning (42, 4) D 0.888 vs 0.934; Multiscale 1.0 vs 1.0.
FractureSurfaces (ma_roma 6/6 success, roma 0/6) and SlipPartitioning (0 successes) skipped: one class only. Small n; CIs wide (in JSON).

## The 42 confident false accepts (ma_roma; assert 42 passed; Youden cut 0.1711 on SameSlice+SerialSectioning; 19 held-out successes also accepted by S1)
D_strict of the 42: median 7.6 px (q25 3.2, q75 13.2, q90 40.3, max 78.4); <1 px 12%, <5 px 40%, <20 px 86%. Accepted successes: median 2.0 px (<1 px 37%, <5 79%).
D cut-off (reject if D > t) | false accepts caught of 42 | accepted successes rejected of 19
 t=1: 37 | 12 (63%);  t=2: 34 | 8 (42%);  t=5: 25 | 4 (21%);  t=10: 17 | 3 (16%);  t=20: 6 | 1 (5%).
Seeds 0+1 only (D2): t=1: 34 | 11;  t=2: 31 | 8;  t=5: 20 | 5;  t=10: 12 | 4;  t=20: 6 | 1.
Reading: a D cut-off does catch many of the 42, but the false accepts are only moderately less stable than true successes, so catching most (>=34/42) costs
rejecting about 40-60% of the accepted successes. Caveat: the 42 are selected on S1 and the 19 successes are only the S1-accepted held-out ones (small denominator, no CI); the cut-offs
were swept post hoc.
Side note roma (same recipe, not the 42): Youden cut 0.272, 9 held-out false accepts, 16 accepted successes; t=1 catches 8/9 but rejects 8/16; t=5 catches 3/9, rejects 3/16.
