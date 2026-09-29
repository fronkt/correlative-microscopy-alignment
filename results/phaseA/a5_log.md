# A5 log: the RoMa-family-only S1 pick rule (exploratory, data already seen)

Script: `scripts/phaseA_a5_family_rule.py` -> `a5_family_rule.json`, `a5_wins_losses.csv`. Run with `PYTHONPATH=src`.
Best single candidate = `ma_roma|pyramid_v2|none|s0` (47/187). Success = mu_ed <= 20 px, failed runs = failure.
Ties in argmax go to the first candidate in sorted voter order (as `idxmax` in E4). Pairs with no eligible candidate = failure.

## Reproduction of E4 (exact)
The 12 = the 6 `roma` + 6 `ma_roma` voting candidates (direct x {none, invert, histmatch, clahe, gradmag} + pyramid_v2).
MatchAnything, LoFTR and SIFT are NOT in the set. Result: 56/187, best single 47, won 14, lost 5, McNemar p = 0.063568,
oracle 80, all identical to `results/triage/exploratory.json` (asserted in the script). 29 pairs tie at the max S1
within the 12; in none of them does the outcome depend on the tie-break.

## Every variant tried (McNemar vs best single, exact two-sided)
| # | variant | SR20 n | won | lost | p |
|---|---|---|---|---|---|
| 1 | H3 select_S1, all 15 (prior work) | 42 | 6 | 11 | 0.332 |
| 2 | H3 select_S2 (prior work) | 40 | 4 | 11 | 0.118 |
| 3 | H3 select_S3 (prior work) | 47 | 5 | 5 | 1.0 |
| 4 | E4 = R2, S1 over dense 12 (prior work) | 56 | 14 | 5 | 0.0636 |
| 5 | R1 S1 over all 15, n_matches >= 50 | 46 | 6 | 7 | 1.0 |
| 6 | R1 ... >= 100 | 46 | 6 | 7 | 1.0 |
| 7 | R1 ... >= 200 | 46 | 6 | 7 | 1.0 |
| 8 | R1 ... >= 500 | 46 | 6 | 7 | 1.0 |
| 9 | R1 ... >= 1000 | 56 | 14 | 5 | 0.0636 |
| 10 | R1 ... >= 5000 | 56 | 14 | 5 | 0.0636 |
| 11 | R3a S1 over dense direct 10 (no pyramid) | 54 | 13 | 6 | 0.167 |
| 12 | R3b S1 over the 2 pyramid_v2 only | 51 | 9 | 5 | 0.424 |
| 13 | R3c (extra) S1 over roma-only 6 | 49 | 18 | 16 | 0.864 |
| 14 | R3d (extra) S1 over ma_roma-only 6 | 59 | 16 | 4 | 0.0118 |
| 15 | R4 max n_inliers over all 15 | 56 | 14 | 5 | 0.0636 |
| 16 | R4b (extra) max n_inliers over dense 12 | 56 | 14 | 5 | 0.0636 |

Rows 2 and 3 are from `results/triage/summary.json` (H3), counted because they were also tried on these 187 pairs.
Dead ends (no gain over best single): R1 with N <= 500, R3b, R3c, and R1 in general below 1000.
Not counted as variants: leave-one-group-out, leave-one-scene-out, bootstrap, tie check (sensitivity of one rule).
R3c/R3d/R4b were added after seeing R3a/R3b and R4, so they are post hoc; R3d is the smallest p of all (0.0118) and is
the least trustworthy for that reason.

## Findings
- Wins/losses (`a5_wins_losses.csv`): 10 of 14 wins are in ONE scene (MoTaTiZrHf-HEA-DDRX-1100C, TEM), where best single
  is at 177-192 px (a confident wrong lock) and the picked RoMa/MA-RoMa transform variant lands at 12-19 px, i.e. within
  1-8 px of the 20 px cut. The other 4 wins come from 4 different scenes. Of the 5 losses, 2 are borderline (20.98, 21.76
  px vs best single 19.5, 19.8) and 3 are gross (492, 1055, 494 px vs 11-20 px).
- Leave-one-group-out: dropping DislocationCharacterization -> gain 2 (won 4, lost 2, p 0.69); dropping SerialSectioning
  -> gain 7 (p 0.14); Multiscale or SameSlice -> gain 9 (p 0.049); FractureSurfaces or SlipPartitioning (no wins or
  losses there) -> unchanged (p 0.064). Leave-one-scene-out (35 scenes): gain 1 to 10, p 0.031 to 1.0; dropping the
  TEM 1100C scene alone takes the gain from 9 to 1.
- Mechanism: the rule works by excluding candidates with few matches, whose S1 is inflated (SIFT median 99 matches, 79 of
  187 argmax picks; S1 median 0.31 vs 0.13 for dense). Restricting all 15 to n_matches >= 1000 (or 5000) reproduces E4's
  56/14/5 exactly, while thresholds <= 500 do not help (SIFT still wins 56 pairs at N=500). At N=1000 the only
  difference from E4 is that MatchAnything (median 410 matches, but >= 1000 on some pairs) is admitted for 5 picks with
  no change in outcome. So "exclude low n_matches" and "exclude sparse matchers" are indistinguishable on this data; the
  useful cut lies between 500 and 1000 on a coarse grid. Max n_inliers over all 15 gives the identical picks (dense
  matchers cap at 10,000 matches, so this is nearly "prefer the dense matchers at the cap").
- Direct vs pyramid: direct-only 54 (p 0.167), pyramid-only 51 (p 0.42): the gain needs the input-transform variants
  (146 of the 187 E4 picks are invert/gradmag/histmatch/clahe variants, 28 untransformed direct, 13 pyramid), i.e. it is mostly selection among input-transform variants.
- Bootstrap (scene-clustered, B=10,000): gain +4.8 pts, 95% CI [-1.6, +12.4] for R2, identical for best R1 (N>=1000,
  same outcome vector); 15% of resamples have gain <= 0.
- Multiplicity: 16 pick rules tried against the best single on these 187 pairs (table above). Best p among them is
  0.0118 (R3d, post hoc); Bonferroni x16 = 0.19. E4's own p 0.0636 x16 = 1.0. Alpha per test would be 0.0031.
  Real forking-path count is higher since "dense-only" itself was chosen after seeing SIFT dominate the picks (E3).
