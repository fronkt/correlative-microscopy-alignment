# A3 log: per-subclass calibration from k hand-checked pairs (exploratory, data already seen)

Script: scripts/phaseA_a3_subclass_calibration.py (RNG seed 20260929, 1,000 draws per subclass x k, same draws for all variants).
Scored candidate: MA-RoMa, core, direct, seed 0 (187 pairs). Success = mu_ed <= 20 px; failed run = failure.
Global cut-off recomputed: Youden on SameSlice+SerialSectioning = 0.1711 (assert |x-0.171|<0.002 passed).
Sanity: global cut-off accepts 89 pairs, 50 failures among them; 42 confident false accepts outside design groups, all 42 in TEM (assert-free check printed).

## Variants run (complete list; nothing else was tried)
1. Rule A (pre-fixed): both classes in the k -> midpoint(min S1 of successes, max S1 of failures); if they overlap -> Youden on the k;
   all-one-label -> global cut-off. Accept if S1 >= cut.
2. Rule B (logged separately): all k fail -> reject whole subclass; all k succeed -> accept all; else midpoint/Youden as in A.
3. Global cut-off on the same held-out pairs (baseline).
Choices made and not tuned: failed-run S1 set to 0 (analyze_triage uses -inf; identical ranking, finite for midpoints);
"overlap" = min success S1 <= max failure S1. Rates use NaN-skipping over draws when a denominator is empty (held-out has no failures/successes/acceptances),
so per-subclass means are conditional on the denominator existing. Pooled = counts summed over subclasses per draw (pairs weighted equally), then rates.
No dead ends beyond this; no variant was dropped after looking at results.

## Subclass sizes (n, successes, within-subclass AUROC of S1)
Only subclasses with size > k enter a given k (k=1: 17 subclasses, k=2: 16, k=3: 14, k=5: 11).
Too small to say anything (n<6 or <2 successes or <2 failures): 14 of 19 flagged in JSON (`too_small`).
6 of 19 subclasses have ZERO successes (n = 1, 18, 27, 4, 16, 9): S1 has no within-subclass job there; only the subclass identity matters.
AUROC defined for only 10 subclasses; the non-TEM ones with n>=6 are 0.96-1.00 (AF9628 1.00 n=15; CoNi-AM90 SameSlice 1.00 n=6; In718-SolStr 0.96 n=10; TRIP1 1.00 n=7).
TEM: 1100C n=32, 4 succ, AUROC 0.598; 900C-largeFOV n=18, 1 succ, AUROC 0.647; 900C n=16, 0 succ (undefined); Ti3AlC2 n=3, 3 succ (undefined).
=> Within TEM there is essentially no S1 separation; a per-subclass cut-off cannot fix it.

## Pooled results (mean over draws; held-out pairs of subclasses with n>k)
| k | labelled pairs | global FA / rec / prec | Rule A FA / rec / prec | Rule B FA / rec / prec | failures saved A / B | successes lost A / B |
|---|---|---|---|---|---|---|
| 1 | 17 | .360 / .884 / .408 | .360 / .884 / .408 | .119 / .570 / .606 | 0 / 31.6 | 0 / 11.8 |
| 2 | 32 | .370 / .885 / .378 | .330 / .841 / .396 | .083 / .652 / .701 | 4.9 / 34.5 | 1.4 / 7.3 |
| 3 | 42 | .380 / .887 / .347 | .325 / .848 / .377 | .075 / .698 / .723 | 6.0 / 33.4 | 1.0 / 4.8 |
| 5 | 55 | .394 / .902 / .295 | .303 / .855 / .348 | .096 / .724 / .641 | 8.4 / 27.4 | 0.8 / 3.1 |
(Global baseline changes with k only because the eligible subclass set and held-out pairs change.)

## TEM (4 DislocationCharacterization subclasses pooled; k=1: 4, k=2: 4, k>=3: 3 eligible since Ti3AlC2 has n=3)
| k | global FA / rec | Rule A FA / rec | Rule B FA / rec / prec | failures saved A / B | successes lost A / B |
|---|---|---|---|---|---|
| 1 | .693 / 1.00 | .693 / 1.00 | .074 / .356 / .853 | 0 / 35.9 | 0 / 4.5 |
| 2 | .698 / 1.00 | .629 / .940 | .076 / .272 / .707 | 3.8 / 34.4 | 0.3 / 4.2 |
| 3 | .703 / 1.00 | .605 / .899 | .090 / .148 / .074 | 5.2 / 32.1 | 0.4 / 3.9 |
| 5 | .716 / 1.00 | .531 / .799 | .153 / .252 / .080 | 8.8 / 26.4 | 0.7 / 3.2 |
Rule A barely helps in TEM (FA .70 -> .60 at k=3). Rule B cuts FA to about .08-.15 but recall falls to .15-.36 and precision is low or unstable at k>=3 (few successes remain).

## Cost caveat, k=3
Rule A labels 42 pairs (3 x 14 eligible subclasses; 57 if all 19 counted) and saves 6.0 of the 41.7 failures the global cut-off accepts on the same held-out pairs (14%; 109.9 held-out failures), losing 1.0 success.
Rule B labels the same 42 pairs and saves 33.4 failure acceptances, losing 4.8 successes: ~0.8 failures avoided per labelled pair, but it works mainly because 6/19 subclasses have zero successes and TEM is nearly all failures (8 of 69 TEM pairs succeed, 5 of 66 excluding the 3 Ti3AlC2 pairs), i.e. it exploits the subclass base rate (a 3-image spot check discovers "this image type never works"), not S1 separation. For 3-pair labelling at the subclass level, most of the labelled pairs (42) are themselves pairs that got hand-verified and removed from the held-out set.

## Verdict
Rule A (midpoint cut-off) gives a small gain and only where S1 separates within subclass (mostly non-TEM); useless in TEM (AUROC ~0.6). Rule B (spot check to reject a subclass) removes most false accepts but is really subclass-level rejection driven by base rates and costs 3-5 true successes; exploratory only, nothing confirmed.
