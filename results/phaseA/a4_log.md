# A4 log: matcher-comparable score (exploratory; data already seen, nothing here is confirmatory)

Script: scripts/phaseA_a4_matcher_score.py. Outputs: a4_matcher_score.json, a4_scores.csv (voting rows only).

## Code facts (read, not guessed)
- Estimator: cv2.USAC_MAGSAC (MAGSAC++), maxIters 10000, confidence 0.999, family "auto" (homography vs affine by BIC), threshold 5.5 px (src/cma/estimators/consensus.py).
- Frame: fit_transform(src_xy=target pts, dst_xy=source pts); mask and residuals are in the SOURCE image frame. So p = pi*5.5^2/(w_s*h_s).
- pyramid_v2 tile/zoom correspondences are mapped to the full source frame before the fit; full-source area is used (zoom-crop null would be tighter; not stored).
- Reproduced: best single = ma_roma|pyramid_v2 47/187; pick-by-S1 42; oracle 80. Self-check (3 toy cases vs exact Fraction arithmetic) passes.

## Variants tried (all reported)
1. nfa4 = -logNFA, N_tests = C(n_matches,4), p from source area. (primary)
2. nfa1 = same with N_tests = 1.
3. nfa4t = p from TARGET area (sensitivity; dead end, no change in conclusions).
4. wilson = Wilson 95% lower bound of n_inliers/n_matches.
5. ninl = raw n_inliers.
6. S1 reference.
Not tried: C(n,3) for affine fits (family stored only as a label; ties to variant 1/4 ordering anyway), zoom-crop area.
scipy binom.logsf returned -inf (underflow) on 12,391 of ~15k calls (all dense runs); replaced by logsumexp(logpmf) fallback, so no silent -inf.

## Saturation check (voting rows, ok runs). Result: SATURATED for dense matchers
-logNFA vs n_inliers Spearman: ma_roma 0.995, roma 0.994, loftr 0.964, sift 0.910, matchanything 0.861; all voting 0.993.
-logNFA vs S1 Spearman: ma_roma 0.995, roma 0.994 (dense matchers cap n_matches at 10,000, so S1 ~ n_inliers there), loftr 0.877, sift 0.775, matchanything 0.718; all voting 0.777.
Median -logNFA: roma 9,056, ma_roma 11,806, loftr 58, matchanything 35, sift 39. 84% of voting rows > 700 nats. nfa1 and nfa4 are near-identical (picks identical).
So -logNFA is essentially a log-scale n_inliers for dense matchers; it ranks dense above sparse by construction (p is tiny, images are large).

## Pooled AUROC (2,805 voting rows, scene-clustered B=10,000)
S1 0.862 [0.764, 0.947]; nfa4 0.883 [0.819, 0.942]; nfa1 0.883; nfa4t 0.884; wilson 0.870 [0.772, 0.957]; ninl 0.888 [0.827, 0.941].
Delta vs S1: nfa4 +0.021 [-0.018, +0.063]; ninl +0.027 [-0.015, +0.068]; wilson +0.009 [+0.005, +0.013] (excludes 0, tiny).
Per-candidate: MA-RoMa direct AUROC S1 = wilson = ninl = 0.854 (n_matches constant 10,000), nfa4 0.844. SIFT direct (4 successes only): S1 0.622, nfa4 0.649, ninl 0.672.

## Per-pair pick (SR@20 of 187)
| score | n | vs best single 47 (won/lost, McNemar p) | vs S1 42 (won/lost, p) | SIFT picks |
|---|---|---|---|---|
| S1 | 42 | -5 (p 0.33) | - | 79 |
| nfa4 / nfa1 / nfa4t | 56 | +9 (14/5, p 0.064; rate diff CI [-0.016, +0.124]) | +14 (14/0, p 0.0001; CI [+0.008, +0.161]) | 1 |
| ninl | 56 | +9 (same as nfa4, p 0.064) | +14 (14/0) | 0 |
| wilson | 48 | +1 (p 1.0) | +6 (6/0, p 0.031) | 65 |
Oracle 80. nfa4/ninl picks: roma 105, ma_roma 81, sift 1 (nfa4) or 0 (ninl); zero LoFTR/MatchAnything picks.
Wins vs best single by picked backbone (nfa4): roma +10/-4, ma_roma +4/-1. Wilson still picks SIFT 65 times (2 losses by SIFT picks).

## Verdict
Wilson does not fix the sparse bias (65 SIFT picks, 48/187). -logNFA fixes it but only because it saturates into raw n_inliers: it never picks a sparse matcher (1 SIFT pick), gets 56/187, the same as picking by n_inliers. Gain over best single (+9) is not significant (McNemar p 0.064, CI includes 0); gain over S1 (+14, 14/0) is. The a-contrario null adds nothing beyond n_inliers here. Caveat: a "score" that always picks a dense matcher is really a pool-restriction rule; choosing among dense candidates is where S1 already equals n_inliers.
