"""Cross-modal appearance divergence as a measured axis (Reviewer 2, R2.1).

The manuscript isolates field-of-view overlap with a controlled ladder
(scripts/run_fov_ladder.py) but infers "appearance divergence" by
elimination. This script turns appearance into a *measured* per-pair scalar so
the two axes can be compared on the same 187-pair benchmark, and so the
confound between them can be reported as a number instead of an assertion.

WHAT IS MEASURED
----------------
For each pair the ground-truth correspondences are used to estimate a global
affine source -> target, the target image is warped back into the source
frame, and normalised mutual information (NMI) is computed on the pixels that
are valid in both frames. High NMI = the two modalities carry statistically
similar intensity structure over the same physical region; low NMI = the
modalities disagree about what the same material looks like. NMI is used (not
raw MI) because MI grows with the entropy of the individual images, which
varies wildly across SEM / EBSD / OM.

PINNED ESTIMATOR CHOICES (change any of these and the number changes)
--------------------------------------------------------------------
1.  GT geometry: cv2.estimateAffine2D(src_xy, tgt_xy, method=RANSAC,
    ransacReprojThreshold=5.0). Affine, not homography: the GT point sets are
    small and a homography overfits. cv2.setRNGSeed(0) is called once so that
    RANSAC is deterministic across runs.
2.  Colour: any 3-channel image is reduced by an unweighted channel mean.
3.  Overlap: the warped target's validity mask, intersected with np.isfinite
    on BOTH images, cropped to its bounding box. Pairs with < 500 valid pixels
    before resampling, or < 200 after, are failures -- not silently dropped.
4.  Resolution cap: the overlap crop's long side is capped at 384 px with
    cv2.INTER_AREA. Without a cap, NMI is partly a function of image size
    (more pixels -> less histogram noise), which would smuggle a magnification
    effect into the appearance axis. The mask is resampled with the same
    filter and re-thresholded at > 0.999 so no partially interpolated border
    pixel enters the histogram.
5.  Histogram: bins=32 in both the joint histogram
    (cma.metrics.mutual_information, already used by the classical control
    pipeline) and in the marginal entropies.
6.  Normalisation: NMI = MI / min(H(a), H(b)). min- rather than mean- or
    sqrt-normalisation, so a pair only counts as "similar" if the *less*
    informative of the two images is explained.
7.  The marginal entropies use min-max scaling to [0, 1] before binning,
    whereas mutual_information internally uses 1--99th percentile clipping.
    These two normalisations are NOT the same, so NMI here is not guaranteed
    to lie in [0, 1]. This is kept deliberately: it is the estimator whose
    correlation with FOV was pre-verified. --variant marginal-entropy
    recomputes the internally consistent version (both marginals taken from
    the same joint histogram as the MI) as a robustness check and prints, but
    does not commit, its correlation.

NON-FINITE GUARD
----------------
The estimator is fragile in one specific way. If non-finite pixels reach
np.histogram2d they fall outside every bin, silently shrink the effective
sample, and (measured) can flip the sign of the NMI/FOV correlation.
Non-finite pixels are therefore removed from the overlap mask *before* any
histogram is formed, again after resampling (INTER_AREA propagates NaN), and
the script asserts that all 187 pairs resolve to a finite NMI. It exits
non-zero rather than writing a partially-NaN CSV.

USAGE
-----
    python scripts/appearance_axis.py                 # analyse (measures if no CSV)
    python scripts/appearance_axis.py --recompute     # force re-measure (~3 min, CPU)
    python scripts/appearance_axis.py --recompute --variant marginal-entropy
    python scripts/appearance_axis.py --analysis-only

Writes results/appearance_nmi.csv (pair_id, nmi, n_overlap_px) and prints the
analysis reproduced in reports/appearance_axis.md.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "data" / "AmalgaMatch"
NMI_CSV = ROOT / "results" / "appearance_nmi.csv"
FOV_CSV = ROOT / "results" / "fov_ratios.csv"
BASELINES = ROOT / "results" / "baselines_A.csv"

BINS = 32
MAX_SIDE = 384
RANSAC_THRESH = 5.0
MIN_OVERLAP_PX = 500
MIN_SAMPLE_PX = 200
N_PAIRS = 187
SR_THRESH = 10.0
B = 10_000
SEED = 0


# --------------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------------
def _entropy_minmax(x: np.ndarray) -> float:
    """Shannon entropy (nats) of x after min-max scaling into BINS bins."""
    span = np.ptp(x)
    y = (x - x.min()) / (span + 1e-12)
    hist, _ = np.histogram(y, bins=BINS, range=(0.0, 1.0))
    p = hist / max(hist.sum(), 1e-12)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _norm_pct(x: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(x, [1, 99])
    if hi <= lo:
        return np.clip(x - x.min(), 0, 1)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _nmi_marginal_consistent(a: np.ndarray, b: np.ndarray) -> float:
    """Robustness variant: MI and both marginals from ONE joint histogram."""
    joint, _, _ = np.histogram2d(
        _norm_pct(a), _norm_pct(b), bins=BINS, range=[[0, 1], [0, 1]]
    )
    pj = joint / max(joint.sum(), 1e-12)
    pa = pj.sum(axis=1)
    pb = pj.sum(axis=0)
    denom = pa[:, None] * pb[None, :]
    nz = (pj > 0) & (denom > 0)
    mi = float((pj[nz] * np.log(pj[nz] / denom[nz])).sum())
    ha = float(-(pa[pa > 0] * np.log(pa[pa > 0])).sum())
    hb = float(-(pb[pb > 0] * np.log(pb[pb > 0])).sum())
    return mi / (min(ha, hb) + 1e-12)


def measure(variant: str = "pinned") -> list[tuple[str, float, int, float]]:
    """Return [(pair_id, nmi, n_overlap_px, nmi_variant)] for all 187 pairs."""
    import cv2

    from cma.data.amalgamatch import AmalgaMatchLoader
    from cma.metrics.mutual_information import mutual_information

    cv2.setRNGSeed(SEED)
    ds = AmalgaMatchLoader(DATA)
    rows: list[tuple[str, float, int, float]] = []
    failures: list[str] = []
    t0 = time.time()

    for i, rec in enumerate(ds.records):
        pid = rec.pair_id
        try:
            pair = ds.load_pair(rec)
            src_img = pair.source
            tgt_img = pair.target
            if src_img.ndim == 3:
                src_img = src_img.mean(axis=2)
            if tgt_img.ndim == 3:
                tgt_img = tgt_img.mean(axis=2)
            src_img = src_img.astype(np.float32)
            tgt_img = tgt_img.astype(np.float32)

            affine, _ = cv2.estimateAffine2D(
                pair.gt.src_xy.astype(np.float32),
                pair.gt.tgt_xy.astype(np.float32),
                method=cv2.RANSAC,
                ransacReprojThreshold=RANSAC_THRESH,
            )
            if affine is None:
                failures.append(pid + ": affine estimation returned None")
                continue

            h, w = src_img.shape
            warped = cv2.warpAffine(
                tgt_img, affine, (w, h),
                flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR, borderValue=0.0,
            )
            mask = cv2.warpAffine(
                np.ones_like(tgt_img), affine, (w, h),
                flags=cv2.WARP_INVERSE_MAP | cv2.INTER_NEAREST, borderValue=0.0,
            ) > 0.5
            # GUARD: non-finite pixels must never reach a histogram.
            mask &= np.isfinite(src_img) & np.isfinite(warped)
            n_mask = int(mask.sum())
            if n_mask < MIN_OVERLAP_PX:
                failures.append(pid + f": overlap {n_mask} px < {MIN_OVERLAP_PX}")
                continue

            ys, xs = np.where(mask)
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            a = src_img[y0:y1, x0:x1].copy()
            b = warped[y0:y1, x0:x1].copy()
            m = mask[y0:y1, x0:x1].astype(np.float32)
            scale = MAX_SIDE / max(a.shape)
            if scale < 1.0:
                a = cv2.resize(a, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                b = cv2.resize(b, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                m = cv2.resize(m, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            keep = m > 0.999
            av = a[keep].astype(np.float64)
            bv = b[keep].astype(np.float64)
            # GUARD again after resampling: INTER_AREA propagates NaN.
            finite = np.isfinite(av) & np.isfinite(bv)
            av, bv = av[finite], bv[finite]
            if av.size < MIN_SAMPLE_PX:
                failures.append(pid + f": {av.size} finite overlap px < {MIN_SAMPLE_PX}")
                continue

            mi = mutual_information(av, bv, bins=BINS)
            den = min(_entropy_minmax(av), _entropy_minmax(bv)) + 1e-12
            nmi = float(mi / den)
            if not np.isfinite(nmi):
                failures.append(pid + f": non-finite NMI (MI={mi}, den={den})")
                continue
            alt = (
                _nmi_marginal_consistent(av, bv)
                if variant == "marginal-entropy"
                else float("nan")
            )
            rows.append((pid, nmi, int(av.size), alt))
        except Exception as exc:  # noqa: BLE001 - report, never mask
            failures.append(pid + f": {type(exc).__name__}: {exc}")
        if (i + 1) % 40 == 0:
            print(f"  ... {i + 1}/{len(ds.records)} ({time.time() - t0:.0f}s)", flush=True)

    print(f"measured {len(rows)}/{len(ds.records)} pairs in {time.time() - t0:.0f}s")
    for f in failures:
        print("  FAILED " + f, file=sys.stderr)
    if len(rows) != N_PAIRS:
        raise SystemExit(
            f"ABORT: {len(rows)}/{N_PAIRS} pairs resolved to a finite NMI. "
            "A partial appearance axis is not a measurement; fix the failures above."
        )
    assert all(np.isfinite(r[1]) for r in rows), "non-finite NMI survived the guard"
    return rows


def write_csv(rows) -> None:
    NMI_CSV.parent.mkdir(parents=True, exist_ok=True)
    with NMI_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "nmi", "n_overlap_px"])
        for pid, nmi, npx, _ in sorted(rows):
            w.writerow([pid, f"{nmi:.6f}", npx])
    print(f"wrote {NMI_CSV.relative_to(ROOT)} ({len(rows)} rows)")


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------
def load_nmi() -> dict[str, float]:
    with NMI_CSV.open(newline="", encoding="utf-8") as f:
        return {r["pair_id"]: float(r["nmi"]) for r in csv.DictReader(f)}


def load_fov() -> dict[str, float]:
    with FOV_CSV.open(newline="", encoding="utf-8") as f:
        return {r["pair_id"]: float(r["fov_area_ratio"]) for r in csv.DictReader(f)}


def load_err(backbone: str, mode: str) -> dict[str, float]:
    """Per-pair error, TPS-refined with fallback to raw (paper convention)."""
    out: dict[str, float] = {}
    with BASELINES.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["backbone"] != backbone or r["mode"] != mode:
                continue
            if r["status"] != "ok":
                out[r["pair_id"]] = float("inf")
                continue
            v = r["mu_ed_tps"] or r["mu_ed"]
            out[r["pair_id"]] = float(v) if v else float("inf")
    return out


def scene_of(pair_id: str) -> str:
    """eval_<subclass>_<sceneidx>#<pairidx> -> eval_<subclass>_<sceneidx>."""
    return pair_id.rsplit("#", 1)[0]


def cluster_bootstrap(success, group_hi, scenes, rng):
    """Delta SR@10 (hi - lo) with a 95% CI from a bootstrap over SCENES.

    Pairs inside one scene share a source image and are not independent, so
    the resampling unit is the scene, not the pair. B=10,000 resamples,
    percentile 95% CI on the difference, two-sided p.
    """
    success = np.asarray(success, dtype=float)
    group_hi = np.asarray(group_hi, dtype=bool)
    scenes = np.asarray(scenes)
    uniq = sorted(set(scenes.tolist()))
    idx_by_scene = [np.flatnonzero(scenes == s) for s in uniq]
    d_obs = float(success[group_hi].mean() - success[~group_hi].mean())
    draws = rng.integers(0, len(uniq), size=(B, len(uniq)))
    deltas = []
    for row in draws:
        sel = np.concatenate([idx_by_scene[j] for j in row])
        g = group_hi[sel]
        if g.all() or not g.any():
            continue
        s = success[sel]
        deltas.append(s[g].mean() - s[~g].mean())
    d = np.asarray(deltas)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = min(1.0, 2.0 * min(float((d <= 0).mean()), float((d >= 0).mean())))
    return d_obs, float(lo), float(hi), p, len(d)


def analyse() -> None:
    from scipy import stats

    nmi = load_nmi()
    fov = load_fov()
    ids = sorted(set(nmi) & set(fov))
    assert len(ids) == N_PAIRS, f"{len(ids)} pairs joined, expected {N_PAIRS}"

    x = np.array([np.log10(fov[i]) for i in ids])
    y = np.array([nmi[i] for i in ids])
    scenes = [scene_of(i) for i in ids]

    print("=" * 74)
    print("1. APPEARANCE AXIS (NMI) -- distribution")
    print("=" * 74)
    print(f"n = {len(ids)} pairs over {len(set(scenes))} scenes")
    q = np.percentile(y, [0, 25, 50, 75, 100])
    print(f"NMI  min {q[0]:.4f}  q25 {q[1]:.4f}  median {q[2]:.4f}  "
          f"q75 {q[3]:.4f}  max {q[4]:.4f}  mean {y.mean():.4f}")

    print()
    print("=" * 74)
    print("2. THE CONFOUND: is appearance divergence entangled with FOV?")
    print("=" * 74)
    r, p = stats.pearsonr(x, y)
    print(f"Pearson r[log10(GT area ratio), NMI] = {r:+.4f}   p = {p:.4f}   n = {len(ids)}")
    rs, ps = stats.spearmanr(x, y)
    print(f"Spearman rho                         = {rs:+.4f}   p = {ps:.4f}")

    ar = np.array([fov[i] for i in ids])
    low = y[ar < 0.25]
    high = y[ar >= 0.5]
    u, pu = stats.mannwhitneyu(low, high, alternative="two-sided")
    print(f"median NMI, area ratio  < 0.25 : {np.median(low):.4f}  (n={low.size})")
    print(f"median NMI, area ratio >= 0.50 : {np.median(high):.4f}  (n={high.size})")
    print(f"Mann-Whitney U = {u:.1f}, p = {pu:.4f} (two-sided, not scene-clustered)")
    print("Per the plot_baselines.py bins [(0,.05),(.05,.25),(.25,.5),(.5,10)]:")
    for lo_b, hi_b in [(0.0, 0.05), (0.05, 0.25), (0.25, 0.5), (0.5, 10.0)]:
        sel = y[(ar >= lo_b) & (ar < hi_b)]
        print(f"  [{lo_b:.2f},{hi_b:.2f})  n={sel.size:3d}  median NMI {np.median(sel):.4f}")

    # Conservative re-test at the scene level: one NMI (the scene median) and
    # one area ratio (the scene median) per scene, so no pair is counted twice.
    sc = np.asarray(scenes)
    s_nmi, s_ar = [], []
    for s in sorted(set(scenes)):
        k = sc == s
        s_nmi.append(float(np.median(y[k])))
        s_ar.append(float(np.median(ar[k])))
    s_nmi, s_ar = np.array(s_nmi), np.array(s_ar)
    rsc, psc = stats.pearsonr(np.log10(s_ar), s_nmi)
    lo_s, hi_s = s_nmi[s_ar < 0.25], s_nmi[s_ar >= 0.5]
    us, pus = stats.mannwhitneyu(lo_s, hi_s, alternative="two-sided")
    print(f"scene-level (n={s_nmi.size} scenes, median-aggregated): "
          f"Pearson r = {rsc:+.4f}, p = {psc:.4f}")
    print(f"  median scene NMI  <0.25: {np.median(lo_s):.4f} (n={lo_s.size})   "
          f">=0.50: {np.median(hi_s):.4f} (n={hi_s.size})   "
          f"Mann-Whitney U = {us:.1f}, p = {pus:.4f}")

    # Is the correlation just a sample-size artefact? Narrow-FOV pairs have
    # smaller overlaps, and MI estimators are biased at small sample sizes.
    with NMI_CSV.open(newline="", encoding="utf-8") as f:
        npx = {r["pair_id"]: int(r["n_overlap_px"]) for r in csv.DictReader(f)}
    z = np.array([npx[i] for i in ids], dtype=float)

    def _resid(v, on):
        design = np.c_[np.ones_like(on), on]
        return v - design @ np.linalg.lstsq(design, v, rcond=None)[0]

    r_pz, p_pz = stats.pearsonr(z, y)
    r_xz, p_xz = stats.pearsonr(z, x)
    r_par, p_par = stats.pearsonr(_resid(x, z), _resid(y, z))
    print(f"overlap-size control: r[n_overlap_px, NMI] = {r_pz:+.4f} (p={p_pz:.4f}); "
          f"r[n_overlap_px, log10 area ratio] = {r_xz:+.4f} (p={p_xz:.4f})")
    print(f"  partial r[log10 area ratio, NMI | n_overlap_px] = {r_par:+.4f} (p={p_par:.4f}); "
          f"{int((z < 100_000).sum())}/{z.size} pairs sit below the 384 px cap")

    print()
    print("=" * 74)
    print("3. THE 2x2: SR@10px under a median split on BOTH axes")
    print("=" * 74)
    fov_med = float(np.median(x))
    nmi_med = float(np.median(y))
    hi_fov = x >= fov_med
    hi_nmi = y >= nmi_med
    print(f"median log10(area ratio) = {fov_med:+.4f} (area ratio {10 ** fov_med:.4f}); "
          f"high-FOV n = {int(hi_fov.sum())}, low-FOV n = {int((~hi_fov).sum())}")
    print(f"median NMI               = {nmi_med:.4f}; "
          f"high-NMI n = {int(hi_nmi.sum())}, low-NMI n = {int((~hi_nmi).sum())}")

    rng_master = np.random.default_rng(SEED)
    for backbone, mode in [("roma", "direct"), ("ma_roma", "direct")]:
        err = load_err(backbone, mode)
        missing = [i for i in ids if i not in err]
        assert not missing, f"{backbone}/{mode} missing {len(missing)} pairs"
        succ = np.array([err[i] < SR_THRESH for i in ids], dtype=float)
        print()
        print(f"--- {backbone}/{mode}   SR@10 overall {succ.mean():.4f} "
              f"({int(succ.sum())}/{len(ids)}) ---")
        print(f"{'':<12}{'low NMI':>18}{'high NMI':>18}{'row':>18}")
        for name, fmask in [("high FOV", hi_fov), ("low FOV", ~hi_fov)]:
            cells = []
            for nmask in [~hi_nmi, hi_nmi]:
                s = succ[fmask & nmask]
                cells.append(f"{int(s.sum())}/{s.size}={s.mean():.3f}")
            rv = succ[fmask]
            rowtxt = f"{int(rv.sum())}/{rv.size}={rv.mean():.3f}"
            print(f"{name:<12}{cells[0]:>18}{cells[1]:>18}{rowtxt:>18}")
        cols = []
        for nmask in [~hi_nmi, hi_nmi]:
            s = succ[nmask]
            cols.append(f"{int(s.sum())}/{s.size}={s.mean():.3f}")
        print(f"{'col':<12}{cols[0]:>18}{cols[1]:>18}")

        # Why the scene-clustered CIs are wide: successes are not spread over
        # scenes, they pile up in a handful of them.
        sc_arr = np.asarray(scenes)
        per_scene = np.array([succ[sc_arr == s].sum() for s in sorted(set(scenes))])
        nz = int((per_scene > 0).sum())
        top = np.sort(per_scene)[::-1]
        print(f"  successes live in {nz}/{per_scene.size} scenes; "
              f"top-3 scenes hold {int(top[:3].sum())}/{int(per_scene.sum())} of them")

        print(f"  effects, bootstrap CLUSTERED ON SCENE (B={B}, 95% CI, two-sided):")
        tests = [
            ("FOV        (high - low), marginal", hi_fov, np.ones(len(ids), bool)),
            ("appearance (high - low), marginal", hi_nmi, np.ones(len(ids), bool)),
            ("FOV  effect | low  NMI           ", hi_fov, ~hi_nmi),
            ("FOV  effect | high NMI           ", hi_fov, hi_nmi),
            ("NMI  effect | low  FOV           ", hi_nmi, ~hi_fov),
            ("NMI  effect | high FOV           ", hi_nmi, hi_fov),
        ]
        for label, gmask, sub in tests:
            sel = np.flatnonzero(sub)
            rng = np.random.default_rng(int(rng_master.integers(0, 2 ** 31)))
            d, lo_ci, hi_ci, pv, nrep = cluster_bootstrap(
                succ[sel], gmask[sel], [scenes[k] for k in sel], rng
            )
            print(f"    {label}  delta {d:+.4f}  95% CI [{lo_ci:+.4f}, {hi_ci:+.4f}]  "
                  f"p={pv:.4f}  ({nrep}/{B} usable replicates)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure the cross-modal appearance axis.")
    ap.add_argument("--recompute", action="store_true",
                    help="re-measure NMI from images even if the CSV exists")
    ap.add_argument("--variant", choices=["pinned", "marginal-entropy"], default="pinned",
                    help="also compute the internally consistent NMI as a robustness "
                         "check (printed, never committed); needs --recompute")
    ap.add_argument("--analysis-only", action="store_true",
                    help="skip measurement, analyse the committed CSV")
    args = ap.parse_args()

    if args.analysis_only and not NMI_CSV.exists():
        raise SystemExit(f"{NMI_CSV} missing; run without --analysis-only")
    if not args.analysis_only and (args.recompute or not NMI_CSV.exists()):
        rows = measure(variant=args.variant)
        write_csv(rows)
        if args.variant == "marginal-entropy":
            from scipy import stats
            fov = load_fov()
            xa = np.array([np.log10(fov[p]) for p, _, _, _ in rows])
            yp = np.array([v for _, v, _, _ in rows])
            ya = np.array([v for _, _, _, v in rows])
            r1, p1 = stats.pearsonr(xa, yp)
            r2, p2 = stats.pearsonr(xa, ya)
            print(f"ROBUSTNESS pinned NMI          r={r1:+.4f} p={p1:.4f}  "
                  f"(median {np.median(yp):.4f}, max {yp.max():.4f})")
            print(f"ROBUSTNESS marginal-consistent r={r2:+.4f} p={p2:.4f}  "
                  f"(median {np.median(ya):.4f}, max {ya.max():.4f})")
    analyse()


if __name__ == "__main__":
    main()
