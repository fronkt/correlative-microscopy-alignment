"""Phase A / A1: the 42 confident TEM false accepts (exploratory; nothing here is confirmatory).

Convention (verified in scripts/run_triage_candidates.py): H maps TARGET (narrow-FOV) pixels into SOURCE
(wide-FOV) pixels; mu_ed = mean |H(tgt_gt) - src_gt| in source px.  For estimate H_e and GT-fit H_g, a target
point x lands at s_e = H_e x (estimate) and s_g = H_g x (truth), so s_g = H_g H_e^-1 s_e  ->  H_rel = H_g @ inv(H_e)
acts in the SOURCE frame and maps the estimated position to the true one.

Writes results/phaseA/a1_false_accepts.json and a1_per_pair.csv.  Log is written by hand (a1_log.md).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cma.triage import auroc, grid_points, project, transform_distance, youden_cutoff  # noqa: E402

OUT = ROOT / "results" / "phaseA"
OUT.mkdir(parents=True, exist_ok=True)
DATA = Path(r"C:\Users\frank\Documents\correlative-microscopy-alignment\data\AmalgaMatch")
DESIGN = ("SameSlice", "SerialSectioning")
TH = 20.0
SEED_LAT = 0
DOWN_MAX = 512  # autocorrelation image: long side downsampled to this many px (primary)


def Hof(s):
    return np.array(json.loads(s)).reshape(3, 3)


def load():
    d = pd.read_csv(ROOT / "results" / "triage" / "candidates.csv")
    d["cand"] = (d.backbone + "|" + d["mode"] + "|" + d["transform"].astype(str) + "|s" + d.seed.astype(int).astype(str))
    ok = d.status.eq("ok")
    d["err"] = np.where(ok, pd.to_numeric(d.mu_ed, errors="coerce"), np.inf)
    d["err"] = d.err.fillna(np.inf)
    nm = pd.to_numeric(d.n_matches, errors="coerce")
    ni = pd.to_numeric(d.n_inliers, errors="coerce")
    d["S1"] = np.where(ok & (nm > 0), ni / nm, -np.inf)
    return d


def define_42(d):
    prim = d[(d.backbone == "ma_roma") & (d["mode"] == "direct") & (d.pool == "core") & (d.seed == 0)
             & (d["transform"] == "none")].set_index("pair_id")
    assert len(prim) == 187
    des = ~prim.group.isin(DESIGN)
    cut = youden_cutoff(prim.S1[~des].values, (prim.err[~des] <= TH).values)
    held = prim[des]
    fa = held[(held.S1 >= cut) & (held.err > TH)]
    return prim, cut, fa


def decompose(Hrel, c_src):
    """Decompose the source-frame correction Hrel about the point c_src (estimated position of target centre)."""
    Hn = Hrel / Hrel[2, 2]
    q = project(Hn, c_src[None])[0]
    T0 = np.array([[1, 0, -c_src[0]], [0, 1, -c_src[1]], [0, 0, 1.0]])
    T1 = np.array([[1, 0, q[0]], [0, 1, q[1]], [0, 0, 1.0]])
    Hc = T1 @ Hn @ T0
    Hc = Hc / Hc[2, 2]
    # after re-centring, Hc maps c_src -> q ; its translation column is q - L c_src... use the local linearisation
    # (Jacobian of the homography at c_src) as the linear part: exact for affine, first order for perspective
    x, y = c_src
    w = Hn[2, 0] * x + Hn[2, 1] * y + Hn[2, 2]
    J = np.zeros((2, 2))
    px = Hn[0, 0] * x + Hn[0, 1] * y + Hn[0, 2]
    py = Hn[1, 0] * x + Hn[1, 1] * y + Hn[1, 2]
    for k, (a, b) in enumerate(((Hn[0, 0], Hn[0, 1]), (Hn[1, 0], Hn[1, 1]))):
        num = px if k == 0 else py
        J[k, 0] = (a * w - num * Hn[2, 0]) / w**2
        J[k, 1] = (b * w - num * Hn[2, 1]) / w**2
    U, S, Vt = np.linalg.svd(J)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R = U @ np.diag([1, -1]) @ Vt
    rot = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    scale = float(np.sqrt(abs(np.linalg.det(J))))
    aniso = float(S[0] / S[1]) if S[1] > 0 else float("inf")
    P = R.T @ J  # symmetric stretch
    shear = float(abs(P[0, 1] + P[1, 0]) / 2 / scale)
    return dict(trans=(q - c_src), rot_deg=rot, scale=scale, aniso=aniso, shear=shear,
                persp=float(np.hypot(Hn[2, 0], Hn[2, 1])))


def read_gray(path, max_side):
    s = str(path)
    import os
    a = os.path.abspath(s)
    if os.name == "nt" and len(a) > 240 and not a.startswith("\\\\?\\"):
        a = "\\\\?\\" + a
    buf = np.fromfile(a, dtype=np.uint8)
    im = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if im.ndim == 3:
        im = cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2GRAY)
    im = im.astype(np.float32)
    f = min(1.0, max_side / max(im.shape))
    if f < 1.0:
        im = cv2.resize(im, (int(round(im.shape[1] * f)), int(round(im.shape[0] * f))), interpolation=cv2.INTER_AREA)
    return im, f


def autocorr_peaks(im, k=3, min_dist=6, mode="highpass", hp_sigma=8.0, max_frac=0.5, r_min=16, min_overlap=0.3):
    """Window-corrected FFT autocorrelation (a masked/normalised cross-correlation, so lag-overlap decay is divided
    out); top-k non-origin local maxima outside the central lobe, one of each +/- pair.
    Returns (peaks, ac) with peaks = [(dx, dy, height, z)] in DOWNSAMPLED-image px.  mode 'highpass' subtracts a
    Gaussian blur (hp_sigma px) first so slow illumination/contrast gradients do not dominate; 'raw' does not."""
    a = im.astype(np.float32)
    if mode == "highpass":
        a = a - cv2.GaussianBlur(a, (0, 0), hp_sigma)
    a = a - a.mean()
    h, w = a.shape
    win = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    a = a * win

    def acorr(x):
        F = np.fft.rfft2(x, s=(2 * h, 2 * w))
        return np.fft.fftshift(np.fft.irfft2(F * np.conj(F), s=(2 * h, 2 * w)))

    ac = acorr(a)
    nrm = acorr(win * win)  # overlap of the window energy at each lag
    ac = (ac / np.maximum(nrm, 1e-3 * nrm.max()) / (ac[h, w] / nrm[h, w])).astype(np.float32)
    cy, cx = h, w
    yy, xx = np.mgrid[0:2 * h, 0:2 * w]
    rr = np.hypot(yy - cy, xx - cx)
    valid = ((np.abs(yy - cy) <= max_frac * h) & (np.abs(xx - cx) <= max_frac * w) & (rr >= r_min)
             & (nrm >= min_overlap * nrm[h, w]))
    mu, sd = float(ac[valid].mean()), float(ac[valid].std())
    mx = cv2.dilate(ac, np.ones((2 * min_dist + 1, 2 * min_dist + 1), np.uint8))
    cand = np.argwhere((ac == mx) & valid & (yy >= cy) & ~((yy == cy) & (xx < cx)))
    res = [(float(x - cx), float(y - cy), float(ac[y, x]), float((ac[y, x] - mu) / sd)) for y, x in cand]
    res.sort(key=lambda t: -t[2])
    return res[:k], ac


def lattice_stat(T, gens, K=6):
    """Mean over translations of min distance to {sum m_i g_i, |m_i|<=K} using first two generators, normalised
    by the mean generator length."""
    G = np.array(gens[:2], dtype=float).reshape(-1, 2)
    ms = np.arange(-K, K + 1)
    if len(G) == 0:  # no peak found: the lattice is just the origin
        return float(np.linalg.norm(T, axis=1).mean())
    M = np.array([(a, b) for a in ms for b in (ms if len(G) == 2 else [0])], dtype=float)[:, :len(G)]
    L = M @ G  # (n,2)
    dd = np.linalg.norm(T[:, None, :] - L[None], axis=2).min(axis=1)
    return float(dd.mean())


def lattice_test(T, gens, n_perm=1000, seed=SEED_LAT, K=6):
    """Permutation null: rotate the whole generator set by a random angle (lengths and relative angles kept), and
    separately draw each generator with an independent random angle (lengths kept)."""
    rng = np.random.default_rng(seed)
    obs = lattice_stat(T, gens, K)
    g = np.array(gens, dtype=float)
    ln = np.linalg.norm(g, axis=1)
    null_rot, null_ind = [], []
    for _ in range(n_perm):
        th = rng.uniform(0, 2 * np.pi)
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        null_rot.append(lattice_stat(T, list(g @ R.T), K))
        ang = rng.uniform(0, 2 * np.pi, len(g))
        null_ind.append(lattice_stat(T, list(np.stack([ln * np.cos(ang), ln * np.sin(ang)], 1)), K))
    null_rot, null_ind = np.array(null_rot), np.array(null_ind)
    return dict(obs=obs, p_rot=float((1 + (null_rot <= obs).sum()) / (n_perm + 1)),
                p_indep=float((1 + (null_ind <= obs).sum()) / (n_perm + 1)),
                null_rot_median=float(np.median(null_rot)), null_ind_median=float(np.median(null_ind)))


def main():
    d = load()
    prim, cut, fa = define_42(d)
    out = {"cutoff": float(cut), "n_false_accepts": int(len(fa))}
    assert len(fa) == 42, len(fa)
    assert set(fa.group) == {"DislocationCharacterization"}
    assert fa.subclass.str.startswith("MoTaTiZrHf").all()
    assert fa.scene.nunique() == 3
    out["scenes"] = fa.scene.value_counts().to_dict()

    gt_h = d[(d.pool == "gt") & (d["mode"] == "homography")].set_index("pair_id")
    others = d[d.pool.isin(["core", "transform", "control"])]

    # ---------------- (iv) thresholds
    e = fa.err.values
    out["iv"] = {"n_gt25": int((e > 25).sum()), "n_gt30": int((e > 30).sum()), "n_20_30": int(((e > 20) & (e <= 30)).sum()),
                 "n_gt100": int((e > 100).sum()), "n_gt150": int((e > 150).sum()),
                 "gtfit_hom_lt20": int((gt_h.loc[fa.index, "err"] < TH).sum()),
                 "gtfit_hom_err_max": float(gt_h.loc[fa.index, "err"].max()),
                 "gtfit_hom_err_median": float(gt_h.loc[fa.index, "err"].median()),
                 "gtfit_hom_over20_pairs": [p for p in fa.index if gt_h.loc[p, "err"] >= TH]}
    aff = d[(d.pool == "gt") & (d["mode"] == "affine")].set_index("pair_id")
    out["iv"]["gtfit_affine_lt20"] = int((aff.loc[fa.index, "err"] < TH).sum())
    out["iv"]["est_family_counts"] = fa.family.value_counts().to_dict()

    # ---------------- (i) shared vs matcher-specific
    rows = []
    for pid, r in fa.iterrows():
        Hm = Hof(r.H)
        pts = grid_points(int(r.h_t), int(r.w_t), 5)
        sub = others[(others.pair_id == pid) & ~((others.cand == r.cand))]
        for _, o in sub.iterrows():
            if o.status != "ok" or not isinstance(o.H, str):
                cls, dist = "FAILED_RUN", np.inf
            else:
                dist = transform_distance(Hof(o.H), Hm, pts)
                if o.err <= TH:
                    cls = "CORRECT"
                elif dist < TH and o.err > TH:
                    cls = "SAME-WRONG"
                else:
                    cls = "ELSEWHERE"
            rows.append(dict(pair_id=pid, cand=o.cand, backbone=o.backbone, mode=o["mode"], transform=o["transform"],
                             seed=int(o.seed), pool=o.pool, dist_to_maroma=dist, own_err=o.err, cls=cls))
    R = pd.DataFrame(rows)
    R["group_name"] = np.where(R.pool == "control", "seeds(" + R.backbone + ")",
                       np.where(R.pool == "transform", "transforms(" + R.backbone + ")",
                                R.backbone + "|" + R["mode"]))
    out["i"] = {
        "overall": R.cls.value_counts().to_dict(),
        "by_group": {k: v.cls.value_counts().to_dict() for k, v in R.groupby("group_name")},
        "by_backbone": {k: v.cls.value_counts().to_dict() for k, v in R.groupby("backbone")},
    }
    perp = R.groupby("pair_id").cls.value_counts().unstack(fill_value=0)
    for c in ("SAME-WRONG", "CORRECT", "ELSEWHERE", "FAILED_RUN"):
        if c not in perp:
            perp[c] = 0
    perp["n_other"] = perp.sum(axis=1)
    perp["frac_same_wrong"] = perp["SAME-WRONG"] / perp.n_other
    out["i"]["pairs_with_any_correct"] = int((perp.CORRECT > 0).sum())
    out["i"]["pairs_majority_same_wrong"] = int((perp.frac_same_wrong > 0.5).sum())
    out["i"]["pairs_no_same_wrong"] = int((perp["SAME-WRONG"] == 0).sum())
    out["i"]["frac_same_wrong_median"] = float(perp.frac_same_wrong.median())
    # how many of the SAME-WRONG are just tiny-displacement (near misses share the same near-miss)
    out["i"]["dist_of_others_median"] = float(R.dist_to_maroma.replace(np.inf, np.nan).median())
    # same-wrong among non-dense (sift, loftr, matchanything)
    nd = R[R.backbone.isin(["sift", "loftr", "matchanything"])]
    out["i"]["non_ma_roma_family_same_wrong"] = nd.groupby("backbone").cls.value_counts().unstack(fill_value=0).to_dict("index")
    # per pair: is there a "consensus attractor": largest cluster of other candidates' H within 20 px of each other
    att = {}
    for pid, g in R.groupby("pair_id"):
        r = fa.loc[pid]
        pts = grid_points(int(r.h_t), int(r.w_t), 5)
        Hs = {c: Hof(others[(others.pair_id == pid) & (others.cand == c)].H.iloc[0])
              for c in g.cand[g.cls != "FAILED_RUN"]}
        cs = list(Hs)
        best, bestc = 0, None
        for c in cs:
            n = sum(transform_distance(Hs[c], Hs[c2], pts) < TH for c2 in cs)
            if n > best:
                best, bestc = n, c
        wrong_cluster = float(np.mean([others[(others.pair_id == pid) & (others.cand == c2)].mu_ed.astype(float).iloc[0] > TH
                                       for c2 in cs if transform_distance(Hs[bestc], Hs[c2], pts) < TH]))
        att[pid] = dict(largest_cluster=best, n_cands=len(cs), cluster_all_wrong_frac=wrong_cluster,
                        cluster_center_err=float(others[(others.pair_id == pid) & (others.cand == bestc)].mu_ed.astype(float).iloc[0]))
    out["i"]["other_cands_clusters"] = {
        "median_largest_cluster_frac": float(np.median([v["largest_cluster"] / v["n_cands"] for v in att.values()])),
        "pairs_where_largest_other_cluster_is_wrong": int(sum(v["cluster_center_err"] > TH for v in att.values()))}

    # ---------------- (ii) error geometry
    geo = []
    for pid, r in fa.iterrows():
        He, Hg = Hof(r.H), Hof(gt_h.loc[pid].H)
        Hrel = Hg @ np.linalg.inv(He)
        c_t = np.array([(r.w_t - 1) / 2.0, (r.h_t - 1) / 2.0])
        c_src = project(He, c_t[None])[0]
        dc = decompose(Hrel, c_src)
        # centre displacement directly (truth minus estimate at target centre) as a cross-check
        direct = project(Hg, c_t[None])[0] - project(He, c_t[None])[0]
        assert np.allclose(direct, dc["trans"], atol=1e-6), (direct, dc["trans"])
        pts = grid_points(int(r.h_t), int(r.w_t), 5)
        disp = project(Hg, pts) - project(He, pts)
        mean_disp = disp.mean(axis=0)
        resid = np.linalg.norm(disp - mean_disp, axis=1).mean()  # part of error not explained by a single translation
        tot = np.linalg.norm(disp, axis=1).mean()
        geo.append(dict(pair_id=pid, tx=dc["trans"][0], ty=dc["trans"][1], tnorm=float(np.hypot(*dc["trans"])),
                        w_s=int(r.w_s), h_s=int(r.h_s), tfrac_w=float(dc["trans"][0] / r.w_s), tfrac_h=float(dc["trans"][1] / r.h_s),
                        tfrac_diag=float(np.hypot(*dc["trans"]) / np.hypot(r.w_s, r.h_s)),
                        rot_deg=dc["rot_deg"], scale=dc["scale"], aniso=dc["aniso"], shear=dc["shear"], persp=dc["persp"],
                        grid_mean_disp_x=float(mean_disp[0]), grid_mean_disp_y=float(mean_disp[1]),
                        grid_mean_err=float(tot), grid_resid_after_translation=float(resid),
                        translation_share=float(1 - resid / tot) if tot > 0 else np.nan,
                        est_family=r.family, err=float(r.err), S1=float(r.S1)))
    G = pd.DataFrame(geo).set_index("pair_id")
    out["ii"] = {
        "translation_norm_px": {"min": float(G.tnorm.min()), "median": float(G.tnorm.median()), "max": float(G.tnorm.max())},
        "translation_frac_of_source_diag_median": float(G.tfrac_diag.median()),
        "rot_deg_abs_median": float(G.rot_deg.abs().median()), "rot_deg_abs_max": float(G.rot_deg.abs().max()),
        "scale_median": float(G.scale.median()), "scale_min": float(G.scale.min()), "scale_max": float(G.scale.max()),
        "aniso_median": float(G.aniso.median()), "aniso_max": float(G.aniso.max()),
        "shear_median": float(G.shear.median()), "shear_max": float(G.shear.max()),
        "persp_median": float(G.persp.median()), "persp_max": float(G.persp.max()),
        "translation_share_median": float(G.translation_share.median()),
        "translation_share_min": float(G.translation_share.min()),
        "n_translation_share_ge_0.8": int((G.translation_share >= 0.8).sum()),
        "n_translation_share_ge_0.9": int((G.translation_share >= 0.9).sum()),
        "per_scene": {},
    }
    # do the translations cluster?  (hierarchical, 20 px linkage) overall and by scene
    from scipy.cluster.hierarchy import fcluster, linkage
    Tall = G[["tx", "ty"]].values
    for lab, thr in (("20px", 20.0), ("50px", 50.0)):
        cl = fcluster(linkage(Tall, "single"), thr, criterion="distance")
        out["ii"][f"single_linkage_{lab}_nclusters"] = int(len(np.unique(cl)))
        out["ii"][f"single_linkage_{lab}_sizes"] = sorted(np.bincount(cl)[1:].tolist(), reverse=True)
    for sc, gg in fa.groupby("scene"):
        T = G.loc[gg.index, ["tx", "ty"]].values
        cl = fcluster(linkage(T, "single"), 20.0, criterion="distance") if len(T) > 1 else np.array([1])
        out["ii"]["per_scene"][sc] = {"n": int(len(T)), "tx": T[:, 0].round(1).tolist(), "ty": T[:, 1].round(1).tolist(),
                                      "cluster20_sizes": sorted(np.bincount(cl)[1:].tolist(), reverse=True),
                                      "mean_vec": T.mean(axis=0).round(1).tolist(), "std_vec": T.std(axis=0).round(1).tolist()}
    out["ii"]["translations"] = {p: [round(float(a), 1), round(float(b), 1)] for p, a, b in zip(G.index, G.tx, G.ty)}

    # ---------------- (iii) repeating structure, per distinct SOURCE image (error translations live in the source frame)
    from cma.data.amalgamatch import AmalgaMatchLoader
    L = AmalgaMatchLoader(DATA)
    recmap = {r_.pair_id: r_ for r_ in L.records}
    cache = {}

    def gens_for(path, variant):
        key_ = (str(path), variant)
        if key_ not in cache:
            mode, ds = variant
            im, f = read_gray(path, ds)
            pk, _ = autocorr_peaks(im, mode=mode, hp_sigma=8.0 * ds / 512.0)
            cache[key_] = [(dx / f, dy / f, hgt, z) for dx, dy, hgt, z in pk]
        return cache[key_]

    def rotm(th):
        return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

    variants = {"primary_highpass_512": ("highpass", 512), "raw_512": ("raw", 512), "highpass_256": ("highpass", 256),
                "highpass_1024": ("highpass", 1024)}
    out["iii_variants"] = {}
    srcs = {}
    for pid in fa.index:
        srcs.setdefault(str(recmap[pid].source_path), []).append(pid)
    order = [p for pids in srcs.values() for p in pids]
    sid = [si for si, pids in enumerate(srcs.values()) for _ in pids]
    T_all = np.array([G.loc[p, ["tx", "ty"]].values.astype(float) for p in order])
    scn = np.array([fa.loc[p, "scene"] for p in order])
    for vname, var in variants.items():
        vinfo = {"sources": {}}
        gens_all = []
        for si, (sp, pids) in enumerate(srcs.items()):
            gs = gens_for(sp, var)
            vinfo["sources"][Path(sp).name] = {"n_pairs": len(pids), "peaks_srcpx(dx,dy,height,z)": [[round(v, 2) for v in g] for g in gs]}
            gens_all += [[g[:2] for g in gs] for _ in pids]
        rng = np.random.default_rng(SEED_LAT)

        def pooled(gen_list, idx=None):
            idx = range(len(T_all)) if idx is None else idx
            return float(np.mean([lattice_stat(T_all[i:i + 1], gen_list[i]) for i in idx]))

        obs = pooled(gens_all)
        n_perm = 1000
        nr, ni = [], []
        ns = max(sid) + 1
        for _ in range(n_perm):
            ths = rng.uniform(0, 2 * np.pi, ns)
            angs = rng.uniform(0, 2 * np.pi, (ns, 3))
            gl_r, gl_i = [], []
            for i in range(len(T_all)):
                g = np.array(gens_all[i], dtype=float).reshape(-1, 2)
                gl_r.append(list(g @ rotm(ths[sid[i]]).T))
                ln = np.linalg.norm(g, axis=1)
                a_ = angs[sid[i]][:len(g)]
                gl_i.append(list(np.stack([ln * np.cos(a_), ln * np.sin(a_)], 1)))
            nr.append(pooled(gl_r))
            ni.append(pooled(gl_i))
        nr, ni = np.array(nr), np.array(ni)
        vinfo.update(obs_mean_lattice_dist_px=obs, null_rot_median=float(np.median(nr)), null_indep_median=float(np.median(ni)),
                     p_rot=float((1 + (nr <= obs).sum()) / (n_perm + 1)), p_indep=float((1 + (ni <= obs).sum()) / (n_perm + 1)),
                     mean_T_norm=float(np.linalg.norm(T_all, axis=1).mean()), n_distinct_sources=len(srcs))
        per_scene = {}
        for sc in np.unique(scn):
            idx = np.flatnonzero(scn == sc)
            o = pooled(gens_all, idx)
            rr_ = np.array([pooled([list(np.array(g, dtype=float).reshape(-1, 2) @ rotm(rng.uniform(0, 2 * np.pi)).T) for g in gens_all], idx) for _ in range(300)])
            per_scene[sc] = dict(n=int(len(idx)), obs=o, null_rot_median=float(np.median(rr_)), p_rot=float((1 + (rr_ <= o).sum()) / 301))
        vinfo["per_scene"] = per_scene
        out["iii_variants"][vname] = vinfo

    # ---------------- identity-attractor probe (found while doing (ii): the estimate is near "no displacement")
    def ident_dist(Hm, r):
        pts_ = grid_points(int(r.h_t), int(r.w_t), 5)
        return float(np.linalg.norm(project(Hm, pts_) - pts_, axis=1).mean())

    idd = {pid: ident_dist(Hof(r.H), r) for pid, r in fa.iterrows()}
    gt_shift = {pid: ident_dist(Hof(gt_h.loc[pid].H), gt_h.loc[pid]) for pid in fa.index}
    ida = np.array(list(idd.values()))
    out["identity"] = {
        "est_dist_to_identity_px": {"min": float(ida.min()), "median": float(np.median(ida)), "max": float(ida.max())},
        "n_est_within20_of_identity": int((ida < 20).sum()), "n_est_within30_of_identity": int((ida < 30).sum()),
        "gt_shift_from_identity_px": {"min": float(min(gt_shift.values())), "median": float(np.median(list(gt_shift.values()))),
                                      "max": float(max(gt_shift.values()))},
        "all_42_same_image_size_and_pixel_size": bool(all(
            fa.loc[p, "h_s"] == fa.loc[p, "h_t"] and fa.loc[p, "w_s"] == fa.loc[p, "w_t"]
            and abs(recmap[p].source_pixel_nm - recmap[p].target_pixel_nm) < 1e-9 for p in fa.index)),
    }
    ss = prim[(prim.h_s == prim.h_t) & (prim.w_s == prim.w_t)]
    e_id = np.array([ident_dist(Hof(r.H), r) if r.status == "ok" else np.inf for _, r in ss.iterrows()])
    g_id = np.array([ident_dist(Hof(gt_h.loc[p].H), gt_h.loc[p]) for p in ss.index])
    out["identity"]["same_size_pairs"] = {
        "n": int(len(ss)), "groups": ss.group.value_counts().to_dict(),
        "est_within20_of_identity": int((e_id < 20).sum()),
        "identity_would_succeed_(gt_shift<20)": int((g_id < 20).sum()),
        "successes_of_primary": int((ss.err <= TH).sum()),
        "near_identity_and_S1_ge_cut": int(((e_id < 20) & (ss.S1.values >= cut)).sum())}
    # SIFT re-run probe: one pair per distinct (scene, GT-shift) configuration; where do the raw matches sit?
    from cma.matchers.sift import SIFTMatcher
    cfg = {}
    for pid in fa.index:
        cfg.setdefault((pid.split("#")[0], round(gt_shift[pid] / 15)), pid)
    probe = {}
    sm = SIFTMatcher()
    for (sc, _), pid in cfg.items():
        pair = L.load_pair(recmap[pid])
        corr = sm.match(pair.source, pair.target)
        a, b = corr.a_xy, corr.b_xy  # a in source, b in target
        Hg = Hof(gt_h.loc[pid].H)
        disp = np.linalg.norm(a - b, axis=1)
        to_gt = np.linalg.norm(a - project(Hg, b), axis=1)
        near0 = disp < 5
        hist = np.zeros((4, 4), int)
        for (x, y) in b[near0]:
            hist[min(3, int(y / pair.target.shape[0] * 4)), min(3, int(x / pair.target.shape[1] * 4))] += 1
        probe[pid] = {"n_raw_matches": int(len(a)), "frac_zero_displacement_lt5px": float(near0.mean()) if len(a) else None,
                      "n_zero_displacement": int(near0.sum()),
                      "frac_consistent_with_GT_lt10px": float((to_gt < 10).mean()) if len(a) else None,
                      "n_consistent_with_GT": int((to_gt < 10).sum()),
                      "zero_disp_matches_grid4x4_counts(rows=y)": hist.tolist(), "gt_shift_px": gt_shift[pid],
                      "ma_roma_dist_to_identity": idd[pid],
                      "zero_disp_frac_in_bottom_9pct_of_image": float((b[near0][:, 1] > 0.91 * pair.target.shape[0]).mean()) if near0.any() else None,
                      "gt_points_max_y_frac": float(pair.gt.tgt_xy[:, 1].max() / pair.target.shape[0])}
    out["identity"]["sift_probe"] = probe

    # Causal test (SIFT only, CPU): crop the bottom metadata banner (burned-in text/scale bar, same pixel position in
    # every image) from BOTH images before matching; coordinates of the remaining pixels are unchanged.
    from cma.estimators import fit_transform
    from cma.triage import pair_seed, seed_everything
    from cma.metrics import registration_metrics
    RANSAC_PX = 5.5

    def sift_err(pair, pid, crop):
        hs = int(round(pair.source.shape[0] * (1 - crop)))
        ht = int(round(pair.target.shape[0] * (1 - crop)))
        seed_everything(pair_seed(0, pid))
        corr = sm.match(pair.source[:hs], pair.target[:ht])
        if len(corr) < 4:
            return np.inf, None, len(corr)
        try:
            est = fit_transform(src_xy=corr.b_xy, dst_xy=corr.a_xy, family="auto", ransac_threshold_px=RANSAC_PX)
        except Exception:
            return np.inf, None, len(corr)
        Hm = est.as_3x3()
        return float(registration_metrics(project(Hm, pair.gt.tgt_xy), pair.gt.src_xy).mu_err), Hm, len(corr)

    crop_res = {}
    sift_stored = d[(d.backbone == "sift") & (d.pool == "core")].set_index("pair_id")
    for pid in fa.index:
        pair = L.load_pair(recmap[pid])
        row = {"sift_stored_err": float(sift_stored.loc[pid, "err"])}
        for crop in (0.0, 0.09, 0.15):
            e_, Hm, nm_ = sift_err(pair, pid, crop)
            row[f"sift_err_crop{crop}"] = e_
            row[f"sift_n_matches_crop{crop}"] = nm_
            if Hm is not None:
                row[f"sift_dist_to_identity_crop{crop}"] = ident_dist(Hm, fa.loc[pid])
        crop_res[pid] = row
    C = pd.DataFrame(crop_res).T
    out["identity"]["sift_crop_experiment"] = {
        "note": "SIFT rerun on the 42 MA-RoMa false-accept pairs; crop = fraction of image height removed at the bottom of both images",
        "n_pairs": int(len(C)),
        "stored_sift_success_(<=20px)": int((C.sift_stored_err <= TH).sum()),
        **{f"rerun_crop{c}_success_(<=20px)": int((C[f"sift_err_crop{c}"].astype(float) <= TH).sum()) for c in (0.0, 0.09, 0.15)},
        **{f"rerun_crop{c}_near_identity_(<20px)": int((C[f"sift_dist_to_identity_crop{c}"].astype(float) < 20).sum()) for c in (0.0, 0.09, 0.15)},
        "rerun_matches_stored_err_within_1px": int((np.abs(C["sift_err_crop0.0"].astype(float) - C.sift_stored_err) < 1).sum()),
    }
    C.to_csv(OUT / "a1_sift_crop_experiment.csv")


    G.to_csv(OUT / "a1_geometry.csv")
    # ---------------- per-pair csv
    pp = G.join(perp[["SAME-WRONG", "CORRECT", "ELSEWHERE", "FAILED_RUN", "n_other", "frac_same_wrong"]])
    pp["gtfit_hom_err"] = gt_h.loc[pp.index, "err"]
    pp["gtfit_aff_err"] = aff.loc[pp.index, "err"]
    pp["cluster_center_err"] = [att[p]["cluster_center_err"] for p in pp.index]
    pp["largest_other_cluster"] = [att[p]["largest_cluster"] for p in pp.index]
    pp["scene"] = fa.loc[pp.index, "scene"]
    pp.to_csv(OUT / "a1_per_pair.csv")
    R.to_csv(OUT / "a1_candidates_vs_maroma.csv", index=False)
    (OUT / "a1_false_accepts.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("cutoff", "n_false_accepts", "iv", "i")}, indent=1, default=float))


if __name__ == "__main__":
    main()
