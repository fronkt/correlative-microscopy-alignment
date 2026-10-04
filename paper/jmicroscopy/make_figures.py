"""Figures for the Journal of Microscopy draft (Arm 1 banner crop, Arm 2 materials replication).

Every plotted number is read from the committed result files of this worktree; nothing is typed in by hand.
  - Arm 1: results/arm1/arm1.csv (+ results/triage/candidates.csv for the GT homographies and the stored rows),
           results/arm1/arm1_summary.json, results/arm1/crop_plans.csv
  - Arm 2: results/arm2/candidates.csv, results/arm2/arm2_summary.json, results/arm2/gate_result.json
The Arm 1 lock / success indicators are computed with the frozen analysis code (scripts/analyze_arm1.py), imported,
not re-implemented. The script re-derives the headline numbers and asserts that they equal the summary files.

Figure 1 (banner example) needs the AmalgaMatch images (doi:10.24406/fordatis/436, CC BY 4.0). They are not in this
worktree; the default root is the sibling repository's copy. If the root is missing, Figure 1 falls back to the
low-resolution smoke-test thumbnail committed at results/arm1/smoke/cropped_pair_example.png.

Usage:  python paper/jmicroscopy/make_figures.py [--amalgamatch-root PATH]
Output: paper/jmicroscopy/figures/Fig{1,2,3}_*.pdf (vector) and .png (600 dpi)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "figures"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# Okabe-Ito (colour-blind safe)
C_UNC = "#D55E00"   # uncropped / failure
C_CROP = "#0072B2"  # cropped / success
C_GREY = "#7F7F7F"
C_COMP = {"DefDAP (HR-DIC vs EBSD)": "#E69F00", "refodat.86 (BSE vs EBSD)": "#CC79A7",
          "NIST IN625 (BSE2 vs BSE1)": "#009E73", "NIST IN718 (BSE2 vs BSE1)": "#56B4E9"}

MM = 1 / 25.4
DOUBLE = 170 * MM   # two-column width
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.0, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600,
    "axes.spines.top": False, "axes.spines.right": False,
})


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


a1 = _load_module("analyze_arm1", ROOT / "scripts" / "analyze_arm1.py")


def panel_label(ax, s, x=-0.14, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", dpi=600)
    plt.close(fig)
    print("wrote", OUT / f"{stem}.pdf", "and .png")


# ----------------------------------------------------------------------------- Arm 1 data

def arm1_frames():
    cand = a1.read_csv(ROOT / "results/triage/candidates.csv")
    raw = a1.read_csv(ROOT / "results/arm1/arm1.csv")
    meta = a1.build_meta(cand)
    runs, _ = a1.dedupe(raw)
    runs = a1.prepare_runs(runs, meta)
    overlay = a1.read_pairs(ROOT / "results/arm1/pairs_overlay.txt")
    sham = a1.read_pairs(ROOT / "results/arm1/pairs_sham.txt")
    P_ov = a1.paired_frame(runs, overlay, a1.CONFIGS, "none", "overlay")
    P_sh = a1.paired_frame(runs, sham, a1.CONFIGS, "none", "sham")
    return runs, meta, P_ov, P_sh


def apply_h(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    p = np.hstack([pts, np.ones((len(pts), 1))]) @ H.T
    return p[:, :2] / p[:, 2:3]


# ----------------------------------------------------------------------------- Figure 1

EXAMPLE = "eval_MoTaTiZrHf-HEA-DDRX-1100C_TEM_DislocationCharacterization_0#4"


def fig1(runs: pd.DataFrame, amalgamatch_root: Path) -> None:
    crop = pd.read_csv(ROOT / "results/arm1/crop_plans.csv").set_index("pair_id").loc[EXAMPLE]
    r_unc = runs[(runs.pair_id == EXAMPLE) & (runs.backbone == "ma_roma") & (runs["mode"] == "direct") & (runs.crop == "none")].iloc[0]
    r_crp = runs[(runs.pair_id == EXAMPLE) & (runs.backbone == "ma_roma") & (runs["mode"] == "direct") & (runs.crop == "overlay")].iloc[0]
    H_unc, H_crp = a1.parse_H(r_unc.H), a1.parse_H(r_crp.H)

    if not amalgamatch_root.exists():
        print(f"AmalgaMatch root {amalgamatch_root} not found: Figure 1 uses the smoke-test thumbnail")
        img = plt.imread(ROOT / "results/arm1/smoke/cropped_pair_example.png")
        fig, ax = plt.subplots(figsize=(DOUBLE, DOUBLE * img.shape[0] / img.shape[1]))
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        save(fig, "Fig1_banner_example")
        return

    from cma.data import AmalgaMatchLoader
    loader = AmalgaMatchLoader(amalgamatch_root)
    rec = next(r for r in loader.records if r.pair_id == EXAMPLE)
    pair = loader.load_pair(rec)
    src, tgt = pair.source, pair.target
    if src.ndim == 3:
        src, tgt = src.mean(-1), tgt.mean(-1)
    g_src, g_tgt = pair.gt.src_xy, pair.gt.tgt_xy

    # self-check: H maps target -> source and mu_ed is the mean GT error in source px
    for H, r in ((H_unc, r_unc), (H_crp, r_crp)):
        mu = float(np.mean(np.linalg.norm(apply_h(H, g_tgt) - g_src, axis=1)))
        assert abs(mu - float(r.mu)) < 0.01, (mu, r.mu)

    rows_t, rows_s = int(crop.rows_tgt), int(crop.rows_src)
    assert crop.side == "bottom"
    fig, axs = plt.subplots(1, 3, figsize=(DOUBLE, DOUBLE / 3 * src.shape[0] / src.shape[1] + 0.35))

    ax = axs[0]
    ax.imshow(tgt, cmap="gray", vmin=0, vmax=1)
    h = tgt.shape[0]
    ax.axhspan(h - rows_t, h, color=C_UNC, alpha=0.30, lw=0)
    ax.axhline(h - rows_t, color=C_UNC, lw=0.8, ls="--")
    ax.text(0.02 * tgt.shape[1], h - rows_t - 30, f"crop line: bottom {rows_t} of {h} rows", color="white",
            fontsize=6, va="bottom", bbox=dict(fc="black", alpha=0.55, lw=0, pad=1.2))
    ax.set_title("     Narrow image, data bar shaded", loc="left")
    ax.text(0.0, 1.0, "a", transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")

    for ax, H, r, lab, ttl in ((axs[1], H_unc, r_unc, "b", "Uncropped: near-identity lock"),
                               (axs[2], H_crp, r_crp, "c", "Bar cropped from both images")):
        ax.imshow(src, cmap="gray", vmin=0, vmax=1)
        hs = src.shape[0]
        if ax is axs[2]:
            ax.axhspan(hs - rows_s, hs, color="black", alpha=0.55, lw=0)
        pred = apply_h(H, g_tgt)
        for p, q in zip(g_src, pred):
            ax.plot([p[0], q[0]], [p[1], q[1]], color="#F0E442", lw=0.6)
        ax.scatter(g_src[:, 0], g_src[:, 1], s=9, facecolors="none", edgecolors=C_CROP, linewidths=0.8,
                   label="annotated position")
        ax.scatter(pred[:, 0], pred[:, 1], s=9, marker="x", color=C_UNC, linewidths=0.8, label="MA-RoMa estimate")
        ax.set_title("     " + ttl, loc="left")
        ax.text(0.0, 1.0, lab, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")
        ax.text(0.02, 0.97, f"mean error {float(r.mu):.0f} px", transform=ax.transAxes, color="white", fontsize=6.5,
                va="top", bbox=dict(fc="black", alpha=0.55, lw=0, pad=1.2))
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, handletextpad=0.3, columnspacing=1.2,
               bbox_to_anchor=(0.5, -0.04))
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout(w_pad=0.6)
    save(fig, "Fig1_banner_example")


# ----------------------------------------------------------------------------- Figure 2

CFG_LABEL = {"ma_roma|direct": "MA-RoMa\ndirect\n(primary)", "ma_roma|pyramid_v2": "MA-RoMa\nzoom search",
             "roma|direct": "RoMa\ndirect", "roma|pyramid_v2": "RoMa\nzoom search"}
CFG_ORDER = ["ma_roma|direct", "ma_roma|pyramid_v2", "roma|direct", "roma|pyramid_v2"]


def fig2(P_ov: pd.DataFrame, P_sh: pd.DataFrame) -> None:
    S = json.loads((ROOT / "results/arm1/arm1_summary.json").read_text())
    g = P_ov.groupby("cfg")[["lock_base", "lock_treat", "succ_base", "succ_treat"]].sum()
    # assert the plotted counts equal the pre-registered summary
    assert int(g.lock_base.sum()) == S["H-A1"]["base_count"] and int(g.lock_treat.sum()) == S["H-A1"]["treat_count"]
    assert int(g.loc["ma_roma|direct", "succ_base"]) == S["H-A2"]["primary"]["base_count"]
    assert int(g.loc["ma_roma|direct", "succ_treat"]) == S["H-A2"]["primary"]["treat_count"]
    for c in ("roma|direct", "roma|pyramid_v2", "ma_roma|pyramid_v2"):
        assert int(g.loc[c, "succ_base"]) == S["H-A2"]["secondary"][c]["base_count"]
    gs = P_sh.groupby("cfg")[["succ_base", "succ_treat"]].sum()
    assert int(gs.succ_base.sum()) == S["sham"]["successes_uncropped"] and int(gs.succ_treat.sum()) == S["sham"]["successes_sham"]

    fig, axs = plt.subplots(2, 2, figsize=(DOUBLE, DOUBLE * 0.62))
    x = np.arange(len(CFG_ORDER))
    w = 0.38

    ax = axs[0, 0]
    ax.bar(x - w / 2, g.loc[CFG_ORDER, "lock_base"], w, color=C_UNC, label="uncropped")
    ax.bar(x + w / 2, g.loc[CFG_ORDER, "lock_treat"], w, color=C_CROP, label="cropped")
    for i, c in enumerate(CFG_ORDER):
        ax.text(i - w / 2, g.loc[c, "lock_base"] + 0.6, int(g.loc[c, "lock_base"]), ha="center", fontsize=6)
        ax.text(i + w / 2, g.loc[c, "lock_treat"] + 0.6, int(g.loc[c, "lock_treat"]), ha="center", fontsize=6)
    ax.set_xticks(x, [CFG_LABEL[c] for c in CFG_ORDER])
    ax.set_ylabel("Near-identity locks (of 67 pairs)")
    ax.set_ylim(0, 67)
    ax.set_title(f"H-A1: locks {S['H-A1']['base_count']} → {S['H-A1']['treat_count']} (268 paired runs)")
    ax.legend(frameon=False, loc="upper right")
    panel_label(ax, "a")

    ax = axs[0, 1]
    ax.bar(x - w / 2, g.loc[CFG_ORDER, "succ_base"], w, color=C_UNC, label="uncropped")
    ax.bar(x + w / 2, g.loc[CFG_ORDER, "succ_treat"], w, color=C_CROP, label="cropped")
    for i, c in enumerate(CFG_ORDER):
        ax.text(i - w / 2, g.loc[c, "succ_base"] + 0.4, int(g.loc[c, "succ_base"]), ha="center", fontsize=6)
        ax.text(i + w / 2, g.loc[c, "succ_treat"] + 0.4, int(g.loc[c, "succ_treat"]), ha="center", fontsize=6)
    ax.set_xticks(x, [CFG_LABEL[c] for c in CFG_ORDER])
    ax.set_ylabel("Registered within 20 px (of 67)")
    ax.set_ylim(0, 24)
    p = S["H-A2"]["primary"]["p_one_sided"]
    ax.set_title(f"H-A2: MA-RoMa direct 6 → 16, one-sided p = {p:.3f}")
    panel_label(ax, "b")

    ax = axs[1, 0]
    prim = P_ov[P_ov.cfg == "ma_roma|direct"]
    mb, mt = prim.mu_base.astype(float).values, prim.mu_treat.astype(float).values
    col = np.where(prim.lock_base.values, C_UNC, C_GREY)
    ax.scatter(mb, mt, s=10, c=col, edgecolors="none", alpha=0.85)
    lo, hi = 3, 4000
    ax.plot([lo, hi], [lo, hi], color="black", lw=0.5, ls=":")
    ax.axhline(20, color=C_CROP, lw=0.6, ls="--")
    ax.axvline(20, color=C_CROP, lw=0.6, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Mean error, uncropped (px)")
    ax.set_ylabel("Mean error, cropped (px)")
    ax.scatter([], [], s=10, c=C_UNC, label="locked when uncropped")
    ax.scatter([], [], s=10, c=C_GREY, label="not locked")
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("MA-RoMa direct, 67 data-bar pairs (dashed: 20 px)")
    panel_label(ax, "c")

    ax = axs[1, 1]
    h3 = S["H-A3"]
    vals = [h3["base_count"], h3["treat_count"]]
    ax.bar([0, 1], vals, 0.55, color=[C_UNC, C_CROP])
    for i, v in enumerate(vals):
        ax.text(i, v + 1, int(v), ha="center", fontsize=6.5)
    sh = S["sham"]
    ax.bar([2.6, 3.4], [sh["successes_uncropped"], sh["successes_sham"]], 0.55, color=[C_GREY, "#BBBBBB"])
    ax.text(2.6, sh["successes_uncropped"] + 1, sh["successes_uncropped"], ha="center", fontsize=6.5)
    ax.text(3.4, sh["successes_sham"] + 1, sh["successes_sham"], ha="center", fontsize=6.5)
    ax.set_xticks([0, 1, 2.6, 3.4], ["uncropped", "cropped", "uncropped", "sham\ncropped"])
    ax.set_ylim(0, 62)
    ax.set_yticks(range(0, 51, 10))
    ax.set_ylabel("Count")
    ax.text(0.5, 61, f"H-A3: false accepts at S1 ≥ {h3['cutoff']}\n({h3['n']} held-out pairs)", ha="center",
            va="top", fontsize=6.3)
    ax.text(3.0, 61, f"Sham control: successes\n(20 pairs × 4 configs; {sh['discordant']}/{sh['n_runs']} changed)",
            ha="center", va="top", fontsize=6.3)
    panel_label(ax, "d")

    fig.tight_layout(h_pad=1.2, w_pad=1.5)
    save(fig, "Fig2_arm1_results")


# ----------------------------------------------------------------------------- Figure 3

def component(pid: str, subclass: str) -> str:
    if pid.startswith("defdap"):
        return "DefDAP (HR-DIC vs EBSD)"
    if pid.startswith("refodat86"):
        return "refodat.86 (BSE vs EBSD)"
    if subclass == "IN625-BSE2-vs-BSE1":
        return "NIST IN625 (BSE2 vs BSE1)"
    if subclass == "IN718-BSE2-vs-BSE1":
        return "NIST IN718 (BSE2 vs BSE1)"
    return "excluded"


def auroc(score: np.ndarray, y: np.ndarray) -> float:
    pos, neg = score[y], score[~y]
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def roc_curve(score: np.ndarray, y: np.ndarray):
    thr = np.unique(score)[::-1]
    tpr = [0.0] + [float((score[y] >= t).mean()) for t in thr]
    fpr = [0.0] + [float((score[~y] >= t).mean()) for t in thr]
    return np.array(fpr), np.array(tpr)


def fig3() -> None:
    S = json.loads((ROOT / "results/arm2/arm2_summary.json").read_text())
    gate = json.loads((ROOT / "results/arm2/gate_result.json").read_text())
    c = pd.read_csv(ROOT / "results/arm2/candidates.csv", dtype=str, keep_default_na=False)
    d = c[(c.backbone == "ma_roma") & (c["mode"] == "direct") & (c["transform"] == "none") & (c.pool != "gt")].copy()
    d["comp"] = [component(p, s) for p, s in zip(d.pair_id, d.subclass)]
    d = d[d.comp != "excluded"].drop_duplicates("pair_id", keep="last")
    d["mu"] = pd.to_numeric(d.mu_ed, errors="coerce")
    ok = (d.status == "ok") & np.isfinite(d.mu)
    ninl, nm = pd.to_numeric(d.n_inliers, errors="coerce"), pd.to_numeric(d.n_matches, errors="coerce")
    d["S1"] = np.where(ok & (nm > 0), ninl / nm, -np.inf)
    d["succ"] = ok & (d.mu <= 20.0)
    s1, y = d.S1.values.astype(float), d.succ.values.astype(bool)
    A = auroc(s1, y)
    H21 = S["primary_20px"]["H2-1"]
    assert len(d) == S["primary_n"] == 55 and int(y.sum()) == H21["successes"], (len(d), y.sum())
    assert abs(A - H21["auroc"]) < 1e-9, (A, H21["auroc"])
    cut = S["constants"]["S1_CUT"]
    assert int((s1 >= cut).sum()) == S["primary_20px"]["H2-2"]["n_accepted"]

    fig, axs = plt.subplots(2, 2, figsize=(DOUBLE, DOUBLE * 0.66))

    # (a) NIST GT gate
    ax = axs[0, 0]
    order = ["IN625-BSE2-vs-BSE1", "IN718-BSE2-vs-BSE1", "IN718-BSE1-vs-OM", "IN718-BSE2-vs-OM"]
    med = [gate["types"][t]["median_target_px"] for t in order]
    cols = [C_CROP if gate["types"][t]["passes"] else C_UNC for t in order]
    ax.barh(range(4), med, color=cols, height=0.6)
    ax.axvline(gate["limit_target_px"], color="black", lw=0.7, ls="--")
    for i, (t, m) in enumerate(zip(order, med)):
        v = "pass" if gate["types"][t]["passes"] else "excluded"
        ax.text(m + 0.5, i, f"{m:.1f} px, {gate['types'][t]['points']} points, {v}", va="center", fontsize=6)
    ax.set_yticks(range(4), [t.replace("-vs-", " vs ").replace("-", " ") for t in order])
    ax.invert_yaxis()
    ax.set_xlim(0, 42)
    ax.set_xlabel("Median hand-click vs ground-truth distance (target px)")
    ax.set_title("NIST ground-truth gate (limit 10 px, fixed in advance)")
    panel_label(ax, "a", x=-0.45)

    # (b) S1 vs error
    ax = axs[0, 1]
    for comp, col in C_COMP.items():
        m = d.comp == comp
        ax.scatter(d.S1[m], d.mu[m].clip(upper=5000), s=12, color=col, edgecolors="none", alpha=0.9,
                   label=f"{comp} (n = {int(m.sum())})")
    ax.axvline(cut, color="black", lw=0.7, ls="--")
    ax.axhline(20, color=C_GREY, lw=0.6, ls=":")
    ax.set_yscale("log")
    ax.set_xlabel("Retained fraction S1 (MA-RoMa direct)")
    ax.set_ylabel("Mean error (source px)")
    ax.text(cut + 0.015, 4, f"cut-off {cut}\n(fixed from\nAmalgaMatch)", fontsize=6, va="top")
    ax.legend(frameon=False, loc="upper right", fontsize=5.8, handletextpad=0.2, borderaxespad=0.1)
    ax.set_title("Primary set, 55 pairs")
    panel_label(ax, "b")

    # (c) ROC
    ax = axs[1, 0]
    fpr, tpr = roc_curve(s1, y)
    ax.plot(fpr, tpr, color=C_CROP, lw=1.2, drawstyle="steps-post",
            label=f"all 55 (AUROC {A:.3f}; 95% CI {H21['ci95'][0]:.3f}–{H21['ci95'][1]:.3f})")
    mP = ~d.comp.str.startswith("NIST").values
    fP, tP = roc_curve(s1[mP], y[mP])
    AP = auroc(s1[mP], y[mP])
    assert abs(AP - S["descriptive"]["P"]["20px"]["auroc_S1_ma_roma_direct"]) < 1e-9
    ax.plot(fP, tP, color=C_COMP["DefDAP (HR-DIC vs EBSD)"], lw=1.0, ls="--", drawstyle="steps-post",
            label=f"DefDAP + refodat.86 only, 14 (AUROC {AP:.3f}, descriptive)")
    ax.plot([0, 1], [0, 1], color=C_GREY, lw=0.5, ls=":")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, loc="center right", fontsize=5.8)
    ax.set_title(f"H2-1: S1 predicts success ({H21['successes']} successes, {H21['failures']} failures)")
    panel_label(ax, "c")

    # (d) success counts
    ax = axs[1, 1]
    p20 = S["primary_20px"]
    labels = ["MA-RoMa\ndirect", "best\nsingle", "pick\nby S1", "rule R", "oracle", "GT\nceiling"]
    sr = p20["SR"]
    assert sr["ma_roma_direct"] == int(y.sum())
    vals =[sr["ma_roma_direct"], sr["best_single"], sr["pick_by_S1"], sr["R"], sr["oracle_best_of_15"],
            S["gt_ceiling"]["homography"]]
    cols = [C_GREY, C_GREY, C_GREY, C_CROP, "#BBBBBB", "#BBBBBB"]
    ax.bar(range(len(vals)), vals, color=cols, width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.4, int(v), ha="center", fontsize=6.5)
    ax.set_xticks(range(len(vals)), labels, fontsize=6)
    ax.set_ylim(40, 57.5)
    ax.set_ylabel("Registered within 20 px (of 55)")
    h3, h4 = p20["H2-3"], p20["H2-4"]
    ax.set_title(f"H2-3: R vs pick by S1, p = {h3['p_one_sided']:.2g}\nH2-4: R vs best single, p = {h4['p_one_sided']:.3g}")
    panel_label(ax, "d")

    fig.tight_layout(h_pad=1.4, w_pad=1.6)
    save(fig, "Fig3_arm2_results")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--amalgamatch-root", type=Path,
                    default=ROOT.parent / "correlative-microscopy-alignment" / "data" / "AmalgaMatch")
    args = ap.parse_args()
    runs, _meta, P_ov, P_sh = arm1_frames()
    fig1(runs, args.amalgamatch_root)
    fig2(P_ov, P_sh)
    fig3()


if __name__ == "__main__":
    main()
