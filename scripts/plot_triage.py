"""Figures for the CJSJ triage paper, drawn only from results/triage/{summary.json, per_pair.csv, candidates.csv}.

Fig. 2  retained fraction (S1) vs registration error for MA-RoMa, by group, with the transferred cut-off
Fig. 3  (a) success rate among accepted registrations vs coverage; (b) per-pair selection vs best single
Fig. 1  example overlays (accepted-correct / rejected-wrong) are drawn by --examples, which needs the dataset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from cma.triage import accepted_success  # noqa: E402

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 8, "axes.linewidth": 0.6,
                     "savefig.dpi": 300, "pdf.fonttype": 42})
GROUP_ORDER = ["SameSlice", "SerialSectioning", "Multiscale", "DislocationCharacterization",
               "FractureSurfaces", "SlipPartitioning"]
GROUP_LABEL = {"SameSlice": "Same slice", "SerialSectioning": "Serial sectioning", "Multiscale": "Multiscale",
               "DislocationCharacterization": "Dislocation (TEM)", "FractureSurfaces": "Fracture surfaces",
               "SlipPartitioning": "Slip partitioning"}
COLORS = ["#0072B2", "#56B4E9", "#E69F00", "#D55E00", "#009E73", "#CC79A7"]  # Okabe-Ito


def fig2(pp: pd.DataFrame, summ: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    cut = summ["H1"]["transfer"]["cutoff"]
    for g, col in zip(GROUP_ORDER, COLORS):
        d = pp[pp.group == g]
        if d.empty:
            continue
        design = g in ("SameSlice", "SerialSectioning")
        err = np.clip(d.err_primary.replace(np.inf, 1e5), 0.3, 1e5)
        ax.scatter(d.S1_primary, err, s=12, c=col, marker="o" if design else "^", lw=0.3,
                   edgecolors="k", label=GROUP_LABEL[g] + (" *" if design else ""), alpha=0.85)
    ax.set_yscale("log")
    ax.axhline(20, color="0.3", ls="--", lw=0.7)
    ax.axvline(cut, color="0.3", ls=":", lw=0.9)
    ax.text(cut, ax.get_ylim()[1], f" cut-off {cut:.3f}", va="top", ha="left", fontsize=7)
    ax.set_xlabel("Retained fraction of correspondences (inliers / proposed)")
    ax.set_ylabel("Registration error (pixels)")
    ax.legend(fontsize=6, frameon=False, loc="upper right", handletextpad=0.2)
    fig.tight_layout()
    fig.savefig(out / "fig2_retained_fraction.png")
    fig.savefig(out / "fig2_retained_fraction.pdf")
    plt.close(fig)


def fig3(pp: pd.DataFrame, summ: dict, out: Path) -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 2.5), gridspec_kw={"width_ratios": [1.2, 1]})
    y = (pp.err_primary <= 20).values
    cov = np.linspace(0.05, 1.0, 96)
    for col, label, key in [("#0072B2", "Retained fraction (S1)", "S1_primary"),
                            ("#D55E00", "Combined score (S3)", "S3_primary")]:
        s = pp[key].replace([np.inf, -np.inf], np.nan).fillna(-1e9).values
        a.plot(cov * 100, [accepted_success(s, y, c) * 100 for c in cov], color=col, lw=1.2, label=label)
    a.axhline(y.mean() * 100, color="0.4", ls="--", lw=0.8, label="Accept everything")
    a.set_xlabel("Registrations accepted (% of pairs, highest score first)")
    a.set_ylabel("Accepted that are correct (%)")
    a.set_ylim(0, 102)
    a.legend(fontsize=6.5, frameon=False)
    a.set_title("(a) Triage, MA-RoMa", fontsize=8)

    h3 = summ["H3"]
    names = ["Best single\nmethod", "Pick by\nS1", "Pick by\nS2", "Pick by\nS3", "Oracle\n(any correct)"]
    vals = [h3["best_single_sr20"], h3["select_S1"]["sr20"], h3["select_S2"]["sr20"],
            h3["select_S3"]["sr20"], h3["oracle_sr20"]]
    cols = ["0.6", "#0072B2", "#009E73", "#D55E00", "0.85"]
    bars = b.bar(range(5), np.array(vals) * 100, color=cols, edgecolor="k", lw=0.4)
    for r, v in zip(bars, vals):
        b.text(r.get_x() + r.get_width() / 2, v * 100 + 1, f"{v * 100:.1f}", ha="center", fontsize=6.5)
    b.set_xticks(range(5), names, fontsize=6.5)
    b.set_ylabel("Pairs registered within 20 px (%)")
    b.set_title("(b) Choosing one registration per pair", fontsize=8)
    b.set_ylim(0, max(vals) * 100 + 10)
    fig.tight_layout()
    fig.savefig(out / "fig3_triage_selection.png")
    fig.savefig(out / "fig3_triage_selection.pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="results/triage")
    ap.add_argument("--out", default="paper/cjsj/figures")
    args = ap.parse_args()
    ind, out = Path(args.indir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summ = json.loads((ind / "summary.json").read_text(encoding="utf-8"))
    pp = pd.read_csv(ind / "per_pair.csv")
    for c in [c for c in pp.columns if c.startswith(("err_", "S"))]:
        pp[c] = pd.to_numeric(pp[c], errors="coerce")
    fig2(pp, summ, out)
    fig3(pp, summ, out)
    print("figures ->", out)


if __name__ == "__main__":
    main()
