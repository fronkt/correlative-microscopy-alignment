"""Arm 1 gate: the frozen cma.overlay detector must reproduce A6's per-image flags exactly on all 156 images,
and the derived crop plans must agree with A6's per-pair SHARED_OVERLAY (67 pairs).
Writes results/arm1/detector_check.csv + crop_plans.csv (the frozen plan per overlay pair)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cma.overlay import analyze, detect_overlay, overlay_rule, read_gray_u8  # noqa: E402

OUT = ROOT / "results" / "arm1"
A6 = ROOT / "results" / "phaseA"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pi = pd.read_csv(A6 / "a6_per_image.csv", keep_default_na=False)
    rows, det = [], {}
    for r in pi.itertuples():
        im = read_gray_u8(r.path)
        a = analyze(im)
        d = detect_overlay(im)
        det[r.path] = a
        exp_edges = [e for e in r.band_edges.split("|") if e]
        exp_lab = [e for e in r.label_corners.split("|") if e]
        exp_frac = json.loads(r.band_frac)
        ok = (a["overlay"] == bool(r.overlay) and a["band_edges"] == exp_edges and a["label_corners"] == exp_lab
              and a["band_frac"] == exp_frac and (a["h"], a["w"]) == (r.h, r.w))
        rows.append(dict(idx=r.idx, file=r.file, match=ok, overlay=a["overlay"], exp_overlay=bool(r.overlay),
                         edges="|".join(a["band_edges"]), exp_edges="|".join(exp_edges),
                         labels="|".join(a["label_corners"]), exp_labels="|".join(exp_lab),
                         detect_overlay=str(d)))
    chk = pd.DataFrame(rows)
    chk.to_csv(OUT / "detector_check.csv", index=False)
    bad = chk[~chk.match]
    print(f"images {len(chk)}, mismatches {len(bad)}")
    if len(bad):
        print(bad.to_string())
        return 1

    pp = pd.read_csv(A6 / "a6_per_pair.csv", keep_default_na=False)
    P = pd.read_csv(A6 / "a6_per_pair.csv", keep_default_na=False)
    # image path per pair: use the loader records
    from cma.data.amalgamatch import AmalgaMatchLoader
    L = AmalgaMatchLoader(Path(r"C:\Users\frank\Documents\correlative-microscopy-alignment\data\AmalgaMatch"))
    plans, n_rule_flag = [], 0
    for rec in L.records:
        as_, at_ = det[str(rec.source_path)], det[str(rec.target_path)]
        rule = overlay_rule(as_, at_)
        a6 = bool(P.set_index("pair_id").loc[rec.pair_id, "SHARED_OVERLAY"])
        if (rule is not None) != a6:
            print("PAIR MISMATCH", rec.pair_id, rule, a6)
            return 1
        n_rule_flag += rule is not None
        if rule is not None:
            plans.append(dict(pair_id=rec.pair_id, side=rule.side, rows_src=rule.rows_src, rows_tgt=rule.rows_tgt,
                              hw_src=f"{as_['h']}x{as_['w']}", hw_tgt=f"{at_['h']}x{at_['w']}",
                              frac_src=round(rule.rows_src / as_["h"], 4), frac_tgt=round(rule.rows_tgt / at_["h"], 4)))
    pd.DataFrame(plans).to_csv(OUT / "crop_plans.csv", index=False)
    print(f"pairs with an overlay rule: {n_rule_flag} (A6 SHARED_OVERLAY: {int(pp.SHARED_OVERLAY.sum())})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
