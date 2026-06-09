"""Compare the pyramid pipeline (SIFT backbone, stand-in for RoMa/ELoFTR) vs
classical Control B (SIFT direct + optional MMI refinement) on synthetic
correlative pairs across FOV ratios.

Usage:
    python scripts/run_fov_sweep.py [--n 8] [--out results/fov_sweep.csv]
    python scripts/run_fov_sweep.py --methods pyramid,classical,classical_mi
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

from cma.eval import SweepConfig, classical_method, fov_sweep, pyramid_method
from cma.matchers import SIFTMatcher


def _pyramid_sift(cfg):
    return pyramid_method(SIFTMatcher(), cfg)


def _pyramid_loftr(cfg):
    # Import lazily so the SIFT-only path works without torch installed.
    from cma.matchers import LoFTRMatcher
    return pyramid_method(LoFTRMatcher(), cfg)


def _pyramid_matchanything(cfg):
    from cma.matchers import MatchAnythingMatcher
    return pyramid_method(MatchAnythingMatcher(), cfg)


def _pyramid_roma(cfg):
    from cma.matchers import RoMaMatcher
    return pyramid_method(RoMaMatcher(), cfg)


def _classical_roma(cfg):
    from cma.data import ImagePair
    from cma.eval.sweep import MethodResult
    from cma.matchers import RoMaMatcher
    from cma.pipeline.classical import classical_register

    matcher = RoMaMatcher()

    def _run(pair: ImagePair) -> MethodResult:
        res = classical_register(
            pair.source,
            pair.target,
            matcher=matcher,
            family=cfg.family,
            ransac_threshold_px=cfg.ransac_threshold_px,
            refine_with_mi=False,
        )
        return MethodResult(
            H_target_to_source=res.H_target_to_source,
            n_correspondences=res.n_correspondences,
            n_tiles=1,
            family=res.transform.family,
        )

    return _run


def _classical_matchanything(cfg):
    """Direct (non-pyramid) MatchAnything — the matcher is scale-aware itself."""
    from cma.data import ImagePair
    from cma.eval.sweep import MethodResult
    from cma.matchers import MatchAnythingMatcher
    from cma.pipeline.classical import classical_register

    matcher = MatchAnythingMatcher()

    def _run(pair: ImagePair) -> MethodResult:
        res = classical_register(
            pair.source,
            pair.target,
            matcher=matcher,
            family=cfg.family,
            ransac_threshold_px=cfg.ransac_threshold_px,
            refine_with_mi=False,
        )
        return MethodResult(
            H_target_to_source=res.H_target_to_source,
            n_correspondences=res.n_correspondences,
            n_tiles=1,
            family=res.transform.family,
        )

    return _run


METHODS = {
    "pyramid": ("pyramid_sift", _pyramid_sift),
    "pyramid_loftr": ("pyramid_loftr", _pyramid_loftr),
    "pyramid_roma": ("pyramid_roma", _pyramid_roma),
    "pyramid_matchanything": ("pyramid_matchanything", _pyramid_matchanything),
    "classical_matchanything": ("classical_matchanything", _classical_matchanything),
    "classical_roma": ("classical_roma", _classical_roma),
    "classical": ("classical_sift", lambda cfg: classical_method(False, cfg)),
    "classical_mi": ("classical_sift_mi", lambda cfg: classical_method(True, cfg)),
}


def _summarise(rows: list) -> list[dict]:
    grouped: dict[tuple[str, float], list] = defaultdict(list)
    for r in rows:
        grouped[(r.backbone, r.fov_ratio)].append(r)
    out = []
    # Sorted iteration: by method name, then by descending FOV ratio
    keys = sorted(grouped.keys(), key=lambda k: (k[0], -k[1]))
    for method, fov in keys:
        items = grouped[(method, fov)]
        successes = [r for r in items if r.success]
        if successes:
            mu_errs = [r.mu_err for r in successes]
            p5s = [r.p_match_at_5 for r in successes]
            rt = [r.runtime_s for r in successes]
            out.append(
                {
                    "method": method,
                    "fov_ratio": fov,
                    "n_pairs": len(items),
                    "success_rate": len(successes) / len(items),
                    "mean_mu_err": statistics.fmean(mu_errs),
                    "median_mu_err": statistics.median(mu_errs),
                    "mean_p_match_at_5": statistics.fmean(p5s),
                    "mean_runtime_s": statistics.fmean(rt),
                }
            )
        else:
            out.append(
                {
                    "method": method,
                    "fov_ratio": fov,
                    "n_pairs": len(items),
                    "success_rate": 0.0,
                    "mean_mu_err": float("nan"),
                    "median_mu_err": float("nan"),
                    "mean_p_match_at_5": float("nan"),
                    "mean_runtime_s": float("nan"),
                }
            )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8, help="pairs per FOV ratio")
    parser.add_argument(
        "--out", type=Path, default=Path("results/fov_sweep.csv")
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="pyramid,classical",
        help="comma-separated subset of: " + ",".join(METHODS),
    )
    parser.add_argument(
        "--cross-modal",
        type=str,
        default=None,
        choices=[None, "invert", "gamma", "edge", "smooth", "stack"],
        help="apply a cross-modal contrast transform to each target",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="noise",
        choices=["noise", "natural"],
        help="source content: layered noise (default) or skimage astronaut",
    )
    args = parser.parse_args()

    cfg = SweepConfig(
        n_pairs=args.n, cross_modal=args.cross_modal, source=args.source
    )
    all_rows: list = []
    for key in args.methods.split(","):
        key = key.strip()
        if key not in METHODS:
            raise SystemExit(f"unknown method '{key}', expected one of {list(METHODS)}")
        name, builder = METHODS[key]
        method = builder(cfg)
        print(f"running {name} ...")
        rows = fov_sweep(name, method, config=cfg)
        all_rows.extend(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(all_rows[0].as_dict().keys())
    with args.out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_rows:
            writer.writerow(r.as_dict())
    print(f"\nwrote {len(all_rows)} rows -> {args.out}")

    summary = _summarise(all_rows)
    print("\nFOV sweep summary:")
    print(
        f"{'method':<22} {'fov':>6} {'n':>4} {'succ':>6} {'mean_mu_err':>12} "
        f"{'med_mu_err':>11} {'P@5':>6} {'runtime_s':>10}"
    )
    for s in summary:
        print(
            f"{s['method']:<22} {s['fov_ratio']:>6.3f} {s['n_pairs']:>4d} "
            f"{s['success_rate']:>6.2f} {s['mean_mu_err']:>12.3f} "
            f"{s['median_mu_err']:>11.3f} {s['mean_p_match_at_5']:>6.3f} "
            f"{s['mean_runtime_s']:>10.3f}"
        )


if __name__ == "__main__":
    main()
