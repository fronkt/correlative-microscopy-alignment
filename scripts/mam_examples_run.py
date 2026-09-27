"""Re-run the handful of matches that the worked-example figures display.

The benchmark runs (results/*.csv) did not keep per-pair correspondences or fitted
transforms, so a figure that SHOWS a registration needs the match re-run. These re-runs
are for display only: they run on CPU with a fixed seed, and a dense matcher samples its
10,000 correspondences at random, so a re-run's error need not equal the benchmark row.
Each cache file stores both, and every legend quotes the re-run value it draws.

Jobs (name -> what the figure needs):
  pipeline_5842     RoMa, direct, 5842WCu SE/BSE multiscale pair (Fig. 2)
  tile_inside       RoMa on the level-0 tile of CoNi-AM67 SEM/EBSD 1#0 that lies most
  tile_outside      inside / wholly outside the narrow image's true outline (Fig. 3);
                    these also keep the matcher's dense certainty over the tile
  gal_appearance    MatchAnything-RoMa, direct, STEM dark-field vs annular dark-field (Fig. 4)
  gal_fov           MatchAnything-RoMa, direct, smallest-FOV pair in the benchmark (Fig. 4)
  gal_c103_zs       MatchAnything-RoMa zero-shot and fine-tuned on a held-out C103
  gal_c103_ft       SEM / optical pair (Fig. 4)

Resumable: a job whose cache file exists is skipped. Usage:
    python scripts/mam_examples_run.py [job ...]
"""

from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mam_example_lib as L  # noqa: E402

from cma.estimators import fit_transform  # noqa: E402
from cma.pyramid import build  # noqa: E402

RANSAC_PX = 5.5  # benchmark protocol, as run_baselines_A.py
SEED = 0
FT_WEIGHTS = L.ROOT / "checkpoints" / "ma_roma_ft.pth"

PAIRS = {
    "5842": "eval_5842WCu-Spalled_SEM-SE_SEM-BSE_Multiscale_0#0",
    "tiles": "eval_CoNi-AM67_SEM-EBSD_SameSliceSerialSectioning_1#0",
    "appearance": "eval_MoTaTiZrHf-HEA-DDRX-900C_TEM_DislocationCharacterization_0#4",
    "fov": "eval_CoNi-AM67_OM-SEM_Multiscale_0#2",
    "c103": "eval_C103_SEM-SE_LOM-DC-Height_FractureSurfaces_1#0",
}
JOBS = {
    "pipeline_5842": ("roma", PAIRS["5842"], "direct"),
    "tile_inside": ("roma", PAIRS["tiles"], "tile_inside"),
    "tile_outside": ("roma", PAIRS["tiles"], "tile_outside"),
    "gal_appearance": ("ma_roma", PAIRS["appearance"], "direct"),
    "gal_fov": ("ma_roma", PAIRS["fov"], "direct"),
    "gal_c103_zs": ("ma_roma", PAIRS["c103"], "direct"),
    "gal_c103_ft": ("ma_roma_ft", PAIRS["c103"], "direct"),
}

_MATCHERS: dict = {}


def matcher(name: str):
    if name not in _MATCHERS:
        from cma.matchers import RoMaMatcher
        if name == "roma":
            _MATCHERS[name] = RoMaMatcher(device="cpu")
        elif name == "ma_roma":
            _MATCHERS[name] = RoMaMatcher(device="cpu", variant="ma_outdoor")
        elif name == "ma_roma_ft":
            _MATCHERS[name] = RoMaMatcher(device="cpu", variant="ma_outdoor", weights_path=str(FT_WEIGHTS))
    return _MATCHERS[name]


def match_dense(m, image_a: np.ndarray, image_b: np.ndarray):
    """RoMaMatcher.match, but also return RoMa's dense certainty over image A.

    Same resize / sample / rescale path as cma.matchers.roma.RoMaMatcher.match.
    """
    import torch
    from cma.matchers.roma import _save_for_roma

    torch.manual_seed(SEED)
    with tempfile.TemporaryDirectory() as td:
        a_path, b_path = Path(td) / "a.png", Path(td) / "b.png"
        _, scale_a, sz_a = _save_for_roma(image_a, a_path, m.max_long_side)
        _, scale_b, sz_b = _save_for_roma(image_b, b_path, m.max_long_side)
        with torch.inference_mode():
            warp, certainty = m._model.match(str(a_path), str(b_path), device=m.device)
            matches, conf = m._model.sample(warp, certainty)
            kp_a, kp_b = m._model.to_pixel_coordinates(matches, sz_a[0], sz_a[1], sz_b[0], sz_b[1])
    cert = certainty.detach().cpu().numpy()
    if cert.ndim == 3:  # romatch returns a batch axis: (1, H, W)
        cert = cert[0]
    assert cert.ndim == 2, cert.shape
    # Symmetric RoMa (the default) concatenates A->B and B->A along the WIDTH; the
    # left half is the certainty that each pixel of A has a counterpart in B.
    if m._model.symmetric:
        assert cert.shape[1] == 2 * cert.shape[0], cert.shape
        cert = cert[:, : cert.shape[1] // 2]
    cert_a = cert
    return (kp_a.cpu().numpy() / scale_a, kp_b.cpu().numpy() / scale_b,
            conf.cpu().numpy(), cert_a)


def footprint_mask(pair, shape_scale: float = 0.25) -> tuple[np.ndarray, float]:
    """Boolean mask (downscaled) of the narrow image's true outline inside the source."""
    h, w = pair.source.shape[:2]
    hh, ww = max(1, int(h * shape_scale)), max(1, int(w * shape_scale))
    poly = L.outline(L.gt_homography(pair), pair.target.shape)[:4] * shape_scale
    mask = np.zeros((hh, ww), np.uint8)
    cv2.fillPoly(mask, [np.round(poly).astype(np.int32)], 1)
    return mask.astype(bool), shape_scale


def pick_tiles(pair):
    """Level-0 tiles with the largest / zero overlap with the true outline."""
    tiles = build(pair.source, pair.scale_ratio, tile_size=int(min(pair.target.shape[:2])), overlap=0.5)
    mask, f = footprint_mask(pair)
    scored = []
    for i, t in enumerate(tiles):
        if t.level != 0:
            continue
        x0, y0 = int(t.x0 * f), int(t.y0 * f)
        s = max(1, int(t.tile_size * f))
        frac = float(mask[y0:y0 + s, x0:x0 + s].mean())
        scored.append((frac, float(t.image.std()), i))
    inside = max(scored)[2]
    outside_candidates = [c for c in scored if c[0] == 0.0]
    outside = max(outside_candidates, key=lambda c: c[1])[2]  # most texture among disjoint tiles
    return tiles, inside, outside, {i: fr for fr, _, i in scored}


def run(job: str) -> None:
    out = L.cache_path(job)
    if out.exists():
        print(f"[skip] {job}: {out.name} exists")
        return
    mname, pid, how = JOBS[job]
    pair, rec = L.load(pid)
    m = matcher(mname)
    t0 = time.time()
    extra = {}
    if how == "direct":
        a_xy, b_xy, conf, cert = match_dense(m, pair.source, pair.target)
        src_xy, tgt_xy = a_xy, b_xy
    else:
        tiles, i_in, i_out, fracs = pick_tiles(pair)
        i = i_in if how == "tile_inside" else i_out
        tile = tiles[i]
        a_xy, b_xy, conf, cert = match_dense(m, tile.image, pair.target)
        src_xy, tgt_xy = tile.tile_to_source(a_xy), b_xy
        extra = dict(tile_index=i, tile_level=tile.level, tile_x0=tile.x0, tile_y0=tile.y0,
                     tile_size=tile.tile_size, tile_overlap_frac=fracs[i], n_tiles=len(tiles))
    est = fit_transform(src_xy=tgt_xy, dst_xy=src_xy, family="auto", ransac_threshold_px=RANSAC_PX)
    H = est.as_3x3()
    proj = L.apply_h(H, pair.gt.tgt_xy)
    ed = np.linalg.norm(proj - pair.gt.src_xy, axis=1)
    np.savez_compressed(
        out, pair_id=pid, matcher=mname, how=how,
        src_xy=src_xy.astype(np.float32), tgt_xy=tgt_xy.astype(np.float32),
        conf=conf.astype(np.float32), certainty=cert.astype(np.float16),
        H=H, inliers=est.inliers, family=est.family, mu_ed=float(ed.mean()),
        gt_proj=proj.astype(np.float32), seconds=time.time() - t0,
        sample_thresh=float(m._model.sample_thresh),
        frac_above_thresh=float((cert > m._model.sample_thresh).mean()), **extra)
    print(f"[done] {job}: {len(src_xy)} matches, inliers {est.n_inliers} ({est.family}), "
          f"mean error {ed.mean():.1f} px, dense certainty median {np.median(cert):.3f}, "
          f"above sample threshold {m._model.sample_thresh}: {(cert > m._model.sample_thresh).mean():.2f}, "
          f"{time.time() - t0:.0f} s {extra}", flush=True)


if __name__ == "__main__":
    L.configure_threads()
    for job in (sys.argv[1:] or list(JOBS)):
        run(job)
