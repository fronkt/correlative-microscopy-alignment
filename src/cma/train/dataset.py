"""Training samples for MA-RoMa fine-tuning from sparse AmalgaMatch GT.

GT is sparse (median ~37 points/pair), but RoMa's loss wants a dense
A-grid -> B warp. We densify per pair: fit a thin-plate spline (affine
for <16 points) from GT source->target coords, evaluate it on a regular
grid over the source, and supervise only inside the GT support region
(bounding box of GT source points, 10% dilated) where the mapped point
lands inside the target. The GT global-affine residual is ~10 px, which
matches the coarse regime the decoder learns at 560 res.

Augmentation: random FOV crop of the target (the failure mode the FOV
ladder exposed) and independent photometric jitter (gamma / brightness /
contrast) per image. Geometry of the warp is updated for the crop.
"""

from __future__ import annotations

import numpy as np
import torch

from cma.data.amalgamatch import AmalgaMatchLoader

WARP_GRID = 140  # warp stored at this res; the loss interpolates per scale
TRAIN_RES = 560


def _to_rgb01(img: np.ndarray) -> np.ndarray:
    arr = img
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    return np.clip(arr.astype(np.float32), 0.0, 1.0)


def _fit_src_to_tgt(src_xy: np.ndarray, tgt_xy: np.ndarray):
    """Return f(P)->tgt coords. TPS for >=16 points, else least-squares affine."""
    if len(src_xy) >= 16:
        from scipy.interpolate import RBFInterpolator
        return RBFInterpolator(src_xy, tgt_xy, kernel="thin_plate_spline",
                               smoothing=1.0)
    A = np.hstack([src_xy, np.ones((len(src_xy), 1))])
    M, *_ = np.linalg.lstsq(A, tgt_xy, rcond=None)
    return lambda P: np.hstack([P, np.ones((len(P), 1))]) @ M


class WarpPairDataset(torch.utils.data.Dataset):
    """Yields RoMa-style batches with dense pseudo-GT warps."""

    def __init__(self, root: str, pair_ids: list[str], augment: bool = True,
                 seed: int = 0) -> None:
        loader = AmalgaMatchLoader(root)
        wanted = set(pair_ids)
        self.recs = [r for r in loader.records if r.pair_id in wanted]
        self.loader = loader
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        from romatch.utils.utils import get_tuple_transform_ops
        self.transform = get_tuple_transform_ops(
            resize=(TRAIN_RES, TRAIN_RES), normalize=True)

    def __len__(self) -> int:
        return len(self.recs)

    def _photometric(self, img: np.ndarray) -> np.ndarray:
        g = self.rng.uniform(0.7, 1.4)
        c = self.rng.uniform(0.8, 1.2)
        b = self.rng.uniform(-0.08, 0.08)
        return np.clip(((img ** g) - 0.5) * c + 0.5 + b, 0.0, 1.0)

    def __getitem__(self, idx: int) -> dict:
        rec = self.recs[idx]
        pair = self.loader.load_pair(rec)
        A = _to_rgb01(pair.source)
        B = _to_rgb01(pair.target)
        src_xy = pair.gt.src_xy.copy()
        tgt_xy = pair.gt.tgt_xy.copy()

        # FOV crop augmentation on the target
        if self.augment and self.rng.random() < 0.5 and len(tgt_xy) >= 4:
            hb, wb = B.shape[:2]
            f = float(np.sqrt(self.rng.uniform(0.15, 0.8)))
            ch, cw = max(16, round(hb * f)), max(16, round(wb * f))
            anchor = tgt_xy[self.rng.integers(len(tgt_xy))]
            x0 = int(np.clip(round(anchor[0] - cw / 2), 0, wb - cw))
            y0 = int(np.clip(round(anchor[1] - ch / 2), 0, hb - ch))
            B = B[y0:y0 + ch, x0:x0 + cw]
            tgt_xy = tgt_xy - [x0, y0]

        if self.augment:
            A = self._photometric(A)
            B = self._photometric(B)

        # dense pseudo-GT warp on a regular grid over A
        ha, wa = A.shape[:2]
        hb, wb = B.shape[:2]
        f = _fit_src_to_tgt(src_xy, tgt_xy)
        gx = np.linspace(0, wa - 1, WARP_GRID)
        gy = np.linspace(0, ha - 1, WARP_GRID)
        G = np.stack(np.meshgrid(gx, gy), axis=-1).reshape(-1, 2)  # (N,2) x,y
        mapped = np.asarray(f(G), dtype=np.float64)

        x_lo, y_lo = src_xy.min(axis=0)
        x_hi, y_hi = src_xy.max(axis=0)
        mx, my = 0.10 * (x_hi - x_lo) + 1, 0.10 * (y_hi - y_lo) + 1
        in_support = ((G[:, 0] >= x_lo - mx) & (G[:, 0] <= x_hi + mx)
                      & (G[:, 1] >= y_lo - my) & (G[:, 1] <= y_hi + my))
        in_target = ((mapped[:, 0] >= 0) & (mapped[:, 0] <= wb - 1)
                     & (mapped[:, 1] >= 0) & (mapped[:, 1] <= hb - 1))
        prob = (in_support & in_target).astype(np.float32)

        # normalize to [-1, 1] in B coords (RoMa pixel-center convention:
        # coord = 2*(px+0.5)/size - 1, the inverse of to_pixel_coordinates)
        warp = np.empty_like(mapped, dtype=np.float32)
        warp[:, 0] = 2 * (mapped[:, 0] + 0.5) / wb - 1
        warp[:, 1] = 2 * (mapped[:, 1] + 0.5) / hb - 1
        warp = np.clip(warp, -1.5, 1.5)

        from PIL import Image
        pil_a = Image.fromarray((A * 255).astype(np.uint8))
        pil_b = Image.fromarray((B * 255).astype(np.uint8))
        im_A, im_B = self.transform((pil_a, pil_b))
        return {
            "im_A": im_A,
            "im_B": im_B,
            "gt_warp": torch.from_numpy(
                warp.reshape(WARP_GRID, WARP_GRID, 2)),
            "gt_prob": torch.from_numpy(
                prob.reshape(WARP_GRID, WARP_GRID)),
            "pair_id": rec.pair_id,
        }
