"""RoMa matcher (Parskatt/RoMa) via the `romatch` PyPI package.

RoMa is a dense robust matcher that outputs a per-pixel warp plus a
certainty mask, then sparsifies via `model.sample(...)`. We expose the
standard `Matcher` ABC so it composes with the pyramid pipeline.

Install: `pip install romatch`. Pretrained weights download on first use.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from cma.matchers.base import Correspondences, Matcher
from cma.matchers.foundational import MatcherNotInstalled

try:
    import torch
    from PIL import Image
    from romatch import roma_outdoor, tiny_roma_v1_outdoor
    _ROMATCH_AVAILABLE = True
except ImportError:
    _ROMATCH_AVAILABLE = False


class RoMaMatcher(Matcher):
    """Parskatt/RoMa outdoor (or tiny). Dense warp + certainty -> sparse."""

    name = "roma"

    def __init__(
        self,
        variant: str = "outdoor",
        device: str = "cuda",
        sample_thresh: float | None = None,
        max_long_side: int = 832,
    ) -> None:
        if not _ROMATCH_AVAILABLE:
            raise MatcherNotInstalled(
                "RoMaMatcher needs `romatch`. `pip install romatch` (with the "
                "project's torch already installed)."
            )
        if variant not in ("outdoor", "tiny_outdoor"):
            raise ValueError(f"variant must be 'outdoor' or 'tiny_outdoor', got {variant}")
        self.variant = variant
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_long_side = int(max_long_side)
        loader = roma_outdoor if variant == "outdoor" else tiny_roma_v1_outdoor
        self._model = loader(device=self.device)
        if sample_thresh is not None:
            self._model.sample_thresh = float(sample_thresh)

    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> Correspondences:
        # romatch.match() expects two image file paths. Write to a per-call
        # temp directory so concurrent calls don't collide.
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            a_path = td_path / "a.png"
            b_path = td_path / "b.png"
            pil_a, scale_a, sz_a = _save_for_roma(image_a, a_path, self.max_long_side)
            pil_b, scale_b, sz_b = _save_for_roma(image_b, b_path, self.max_long_side)

            with torch.inference_mode():
                warp, certainty = self._model.match(
                    str(a_path), str(b_path), device=self.device
                )
                matches, conf = self._model.sample(warp, certainty)
                kp_a, kp_b = self._model.to_pixel_coordinates(
                    matches, sz_a[0], sz_a[1], sz_b[0], sz_b[1]
                )

        kp_a = kp_a.detach().cpu().numpy().astype(np.float64)
        kp_b = kp_b.detach().cpu().numpy().astype(np.float64)
        conf = conf.detach().cpu().numpy().astype(np.float64)

        # Rescale back to the caller's original pixel coordinates
        kp_a = kp_a / scale_a
        kp_b = kp_b / scale_b

        if kp_a.shape[0] == 0:
            return _empty()
        return Correspondences(a_xy=kp_a, b_xy=kp_b, confidence=conf)


def _save_for_roma(
    img: np.ndarray, path: Path, max_long_side: int
) -> tuple["Image.Image", np.ndarray, tuple[int, int]]:
    """Save numpy image to disk as RGB PNG; downscale if long side > max."""
    arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0) * 255.0
        arr = arr.astype(np.uint8)

    h, w = arr.shape[:2]
    long_side = max(h, w)
    if long_side > max_long_side:
        scale = max_long_side / long_side
        new_h = max(1, int(round(h * scale)))
        new_w = max(1, int(round(w * scale)))
        pil = Image.fromarray(arr).resize((new_w, new_h), Image.BILINEAR)
    else:
        pil = Image.fromarray(arr)
        new_h, new_w = h, w
    pil.save(path)
    scale_xy = np.array([new_w / w, new_h / h], dtype=np.float64)
    # romatch.to_pixel_coordinates takes (H, W) — return the on-disk shape
    return pil, scale_xy, (new_h, new_w)


def _empty() -> Correspondences:
    return Correspondences(
        a_xy=np.zeros((0, 2), dtype=np.float64),
        b_xy=np.zeros((0, 2), dtype=np.float64),
        confidence=np.zeros((0,), dtype=np.float64),
    )
