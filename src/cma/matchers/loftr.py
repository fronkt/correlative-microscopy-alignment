"""LoFTR matcher backed by kornia (auto-downloads pretrained weights).

LoFTR (Sun et al., 2021) is the canonical transformer-based semi-dense matcher
and the direct predecessor of ELoFTR. It's a useful first foundational
backbone because kornia ships pretrained outdoor/indoor weights and no
vendoring is required.

torch + kornia must be installed (`pip install -e .[torch]`). Construction
raises `MatcherNotInstalled` otherwise.
"""

from __future__ import annotations

import numpy as np

from cma.matchers.base import Correspondences, Matcher
from cma.matchers.foundational import MatcherNotInstalled

try:
    import torch
    import kornia.feature as kf
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


class LoFTRMatcher(Matcher):
    """kornia-backed LoFTR. Pretrained on outdoor or indoor scenes."""

    name = "loftr"

    def __init__(
        self,
        pretrained: str = "outdoor",
        device: str = "cuda",
        max_long_side: int = 832,
        confidence_threshold: float = 0.2,
    ) -> None:
        if not _TORCH_AVAILABLE:
            raise MatcherNotInstalled(
                "LoFTRMatcher needs torch + kornia. `pip install -e .[torch]`."
            )
        if pretrained not in ("outdoor", "indoor"):
            raise ValueError(f"pretrained must be 'outdoor' or 'indoor', got {pretrained}")
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.max_long_side = int(max_long_side)
        self.confidence_threshold = float(confidence_threshold)
        self._model = kf.LoFTR(pretrained=pretrained).to(self.device).eval()

    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> Correspondences:
        ta, scale_a = _prep(image_a, self.device, self.max_long_side)
        tb, scale_b = _prep(image_b, self.device, self.max_long_side)
        with torch.inference_mode():
            out = self._model({"image0": ta, "image1": tb})

        kp_a = out["keypoints0"].detach().cpu().numpy()
        kp_b = out["keypoints1"].detach().cpu().numpy()
        conf = out["confidence"].detach().cpu().numpy()

        # Rescale back to original input pixel coords
        kp_a = kp_a / scale_a
        kp_b = kp_b / scale_b

        if self.confidence_threshold > 0:
            keep = conf >= self.confidence_threshold
            kp_a = kp_a[keep]
            kp_b = kp_b[keep]
            conf = conf[keep]

        if kp_a.shape[0] == 0:
            return _empty()
        return Correspondences(a_xy=kp_a.astype(np.float64),
                               b_xy=kp_b.astype(np.float64),
                               confidence=conf.astype(np.float64))


def _prep(img: np.ndarray, device, max_long_side: int):
    """Convert HxW or HxWxC image in [0, 1] to (1, 1, h, w) torch grayscale.

    Returns the tensor and the *uniform* scale factor applied (target / source);
    multiply the model's pixel coords by `1/scale` to get back to input pixels.
    """
    if img.ndim == 3:
        img = img.mean(axis=2)
    h, w = img.shape[:2]
    long_side = max(h, w)
    if long_side > max_long_side:
        scale = max_long_side / long_side
    else:
        scale = 1.0
    # LoFTR also requires dims divisible by 8
    new_h = max(8, int(round(h * scale)) // 8 * 8)
    new_w = max(8, int(round(w * scale)) // 8 * 8)
    if (new_h, new_w) != (h, w):
        import cv2
        img_r = cv2.resize(img.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        img_r = img.astype(np.float32)
    scale_x = new_w / w
    scale_y = new_h / h
    if abs(scale_x - scale_y) > 1e-6:
        # Use the (scale_x, scale_y) pair as a uniform fudge — coordinate
        # rescaling below applies (scale_x, scale_y) to (x, y) independently.
        pass
    tensor = torch.from_numpy(img_r).to(device).unsqueeze(0).unsqueeze(0)
    # Return scale as a (2,) array so we can rescale x and y independently
    return tensor, np.array([scale_x, scale_y], dtype=np.float64)


def _empty() -> Correspondences:
    return Correspondences(
        a_xy=np.zeros((0, 2), dtype=np.float64),
        b_xy=np.zeros((0, 2), dtype=np.float64),
        confidence=np.zeros((0,), dtype=np.float64),
    )
