"""MatchAnything (zju-community/matchanything_eloftr) matcher.

Loaded through HuggingFace transformers — no upstream zju3dv repo vendoring
required. Apache-2.0 licensed weights.

Tested combo on the project's GPU box (RTX 5090):
    torch 2.11+cu128 + torchvision 0.26 + transformers 5.10.1
"""

from __future__ import annotations

import numpy as np

from cma.matchers.base import Correspondences, Matcher
from cma.matchers.foundational import MatcherNotInstalled

try:
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForKeypointMatching
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

_DEFAULT_REPO = "zju-community/matchanything_eloftr"


class MatchAnythingMatcher(Matcher):
    """zju3dv MatchAnything (ELoFTR-backbone, large-scale pretraining)."""

    name = "matchanything"

    def __init__(
        self,
        repo_id: str = _DEFAULT_REPO,
        device: str = "cuda",
        confidence_threshold: float = 0.2,
        max_long_side: int = 832,
    ) -> None:
        if not _TRANSFORMERS_AVAILABLE:
            raise MatcherNotInstalled(
                "MatchAnythingMatcher needs `transformers`, `torchvision`, and "
                "`Pillow`. `pip install -e .[torch] && pip install transformers torchvision`."
            )
        self.repo_id = repo_id
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.confidence_threshold = float(confidence_threshold)
        self.max_long_side = int(max_long_side)
        self._processor = AutoImageProcessor.from_pretrained(repo_id)
        self._model = AutoModelForKeypointMatching.from_pretrained(repo_id).to(
            self.device
        ).eval()

    def match(self, image_a: np.ndarray, image_b: np.ndarray) -> Correspondences:
        pil_a, scale_a = _to_pil(image_a, self.max_long_side)
        pil_b, scale_b = _to_pil(image_b, self.max_long_side)

        inputs = self._processor([pil_a, pil_b], return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self._model(**inputs)

        sizes = torch.tensor(
            [[[pil_a.height, pil_a.width], [pil_b.height, pil_b.width]]],
            device=self.device,
        )
        post = self._processor.post_process_keypoint_matching(
            out, sizes, threshold=self.confidence_threshold
        )
        if not post:
            return _empty()

        kp_a = post[0]["keypoints0"].detach().cpu().numpy().astype(np.float64)
        kp_b = post[0]["keypoints1"].detach().cpu().numpy().astype(np.float64)
        conf = post[0]["matching_scores"].detach().cpu().numpy().astype(np.float64)

        if kp_a.shape[0] == 0:
            return _empty()

        # Rescale back to the original input image's pixel coords
        kp_a = kp_a / scale_a
        kp_b = kp_b / scale_b
        return Correspondences(a_xy=kp_a, b_xy=kp_b, confidence=conf)


def _to_pil(img: np.ndarray, max_long_side: int) -> tuple["Image.Image", np.ndarray]:
    """Convert a float [0, 1] numpy image to a PIL RGB image, downscaling so
    that the long side is at most `max_long_side`. Returns (pil, scale_xy)
    where scale_xy is (scale_x, scale_y) applied from input to PIL coords.
    """
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
    scale_xy = np.array([new_w / w, new_h / h], dtype=np.float64)
    return pil, scale_xy


def _empty() -> Correspondences:
    return Correspondences(
        a_xy=np.zeros((0, 2), dtype=np.float64),
        b_xy=np.zeros((0, 2), dtype=np.float64),
        confidence=np.zeros((0,), dtype=np.float64),
    )
