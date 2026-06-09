"""Synthetic image-pair generator for tests and FOV sensitivity sweeps.

Two modes are supported:
  * `synthesize_pair`  : target is a rotated, translated crop of the source
                         with optional gaussian noise. Same-modality.
  * `synthesize_cross_modal_pair`: as above, then applies a non-trivial
                                   contrast/structural transform to the
                                   target that simulates cross-modality
                                   (SEM vs EBSD vs AFM topography). This
                                   is the stress-test setup where the
                                   pyramid+foundational wrapper is
                                   expected to outperform classical SIFT.
"""

from __future__ import annotations

from typing import Literal

import cv2
import numpy as np

from cma.data.types import ImagePair, KeypointSet

CrossModalMode = Literal["invert", "gamma", "edge", "smooth", "stack"]


def _prepare_source(img: np.ndarray, size: int) -> np.ndarray:
    """Convert an arbitrary input image into a (size, size) float32 in [0, 1]."""
    arr = np.asarray(img)
    if arr.ndim == 3:
        # RGB -> grayscale via standard luma weights
        if arr.shape[2] >= 3:
            arr = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2])
        else:
            arr = arr[..., 0]
    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    arr = cv2.resize(arr, (size, size), interpolation=cv2.INTER_AREA)
    return np.clip(arr, 0.0, 1.0)


def natural_source_image(size: int = 1024) -> np.ndarray:
    """Return a natural photograph at `(size, size)` (skimage.data.astronaut)."""
    from skimage.data import astronaut
    return _prepare_source(astronaut(), size)


def _textured_image(size: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Layered noise: gives the matcher repeatable, content-rich texture
    low = rng.random((size // 16, size // 16), dtype=np.float32)
    mid = rng.random((size // 4, size // 4), dtype=np.float32)
    hi = rng.random((size, size), dtype=np.float32)
    low_up = cv2.resize(low, (size, size), interpolation=cv2.INTER_CUBIC)
    mid_up = cv2.resize(mid, (size, size), interpolation=cv2.INTER_CUBIC)
    img = 0.5 * low_up + 0.3 * mid_up + 0.2 * hi
    img = (img - img.min()) / (img.max() - img.min() + 1e-9)
    return img.astype(np.float32)


def synthesize_pair(
    source_size: int = 1024,
    fov_ratio: float = 0.2,
    target_size: int = 256,
    seed: int = 0,
    rotation_deg: float = 7.0,
    noise_sigma: float = 0.01,
    source_image: np.ndarray | None = None,
) -> tuple[ImagePair, np.ndarray]:
    """Build a synthetic correlative pair with known ground-truth homography.

    The target image is a rotated, translated crop of the source. The returned
    `H_gt` maps target coordinates into source coordinates: x_s = H_gt @ x_t.

    Args:
        source_size: side length of I_s in pixels.
        fov_ratio: target FOV as a fraction of source FOV (area ratio).
        target_size: side length of I_t in pixels.
        seed: RNG seed.
        rotation_deg: target-vs-source rotation, degrees.
        noise_sigma: additive gaussian noise on I_t (std in [0, 1] intensity).
        source_image: optional source image to use instead of layered noise.
            Will be resized to (source_size, source_size). Useful for testing
            matchers that need natural-image priors (e.g. MatchAnything).
    """
    if not 0 < fov_ratio <= 1:
        raise ValueError(f"fov_ratio must be in (0, 1], got {fov_ratio}")

    rng = np.random.default_rng(seed)
    if source_image is None:
        I_s = _textured_image(source_size, seed)
    else:
        I_s = _prepare_source(source_image, source_size)

    # Target physical extent in source pixels: side length s_t_in_s
    s_t_in_s = source_size * np.sqrt(fov_ratio)
    # Center of target window in source coords (kept clear of borders when
    # geometry allows; otherwise centered, with warpPerspective reflecting at
    # the boundary).
    margin = s_t_in_s / np.sqrt(2) + 4
    if 2 * margin >= source_size:
        cx = source_size / 2.0
        cy = source_size / 2.0
    else:
        cx = float(rng.uniform(margin, source_size - margin))
        cy = float(rng.uniform(margin, source_size - margin))

    # Scale ratio = pix_size(I_t) / pix_size(I_s) = s_t_in_s / target_size
    scale_ratio = s_t_in_s / target_size

    theta = np.deg2rad(rotation_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    # H_gt: target(x,y) -> source(X,Y)
    # Compose translate(cx,cy) * rotate(theta) * scale(scale_ratio) * translate(-target_size/2,-target_size/2)
    R = np.array(
        [
            [cos_t, -sin_t, 0.0],
            [sin_t, cos_t, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    S = np.diag([scale_ratio, scale_ratio, 1.0])
    T_pre = np.array(
        [
            [1, 0, -target_size / 2.0],
            [0, 1, -target_size / 2.0],
            [0, 0, 1.0],
        ],
        dtype=np.float64,
    )
    T_post = np.array(
        [
            [1, 0, cx],
            [0, 1, cy],
            [0, 0, 1.0],
        ],
        dtype=np.float64,
    )
    H_gt = T_post @ R @ S @ T_pre

    # Sample I_t by warping I_s into the target frame using H_gt
    # cv2.warpPerspective requires the matrix that maps src->dst, i.e. inverse of H_gt
    H_inv = np.linalg.inv(H_gt)
    I_t = cv2.warpPerspective(
        I_s,
        H_inv,
        (target_size, target_size),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    if noise_sigma > 0:
        I_t = I_t + rng.normal(0.0, noise_sigma, size=I_t.shape).astype(np.float32)
        I_t = np.clip(I_t, 0.0, 1.0)

    # Ground-truth keypoint set: regular grid in target, projected to source
    n = 8
    xs, ys = np.meshgrid(
        np.linspace(8, target_size - 8, n),
        np.linspace(8, target_size - 8, n),
    )
    tgt_xy = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float64)
    src_xy = _apply_h(H_gt, tgt_xy)
    gt = KeypointSet(src_xy=src_xy, tgt_xy=tgt_xy)

    pair = ImagePair(
        source=I_s,
        target=I_t,
        scale_ratio=scale_ratio,
        gt=gt,
        metadata={"fov_ratio": fov_ratio, "rotation_deg": rotation_deg, "seed": seed},
    )
    return pair, H_gt


def _apply_h(H: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Apply 3x3 homography to (N, 2) points."""
    ones = np.ones((xy.shape[0], 1), dtype=xy.dtype)
    hom = np.concatenate([xy, ones], axis=1)
    proj = hom @ H.T
    return proj[:, :2] / proj[:, 2:3]


def _cross_modal_transform(
    img: np.ndarray,
    mode: CrossModalMode,
    seed: int,
) -> np.ndarray:
    """Apply a contrast/structural distortion that mimics modality mismatch."""
    rng = np.random.default_rng(seed)
    x = img.astype(np.float32)
    if mode == "invert":
        # Hard contrast inversion (e.g. SE vs BSE-like polarity flip)
        return 1.0 - x
    if mode == "gamma":
        gamma = float(rng.uniform(0.3, 3.0))
        return np.clip(x, 0.0, 1.0) ** gamma
    if mode == "edge":
        # Gradient-magnitude (edges only) — simulates EBSD boundary maps
        gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        if mag.max() > 0:
            mag = mag / mag.max()
        return mag
    if mode == "smooth":
        # Heavy gaussian blur — simulates AFM topography vs SEM detail
        sigma = float(rng.uniform(2.0, 4.0))
        k = int(2 * round(3 * sigma) + 1)
        return cv2.GaussianBlur(x, (k, k), sigma)
    if mode == "stack":
        # Apply 2-3 of the above to maximally stress the matcher
        order = list(rng.permutation(["invert", "gamma", "edge"]))[:2]
        out = x
        for m in order:
            out = _cross_modal_transform(out, m, seed=seed + 1)
        return out
    raise ValueError(f"unknown cross-modal mode '{mode}'")


def synthesize_cross_modal_pair(
    source_size: int = 1024,
    fov_ratio: float = 0.2,
    target_size: int = 256,
    seed: int = 0,
    rotation_deg: float = 7.0,
    noise_sigma: float = 0.01,
    mode: CrossModalMode = "stack",
    source_image: np.ndarray | None = None,
) -> tuple[ImagePair, np.ndarray]:
    """Same as `synthesize_pair`, but the target gets a cross-modal transform.

    The transform is applied AFTER the warp + noise — so ground-truth keypoint
    correspondences remain valid (geometry is unchanged).
    """
    pair, H_gt = synthesize_pair(
        source_size=source_size,
        fov_ratio=fov_ratio,
        target_size=target_size,
        seed=seed,
        rotation_deg=rotation_deg,
        noise_sigma=noise_sigma,
        source_image=source_image,
    )
    target_x = _cross_modal_transform(pair.target, mode, seed=seed + 1000)
    pair.target = np.clip(target_x.astype(np.float32), 0.0, 1.0)
    pair.metadata["cross_modal_mode"] = mode
    return pair, H_gt
