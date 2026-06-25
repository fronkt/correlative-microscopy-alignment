"""Domain-shift severity metrics — the axes of the paper's central figure.

Two quantitative severities place every (dataset, backbone) point on the plane:

- **Appearance severity** = Fréchet distance between target imagery and a
  backbone-pretraining proxy in *frozen-encoder* feature space (Fréchet DINOv2
  Distance practice; Heusel et al. 2017 for the Fréchet form). Pure math here;
  the DINOv2 extraction wrapper is box-side.
- **Scale severity** = ``|log2(scale_ratio)|`` — symmetric, zero when source and
  target share a resolution, growing with FOV mismatch. Reuses the existing
  ``scale_ratio`` carried by every ``ImagePair``.

``frechet_distance`` and ``feature_stats`` are pure and CPU-unit-tested.
"""

from __future__ import annotations

import numpy as np


def feature_stats(feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean (D,) and covariance (D, D) of an (N, D) feature matrix."""
    feats = np.asarray(feats, dtype=np.float64)
    if feats.ndim != 2:
        raise ValueError(f"feats must be (N, D); got {feats.shape}")
    mu = feats.mean(axis=0)
    cov = np.cov(feats, rowvar=False)
    return mu, np.atleast_2d(cov)


def frechet_distance(mu1: np.ndarray, cov1: np.ndarray,
                     mu2: np.ndarray, cov2: np.ndarray,
                     eps: float = 1e-6) -> float:
    """Fréchet distance between two Gaussians (the FID/FDD formula).

    ``||mu1 - mu2||² + Tr(cov1 + cov2 - 2·sqrt(cov1·cov2))``. Robust to tiny
    numerical asymmetry/imaginary components from the matrix square root.
    """
    from scipy import linalg

    mu1, mu2 = np.asarray(mu1, np.float64), np.asarray(mu2, np.float64)
    cov1, cov2 = np.atleast_2d(cov1).astype(np.float64), np.atleast_2d(cov2).astype(np.float64)
    diff = mu1 - mu2

    def _sqrtm(m: np.ndarray) -> np.ndarray:
        res = linalg.sqrtm(m)  # newer scipy returns the array directly
        return res[0] if isinstance(res, tuple) else res

    covmean = _sqrtm(cov1 @ cov2)
    if not np.isfinite(covmean).all():  # singular product → ridge and retry
        offset = np.eye(cov1.shape[0]) * eps
        covmean = _sqrtm((cov1 + offset) @ (cov2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(cov1) + np.trace(cov2) - 2 * np.trace(covmean))


def scale_severity(scale_ratio: float) -> float:
    """Scale-axis severity ``|log2(scale_ratio)|`` (0 = matched resolution)."""
    if scale_ratio <= 0:
        raise ValueError(f"scale_ratio must be > 0, got {scale_ratio}")
    return float(abs(np.log2(scale_ratio)))


def appearance_severity(target_feats: np.ndarray,
                        ref_mu: np.ndarray, ref_cov: np.ndarray) -> float:
    """Fréchet distance from target features to a precomputed backbone-domain ref."""
    mu, cov = feature_stats(target_feats)
    return frechet_distance(mu, cov, ref_mu, ref_cov)


class DinoFeatureExtractor:
    """Frozen DINOv2 patch-feature extractor for the Fréchet metric (box-side).

    Lazily imports torch/DINOv2 so the pure metrics above stay importable on a
    CPU-only box without pulling weights. Returns (N, D) pooled features.
    """

    def __init__(self, model_name: str = "dinov2_vitb14", device: str = "cuda") -> None:
        self.model_name = model_name
        self._device = device
        self._model = None

    def _ensure(self) -> None:
        if self._model is not None:
            return
        import torch
        self._torch = torch
        self._device_t = torch.device(
            self._device if torch.cuda.is_available() else "cpu")
        self._model = torch.hub.load("facebookresearch/dinov2", self.model_name)
        self._model.eval().to(self._device_t)

    def features(self, images: list[np.ndarray], res: int = 224) -> np.ndarray:
        """CLS-token features for a list of RGB float images → (N, D)."""
        self._ensure()
        torch = self._torch
        import torch.nn.functional as F
        out = []
        with torch.no_grad():
            for img in images:
                a = np.asarray(img, dtype=np.float32)
                if a.ndim == 2:
                    a = np.stack([a] * 3, -1)
                t = torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0).to(self._device_t)
                t = F.interpolate(t, size=(res, res), mode="bilinear", align_corners=False)
                feat = self._model(t)  # (1, D) CLS embedding
                out.append(feat.squeeze(0).cpu().numpy())
        return np.stack(out, axis=0)
