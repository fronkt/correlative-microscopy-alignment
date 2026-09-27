"""Label-free triage of registrations: input transforms, agreement scores, statistics.

Used by the CJSJ study (tasks/todo.md, "label-free triage of registrations"). Everything here is
label-free except the evaluation helpers, which take success labels only to *score* a triage rule.

Conventions: images are float arrays in [0, 1] (the AmalgaMatch loader's output), 2-D or H x W x C.
A transform ``H`` is 3 x 3 and maps target (narrow-FOV) pixel coordinates into source coordinates.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable, Sequence

import cv2
import numpy as np

INPUT_TRANSFORMS = ("none", "invert", "histmatch", "clahe", "gradmag")


# --------------------------------------------------------------------------- input transforms


def to_gray(img: np.ndarray) -> np.ndarray:
    """Float32 2-D grayscale in [0, 1]; colour images use the first three channels."""
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 3:
        if a.shape[2] >= 3:
            a = cv2.cvtColor(np.ascontiguousarray(a[..., :3]), cv2.COLOR_RGB2GRAY)
        else:
            a = a[..., 0]
    return np.clip(a, 0.0, 1.0)


def _clahe(g: np.ndarray) -> np.ndarray:
    u8 = np.round(g * 255.0).astype(np.uint8)
    out = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(u8)
    return out.astype(np.float32) / 255.0


def _gradmag(g: np.ndarray) -> np.ndarray:
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    m = np.sqrt(gx * gx + gy * gy)
    hi = float(np.percentile(m, 99.0))
    return np.clip(m / hi, 0.0, 1.0) if hi > 0 else np.zeros_like(m)


def _histmatch(g: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Map g's grey levels so its histogram matches ref's (CDF matching)."""
    src_vals, src_idx, src_counts = np.unique(g.ravel(), return_inverse=True, return_counts=True)
    ref_vals, ref_counts = np.unique(ref.ravel(), return_counts=True)
    src_cdf = np.cumsum(src_counts).astype(np.float64) / g.size
    ref_cdf = np.cumsum(ref_counts).astype(np.float64) / ref.size
    mapped = np.interp(src_cdf, ref_cdf, ref_vals)
    return mapped[src_idx].reshape(g.shape).astype(np.float32)


def apply_input_transform(name: str, source: np.ndarray, target: np.ndarray
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Return (source', target') with the same pixel geometry as the inputs.

    none      both unchanged
    invert    target contrast inverted (1 - x), source unchanged
    histmatch both grayscale; target's histogram matched to the source's
    clahe     both grayscale, CLAHE (clip 2.0, 8 x 8 tiles)
    gradmag   both grayscale, Sobel gradient magnitude normalised by its 99th percentile
    """
    if name == "none":
        return source, target
    if name == "invert":
        return source, (1.0 - np.clip(np.asarray(target, dtype=np.float32), 0.0, 1.0))
    gs, gt = to_gray(source), to_gray(target)
    if name == "histmatch":
        return gs, _histmatch(gt, gs)
    if name == "clahe":
        return _clahe(gs), _clahe(gt)
    if name == "gradmag":
        return _gradmag(gs), _gradmag(gt)
    raise ValueError(f"unknown input transform: {name}")


# --------------------------------------------------------------------------- seeding


def pair_seed(seed: int, pair_id: str) -> int:
    """Deterministic per-(seed, pair) integer, independent of run order."""
    return int(hashlib.sha256(f"{seed}|{pair_id}".encode()).hexdigest()[:8], 16)


def seed_everything(s: int) -> None:
    random.seed(s)
    np.random.seed(s % (2**32))
    cv2.setRNGSeed(s % (2**31))
    try:
        import torch
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
    except ImportError:
        pass


# --------------------------------------------------------------------------- agreement


def grid_points(h: int, w: int, n: int = 5) -> np.ndarray:
    """n x n grid spanning the image (pixel centres of the extreme rows/columns included)."""
    xs = np.linspace(0.0, w - 1.0, n)
    ys = np.linspace(0.0, h - 1.0, n)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], axis=1)


def project(H: np.ndarray, xy: np.ndarray) -> np.ndarray:
    p = np.hstack([xy, np.ones((len(xy), 1))]) @ np.asarray(H, dtype=np.float64).T
    with np.errstate(divide="ignore", invalid="ignore"):
        return p[:, :2] / p[:, 2:3]


def transform_distance(H1: np.ndarray, H2: np.ndarray, pts: np.ndarray) -> float:
    """Mean distance (source px) between where H1 and H2 send the same target points."""
    d = np.linalg.norm(project(H1, pts) - project(H2, pts), axis=1)
    return float(np.mean(d)) if np.all(np.isfinite(d)) else float("inf")


def agreement_scores(Hs: dict[str, np.ndarray | None], pts: np.ndarray,
                     voters: Sequence[str]) -> dict[str, float]:
    """S2 for every candidate in ``Hs``: minus the median distance to the OTHER valid voters.

    A candidate with no transform, or with no valid voter to compare against, scores -inf.
    """
    out: dict[str, float] = {}
    for c, Hc in Hs.items():
        if Hc is None:
            out[c] = float("-inf")
            continue
        ds = [transform_distance(Hc, Hs[v], pts) for v in voters
              if v != c and Hs.get(v) is not None]
        out[c] = -float(np.median(ds)) if ds else float("-inf")
    return out


# --------------------------------------------------------------------------- statistics


def auroc(score: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney AUROC (ties get average ranks). NaN if one class is empty."""
    score = np.asarray(score, dtype=np.float64)
    y = np.asarray(y).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    from scipy.stats import rankdata
    s = np.where(np.isfinite(score), score, np.where(score > 0, 1e300, -1e300))
    ranks = rankdata(s)  # average ranks for ties
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def cluster_bootstrap(stat: Callable[[np.ndarray], float], clusters: np.ndarray,
                      B: int = 10_000, seed: int = 0) -> np.ndarray:
    """Resample whole clusters with replacement; ``stat`` gets the row indices of a replicate.

    Replicates where ``stat`` returns NaN (e.g. one class missing) are dropped.
    """
    clusters = np.asarray(clusters)
    uniq = np.unique(clusters)
    members = [np.flatnonzero(clusters == u) for u in uniq]
    rng = np.random.default_rng(seed)
    out = np.empty(B, dtype=np.float64)
    for b in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        out[b] = stat(np.concatenate([members[k] for k in pick]))
    return out[np.isfinite(out)]


def ci95(boot: np.ndarray) -> tuple[float, float]:
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p for b discordant pairs one way and c the other."""
    from math import comb
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / 2.0**n
    return float(min(1.0, 2.0 * p))


def youden_cutoff(score: np.ndarray, y: np.ndarray) -> float:
    """Threshold t maximising TPR - FPR for the rule 'accept if score >= t'."""
    score = np.asarray(score, dtype=np.float64)
    y = np.asarray(y).astype(bool)
    best_t, best_j = float("inf"), -np.inf
    for t in np.unique(score[np.isfinite(score)]):
        acc = score >= t
        tpr = acc[y].mean() if y.any() else 0.0
        fpr = acc[~y].mean() if (~y).any() else 0.0
        if tpr - fpr > best_j:
            best_j, best_t = tpr - fpr, float(t)
    return best_t


def accepted_success(score: np.ndarray, y: np.ndarray, coverage: float) -> float:
    """Success rate among the top ``coverage`` fraction of pairs by score (at least one pair)."""
    score = np.asarray(score, dtype=np.float64)
    y = np.asarray(y).astype(float)
    k = max(1, int(round(coverage * len(score))))
    top = np.argsort(-score, kind="mergesort")[:k]
    return float(y[top].mean())


def aurc(score: np.ndarray, y: np.ndarray) -> float:
    """Area under the risk-coverage curve (risk = failure rate among accepted; lower is better)."""
    score = np.asarray(score, dtype=np.float64)
    y = np.asarray(y).astype(float)
    order = np.argsort(-score, kind="mergesort")
    fail = 1.0 - y[order]
    risks = np.cumsum(fail) / np.arange(1, len(fail) + 1)
    return float(risks.mean())
