"""Shared helpers for the worked-example figures of the M&M rewrite.

The rejected manuscript (MAM-26-246) reported failures only as rates; the editor asked for
"examples of the failures". These helpers load individual AmalgaMatch pairs, project the
narrow-field image's outline into the wide-field image under the ground-truth mapping and
under a fitted transform, and prepare images for display. Nothing here computes a number
that appears in a table: table values come from the per-pair result CSVs.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from cma.data import AmalgaMatchLoader

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "AmalgaMatch"
CACHE = ROOT / "results" / "mam_examples"

_LOADER: AmalgaMatchLoader | None = None


def loader() -> AmalgaMatchLoader:
    global _LOADER
    if _LOADER is None:
        _LOADER = AmalgaMatchLoader(DATA)
    return _LOADER


def load(pair_id: str):
    """(ImagePair, record) for one pair_id such as 'eval_C103_..._1#0'."""
    for rec in loader().records:
        if rec.pair_id == pair_id:
            return loader().load_pair(rec), rec
    raise KeyError(pair_id)


def apply_h(H: np.ndarray, xy: np.ndarray) -> np.ndarray:
    hom = np.hstack([xy, np.ones((len(xy), 1))])
    p = hom @ H.T
    return p[:, :2] / p[:, 2:3]


def gt_homography(pair) -> np.ndarray:
    """Least-squares homography target -> source through all GT points.

    Used only to DRAW the true outline of the narrow image; the benchmark's own error is
    measured point by point at the GT correspondences, never through this fit.
    """
    H, _ = cv2.findHomography(pair.gt.tgt_xy.astype(np.float32),
                              pair.gt.src_xy.astype(np.float32), method=0)
    return H


def outline(H: np.ndarray, target_shape) -> np.ndarray:
    """Closed polygon (5, 2) of the target frame's corners mapped by H into the source."""
    h, w = target_shape[:2]
    c = np.array([[0, 0], [w, 0], [w, h], [0, h], [0, 0]], dtype=np.float64)
    return apply_h(H, c)


def display(img: np.ndarray, max_side: int = 900) -> tuple[np.ndarray, float]:
    """Contrast-stretched uint8 copy with the long side capped; returns (image, scale)."""
    a = np.asarray(img, dtype=np.float32)
    if a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]
    lo, hi = np.percentile(a, [0.5, 99.5])
    a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    h, w = a.shape[:2]
    f = min(1.0, max_side / max(h, w))
    if f < 1.0:
        a = cv2.resize(a, (max(1, round(w * f)), max(1, round(h * f))), interpolation=cv2.INTER_AREA)
    return (a * 255).astype(np.uint8), f


def _pair_meta(rec, path: Path) -> dict:
    """The per-image metadata the benchmark ships inside the pair's own eval file."""
    from cma.data.amalgamatch import _load_eval

    data = _load_eval(rec.eval_path)
    subset_dir = DATA / rec.subclass
    for rel, meta in zip(data["image_paths"], data["image_metadata"]):
        if Path(subset_dir / rel).name == path.name:
            return meta
    raise KeyError(f"{path.name} not listed in {rec.eval_path.name}")


# Plain names for the dataset's metadata vocabulary (image_metadata.json, all 19 subsets).
_EBSD_MAPS = (("Inverse Polefigure", "EBSD orientation map"), ("Image Quality", "EBSD image-quality map"),
              ("Confidence Index", "EBSD confidence map"), ("Band Contrast", "EBSD band-contrast map"),
              ("Grain IDs", "EBSD grain map"), ("Pattern Sharpness", "EBSD pattern-sharpness map"),
              ("PRIAS", "EBSD PRIAS map"))
_DETECTORS = {"Backscattered Electron Detector": "SEM, backscattered electrons",
              "Everhart-Thornley Detector": "SEM, secondary electrons",
              "InLens Detector": "SEM, in-lens secondary electrons",
              "Z-Stack Height": "Light optical, height map",
              "Z-Stack Depth Composition": "Light optical, extended focus",
              "High-Angle Annular Dark-Field": "STEM, annular dark field",
              "Darfk-Field (Weak-Beam)": "TEM, weak-beam dark field"}


def modality_label(rec, which: str) -> str:
    """Plain modality name from the dataset's own metadata, e.g. 'SEM, backscattered electrons'."""
    path = rec.source_path if which == "source" else rec.target_path
    m = _pair_meta(rec, path)
    scope, mode = m["Microscope Type"], m.get("Detector/Imaging Mode") or ""
    derived = str(m.get("Derived Modality Type") or "None")
    if "Digital Image Correlation" in derived:
        return "SEM strain map (digital image correlation)"
    if mode == "Electron Backscatter Diffraction":
        for key, lab in _EBSD_MAPS:
            if key in derived:
                return lab
        return "EBSD map"
    if mode in _DETECTORS:
        return _DETECTORS[mode]
    if scope.startswith("Scanning Transmission"):
        return "STEM, " + mode.lower()
    if scope.startswith("Light Optical"):
        return "Light optical, " + mode.lower()
    return f"{scope}, {mode.lower()}"


def short(pair_id: str) -> str:
    return pair_id.replace("eval_", "")


def cache_path(tag: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    return CACHE / f"{tag}.npz"


def configure_threads() -> None:
    """Use the P-cores without starving the machine's other work.

    On Windows a windowless background process is classed as efficiency work and
    confined to the low-power cores; opt this process out (no admin needed).
    """
    import torch
    torch.set_num_threads(int(os.environ.get("CMA_THREADS", "4")))
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class _State(ctypes.Structure):
            _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG),
                        ("StateMask", wintypes.ULONG)]

        state = _State(1, 1, 0)  # EXECUTION_SPEED control, state off = not throttled
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = wintypes.HANDLE
        k32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        k32.SetProcessInformation.restype = wintypes.BOOL
        ok = k32.SetProcessInformation(k32.GetCurrentProcess(), 4, ctypes.byref(state), ctypes.sizeof(state))
        print(f"power throttling opt-out: {'ok' if ok else 'FAILED, error %d' % ctypes.get_last_error()}",
              flush=True)
