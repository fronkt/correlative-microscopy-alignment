"""Box smoke test for the label-free TTA forward (GPU only).

Validates the RoMa-coupled path that CPU-only torch can't exercise:
  1. the decoder exposes adaptable norm-affine params,
  2. tta_adapt's forward (batch build, decoder forward, cycle 2nd pass, the
     three losses) runs and produces a finite, decreasing-ish loss,
  3. adaptation actually moves the params,
  4. the adapted model still matches,
  5. reset() restores the init params (statelessness).

Run on the box: ``python scripts/tta_smoke.py``. No dataset needed — uses a
synthetic structured pair. Prints a diagnostic dump of decoder module types so
that a zero-param result is debuggable in one shot.
"""

from __future__ import annotations

import collections

import cv2
import numpy as np
import torch

from cma.matchers.roma import RoMaMatcher
from cma.train.finetune import build_model
from cma.tta import collect_norm_affine_params, tta_adapt


def _synthetic_pair(seed: int = 0):
    rng = np.random.default_rng(seed)
    base = rng.random((512, 512, 3)).astype(np.float32)
    base = cv2.GaussianBlur(base, (0, 0), 3.0)  # matchable structure
    src = np.clip(base, 0, 1)
    tgt = np.clip(base[64:320, 96:352].copy(), 0, 1)  # sub-window: scale+shift
    return src, tgt


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}")
    model = build_model(torch.device(dev))  # MA-RoMa, decoder grad-enabled

    # diagnostic: what module types are in the decoder, and how many norm params
    types = collections.Counter(type(m).__name__ for m in model.decoder.modules())
    print("decoder module-type histogram (top 12):",
          dict(types.most_common(12)))
    params = collect_norm_affine_params(model.decoder)
    n_elems = sum(p.numel() for p in params)
    print(f"norm-affine params: {len(params)} tensors, {n_elems} elements")
    if not params:
        raise SystemExit(
            "NO norm-affine params in decoder — adapt set must be reconsidered "
            "(see histogram above for the actual norm layer types).")

    before = [p.detach().clone() for p in params]
    adapted, hist, reset = tta_adapt(
        model, *_synthetic_pair(), w_scale=1.0, w_appearance=1.0,
        anchor_lambda=0.1, steps=6, lr=1e-3, device=dev)
    print("loss history:", [round(float(h), 5) for h in hist])
    assert np.all(np.isfinite(hist)), "non-finite loss"

    moved = max(float((p - b).abs().max()) for p, b in zip(params, before, strict=True))
    print(f"max param move during adapt: {moved:.3e}")
    assert moved > 0, "adaptation did not move any param"

    corr = RoMaMatcher(variant="ma_outdoor", device=dev, model=adapted).match(
        *_synthetic_pair())
    print(f"adapted match: {len(corr)} correspondences")
    assert len(corr) > 0, "adapted model produced no matches"

    reset()
    after = max(float((p - b).abs().max()) for p, b in zip(params, before, strict=True))
    print(f"max param delta after reset: {after:.3e} (expect ~0)")
    assert after < 1e-6, "reset did not restore init params"

    print("SMOKE OK")


if __name__ == "__main__":
    main()
