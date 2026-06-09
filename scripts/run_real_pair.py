"""Run one real AmalgaMatch pair through the full pipeline (loader -> register).

Confirms the loader and pipeline meet at the seam. SIFT will not survive hard
cross-modal pairs; pick a same-detector-family subset by default.

Usage: python scripts/run_real_pair.py [subclass] [pair_idx]
"""

from __future__ import annotations

import sys
import time

import numpy as np

from cma.data import AmalgaMatchLoader
from cma.matchers.sift import SIFTMatcher
from cma.metrics import registration_metrics
from cma.pipeline import register

subclass = sys.argv[1] if len(sys.argv) > 1 else "CoNi_SEM-SE_SEM-BSE_Cracks-SameSliceSerialSectioning"
pair_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0

loader = AmalgaMatchLoader("data/AmalgaMatch")
items = list(loader.iter(subclasses=[subclass]))
pair, rec = items[pair_idx]

print(f"pair {rec.pair_id} ({rec.group}/{rec.subclass})")
print(f"  source {pair.source.shape} @ {rec.source_pixel_nm:.1f} nm/px ({rec.source_path.name})")
print(f"  target {pair.target.shape} @ {rec.target_pixel_nm:.1f} nm/px ({rec.target_path.name})")
print(f"  scale_ratio {pair.scale_ratio:.4f}, GT points {len(pair.gt)}, flipped {rec.flipped}")

t0 = time.perf_counter()
result = register(pair.source, pair.target, SIFTMatcher(), pair.scale_ratio)
elapsed = time.perf_counter() - t0

H = result.H_target_to_source
tgt_h = np.hstack([pair.gt.tgt_xy, np.ones((len(pair.gt), 1))])
proj = (H @ tgt_h.T).T
proj = proj[:, :2] / proj[:, 2:3]
m = registration_metrics(proj, pair.gt.src_xy)

print(f"\nregistered in {elapsed:.1f}s: {result.n_tiles} tiles, "
      f"{result.n_correspondences} correspondences, family={result.transform.family}")
print(f"  mu_err   {m.mu_err:9.2f} px")
print(f"  med_err  {m.med_err:9.2f} px")
print(f"  P@1/3/5/10 = {m.p_match_at_1:.2f} / {m.p_match_at_3:.2f} / "
      f"{m.p_match_at_5:.2f} / {m.p_match_at_10:.2f}")
