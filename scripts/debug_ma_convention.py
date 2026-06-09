"""Determine MatchAnything's keypoint convention by matching an image to itself."""

import numpy as np
import torch
from PIL import Image

from cma.data import synthesize_pair
from cma.matchers import MatchAnythingMatcher

pair, _ = synthesize_pair(1024, 0.10, 256, seed=0, rotation_deg=0.0, noise_sigma=0.0)
matcher = MatchAnythingMatcher()

# Match target to itself: keypoints0 and keypoints1 should be IDENTICAL
corr = matcher.match(pair.target, pair.target)
print(f"self-match: {len(corr)} pairs")
print(f"a_xy[0:5]={corr.a_xy[:5]}")
print(f"b_xy[0:5]={corr.b_xy[:5]}")
print(f"max coord in a: {corr.a_xy.max(axis=0) if len(corr) else None}  (target shape={pair.target.shape})")
print(f"a-b diff mean: {np.linalg.norm(corr.a_xy - corr.b_xy, axis=1).mean() if len(corr) else float('nan'):.3f}")

# Match a known crop where we know correspondences exactly
print()
print("Match target[0:128, 0:128] vs target (the top-left quarter):")
sub = pair.target[:128, :128].copy()
corr = matcher.match(sub, pair.target)
print(f"  {len(corr)} pairs")
if len(corr) > 0:
    # Expect: a_xy in [0,128]x[0,128], and b_xy ~= a_xy (since sub IS target's top-left quarter)
    print(f"  a_xy range: x[{corr.a_xy[:,0].min():.1f}, {corr.a_xy[:,0].max():.1f}] "
          f"y[{corr.a_xy[:,1].min():.1f}, {corr.a_xy[:,1].max():.1f}]")
    print(f"  b_xy range: x[{corr.b_xy[:,0].min():.1f}, {corr.b_xy[:,0].max():.1f}] "
          f"y[{corr.b_xy[:,1].min():.1f}, {corr.b_xy[:,1].max():.1f}]")
    # If convention is (x, y): b should be in [0,128]x[0,128] (matching the sub region)
    in_xy_quadrant = ((corr.b_xy[:, 0] < 130) & (corr.b_xy[:, 1] < 130)).mean()
    in_yx_quadrant = ((corr.b_xy[:, 1] < 130) & (corr.b_xy[:, 0] < 130)).mean()
    print(f"  fraction of b in (x<130, y<130) quadrant: {in_xy_quadrant:.2f}")
    # Show first 5 pairs
    print(f"  first 5: a -> b:")
    for i in range(min(5, len(corr))):
        print(f"    ({corr.a_xy[i,0]:6.2f}, {corr.a_xy[i,1]:6.2f}) -> "
              f"({corr.b_xy[i,0]:6.2f}, {corr.b_xy[i,1]:6.2f})  conf={corr.confidence[i]:.3f}")
