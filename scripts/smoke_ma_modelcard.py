"""Run MatchAnything on the model-card's exact example pair (US Capitol).

If this returns hundreds of matches with reasonable scores, the wrapper is
correct and the bad numbers on our synthetic pairs are about the data, not
the wrapper.
"""

import io
import urllib.request

import numpy as np
from PIL import Image

from cma.matchers import MatchAnythingMatcher

URLS = [
    "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_98169888_3347710852.jpg",
    "https://raw.githubusercontent.com/magicleap/SuperGluePretrainedNetwork/refs/heads/master/assets/phototourism_sample_images/united_states_capitol_26757027_6717084061.jpg",
]


def _fetch_image(url: str) -> np.ndarray:
    with urllib.request.urlopen(url) as r:
        data = r.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    arr = np.asarray(img).astype(np.float32) / 255.0
    return arr


def main() -> None:
    print("downloading model-card example pair ...")
    imgs = [_fetch_image(u) for u in URLS]
    print(f"  img0={imgs[0].shape}  img1={imgs[1].shape}")

    print("\nloading MatchAnything ...")
    matcher = MatchAnythingMatcher()

    print("running matcher.match ...")
    corr = matcher.match(imgs[0], imgs[1])
    print(f"  {len(corr)} matches")
    if len(corr):
        print(f"  confidence percentiles: 5={np.percentile(corr.confidence, 5):.3f}  "
              f"50={np.percentile(corr.confidence, 50):.3f}  "
              f"95={np.percentile(corr.confidence, 95):.3f}")
        h_a, w_a = imgs[0].shape[:2]
        h_b, w_b = imgs[1].shape[:2]
        print(f"  a_xy range: x[{corr.a_xy[:,0].min():.1f}, {corr.a_xy[:,0].max():.1f}]  "
              f"y[{corr.a_xy[:,1].min():.1f}, {corr.a_xy[:,1].max():.1f}]  (a={h_a}x{w_a})")
        print(f"  b_xy range: x[{corr.b_xy[:,0].min():.1f}, {corr.b_xy[:,0].max():.1f}]  "
              f"y[{corr.b_xy[:,1].min():.1f}, {corr.b_xy[:,1].max():.1f}]  (b={h_b}x{w_b})")


if __name__ == "__main__":
    main()
