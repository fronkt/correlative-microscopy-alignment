"""Count registration pairs across all AmalgaMatch subsets and sanity-check GT.

Usage: python scripts/count_amalgamatch_pairs.py [root]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/AmalgaMatch")
subsets = sorted(p for p in root.iterdir() if p.is_dir())

total = 0
for subset in subsets:
    sub_pairs = 0
    for npz_path in sorted((subset / "eval_indexs").glob("*.npz")):
        d = np.load(npz_path, allow_pickle=True)
        n = len(d["pair_infos"])
        assert n == len(d["gt_2D_matches"]), npz_path
        sub_pairs += n
        for info, gt in zip(d["pair_infos"], d["gt_2D_matches"]):
            idxs, flag = info
            assert gt.ndim == 2 and gt.shape[1] == 4, (npz_path, gt.shape)
            for i in idxs:
                p = subset / d["image_paths"][i]
                assert p.exists(), p
    gts = [
        gt.shape[0]
        for npz_path in sorted((subset / "eval_indexs").glob("*.npz"))
        for gt in np.load(npz_path, allow_pickle=True)["gt_2D_matches"]
    ]
    print(f"{subset.name}: {sub_pairs} pairs, GT pts/pair {min(gts)}-{max(gts)}")
    total += sub_pairs

print(f"\nTOTAL pairs: {total}")

# what does the second element of pair_info mean? collect distinct values
flags = set()
for subset in subsets:
    for npz_path in sorted((subset / "eval_indexs").glob("*.npz")):
        d = np.load(npz_path, allow_pickle=True)
        for info in d["pair_infos"]:
            flags.add(info[1])
print(f"distinct pair_info[1] values: {sorted(flags)}")

# GT coordinate convention check on one pair with known image sizes
d = np.load(
    next((subsets[0] / "eval_indexs").glob("*.npz")), allow_pickle=True
)
gt = d["gt_2D_matches"][0]
m0, m1 = d["image_metadata"]
print(f"\nimg0 {m0['Resolution Width']}x{m0['Resolution Height']}, "
      f"img1 {m1['Resolution Width']}x{m1['Resolution Height']}")
print("gt col ranges:", [(c, float(gt[:, j].min()), float(gt[:, j].max()))
                         for j, c in enumerate("abcd")])
