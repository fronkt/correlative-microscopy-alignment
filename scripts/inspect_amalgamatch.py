"""Inspect the real AmalgaMatch release: NPZ eval files, metadata, pair counts.

The eval_*.npz files are actually pickled dicts (np.save of a dict object),
so np.load(..., allow_pickle=True) returns a dict directly.

Usage: python scripts/inspect_amalgamatch.py [root]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


def describe(obj, indent=2, depth=0):
    pad = " " * indent * depth
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, np.ndarray):
                print(f"{pad}{k}: ndarray shape={v.shape} dtype={v.dtype}")
                if v.size <= 12 or v.dtype == object:
                    print(f"{pad}  -> {v.ravel()[:6]}")
            elif isinstance(v, (dict, list)):
                print(f"{pad}{k}: {type(v).__name__} len={len(v)}")
                if depth < 3:
                    describe(v, indent, depth + 1)
            else:
                print(f"{pad}{k}: {type(v).__name__} = {v}")
    elif isinstance(obj, list):
        for i, item in enumerate(obj[:3]):
            print(f"{pad}[{i}]: {type(item).__name__}")
            if isinstance(item, (dict, list)) and depth < 3:
                describe(item, indent, depth + 1)
            elif isinstance(item, np.ndarray):
                print(f"{pad}  ndarray shape={item.shape} dtype={item.dtype}")
            else:
                print(f"{pad}  -> {item}")
        if len(obj) > 3:
            print(f"{pad}... ({len(obj)} total)")


root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/AmalgaMatch")
subsets = sorted(p for p in root.iterdir() if p.is_dir())
print(f"{len(subsets)} subsets under {root}\n")

first_npz = sorted((subsets[0] / "eval_indexs").glob("*.npz"))[0]
print(f"=== {first_npz.relative_to(root)} ===")
data = np.load(first_npz, allow_pickle=True)
print(f"top-level type: {type(data).__name__}")
if isinstance(data, np.ndarray):
    print(f"ndarray shape={data.shape} dtype={data.dtype}")
    if data.dtype == object and data.size == 1:
        data = data.item()
        print(f"unwrapped -> {type(data).__name__}")
describe(data)
