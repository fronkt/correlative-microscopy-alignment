"""Scene-level train/val/test split (70/15/15 by PAIR count).

Pairs within a scene share images, so the split unit is the scene
(pair_id prefix before '#'). True stratification is impossible: 13 of 19
subclasses have a single scene, and two single-scene TEM subsets hold 27
and 32 pairs. Strategy: deterministic greedy — walk scenes in descending
pair count and assign each to the split whose pair-count deficit (vs the
70/15/15 target) is largest, with one constraint: every task group with
>= 3 scenes must appear in val and in test.

Writes results/split.json. Run once; the committed file is canonical.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from cma.data.amalgamatch import AmalgaMatchLoader

TARGETS = {"train": 0.70, "val": 0.15, "test": 0.15}

loader = AmalgaMatchLoader("data/AmalgaMatch")
scene_pairs: dict[str, list[str]] = defaultdict(list)
scene_group: dict[str, str] = {}
for rec in loader.records:
    scene = rec.pair_id.split("#")[0]
    scene_pairs[scene].append(rec.pair_id)
    scene_group[scene] = rec.group

n_total = sum(len(v) for v in scene_pairs.values())
group_scene_count = defaultdict(int)
for s, g in scene_group.items():
    group_scene_count[g] += 1
must_cover = {g for g, c in group_scene_count.items() if c >= 3}

assign: dict[str, str] = {}
counts = {k: 0 for k in TARGETS}
covered = {k: set() for k in TARGETS}

# big scenes first so they can be balanced around
ordered = sorted(scene_pairs, key=lambda s: -len(scene_pairs[s]))
for scene in ordered:
    n = len(scene_pairs[scene])
    g = scene_group[scene]

    def deficit(split: str) -> float:
        return TARGETS[split] * n_total - counts[split]

    # forced coverage: if a must-cover group is running out of scenes and
    # val/test still lack it, send this scene there
    remaining_g = [s for s in ordered if scene_group[s] == g and s not in assign]
    forced = None
    for split in ("test", "val"):
        if g in must_cover and g not in covered[split]:
            lacking = [sp for sp in ("test", "val") if g not in covered[sp]]
            if len(remaining_g) <= len(lacking):
                forced = split
                break
    split = forced or max(TARGETS, key=deficit)
    assign[scene] = split
    counts[split] += n
    covered[split].add(g)

out = {
    sp: sorted(pid for s, a in assign.items() if a == sp
               for pid in scene_pairs[s])
    for sp in TARGETS
}
out["scenes"] = {sp: sorted(s for s, a in assign.items() if a == sp)
                 for sp in TARGETS}
Path("results/split.json").write_text(json.dumps(out, indent=1))

print({sp: len(out[sp]) for sp in TARGETS}, "of", n_total, "pairs")
for sp in TARGETS:
    groups = sorted({scene_group[s] for s in out["scenes"][sp]})
    print(f"{sp}: {len(out['scenes'][sp])} scenes, groups: {groups}")
