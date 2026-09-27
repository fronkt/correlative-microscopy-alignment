"""Audit of the RoMa + pyramid v1 run: what was evaluated, what crashed, what tiled.

Found 2026-09-25 while choosing failure examples for the M&M rewrite; the submitted
manuscript (MAM-26-246) got all three of these wrong.

1. The 106 "hard failures" of RoMa + pyramid v1 are not estimator failures. Every one of
   the 106 failed rows carries the same `CUDA error: unknown error`. The run died on the
   82nd pair in loader order and each later pair inherited the dead CUDA context, so v1
   was only ever evaluated on the first 81 pairs.
2. The tile side is the target's shorter side in TARGET pixels. When the source image is
   smaller than that in either dimension, `cma.pyramid.build` reflect-pads it into a
   single square tile; the matcher then sees the specimen next to mirror images of
   itself. Those pairs test padding, not tiling.
3. So the pairs that actually test pool-then-fit tiling are the evaluated pairs with more
   than one tile.

The tile geometry needs no matcher: it depends only on image sizes and the scale ratio,
so it is recomputed here exactly as `register()` computes it.

Writes results/pyramid_v1_tiles.csv. Usage: python scripts/pyramid_v1_audit.py
"""

from __future__ import annotations

import csv
from pathlib import Path

from cma.data import AmalgaMatchLoader
from cma.pyramid import build

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "pyramid_v1_tiles.csv"
CRASH = "CUDA error"


def main() -> None:
    v1 = {}
    with (ROOT / "results" / "baselines_A.csv").open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["backbone"] == "roma" and r["mode"] == "pyramid":
                v1[r["pair_id"]] = r

    loader = AmalgaMatchLoader(ROOT / "data" / "AmalgaMatch")
    rows = []
    for rec in loader.records:
        pair = loader.load_pair(rec)
        tile = int(min(pair.target.shape[:2]))  # exactly as cma.pipeline.register
        tiles = build(pair.source, pair.scale_ratio, tile_size=tile, overlap=0.5)
        h_s, w_s = pair.source.shape[:2]
        r = v1[rec.pair_id]
        crashed = r["status"] != "ok" and CRASH in r["error"]
        if r["status"] == "ok":
            # register() pools one 10,000-match invocation per tile; the row's match
            # count must equal the recomputed tile count times the cap.
            assert int(r["n_matches"]) == 10_000 * len(tiles), rec.pair_id
        rows.append({
            "pair_id": rec.pair_id,
            "subclass": rec.subclass,
            "n_tiles": len(tiles),
            "padded": int(h_s < tile or w_s < tile),
            "source_wh": f"{w_s}x{h_s}",
            "tile_side": tile,
            "v1_status": "evaluated" if r["status"] == "ok" else ("crashed" if crashed else "failed"),
        })

    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    ev = [r for r in rows if r["v1_status"] == "evaluated"]
    cr = [r for r in rows if r["v1_status"] == "crashed"]
    other = [r for r in rows if r["v1_status"] == "failed"]
    tiled = [r for r in ev if r["n_tiles"] > 1]
    single = [r for r in ev if r["n_tiles"] == 1]
    print(f"wrote {len(rows)} rows to {OUT.relative_to(ROOT)}")
    print(f"evaluated {len(ev)}, crashed ({CRASH}) {len(cr)}, other failures {len(other)}")
    print(f"evaluated & tiled (>1 tile): {len(tiled)}; evaluated single tile: {len(single)} "
          f"(all padded: {all(r['padded'] for r in single)})")
    print(f"tile matches still owed on crashed pairs: {sum(r['n_tiles'] for r in cr)}")
    first_crash = min(i for i, r in enumerate(rows) if r["v1_status"] == "crashed")
    assert all(r["v1_status"] == "crashed" for r in rows[first_crash:]), "crash is not a contiguous tail"
    print(f"crash begins at pair {first_crash + 1} of {len(rows)} in loader order and never recovers")


if __name__ == "__main__":
    main()
