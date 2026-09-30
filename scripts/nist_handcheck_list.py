"""Pick the 20 NIST pairs for hand validation (stratified, fixed seed) and write the handcheck pair list.

Strata (pre-declared): IN718 BSE1-vs-OM 6, IN718 BSE2-vs-BSE1 5, IN718 BSE2-vs-OM 5, IN625 BSE2-vs-BSE1 4.
Within a stratum sections are split into terciles by depth; picks are spread over terciles (quota shuffled
by the seed for uneven counts) and the section inside a tercile is drawn at random.  Seed 20260930.
Images shown = the POOL images (materials-bench/pairs/*/{target,source}.png); left = target (smaller FOV),
right = source; gt_h = homography target->source in those PNG pixel coordinates (from the pair record).
"""
import glob
import json
import os
import random

SEED = 20260930
QUOTA = {("718", "bse1-om"): 6, ("718", "bse2-bse1"): 5, ("718", "bse2-om"): 5, ("625", "bse2-bse1"): 4}
REC = r"C:\Users\frank\Documents\materials-bench\records"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools", "handcheck", "nist_handcheck_pairs.json")


def main():
    rng = random.Random(SEED)
    recs = [json.load(open(p, encoding="utf-8")) for p in glob.glob(os.path.join(REC, "NIST_*.json"))]
    out = []
    for (alloy, mp), n in QUOTA.items():
        pool = sorted([r for r in recs if r["alloy"] == alloy and r["modality_pair"] == mp], key=lambda r: r["section"])
        assert len(pool) >= 12, (alloy, mp, len(pool))
        k = len(pool)
        terc = [pool[: k // 3], pool[k // 3: 2 * k // 3], pool[2 * k // 3:]]
        quota = [n // 3] * 3
        for i in rng.sample(range(3), n % 3):
            quota[i] += 1
        for t, q in zip(terc, quota):
            out += rng.sample(t, q)
    rows = [{"pair_id": r["pair_id"].replace("#0", ""), "left_path": r["target_path"], "right_path": r["source_path"],
             "gt_h": r["gt_homography_target_to_source"]} for r in out]
    rng.shuffle(rows)  # random review order so fatigue is not confounded with stratum
    json.dump(rows, open(OUT, "w"), indent=1)
    import csv
    with open(OUT.replace(".json", ".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "left_path", "right_path", "gt_h"])
        for r in rows:
            w.writerow([r["pair_id"], r["left_path"], r["right_path"], json.dumps([x for row in r["gt_h"] for x in row])])
    print(len(rows), "pairs ->", OUT)
    for r in rows:
        print(" ", r["pair_id"])


if __name__ == "__main__":
    main()
