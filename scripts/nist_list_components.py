"""Summarise the RMM component list of the NIST AM Bench 2022 serial-section records.

Usage: python scripts/nist_list_components.py rmm_2767.json [--grep PATTERN]
Only reads a locally downloaded RMM JSON; no network.
"""
import argparse
import collections
import json
import re


def load_components(path):
    d = json.load(open(path, encoding="utf-8"))
    rec = d["ResultData"][0] if "ResultData" in d else d
    return rec, rec.get("components", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path")
    ap.add_argument("--grep", default=None)
    a = ap.parse_args()
    rec, comps = load_components(a.json_path)
    print(rec.get("title"), len(comps), "components")
    if a.grep:
        for c in comps:
            fp = c.get("filepath", c.get("@id", ""))
            if re.search(a.grep, fp):
                print(c.get("size"), fp, c.get("downloadURL"))
        return
    cnt = collections.Counter()
    size = collections.Counter()
    for c in comps:
        fp = c.get("filepath", "")
        k = "/".join(fp.split("/")[:-1])
        cnt[k] += 1
        size[k] += int(c.get("size", 0) or 0)
    for k, v in sorted(cnt.items()):
        print(f"{v:6d} {size[k]/1e9:9.2f} GB  {k}")


if __name__ == "__main__":
    main()
