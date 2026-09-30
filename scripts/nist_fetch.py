"""Download selected per-section files of the NIST AM Bench 2022 serial-sectioning records.

PUBLIC files only, plain HTTPS (data.nist.gov /od/ds/ links -> nist-oar-cache S3 redirect).
Reads the locally cached RMM JSON (rmm_2767.json / rmm_2765.json) to resolve file paths.

Example:
  python scripts/nist_fetch.py --alloy 718 --sections 100 250 --kinds bse1_raw bse1_reg
Kinds: bse1_raw bse1_reg bse2_raw bse2_reg om_raw om_reg ebsd_raw ebsd_reg
Prints total bytes; refuses to exceed --max-gb (default 2).
"""
import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request

ROOT = r"C:\Users\frank\Documents\materials-bench\nist"

# regexes on the component filepath; {s} = section number (unpadded), {s3} = 3-digit padded
KIND_PATTERNS = {
    "718": {
        "bse1_raw": r"^Section Data/BSE_1 Sections/LEROY_0118_BSE_1_{s3}\.tif$",
        "bse1_reg": r"BSE_1_RegisteredtoOptical/LEROY_0118_BSE_1_{s3}warped\.tif$",
        "bse2_raw": r"^Section Data/BSE_2 Sections/LEROY_0118_BSE_2_{s3}\.tif$",
        "bse2_reg": r"BSE_2_Resampled_RegisteredtoBSE1andOptical/LEROY_0118_BSE_2_{s3}warped\.tif$",
        "om_raw": r"^Section Data/Optical Sections/1{s3}\.tif$",
        "om_reg": r"OM_7_Aligned_Cropped/OM_7_Aligned_Cropped_1{s3}\.tif$",
        "ebsd_raw": r"^Section Data/EBSD_Indexed Sections/LEROY_0118_EBSD_1_Section_{s}_DI_REFINED_masked\.ctf$",
        "ebsd_reg": r"EBSD_CTF_RegisteredtoALL/LEROY_0118_EBSD_1_Section_{s}_DI_REFINED_masked_warped\.ctf$",
    },
    "625": {
        "bse1_raw": r"Section Data/BSE_1 Sections/LEROY_0113_BSE_1_{s3}\.tif$",
        "bse1_reg": r"BSE_1_Warped_and_Registered/0{s3}warped\.tif$",
        "bse2_raw": r"Section Data/BSE_2 Sections/LEROY_0113_BSE_2_{s3}\.tif$",
        "bse2_reg": r"BSE_2_Warped_and_Registered/LEROY_0113_BSE_2_0{s3}warped\.tif$",
        "om_raw": r"Section Data/Optical Sections/1{s3}\.tif$",
        "om_reg": r"Optical_Aligned/[^/]*1{s3}\.tif$",
        "ebsd_raw": r"Section Data/EBSD_Indexed Sections/LEROY_0113_EBSD_1_Section_{s}_DI_REFINED_masked\.ctf$",
        "ebsd_reg": r"EBSD_Warped_and_Registered/LEROY_0113_EBSD_1_Section_{s}_DI_REFINED_masked_warped\.ctf$",
    },
}
RMM = {"718": "rmm_2767.json", "625": "rmm_2765.json"}


def components(alloy):
    d = json.load(open(os.path.join(ROOT, RMM[alloy]), encoding="utf-8"))["ResultData"][0]
    return [c for c in d["components"] if c.get("size")]


def resolve(alloy, kind, s, comps):
    pat = re.compile(KIND_PATTERNS[alloy][kind].format(s=s, s3=f"{s:03d}"))
    hits = [c for c in comps if pat.search(c["filepath"])]
    return hits


def fetch(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "python-urllib nist-public-download"})
    with urllib.request.urlopen(req, timeout=300) as r, open(dest + ".part", "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
    os.replace(dest + ".part", dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alloy", choices=["718", "625"], required=True)
    ap.add_argument("--sections", type=int, nargs="+", required=True)
    ap.add_argument("--kinds", nargs="+", required=True, choices=list(KIND_PATTERNS["718"]))
    ap.add_argument("--max-gb", type=float, default=2.0)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    comps = components(a.alloy)
    todo, total = [], 0
    for s in a.sections:
        for k in a.kinds:
            h = resolve(a.alloy, k, s, comps)
            if len(h) != 1:
                print(f"WARN {a.alloy} s{s} {k}: {len(h)} matches", file=sys.stderr)
                continue
            c = h[0]
            todo.append((s, k, c))
            total += int(c["size"])
    print(f"{len(todo)} files, {total/1e6:.1f} MB")
    if total > a.max_gb * 1e9:
        sys.exit("over budget; aborting")
    for s, k, c in todo:
        url = c["downloadURL"]
        if " " in url:
            url = urllib.parse.quote(url, safe=":/%")
        dest = os.path.join(ROOT, a.alloy, f"s{s:03d}", f"{k}__" + os.path.basename(c["filepath"]))
        print(k, s, int(c["size"]) // 1000000, "MB ->", dest)
        if a.dry or (os.path.exists(dest) and os.path.getsize(dest) == int(c["size"])):
            continue
        for attempt in range(200):  # NIST gateway returns 5xx (e.g. 524) under load: retry with backoff
            try:
                fetch(url, dest)
                break
            except Exception as e:  # noqa: BLE001
                print("  retry", attempt, repr(e)[:80], flush=True)
                import time

                time.sleep(min(300, 30 * (attempt + 1)))
        else:
            print("GIVE UP", dest, flush=True)


if __name__ == "__main__":
    main()
