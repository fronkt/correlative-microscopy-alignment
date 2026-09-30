"""Component C: enumerate + download the Li & Shaffer correlative example from the Mendeley public API
(dataset srscfwnrwt v1, CC BY 4.0). Downloads only the 'Correlative analysis example - 10kX/slicebyslice'
files (23 OM + 23 SEM images + 23 BigWarp landmark CSVs)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm2_fetch import BENCH  # noqa: E402

API = "https://data.mendeley.com/public-api/datasets/srscfwnrwt"
OUT = BENCH / "raw" / "mendeley"


def get(url: str):
    # Mendeley 403s python-requests (TLS fingerprint) but serves plain curl.
    out = subprocess.run(["curl", "-sfL", url], capture_output=True, check=True).stdout
    return json.loads(out.decode("utf-8"))


def curl_download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    subprocess.run(["curl", "-sfL", "-o", str(dest), url], check=True)
    print("fetched", dest, dest.stat().st_size, flush=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = get(f"{API}?version=1")
    (OUT / "dataset.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    folders = get(f"{API}/folders/1?version=1")
    byid = {f["id"]: f for f in folders}

    def path_of(fid: str) -> str:
        parts = []
        while fid in byid:
            parts.append(byid[fid]["name"])
            fid = byid[fid].get("parent_id")
        return "/".join(reversed(parts))

    listing = []
    for f in folders:
        files = get(f"{API}/files?folder_id={f['id']}&version=1")
        for x in files:
            listing.append(dict(folder=path_of(f["id"]), name=x["filename"], size=x["content_details"]["size"],
                                url=x["content_details"]["download_url"], sha256=x["content_details"].get("sha256_hash"),
                                id=x["id"]))
    (OUT / "listing.json").write_text(json.dumps(listing, indent=1), encoding="utf-8")
    want = [x for x in listing if "Correlative analysis example - 10kX/slicebyslice" in x["folder"]]
    print(len(listing), "files total;", len(want), "in slicebyslice;",
          sum(x["size"] for x in want) / 1e6, "MB")
    for x in want:
        curl_download(x["url"], OUT / "slicebyslice" / x["name"])


if __name__ == "__main__":
    main()
