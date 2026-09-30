"""Fetch specific members of remote zips (HTTP range reads) or plain URLs into materials-bench.
  python scripts/arm2_fetch.py zip REC "FILE.zip" "member1" "member2" ... --out DIR
  python scripts/arm2_fetch.py url URL --out FILE
Only public Zenodo/Mendeley/refodat/DataCite HTTP. Skips files already present with the right size.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import requests
from remotezip import RemoteZip

BENCH = Path(r"C:\Users\frank\Documents\materials-bench")


def zenodo_url(rec: str, key: str) -> str:
    return f"https://zenodo.org/api/records/{rec}/files/{requests.utils.quote(key)}/content"


def fetch_members(rec: str, zipname: str, members: list[str], out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    got = []
    with RemoteZip(zenodo_url(rec, zipname)) as z:
        names = {i.filename: i for i in z.infolist()}
        for m in members:
            info = names[m]
            dest = out / Path(m).name
            if dest.exists() and dest.stat().st_size == info.file_size:
                got.append(dest)
                continue
            with z.open(info) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 20)
            print("fetched", dest, info.file_size, flush=True)
            got.append(dest)
    return got


def fetch_url(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        size = int(r.headers.get("content-length", -1))
        if dest.exists() and dest.stat().st_size == size:
            return dest
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    print("fetched", dest, dest.stat().st_size, flush=True)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind")
    ap.add_argument("args", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.kind == "zip":
        fetch_members(a.args[0], a.args[1], a.args[2:], Path(a.out))
    else:
        fetch_url(a.args[0], Path(a.out))


if __name__ == "__main__":
    sys.exit(main())
