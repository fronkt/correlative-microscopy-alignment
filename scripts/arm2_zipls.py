"""List Zenodo record files and remote-zip members (HTTP range reads only). Usage:
  python scripts/arm2_zipls.py REC            # list record files
  python scripts/arm2_zipls.py REC FILE [substr]   # list members of zip FILE (filter by substring)
"""
from __future__ import annotations

import sys

import requests
from remotezip import RemoteZip


def files(rec: str) -> list[dict]:
    r = requests.get(f"https://zenodo.org/api/records/{rec}", timeout=60)
    r.raise_for_status()
    j = r.json()
    return [dict(key=f["key"], size=f["size"], url=f["links"]["self"]) for f in j["files"]], j


def main() -> None:
    rec = sys.argv[1]
    fl, j = files(rec)
    if len(sys.argv) == 2:
        print(j["metadata"]["title"], j["metadata"].get("license"))
        for f in fl:
            print(f["size"], f["key"])
        return
    name = sys.argv[2]
    sub = sys.argv[3].lower() if len(sys.argv) > 3 else ""
    f = next(x for x in fl if x["key"] == name)
    with RemoteZip(f["url"]) as z:
        for i in z.infolist():
            if sub in i.filename.lower():
                print(i.file_size, i.filename)


if __name__ == "__main__":
    main()
