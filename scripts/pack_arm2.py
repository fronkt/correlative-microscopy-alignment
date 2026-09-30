"""Pack the Arm 2 materials pool into ONE tar.gz for the GPU box (the pool exists only on this laptop).

  PYTHONPATH=src python scripts/pack_arm2.py [--root C:/Users/frank/Documents/materials-bench] [--out <tar.gz>]

Contents: exactly the manifest's pairs (pairs/<pair>/source.png, target.png, gt_points/<pair>.csv) and a manifest.csv
whose three path columns are RELATIVE (pairs/..., gt_points/...). Extract under /root/matpool and run with
MATPOOL_ROOT=/root/matpool (cma.materials_pool resolves relative paths against the root).

Steps: (1) integrity: every image's sha256 must equal the manifest's sha256_source / sha256_target; (2) deterministic
tar.gz (sorted, mtime 0, no owner) so the sha256 is reproducible for the same pool; (3) sidecar <out>.meta with
SIZE / SHA256 / PAIRS for box_arm2.sh; (4) round trip: extract to a temp dir and load every pair through
MaterialsPoolLoader with MATPOOL_ROOT pointing there (images and GT read, no matcher is run).
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import os
import sys
import tarfile
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cma.materials_pool import DEFAULT_ROOT, ROOT_ENV, resolve_path  # noqa: E402

PATH_COLS = ("source_path", "target_path", "gt_path")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def rel_name(root: Path, p: str) -> str:
    """Archive-relative POSIX name of a manifest path: pairs/<pair>/x.png or gt_points/<pair>.csv."""
    return resolve_path(root, p, rebase_absolute=True).relative_to(root).as_posix()


def build_manifest_text(rows: list[dict], fieldnames: list[str], root: Path) -> str:
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=fieldnames, lineterminator="\n")
    w.writeheader()
    for r in rows:
        r = dict(r)
        for c in PATH_COLS:
            r[c] = rel_name(root, r[c])
        w.writerow(r)
    return out.getvalue()


def pack(root: Path, out: Path) -> dict:
    man = root / "manifest.csv"
    with open(man, newline="", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        fields, rows = list(rd.fieldnames or []), list(rd)
    members: list[tuple[str, Path]] = []
    for r in rows:
        for c in PATH_COLS:
            src = resolve_path(root, r[c], rebase_absolute=True)
            if not src.is_file():
                raise FileNotFoundError(f"{r['pair_id']}: {src}")
            members.append((rel_name(root, r[c]), src))
        for col, name in (("sha256_source", "source_path"), ("sha256_target", "target_path")):
            if r.get(col) and sha256_file(resolve_path(root, r[name], True)) != r[col]:
                raise ValueError(f"{r['pair_id']}: {name} sha256 differs from the manifest")
    members.sort()
    names = [n for n, _ in members]
    assert len(names) == len(set(names)), "duplicate archive member"
    out.parent.mkdir(parents=True, exist_ok=True)
    mtext = build_manifest_text(rows, fields, root).encode("utf-8")
    with open(out, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, compresslevel=6) as gz, \
            tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
        ti = tarfile.TarInfo("manifest.csv")
        ti.size, ti.mtime, ti.mode = len(mtext), 0, 0o644
        tar.addfile(ti, io.BytesIO(mtext))
        for arc, src in members:
            ti = tar.gettarinfo(str(src), arcname=arc)
            ti.mtime, ti.uid, ti.gid, ti.uname, ti.gname, ti.mode = 0, 0, 0, "", "", 0o644
            with open(src, "rb") as f:
                tar.addfile(ti, f)
    size, digest = out.stat().st_size, sha256_file(out)
    meta = out.with_name(out.name + ".meta")
    meta.write_text(f"SIZE={size}\nSHA256={digest}\nPAIRS={len(rows)}\n", encoding="utf-8")
    return dict(size=size, sha256=digest, pairs=len(rows), members=len(members) + 1, meta=str(meta))


def verify_roundtrip(tar_path: Path, expect_pairs: int) -> int:
    """Extract to a temp dir and load every pair through the loader (MATPOOL_ROOT override). Returns pairs loaded."""
    from cma.materials_pool import MaterialsPoolLoader
    with tempfile.TemporaryDirectory() as td:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(td, filter="data") if hasattr(tarfile, "data_filter") else tar.extractall(td)
        old = os.environ.get(ROOT_ENV)
        os.environ[ROOT_ENV] = td
        try:
            loader = MaterialsPoolLoader()
            assert len(loader) == expect_pairs, (len(loader), expect_pairs)
            n = 0
            for pair, rec in loader:
                assert str(rec.source_path).startswith(td) and str(rec.gt_path).startswith(td), rec
                assert pair.source.ndim >= 2 and pair.target.ndim >= 2 and len(pair.gt) >= 4, rec.pair_id
                n += 1
        finally:
            if old is None:
                os.environ.pop(ROOT_ENV, None)
            else:
                os.environ[ROOT_ENV] = old
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT), help="pool folder holding manifest.csv, pairs/, gt_points/")
    ap.add_argument("--out", default="", help="default: <root>/matpool.tar.gz")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()
    root = Path(a.root)
    out = Path(a.out) if a.out else root / "matpool.tar.gz"
    info = pack(root, out)
    print(f"packed {info['pairs']} pairs, {info['members']} members -> {out}")
    print(f"size   {info['size']} bytes ({info['size'] / 1e6:.1f} MB)")
    print(f"sha256 {info['sha256']}")
    print(f"meta   {info['meta']}  (SIZE / SHA256 / PAIRS for box_arm2.sh)")
    if not a.no_verify:
        n = verify_roundtrip(out, info["pairs"])
        print(f"round trip OK: extracted copy loaded {n}/{info['pairs']} pairs via {ROOT_ENV}")


if __name__ == "__main__":
    main()
