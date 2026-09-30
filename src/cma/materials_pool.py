"""Arm-2 materials pool loader (independent real-acquisition, hand-GT correlative pairs).

Same interface as ``cma.data.AmalgaMatchLoader`` (``iter()``, ``load_pair``, ``records``, ``__len__``) so it
plugs into scripts/run_triage_candidates.py (see scripts/arm2_run_candidates.py for a 3-line wrapper).

Layout (created by scripts/arm2_assemble.py; default root C:/Users/frank/Documents/materials-bench)::

    <root>/manifest.csv                 one row per pair (paths, sizes, pixel sizes, licence, sha256, ...)
    <root>/pairs/<pair>/source.png      wide-FOV image (8/16-bit PNG)
    <root>/pairs/<pair>/target.png      narrow-FOV image
    <root>/gt_points/<pair>.csv         src_x,src_y,tgt_x,tgt_y   (pixel-index coordinates, 0-based centres)

Conventions identical to AmalgaMatch: ``pair_id`` = ``"<scene>#<k>"`` (k always 0 here, one pair per scene),
``source`` = larger physical FOV (swap recorded in ``flipped``), GT ``src_xy`` in source pixels, ``tgt_xy`` in
target pixels, and the runner's H maps target -> source, ``mu_ed`` = mean |H(tgt_xy) - src_xy| in SOURCE pixels.
``scale_ratio`` = target_pixel / source_pixel; where pixel sizes are unknown (Component C) it is taken from the
GT affine (sqrt|det| of the least-squares target->source affine).
``group`` = task family, ``subclass`` = specimen/modality label, ``cluster`` = independence unit for
specimen-clustered statistics (records carry it; the runner ignores it).
"""
from __future__ import annotations

import csv
import math
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cma.data.amalgamatch import _read_image_float
from cma.data.types import ImagePair, KeypointSet

DEFAULT_ROOT = Path(r"C:\Users\frank\Documents\materials-bench")
ROOT_ENV = "MATPOOL_ROOT"   # root override: the pool copy to load (e.g. /root/matpool on the box)
_ABS_RE = re.compile(r"^([A-Za-z]:)?/")


def resolve_path(root: Path, p: str, rebase_absolute: bool) -> Path:
    """Resolve a manifest path against ``root``.

    Relative paths (the packed manifest) are joined to ``root``. Absolute paths (the laptop manifest, Windows or
    POSIX) are used as written unless ``rebase_absolute`` (a root override is active), in which case everything from
    the last ``pairs`` / ``gt_points`` segment on is re-rooted under ``root``.
    """
    s = str(p).replace("\\", "/")
    if not _ABS_RE.match(s):
        return root / s
    if not rebase_absolute:
        return Path(p)
    parts = s.split("/")
    idx = max((i for i, seg in enumerate(parts) if seg in ("pairs", "gt_points")), default=None)
    if idx is None:
        raise ValueError(f"cannot re-root absolute manifest path {p!r} under {root}")
    return root.joinpath(*parts[idx:])


@dataclass(frozen=True)
class MaterialsPoolRecord:
    pair_id: str
    group: str
    subclass: str
    cluster: str
    component: str
    source_pixel_nm: float | None
    target_pixel_nm: float | None
    source_path: Path
    target_path: Path
    gt_path: Path
    flipped: bool
    licence: str
    pair_index: int = 0


def _f(v: str) -> float | None:
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _read_gt(path: Path) -> KeypointSet:
    a = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    return KeypointSet(src_xy=a[:, :2].copy(), tgt_xy=a[:, 2:4].copy())


def _gt_scale(gt: KeypointSet) -> float:
    A = np.hstack([gt.tgt_xy, np.ones((len(gt), 1))])
    X, *_ = np.linalg.lstsq(A, gt.src_xy, rcond=None)
    return float(math.sqrt(abs(np.linalg.det(X[:2].T))))


class MaterialsPoolLoader:
    """Iterate over (ImagePair, MaterialsPoolRecord) tuples; ``root`` defaults to the materials-bench folder."""

    def __init__(self, root: str | Path | None = None) -> None:
        override = os.environ.get(ROOT_ENV)
        self._rebase = bool(override)
        self.root = Path(override) if override else (Path(root) if root else DEFAULT_ROOT)
        man = self.root / "manifest.csv"
        if not man.is_file() and override:
            raise FileNotFoundError(f"{ROOT_ENV}={override} but {man} does not exist")
        if not man.is_file():
            # runner passes its AmalgaMatch default (data/AmalgaMatch); fall back to the pool folder
            man = DEFAULT_ROOT / "manifest.csv"
            self.root = DEFAULT_ROOT
        if not man.is_file():
            raise FileNotFoundError(f"materials pool manifest not found at {man}; run scripts/arm2_assemble.py")
        self._records: list[MaterialsPoolRecord] = []
        self._gt: dict[str, KeypointSet] = {}
        with open(man, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rec = MaterialsPoolRecord(
                    pair_id=r["pair_id"], group=r["group"], subclass=r["subclass"], cluster=r["cluster"],
                    component=r["component"], source_pixel_nm=_f(r["source_pixel_nm"]),
                    target_pixel_nm=_f(r["target_pixel_nm"]), source_path=resolve_path(self.root, r["source_path"], self._rebase),
                    target_path=resolve_path(self.root, r["target_path"], self._rebase),
                    gt_path=resolve_path(self.root, r["gt_path"], self._rebase),
                    flipped=r["flipped"] in ("True", "true", "1"), licence=r["licence"])
                self._records.append(rec)
                self._gt[rec.pair_id] = _read_gt(rec.gt_path)

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> list[MaterialsPoolRecord]:
        return list(self._records)

    def __iter__(self) -> Iterator[tuple[ImagePair, MaterialsPoolRecord]]:
        return self.iter()

    def load_pair(self, rec: MaterialsPoolRecord) -> ImagePair:
        return self._load_pair(rec)

    def iter(self, groups: list[str] | None = None, subclasses: list[str] | None = None,
             components: list[str] | None = None) -> Iterator[tuple[ImagePair, MaterialsPoolRecord]]:
        for rec in self._records:
            if groups and rec.group not in set(groups):
                continue
            if subclasses and rec.subclass not in set(subclasses):
                continue
            if components and rec.component not in set(components):
                continue
            yield self._load_pair(rec), rec

    def _load_pair(self, rec: MaterialsPoolRecord) -> ImagePair:
        gt = self._gt[rec.pair_id]
        if rec.source_pixel_nm and rec.target_pixel_nm:
            scale = rec.target_pixel_nm / rec.source_pixel_nm
        else:
            scale = _gt_scale(gt)
        return ImagePair(
            source=_read_image_float(rec.source_path), target=_read_image_float(rec.target_path),
            scale_ratio=scale, gt=gt,
            metadata={"pair_id": rec.pair_id, "group": rec.group, "subclass": rec.subclass, "cluster": rec.cluster,
                      "component": rec.component, "source_pixel_nm": rec.source_pixel_nm,
                      "target_pixel_nm": rec.target_pixel_nm, "flipped": rec.flipped, "licence": rec.licence})
