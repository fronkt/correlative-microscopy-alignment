"""ANHIR cross-stain histology loader (appearance-axis domain #1).

ANHIR (Borovec et al., IEEE TMI 2020) registers consecutive histology slices
stained differently (H&E vs immunostains). Same geometry, near-total appearance
divergence — a clean appearance-axis testbed. GT is manual landmark
correspondences (one CSV per image, paired row-by-row).

Release layout (grand-challenge.org)::

    <root>/
        dataset_medium.csv          # the pair index (configurable)
        <tissue>/<scale>/
            <name>.jpg / .png       # images
            <name>.csv              # landmarks: header ",X,Y" then idx,x,y rows

The dataset CSV columns (case-insensitive): ``Source image``, ``Target image``,
``Source landmarks``, ``Target landmarks``, ``status`` (training/evaluation).
Paths are relative to ``root``. ``status`` lets us hold out the evaluation split.

Appearance, not scale: ``scale_ratio`` is fixed to 1.0 — stains are imaged at
matched resolution, so the shift this domain exercises is purely appearance.
"""

from __future__ import annotations

import contextlib
import csv
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cma.data.amalgamatch import _read_image_float
from cma.data.types import ImagePair, KeypointSet


@dataclass(frozen=True)
class AnhirRecord:
    pair_id: str
    tissue: str
    status: str  # "training" | "evaluation" | "" if absent
    source_path: Path
    target_path: Path
    source_landmarks: Path
    target_landmarks: Path


def _read_landmarks(path: Path) -> np.ndarray:
    """(N, 2) float landmark coords from an ANHIR ',X,Y' CSV."""
    xs: list[tuple[float, float]] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        # tolerate a missing/garbled header by detecting non-numeric first row
        if header is not None and len(header) >= 3:
            # if the first row is numeric it was data, not a header
            with contextlib.suppress(ValueError):
                xs.append((float(header[1]), float(header[2])))
        for row in reader:
            if len(row) < 3:
                continue
            xs.append((float(row[1]), float(row[2])))
    return np.asarray(xs, dtype=np.float64).reshape(-1, 2)


def _find(colnames: list[str], want: str) -> int:
    low = [c.strip().lower() for c in colnames]
    return low.index(want)


class AnhirLoader:
    """Iterate ANHIR pairs with landmark GT, matching the cma loader interface."""

    def __init__(self, root: str | Path, dataset_csv: str = "dataset_medium.csv",
                 status: str | None = None) -> None:
        self.root = Path(root)
        csv_path = self.root / dataset_csv
        if not csv_path.is_file():
            raise FileNotFoundError(
                f"ANHIR dataset CSV not found at {csv_path}. "
                f"Expected layout documented in cma/data/anhir.py."
            )
        self._records: list[AnhirRecord] = []
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            cols = next(reader)
            i_si = _find(cols, "source image")
            i_ti = _find(cols, "target image")
            i_sl = _find(cols, "source landmarks")
            i_tl = _find(cols, "target landmarks")
            i_st = _find(cols, "status") if "status" in [c.strip().lower() for c in cols] else None
            for idx, row in enumerate(reader):
                if not row or len(row) <= max(i_si, i_ti, i_sl, i_tl):
                    continue
                st = row[i_st].strip() if i_st is not None else ""
                if status is not None and st.lower() != status.lower():
                    continue
                src_img = self.root / row[i_si].strip()
                tissue = Path(row[i_si].strip()).parts[0] if row[i_si].strip() else "unknown"
                self._records.append(AnhirRecord(
                    pair_id=f"anhir#{idx}",
                    tissue=tissue,
                    status=st,
                    source_path=src_img,
                    target_path=self.root / row[i_ti].strip(),
                    source_landmarks=self.root / row[i_sl].strip(),
                    target_landmarks=self.root / row[i_tl].strip(),
                ))
        if not self._records:
            raise ValueError(f"no pairs parsed from {csv_path} (status={status!r})")

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> list[AnhirRecord]:
        return list(self._records)

    def load_pair(self, rec: AnhirRecord) -> ImagePair:
        src = _read_image_float(rec.source_path)
        tgt = _read_image_float(rec.target_path)
        src_xy = _read_landmarks(rec.source_landmarks)
        tgt_xy = _read_landmarks(rec.target_landmarks)
        n = min(len(src_xy), len(tgt_xy))  # paired by row; guard ragged files
        gt = KeypointSet(src_xy=src_xy[:n], tgt_xy=tgt_xy[:n])
        return ImagePair(
            source=src, target=tgt, scale_ratio=1.0, gt=gt,
            metadata={"pair_id": rec.pair_id, "tissue": rec.tissue,
                      "status": rec.status, "axis": "appearance"},
        )

    def __iter__(self) -> Iterator[tuple[ImagePair, AnhirRecord]]:
        for rec in self._records:
            yield self.load_pair(rec), rec
