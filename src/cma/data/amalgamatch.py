"""AmalgaMatch dataset loader.

Real release layout (Fordatis DOI 10.24406/fordatis/436, CC-BY-4.0)::

    <root>/
        <SubsetName>/                          # 19 subsets, e.g. CoNi-AM67_OM-SEM_Multiscale
            image_metadata.json                # per-image pixel size / modality info
            eval_indexs/
                eval_<SubsetName>_<idx>.npz    # one per scene; pickled dict (NOT npz archive)
                val_list.txt                   # eval names in the validation split
            scenes/<SubsetName>_<idx>/
                <image files>                  # .tif/.tiff, modality encoded in filename

Each ``eval_*.npz`` is a pickled dict with keys:
    dataset_name    str
    image_paths     list[str], relative to the subset directory
    image_metadata  list[dict], parallel to image_paths
    pair_infos      list[([i, j], flag)], indices into image_paths
    gt_2D_matches   list[(N, 4) float64], columns [x_i, y_i, x_j, y_j]
                    following pair_infos index order

187 pairs total across 19 subsets. The loader orients each pair so that
`source` is the image with the larger physical FOV (pipeline convention:
register narrow-FOV `target` into wide-FOV `source`), swapping GT columns
when needed.

Windows note: several release filenames push absolute paths past the 260-char
MAX_PATH limit, so all file IO here goes through a ``\\\\?\\``-prefixed long
path. Images are read via ``cv2.imdecode`` on bytes for the same reason.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from cma.data.types import ImagePair, KeypointSet

# Task family (group) per subset, keyed by substring of the subset name.
# Order matters: "SameSliceSerialSectioning" must hit SerialSectioning
# before the SameSlice fallback. 6 groups over the 19 subsets.
_GROUP_KEYWORDS: tuple[tuple[str, str], ...] = (
    ("FractureSurfaces", "FractureSurfaces"),
    ("DislocationCharacterization", "DislocationCharacterization"),
    ("SlipPartitioning", "SlipPartitioning"),
    ("SerialSectioning", "SerialSectioning"),
    ("SameSlice", "SameSlice"),
    ("Multiscale", "Multiscale"),
)


@dataclass(frozen=True)
class AmalgaMatchRecord:
    pair_id: str  # "<eval_name>#<pair_index>"
    group: str  # task family, one of 6
    subclass: str  # subset directory name, one of 19
    source_pixel_nm: float
    target_pixel_nm: float
    source_path: Path
    target_path: Path
    eval_path: Path  # pickled-dict file holding GT for this pair
    pair_index: int  # row into pair_infos / gt_2D_matches
    flipped: bool  # True if dataset order was (narrow, wide) and we swapped


class AmalgaMatchLoader:
    """Iterate over (ImagePair, AmalgaMatchRecord) tuples from disk.

    Filtering: pass `groups` or `subclasses` to restrict the iteration.
    GT keypoints are tiny and cached at init; images are read on demand
    to keep memory bounded for the full 187-pair release.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"AmalgaMatch root not found at {self.root}. "
                f"Expected layout documented in cma/data/amalgamatch.py."
            )
        subset_dirs = sorted(
            p for p in self.root.iterdir() if p.is_dir() and (p / "eval_indexs").is_dir()
        )
        if not subset_dirs:
            raise FileNotFoundError(
                f"no subset directories with eval_indexs/ under {self.root}"
            )
        self._records: list[AmalgaMatchRecord] = []
        self._gt: dict[str, KeypointSet] = {}
        for subset_dir in subset_dirs:
            self._index_subset(subset_dir)

    def __len__(self) -> int:
        return len(self._records)

    @property
    def records(self) -> list[AmalgaMatchRecord]:
        return list(self._records)

    def __iter__(self) -> Iterator[tuple[ImagePair, AmalgaMatchRecord]]:
        return self.iter(groups=None, subclasses=None)

    def iter(
        self,
        groups: list[str] | None = None,
        subclasses: list[str] | None = None,
    ) -> Iterator[tuple[ImagePair, AmalgaMatchRecord]]:
        groups_set = set(groups) if groups else None
        subclasses_set = set(subclasses) if subclasses else None
        for rec in self._records:
            if groups_set and rec.group not in groups_set:
                continue
            if subclasses_set and rec.subclass not in subclasses_set:
                continue
            yield self._load_pair(rec), rec

    def _index_subset(self, subset_dir: Path) -> None:
        subclass = subset_dir.name
        group = _derive_group(subclass)
        for eval_path in sorted((subset_dir / "eval_indexs").glob("*.npz")):
            data = _load_eval(eval_path)
            image_paths = data["image_paths"]
            image_meta = data["image_metadata"]
            eval_name = eval_path.stem
            for pair_index, ((i, j), _flag) in enumerate(data["pair_infos"]):
                gt = np.asarray(data["gt_2D_matches"][pair_index], dtype=np.float64)
                path_i = subset_dir / image_paths[i]
                path_j = subset_dir / image_paths[j]
                px_i = float(image_meta[i]["Physical Pixel Size [m]"]) * 1e9
                px_j = float(image_meta[j]["Physical Pixel Size [m]"]) * 1e9
                fov_i = px_i * float(image_meta[i]["Resolution Width"])
                fov_j = px_j * float(image_meta[j]["Resolution Width"])
                flipped = fov_j > fov_i  # source must be the wide-FOV image
                if flipped:
                    path_i, path_j = path_j, path_i
                    px_i, px_j = px_j, px_i
                    gt = gt[:, [2, 3, 0, 1]]
                pair_id = f"{eval_name}#{pair_index}"
                self._records.append(
                    AmalgaMatchRecord(
                        pair_id=pair_id,
                        group=group,
                        subclass=subclass,
                        source_pixel_nm=px_i,
                        target_pixel_nm=px_j,
                        source_path=path_i,
                        target_path=path_j,
                        eval_path=eval_path,
                        pair_index=pair_index,
                        flipped=flipped,
                    )
                )
                self._gt[pair_id] = KeypointSet(
                    src_xy=gt[:, :2].copy(), tgt_xy=gt[:, 2:].copy()
                )

    def _load_pair(self, rec: AmalgaMatchRecord) -> ImagePair:
        I_s = _read_image_float(rec.source_path)
        I_t = _read_image_float(rec.target_path)
        scale_ratio = rec.target_pixel_nm / rec.source_pixel_nm
        return ImagePair(
            source=I_s,
            target=I_t,
            scale_ratio=scale_ratio,
            gt=self._gt[rec.pair_id],
            metadata={
                "pair_id": rec.pair_id,
                "group": rec.group,
                "subclass": rec.subclass,
                "source_pixel_nm": rec.source_pixel_nm,
                "target_pixel_nm": rec.target_pixel_nm,
                "flipped": rec.flipped,
            },
        )


def _derive_group(subset_name: str) -> str:
    for keyword, group in _GROUP_KEYWORDS:
        if keyword in subset_name:
            return group
    raise ValueError(f"cannot derive task group from subset name: {subset_name}")


def _long_path(path: Path) -> str:
    """Absolute path string safe past the Windows 260-char MAX_PATH limit."""
    s = os.path.abspath(path)
    if os.name == "nt" and len(s) > 240 and not s.startswith("\\\\?\\"):
        s = "\\\\?\\" + s
    return s


def _load_eval(eval_path: Path) -> dict:
    # Despite the .npz suffix these are plain pickle files; np.load falls
    # back to pickle and returns the dict directly.
    data = np.load(_long_path(eval_path), allow_pickle=True)
    if not isinstance(data, dict):
        raise ValueError(f"expected pickled dict in {eval_path}, got {type(data)}")
    required = {"image_paths", "image_metadata", "pair_infos", "gt_2D_matches"}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"{eval_path} missing keys: {sorted(missing)}")
    return data


def _read_image_float(path: Path) -> np.ndarray:
    with open(_long_path(path), "rb") as f:
        buf = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"could not decode image: {path}")
    if img.ndim == 3 and img.shape[2] in (3, 4):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if np.issubdtype(img.dtype, np.integer):
        img = img.astype(np.float32) / float(np.iinfo(img.dtype).max)
    else:
        img = img.astype(np.float32)
        lo, hi = float(img.min()), float(img.max())
        if hi > lo:
            img = (img - lo) / (hi - lo)
        else:
            img = np.zeros_like(img)
    return np.clip(img, 0.0, 1.0)


def load_metadata_json(subset_dir: Path) -> list[dict]:
    """Per-image metadata for a subset (same schema as the npz copies)."""
    with open(_long_path(subset_dir / "image_metadata.json"), encoding="utf-8") as f:
        return json.load(f)["images"]
