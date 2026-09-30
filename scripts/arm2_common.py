"""Shared helpers for the arm-2 materials benchmark assembly (no registration algorithms)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

BENCH = Path(r"C:\Users\frank\Documents\materials-bench")
WT = Path(r"C:\Users\frank\Documents\cma-triage-ext")
PAIRS = BENCH / "pairs"
GT_DIR = BENCH / "gt_points"
RECORDS = BENCH / "records"  # one json per pair
THUMBS = WT / "results" / "arm2" / "thumbs"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def to_u8(a: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    a = a.astype(np.float64)
    fin = np.isfinite(a)
    lo, hi = np.percentile(a[fin], [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1
    return (np.clip((np.nan_to_num(a) - lo) / (hi - lo), 0, 1) * 255 + 0.5).astype(np.uint8)


def to_u16(a: np.ndarray, vmax: float) -> np.ndarray:
    return (np.clip(np.nan_to_num(a) / vmax, 0, 1) * 65535 + 0.5).astype(np.uint16)


def write_png(a: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", a)
    assert ok
    path.write_bytes(buf.tobytes())


def read_gray(path: Path) -> np.ndarray:
    buf = np.frombuffer(Path(path).read_bytes(), np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(path)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def write_gt(pair_id: str, src_xy: np.ndarray, tgt_xy: np.ndarray, extra: dict | None = None) -> Path:
    GT_DIR.mkdir(parents=True, exist_ok=True)
    p = GT_DIR / (pair_id.replace("#", "_") + ".csv")
    with open(p, "w", encoding="utf-8") as f:
        f.write("src_x,src_y,tgt_x,tgt_y\n")
        for (a, b), (c, d) in zip(src_xy, tgt_xy):
            f.write(f"{a:.4f},{b:.4f},{c:.4f},{d:.4f}\n")
    return p


def overlay_thumb(pair_id: str, src: np.ndarray, tgt: np.ndarray, src_xy: np.ndarray, tgt_xy: np.ndarray,
                  max_side: int = 640) -> Path:
    """Side-by-side thumbnail (source | target) with numbered GT points. For human inspection only."""
    panels = []
    for img, xy in ((src, src_xy), (tgt, tgt_xy)):
        im = img if img.dtype == np.uint8 else (img.astype(np.float64) / max(float(img.max()), 1) * 255).astype(np.uint8)
        s = max_side / max(im.shape[:2])
        im = cv2.resize(im, (max(1, int(im.shape[1] * s)), max(1, int(im.shape[0] * s))), interpolation=cv2.INTER_AREA)
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        for i, (x, y) in enumerate(xy):
            c = (int(x * s), int(y * s))
            cv2.circle(im, c, 4, (0, 0, 255), 1)
            cv2.putText(im, str(i), (c[0] + 4, c[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        panels.append(im)
    h = max(p.shape[0] for p in panels)
    panels = [cv2.copyMakeBorder(p, 0, h - p.shape[0], 0, 4, cv2.BORDER_CONSTANT, value=(40, 40, 40)) for p in panels]
    out = np.hstack(panels)
    THUMBS.mkdir(parents=True, exist_ok=True)
    p = THUMBS / (pair_id.replace("#", "_") + ".jpg")
    cv2.imwrite(str(p), out, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return p


def save_record(rec: dict) -> None:
    RECORDS.mkdir(parents=True, exist_ok=True)
    (RECORDS / (rec["pair_id"].replace("#", "_") + ".json")).write_text(json.dumps(rec, indent=1), encoding="utf-8")


def finalize_pair(*, pair_id: str, component: str, cluster: str, group: str, subclass: str,
                  img_a: np.ndarray, img_b: np.ndarray, xy_a: np.ndarray, xy_b: np.ndarray,
                  name_a: str, name_b: str, px_a_nm: float | None, px_b_nm: float | None,
                  mod_a: str, mod_b: str, licence: str, source_url: str, files: dict, notes: str,
                  fov_a: float | None = None, fov_b: float | None = None) -> dict:
    """Orient the pair so `source` is the larger physical-FOV image (AmalgaMatch convention), write images,
    GT csv, record. fov_* in any common unit; if unknown falls back to (a = source)."""
    flipped = fov_a is not None and fov_b is not None and fov_b > fov_a
    if flipped:
        img_s, img_t, xy_s, xy_t = img_b, img_a, xy_b, xy_a
        name_s, name_t, px_s, px_t, mod_s, mod_t = name_b, name_a, px_b_nm, px_a_nm, mod_b, mod_a
    else:
        img_s, img_t, xy_s, xy_t = img_a, img_b, xy_a, xy_b
        name_s, name_t, px_s, px_t, mod_s, mod_t = name_a, name_b, px_a_nm, px_b_nm, mod_a, mod_b
    d = PAIRS / pair_id.replace("#", "_")
    ps, pt = d / "source.png", d / "target.png"
    write_png(img_s, ps)
    write_png(img_t, pt)
    gt = write_gt(pair_id, xy_s, xy_t)
    thumb = overlay_thumb(pair_id, img_s if img_s.dtype == np.uint8 else (img_s // 257).astype(np.uint8),
                          img_t if img_t.dtype == np.uint8 else (img_t // 257).astype(np.uint8), xy_s, xy_t)
    rec = dict(pair_id=pair_id, component=component, cluster=cluster, group=group, subclass=subclass,
               source_path=str(ps), target_path=str(pt), gt_path=str(gt),
               source_modality=mod_s, target_modality=mod_t, source_name=name_s, target_name=name_t,
               h_s=int(img_s.shape[0]), w_s=int(img_s.shape[1]), h_t=int(img_t.shape[0]), w_t=int(img_t.shape[1]),
               source_pixel_nm=px_s, target_pixel_nm=px_t, flipped=bool(flipped), n_gt=int(len(xy_s)),
               licence=licence, source_url=source_url, files=files, notes=notes,
               sha256_source=sha256(ps), sha256_target=sha256(pt), thumb=str(thumb))
    save_record(rec)
    return rec
