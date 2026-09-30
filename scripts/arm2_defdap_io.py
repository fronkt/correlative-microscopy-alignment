"""Minimal readers for DefDAP-style data (no DefDAP install): Oxford .cpr/.crc EBSD, DaVis 2D-vector DIC text.

No registration/matching algorithm is used anywhere here.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# Oxford Channel-5 binary field ids -> (name, numpy dtype)
_FIELDS = {3: ("euler1", "<f4"), 4: ("euler2", "<f4"), 5: ("euler3", "<f4"), 6: ("mad", "<f4"),
           7: ("bc", "u1"), 8: ("bs", "u1"), 10: ("bands", "u1"), 11: ("error", "u1"),
           12: ("f12", "<f4")}


def read_cpr(path: Path) -> dict:
    txt = Path(path).read_text(errors="ignore")
    g = lambda k: float(re.search(rf"^{k}=([\d.eE+-]+)", txt, re.M).group(1))  # noqa: E731
    job = txt.split("[Job]")[1].split("[")[0]
    xcells = int(re.search(r"xCells=(\d+)", job).group(1))
    ycells = int(re.search(r"yCells=(\d+)", job).group(1))
    step = float(re.search(r"GridDistX=([\d.eE+-]+)", job).group(1))
    fields_block = txt.split("[Fields]")[1].split("[")[0]
    ids = [int(v) for v in re.findall(r"Field\d+=(\d+)", fields_block)]
    return dict(xcells=xcells, ycells=ycells, step_um=step, fields=ids)


def read_crc(path_no_ext: Path) -> dict[str, np.ndarray]:
    """Return dict of (ycells, xcells) arrays incl. 'phase', 'bc', 'bs', 'mad'; plus meta."""
    p = Path(path_no_ext)
    cpr = read_cpr(Path(str(p) + ".cpr"))
    dt = [("phase", "u1")] + [_FIELDS[i] for i in cpr["fields"]]
    dtype = np.dtype(dt)
    raw = np.fromfile(Path(str(p) + ".crc"), dtype=dtype)
    n = cpr["xcells"] * cpr["ycells"]
    if raw.size != n:
        raise ValueError(f"{p}: crc has {raw.size} px, cpr says {n} (itemsize {dtype.itemsize})")
    out = {k: raw[k].reshape(cpr["ycells"], cpr["xcells"]) for k in raw.dtype.names}
    out["_meta"] = cpr
    return out


def read_dic(path: Path) -> dict:
    """DaVis 2D-vector export: header line '#...', rows 'x y u v' (pixels). Returns grid arrays (ny, nx)."""
    df = pd.read_csv(path, sep=r"\s+", comment="#", header=None, engine="c",
                     dtype=np.float64, na_values=["nan", "NaN", "-nan"])
    x = df[0].to_numpy()
    y = df[1].to_numpy()
    xs = np.unique(x)
    ys = np.unique(y)
    nx, ny = len(xs), len(ys)
    if nx * ny != len(df):
        raise ValueError(f"{path}: {nx}x{ny} != {len(df)} rows")
    # row order differs between exports (x-fastest ascending, or fully reversed): place by coordinate
    sx = float(xs[1] - xs[0])
    sy = float(ys[1] - ys[0])
    ix = np.rint((x - xs[0]) / sx).astype(int)
    iy = np.rint((y - ys[0]) / sy).astype(int)
    u = np.full((ny, nx), np.nan)
    v = np.full((ny, nx), np.nan)
    u[iy, ix] = df[2].to_numpy()
    v[iy, ix] = df[3].to_numpy()
    step = float(xs[1] - xs[0])
    return dict(u=u, v=v, nx=nx, ny=ny, step_px=step, x0=float(xs[0]), y0=float(ys[0]))


def max_shear(u: np.ndarray, v: np.ndarray, step: float) -> np.ndarray:
    """DefDAP-style effective (max) shear strain from grid displacements (small-strain form):
    e11=du/dx, e22=dv/dy, e12=(du/dy+dv/dx)/2, eMaxShear=sqrt(((e11-e22)/2)^2+e12^2). NaN -> 0."""
    u = np.nan_to_num(u)
    v = np.nan_to_num(v)
    du_dy, du_dx = np.gradient(u, step)
    dv_dy, dv_dx = np.gradient(v, step)
    e11, e22, e12 = du_dx, dv_dy, 0.5 * (du_dy + dv_dx)
    return np.sqrt(((e11 - e22) / 2.0) ** 2 + e12**2)
