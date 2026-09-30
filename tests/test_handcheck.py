import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools", "handcheck"))
import handcheck as hc  # noqa: E402


def test_scaling_roundtrip_and_centre_convention():
    sc = (3.0, 2.0)
    x, y = hc.disp_to_full(0, 0, sc)
    assert (x, y) == (1.0, 0.5)  # centre of display pixel 0 = middle of the first 3x2 block
    for px in [(0, 0), (10.3, 7.7), (99, 50)]:
        f = hc.disp_to_full(*px, sc)
        b = hc.full_to_disp(*f, sc)
        assert np.allclose(b, px)


def test_prepare_display_scale_matches_shape():
    g = np.random.default_rng(0).random((900, 1800)).astype(np.float32)
    img, (sx, sy) = hc.prepare_display(g, max_disp=600)
    assert img.dtype == np.uint8 and img.shape == (300, 600)
    assert np.isclose(sx * 600, 1800) and np.isclose(sy * 300, 900)


def test_gt_error_zero_and_offset():
    H = np.array([[1.1, 0.02, 5], [-0.01, 0.9, -3], [1e-6, 0, 1.0]])
    pl = np.array([[10, 20], [300, 400], [1000, 50.0]])
    pr = hc.apply_h(H, pl)
    assert np.allclose(hc.gt_errors(H, pl, pr), 0, atol=1e-9)
    assert np.allclose(hc.gt_errors(H, pl, pr + [3, 4]), 5.0)


def test_csv_save_resume(tmp_path):
    out = str(tmp_path / "c.csv")
    hc.save_pair(out, "A", [(1, 2, 3, 4), (5, 6, 7, 8)], ts="t0")
    hc.save_pair(out, "B", [(9, 9, 9, 9)], ts="t1")
    hc.save_pair(out, "A", [(1, 2, 3, 5)], ts="t2")  # overwrite pair A
    rows = list(csv.DictReader(open(out)))
    assert list(rows[0].keys()) == hc.CSV_FIELDS
    assert [(r["pair_id"], r["k"]) for r in rows] == [("B", "0"), ("A", "0")] or len(rows) == 2
    assert {r["pair_id"] for r in rows} == {"A", "B"}
    hc.save_skip(out + ".skips.csv", "C", "blurry")
    pairs = [{"pair_id": p, "left": "", "right": "", "H": None} for p in "ABCD"]
    assert [p["pair_id"] for p in hc.pending_pairs(pairs, out)] == ["D"]


def _make_pairs(tmp_path, n=2):
    import tifffile

    rng = np.random.default_rng(1)
    pairs = []
    for i in range(n):
        a = (rng.random((400, 600)) * 65535).astype(np.uint16)
        pa, pb = str(tmp_path / f"a{i}.tif"), str(tmp_path / f"b{i}.tif")
        tifffile.imwrite(pa, a)
        tifffile.imwrite(pb, a[::2, ::2])  # right = half-size copy
        H = np.diag([0.5, 0.5, 1.0])
        H[:2, 2] = -0.25  # pixel-centre convention for x/2
        pairs.append({"pair_id": f"p{i}", "left": pa, "right": pb, "H": H})
    return pairs


class Ev:
    def __init__(self, ax, x, y, key=None, button=1):
        self.inaxes, self.xdata, self.ydata, self.key, self.button = ax, x, y, key, button


def test_gui_click_flow_with_agg(tmp_path, monkeypatch):
    pairs = _make_pairs(tmp_path)
    out = str(tmp_path / "out.csv")
    app = hc.HandCheckApp(pairs, out, n_points=3, max_disp=200)
    L, R = app.axes
    sl, sr = app.cur["scales"]
    # wrong-axis click ignored
    app.on_click(Ev(R, 5, 5))
    assert app.cur["pending_left"] is None
    truth = [(20, 30), (100, 60), (150, 90)]  # display coords in LEFT
    for k, (x, y) in enumerate(truth):
        app.on_click(Ev(L, x, y))
        # perfect right-click: half-size image => display coords (rounded scaling) computed from the GT
        xf, yf = hc.disp_to_full(x, y, sl)
        xr, yr = hc.apply_h(pairs[0]["H"], [[xf, yf]])[0]
        dx, dy = hc.full_to_disp(xr, yr, sr)
        app.on_click(Ev(R, dx, dy))
    assert app.cur["finished"]
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 3 and rows[0]["pair_id"] == "p0"
    assert app.report and app.report[0][1].max() < 1e-6
    # undo un-finishes and drops the saved rows; redo works
    app.on_key(Ev(None, 0, 0, key="u"))
    assert not app.cur["finished"] and len(app.cur["clicks"]) == 2
    assert not os.path.exists(out) or "p0" not in {r["pair_id"] for r in csv.DictReader(open(out))}
    app.on_click(Ev(L, 150, 90))
    app.on_click(Ev(R, 75, 45))
    assert app.cur["finished"]
    # n moves on; skip asks the terminal and records the reason
    app.on_key(Ev(None, 0, 0, key="n"))
    assert app.cur["p"]["pair_id"] == "p1"
    monkeypatch.setattr("builtins.input", lambda *_: "out of focus")
    app.on_key(Ev(None, 0, 0, key="s"))
    sk = list(csv.DictReader(open(out + ".skips.csv")))
    assert sk[0]["pair_id"] == "p1" and sk[0]["reason"] == "out of focus"
    assert app.quit
    # resume: nothing pending
    assert hc.pending_pairs(pairs, out) == []


def test_read_pairs_csv(tmp_path):
    p = tmp_path / "p.csv"
    p.write_text('pair_id,left_path,right_path,gt_h\nx,a.tif,b.tif,"[1,0,0,0,1,0,0,0,1]"\ny,c.tif,d.tif,\n')
    ps = hc.read_pairs(str(p))
    assert np.allclose(ps[0]["H"], np.eye(3)) and ps[1]["H"] is None
