"""handcheck: click-based validation of automatic ground-truth transforms for image pairs.

For every pair the left and right image are shown side by side (downsampled for display).
Click a feature in the LEFT image, then the SAME feature in the RIGHT image; repeat N_POINTS times.
Coordinates are stored in FULL-RESOLUTION pixels (pixel-centre convention, (0,0) = centre of the
top-left pixel), so they are directly comparable to a 3x3 homography H that maps left full-res
pixels to right full-res pixels.

Keys:  u undo last click | s skip pair (reason asked in the terminal) | n next pair | q save + quit
Toolbar: zoom/pan are ordinary matplotlib tools; clicks are ignored while a toolbar tool is active
(press the magnifier/arrow button again to deactivate it before clicking).

Pair list (CSV, header required):  pair_id,left_path,right_path[,gt_h]
  gt_h = optional JSON list of 9 numbers (row-major H, left full-res px -> right full-res px).
Output CSV: pair_id,k,x_left,y_left,x_right,y_right,timestamp  (rewritten atomically after each pair)
Skips go to <out>.skips.csv (pair_id,reason,timestamp).  Re-running resumes: finished/skipped
pairs are not shown again.

No image content is ever shown with the GT prediction overlaid (avoids anchoring the clicks).
"""
import argparse
import csv
import datetime as _dt
import json
import os
import sys

import numpy as np

N_POINTS = 6
MAX_DISP = 1200  # longest display side in pixels
CSV_FIELDS = ["pair_id", "k", "x_left", "y_left", "x_right", "y_right", "timestamp"]


# ----------------------------------------------------------------------------- pure helpers
def now_iso():
    return _dt.datetime.now().isoformat(timespec="seconds")


def load_gray(path):
    """Load an image as float32 2-D array (first channel / luminance)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        import tifffile

        a = tifffile.imread(path)
    else:
        import cv2

        a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if a is None:
            raise FileNotFoundError(path)
    a = np.asarray(a)
    while a.ndim > 3:
        a = a[0]
    if a.ndim == 3:
        if a.shape[0] in (3, 4) and a.shape[2] not in (3, 4):
            a = np.moveaxis(a, 0, -1)
        a = a[..., :3].astype(np.float32).mean(axis=2)
    return a.astype(np.float32)


def prepare_display(gray, max_disp=MAX_DISP):
    """Return (uint8 display image, (sx, sy)) with full_px = f(display_px) scale factors sx, sy."""
    import cv2

    h, w = gray.shape
    f = min(1.0, max_disp / max(h, w))
    dw, dh = max(1, round(w * f)), max(1, round(h * f))
    small = cv2.resize(gray, (dw, dh), interpolation=cv2.INTER_AREA) if f < 1 else gray
    lo, hi = np.percentile(small, [1, 99])
    if hi <= lo:
        hi = lo + 1
    img = np.clip((small - lo) / (hi - lo), 0, 1)
    return (img * 255).astype(np.uint8), (w / dw, h / dh)


def disp_to_full(x, y, scale):
    """Display pixel -> full-res pixel, pixel-centre convention."""
    sx, sy = scale
    return (x + 0.5) * sx - 0.5, (y + 0.5) * sy - 0.5


def full_to_disp(x, y, scale):
    sx, sy = scale
    return (x + 0.5) / sx - 0.5, (y + 0.5) / sy - 0.5


def apply_h(H, pts):
    pts = np.atleast_2d(np.asarray(pts, float))
    ph = np.c_[pts, np.ones(len(pts))] @ np.asarray(H, float).T
    return ph[:, :2] / ph[:, 2:3]


def gt_errors(H, left_pts, right_pts):
    """Per-point Euclidean error (right-image full-res px) of clicked pairs under the GT H."""
    pred = apply_h(H, left_pts)
    return np.linalg.norm(pred - np.asarray(right_pts, float), axis=1)


def read_pairs(path):
    if str(path).lower().endswith(".json"):
        return demo_pairs(path)
    pairs = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            H = None
            if r.get("gt_h"):
                H = np.array(json.loads(r["gt_h"]), float).reshape(3, 3)
            pairs.append({"pair_id": r["pair_id"], "left": r["left_path"], "right": r["right_path"], "H": H})
    return pairs


def read_results(path):
    """Return {pair_id: [row dict, ...]} from the output CSV (empty if absent)."""
    out = {}
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                out.setdefault(r["pair_id"], []).append(r)
    return out


def read_skips(path):
    out = {}
    if os.path.exists(path):
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                out[r["pair_id"]] = r
    return out


def _atomic_write(path, header, rows):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    for _ in range(20):  # Windows: another process may briefly hold the target
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            import time

            time.sleep(0.1)
    os.replace(tmp, path)


def save_pair(path, pair_id, clicks, ts=None):
    """Replace all rows of pair_id in the output CSV by clicks = [(xl, yl, xr, yr), ...]."""
    existing = read_results(path)
    existing[pair_id] = [
        {"pair_id": pair_id, "k": k, "x_left": c[0], "y_left": c[1], "x_right": c[2], "y_right": c[3],
         "timestamp": ts or now_iso()}
        for k, c in enumerate(clicks)
    ]
    rows = [[r[f] for f in CSV_FIELDS] for rs in existing.values() for r in rs]
    _atomic_write(path, CSV_FIELDS, rows)


def save_skip(path, pair_id, reason, ts=None):
    sk = read_skips(path)
    sk[pair_id] = {"pair_id": pair_id, "reason": reason, "timestamp": ts or now_iso()}
    _atomic_write(path, ["pair_id", "reason", "timestamp"], [[r["pair_id"], r["reason"], r["timestamp"]] for r in sk.values()])


def pending_pairs(pairs, out_csv):
    done = set(read_results(out_csv)) | set(read_skips(out_csv + ".skips.csv"))
    return [p for p in pairs if p["pair_id"] not in done]


# ----------------------------------------------------------------------------- GUI
class HandCheckApp:
    def __init__(self, pairs, out_csv, n_points=N_POINTS, max_disp=MAX_DISP, backend_show=True):
        import matplotlib.pyplot as plt

        # The tool's own keys (u, s, n, q) must not also fire matplotlib's defaults ('s' = save figure dialog,
        # 'q' = close window without saving). Strip every tool key from every default keymap.
        for name in [k for k in plt.rcParams if k.startswith("keymap.")]:
            plt.rcParams[name] = [x for x in plt.rcParams[name] if x not in ("u", "s", "n", "q")]
        self.plt = plt
        self.pairs = pairs
        self.out_csv = out_csv
        self.skip_csv = out_csv + ".skips.csv"
        self.n_points = n_points
        self.max_disp = max_disp
        self.queue = pending_pairs(pairs, out_csv)
        self.total = len(pairs)
        self.idx = -1
        self.quit = False
        self.report = []  # (pair_id, errors)
        self.fig, self.axes = plt.subplots(1, 2, figsize=(15, 7.5))
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self.backend_show = backend_show
        self.cur = None
        self.next_pair()

    # -- state
    def next_pair(self):
        self.idx += 1
        if self.idx >= len(self.queue):
            self.quit = True
            print("All pairs done.")
            self.plt.close(self.fig)
            return
        p = self.queue[self.idx]
        gl, gr = load_gray(p["left"]), load_gray(p["right"])
        il, sl = prepare_display(gl, self.max_disp)
        ir, sr = prepare_display(gr, self.max_disp)
        self.cur = {"p": p, "scales": (sl, sr), "clicks": [], "pending_left": None, "finished": False,
                    "artists": [], "shapes": (gl.shape, gr.shape)}
        for ax, im, ttl in zip(self.axes, (il, ir), ("LEFT", "RIGHT")):
            ax.clear()
            ax.imshow(im, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            ax.set_title(ttl)
            ax.set_xticks([])
            ax.set_yticks([])
        self.refresh_title()

    def refresh_title(self):
        c = self.cur
        n = len(c["clicks"])
        side = "RIGHT" if c["pending_left"] is not None else "LEFT"
        done_before = self.total - len(self.queue)
        head = f"pair {done_before + self.idx + 1}/{self.total}  [{c['p']['pair_id']}]"
        if c["finished"]:
            msg = "all points recorded - press n (next), u (undo), s (skip), q (quit)"
        else:
            msg = f"click {side} image for point {n + 1}/{self.n_points}   (u undo, s skip, n next, q quit)"
        self.fig.suptitle(head + "\n" + msg)
        self.fig.canvas.draw_idle()

    def redraw_marks(self):
        c = self.cur
        for a in c["artists"]:
            a.remove()
        c["artists"] = []
        sl, sr = c["scales"]
        for k, (xl, yl, xr, yr) in enumerate(c["clicks"]):
            for ax, (x, y), s in ((self.axes[0], (xl, yl), sl), (self.axes[1], (xr, yr), sr)):
                dx, dy = full_to_disp(x, y, s)
                c["artists"].append(ax.plot(dx, dy, "r+", ms=14, mew=1.5)[0])
                c["artists"].append(ax.annotate(str(k + 1), (dx, dy), color="yellow", fontsize=11,
                                                xytext=(5, 5), textcoords="offset points"))
        if c["pending_left"] is not None:
            x, y = c["pending_left"]
            dx, dy = full_to_disp(x, y, sl)
            c["artists"].append(self.axes[0].plot(dx, dy, "c+", ms=14, mew=1.5)[0])

    # -- events
    def on_click(self, ev):
        c = self.cur
        if c is None or ev.inaxes is None or ev.button != 1:
            return
        tb = getattr(self.fig.canvas, "toolbar", None)
        if tb is not None and getattr(tb, "mode", "") not in ("", None) and str(tb.mode) != "":
            return  # zoom/pan active
        if c["finished"]:
            return
        want_left = c["pending_left"] is None
        ax_want = self.axes[0] if want_left else self.axes[1]
        if ev.inaxes is not ax_want:
            print(f"  (expected a click in the {'LEFT' if want_left else 'RIGHT'} image)")
            return
        sl, sr = c["scales"]
        if want_left:
            c["pending_left"] = disp_to_full(ev.xdata, ev.ydata, sl)
        else:
            xr, yr = disp_to_full(ev.xdata, ev.ydata, sr)
            xl, yl = c["pending_left"]
            c["clicks"].append((xl, yl, xr, yr))
            c["pending_left"] = None
            if len(c["clicks"]) >= self.n_points:
                c["finished"] = True
                self.finish_pair()
        self.redraw_marks()
        self.refresh_title()

    def finish_pair(self):
        c = self.cur
        p = c["p"]
        save_pair(self.out_csv, p["pair_id"], c["clicks"])
        if p["H"] is not None:
            arr = np.array(c["clicks"])
            e = gt_errors(p["H"], arr[:, :2], arr[:, 2:])
            self.report.append((p["pair_id"], e))
            print(f"[{p['pair_id']}] error under automatic GT (right-image px): "
                  + " ".join(f"{v:.1f}" for v in e) + f"   median {np.median(e):.1f}  max {e.max():.1f}")
        else:
            print(f"[{p['pair_id']}] saved (no GT supplied for this pair)")

    def on_key(self, ev):
        c = self.cur
        if c is None:
            return
        k = ev.key
        if k == "u":
            if c["pending_left"] is not None:
                c["pending_left"] = None
            elif c["clicks"]:
                c["clicks"].pop()
                c["finished"] = False
                # a saved pair is re-saved only when finished again; drop stale rows now
                if c["p"]["pair_id"] in read_results(self.out_csv):
                    rest = read_results(self.out_csv)
                    rest.pop(c["p"]["pair_id"])
                    rows = [[r[f] for f in CSV_FIELDS] for rs in rest.values() for r in rs]
                    _atomic_write(self.out_csv, CSV_FIELDS, rows)
            self.redraw_marks()
            self.refresh_title()
        elif k == "s":
            reason = input(f"Skip reason for {c['p']['pair_id']}: ").strip() or "unspecified"
            save_skip(self.skip_csv, c["p"]["pair_id"], reason)
            print(f"  skipped ({reason})")
            self.next_pair()
        elif k == "n":
            if not c["finished"]:
                print(f"  need {self.n_points} points first (have {len(c['clicks'])}); use s to skip")
                return
            self.next_pair()
        elif k == "q":
            self.quit = True
            self.plt.close(self.fig)

    def run(self):
        if not self.quit:
            self.plt.show()
        if self.report:
            allerr = np.concatenate([e for _, e in self.report])
            print(f"\nSummary: {len(self.report)} pairs, {len(allerr)} points; "
                  f"median error {np.median(allerr):.1f} px, 90th pct {np.percentile(allerr, 90):.1f} px")


def demo_pairs(demo_json):
    """Pairs listed in a JSON file produced by scripts/nist_fit_homography.py (--emit-pairs)."""
    d = json.load(open(demo_json, encoding="utf-8"))
    return [{"pair_id": r["pair_id"], "left": r["left_path"], "right": r["right_path"],
             "H": np.array(r["gt_h"], float).reshape(3, 3) if r.get("gt_h") else None} for r in d]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", help="pair-list CSV (pair_id,left_path,right_path[,gt_h])")
    ap.add_argument("--out", default="handcheck_clicks.csv", help="output CSV (resume-safe)")
    ap.add_argument("--n-points", type=int, default=N_POINTS)
    ap.add_argument("--max-disp", type=int, default=MAX_DISP)
    ap.add_argument("--demo", action="store_true", help="run on the 2 downloaded NIST demo sections")
    ap.add_argument("--demo-json", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_pairs.json"))
    a = ap.parse_args(argv)
    if a.demo:
        pairs = demo_pairs(a.demo_json)
        if a.out == "handcheck_clicks.csv":
            a.out = "handcheck_demo_clicks.csv"
    elif a.pairs:
        pairs = read_pairs(a.pairs)
    else:
        ap.error("give --pairs FILE or --demo")
    app = HandCheckApp(pairs, a.out, a.n_points, a.max_disp)
    app.run()


if __name__ == "__main__":
    sys.exit(main())
