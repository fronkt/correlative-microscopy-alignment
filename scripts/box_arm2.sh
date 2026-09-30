#!/usr/bin/env bash
# Arm 2 (materials replication): set up a fresh vast.ai box and generate the pre-registered candidate registrations.
# Requires branch triage-ext (prereg Addendum A committed) to be pushed to the PUBLIC repo first.
# The pool exists only on the laptop: copy it up BEFORE or while this runs, e.g.
#   scp -P <port> matpool.tar.gz matpool.tar.gz.meta root@<host>:/root/
# (both files come from `python scripts/pack_arm2.py`). This script waits for /root/matpool.tar.gz to reach the
# expected size, verifies the sha256, extracts to /root/matpool and runs with MATPOOL_ROOT=/root/matpool.
# Expected size / hash: env ARM2_TAR_SIZE and ARM2_TAR_SHA256, or the sidecar /root/matpool.tar.gz.meta
# (lines SIZE=..., SHA256=..., PAIRS=...). Optional env ARM2_PAIRS overrides the pair count (else PAIRS from the meta).
# Run inside tmux:  tmux new-session -d -s arm2 'bash /root/cma/scripts/box_arm2.sh > /root/arm2.log 2>&1'
# Resumable: finished (pair, backbone, mode, transform, seed) rows are skipped on re-run.
set -euo pipefail

cd /root
[ -d cma ] || git clone -q -b triage-ext https://github.com/fronkt/correlative-microscopy-alignment.git cma
cd /root/cma
git pull -q --ff-only || true
echo "commit $(git rev-parse HEAD)"
pip install -q -e ".[dev]" kornia romatch transformers pillow scipy scikit-image
# The pytorch image lacks libxcb, which opencv-python needs; use the headless build, pinned to the
# 4.13 line the local runs used (pip otherwise pulls OpenCV 5.0). Force-reinstall: uninstalling
# opencv-python deletes the cv2 folder the two wheels share.
pip uninstall -y -q opencv-python || true
pip install -q --force-reinstall --no-deps "opencv-python-headless==4.13.0.92"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

TAR=/root/matpool.tar.gz
META=/root/matpool.tar.gz.meta
POOL=/root/matpool
if [ -z "${ARM2_TAR_SIZE:-}" ] || [ -z "${ARM2_TAR_SHA256:-}" ] || [ -z "${ARM2_PAIRS:-}" ]; then
  until [ -s "$META" ]; do echo "waiting for $META (or set ARM2_TAR_SIZE / ARM2_TAR_SHA256 / ARM2_PAIRS)"; sleep 15; done
  # shellcheck disable=SC1090
  meta() { grep -E "^$1=" "$META" | head -1 | cut -d= -f2 | tr -d '\r '; }
  ARM2_TAR_SIZE="${ARM2_TAR_SIZE:-$(meta SIZE)}"
  ARM2_TAR_SHA256="${ARM2_TAR_SHA256:-$(meta SHA256)}"
  ARM2_PAIRS="${ARM2_PAIRS:-$(meta PAIRS)}"
fi
echo "expect size $ARM2_TAR_SIZE sha256 $ARM2_TAR_SHA256 pairs $ARM2_PAIRS"

if [ ! -f "$POOL/manifest.csv" ]; then
  until [ "$(stat -c%s "$TAR" 2>/dev/null || echo 0)" -eq "$ARM2_TAR_SIZE" ]; do
    echo "waiting for pool: $(stat -c%s "$TAR" 2>/dev/null || echo 0)/$ARM2_TAR_SIZE"; sleep 15
  done
  echo "$ARM2_TAR_SHA256  $TAR" | sha256sum -c -
  rm -rf "$POOL"; mkdir -p "$POOL"
  tar -xzf "$TAR" -C "$POOL"
fi
export MATPOOL_ROOT="$POOL"
[ "$(($(grep -c . "$POOL/manifest.csv") - 1))" -eq "$ARM2_PAIRS" ] || { echo "manifest row count != $ARM2_PAIRS"; exit 1; }

mkdir -p results/arm2
python -m pytest -q tests/test_triage.py
# The loader must resolve the extracted copy (relative manifest paths) before any GPU time is spent.
python - <<'PY'
from cma.materials_pool import MaterialsPoolLoader
L = MaterialsPoolLoader()
assert all(r.source_path.is_file() and r.target_path.is_file() and r.gt_path.is_file() for r in L.records)
print("pool loader OK:", len(L), "pairs under", L.root)
PY

# ONE process, backbones strictly sequential (gt, sift, loftr, roma, ma_roma, matchanything): two RoMa-family models
# never share the 24 GB card. (CJSJ ran roma and ma_roma as two parallel processes with 0 CUDA/OOM failures in its
# 5,049 rows, but Arm 2 images are up to ~2.2 kpx and the OOM risk is not worth the ~1 h saved.)
# Pools: gt,core,transform -- NO control pool. Per pair: 2 gt + 7 core + 8 transform = 17 rows.
python scripts/arm2_run_candidates.py --pools gt,core,transform \
  --backbones sift,loftr,roma,ma_roma,matchanything --out results/arm2/candidates.csv > results/arm2/run.log 2>&1 \
  && echo "run exit 0" || echo "run exit $?"

python - <<PY
import csv
rows = list(csv.DictReader(open("results/arm2/candidates.csv", newline="", encoding="utf-8")))
bad = [r for r in rows if r["status"] != "ok"]
exp = $ARM2_PAIRS * 17
print("rows", len(rows), "(expected", exp, "= $ARM2_PAIRS pairs x (15 voting + 2 gt))", "failed", len(bad),
      "MATCH" if len(rows) == exp else "MISMATCH")
PY
echo "ARM2 COMPLETE"
