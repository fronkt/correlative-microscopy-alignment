#!/usr/bin/env bash
# Arm 1 (banner intervention): set up a fresh vast.ai box and run the pre-registered cropped / uncropped reruns.
# Assumes the onstart command already began downloading the dataset from Fordatis to /root/AmalgaMatch_Dataset.zip.
# Requires branch triage-ext (with results/arm1/pairs_*.txt) to be pushed to the PUBLIC repo first.
# Run inside tmux:  tmux new-session -d -s arm1 'bash /root/box_arm1.sh > /root/arm1.log 2>&1'
# The script is resumable: re-running skips finished (pair, backbone, mode, transform, seed, crop) rows.
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

ZIP=/root/AmalgaMatch_Dataset.zip
EXPECTED=4228037938
if [ ! -d data/AmalgaMatch ] || [ -z "$(ls -A data/AmalgaMatch 2>/dev/null)" ]; then
  until [ "$(stat -c%s "$ZIP" 2>/dev/null || echo 0)" -eq "$EXPECTED" ]; do
    echo "waiting for dataset: $(stat -c%s "$ZIP" 2>/dev/null || echo 0)/$EXPECTED"; sleep 15
  done
  mkdir -p data/AmalgaMatch
  python -m zipfile -e "$ZIP" data/AmalgaMatch
  rm -f "$ZIP"
fi
mkdir -p results/arm1
python -m pytest -q tests/test_overlay.py tests/test_triage.py

# Pair lists must be present and of the pre-registered size.
[ "$(grep -c . results/arm1/pairs_overlay.txt)" -eq 67 ]
[ "$(grep -c . results/arm1/pairs_control_ti3alc2.txt)" -eq 3 ]
[ "$(grep -c . results/arm1/pairs_sham.txt)" -eq 20 ]

# ONE process, backbones strictly sequential (roma, then ma_roma): two RoMa-family models never share the 24 GB card.
# Jobs inside run_arm1.py: overlay x {none, overlay}; Ti3AlC2 x none; sham x {none, sham};
# configs per backbone: direct + pyramid_v2, seed 0, transform none  => 708 rows expected.
python scripts/run_arm1.py --backbones roma,ma_roma --out results/arm1/arm1.csv > results/arm1/run.log 2>&1
echo "run exit $?"
python - <<'PY'
import csv
rows = list(csv.DictReader(open("results/arm1/arm1.csv", newline="", encoding="utf-8")))
bad = [r for r in rows if r["status"] != "ok"]
print("rows", len(rows), "(expected 708)", "failed", len(bad))
PY
echo "ARM1 COMPLETE"
