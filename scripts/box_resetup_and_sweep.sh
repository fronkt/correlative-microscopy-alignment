#!/usr/bin/env bash
# Full re-setup after instance recycle + relaunch of the pyramid sweep.
# The committed results/baselines_A.csv arrives with the clone, so the
# runner's resume logic skips all completed (pair, backbone, mode) rows.
set -euo pipefail

wget -q -O /workspace/AmalgaMatch_Dataset.zip \
  https://fordatis.fraunhofer.de/bitstream/fordatis/478/1/AmalgaMatch_Dataset.zip &

git clone -q https://github.com/fronkt/correlative-microscopy-alignment.git /root/cma
sed -i 's/\r$//' /root/cma/scripts/box_setup.sh
bash /root/cma/scripts/box_setup.sh

cd /root/cma
nohup /venv/main/bin/python scripts/run_baselines_A.py \
  --backbones roma,matchanything --mode pyramid \
  --out results/baselines_A.csv > /root/pyramid.log 2>&1 &
echo "SWEEP LAUNCHED"
