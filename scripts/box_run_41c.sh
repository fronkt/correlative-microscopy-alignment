#!/usr/bin/env bash
# 4.1c sweeps: re-fetch dataset, then roma pyramid_v2 with iterated zoom (z3)
# and with certainty gating 0.5 on top (c50). Distinct --tag = distinct rows.
set -euo pipefail
cd /root/cma

if [ ! -d data/AmalgaMatch/CoNi-AM67_OM-SEM_Multiscale ]; then
  wget -q -O /workspace/AmalgaMatch_Dataset.zip \
    https://fordatis.fraunhofer.de/bitstream/fordatis/478/1/AmalgaMatch_Dataset.zip
  mkdir -p data/AmalgaMatch
  unzip -q -o /workspace/AmalgaMatch_Dataset.zip -d data/AmalgaMatch
  rm /workspace/AmalgaMatch_Dataset.zip
fi

/venv/main/bin/python scripts/run_baselines_A.py \
  --backbones roma --mode pyramid_v2 --tag z3 --out results/baselines_A.csv
/venv/main/bin/python scripts/run_baselines_A.py \
  --backbones roma --mode pyramid_v2 --tag c50 --certainty 0.5 \
  --out results/baselines_A.csv
echo "41C SWEEPS COMPLETE"
