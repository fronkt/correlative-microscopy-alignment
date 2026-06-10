#!/usr/bin/env bash
# Pyramid v2 sweep: roma first (the bar-setter), then matchanything.
set -euo pipefail
cd /root/cma
/venv/main/bin/python scripts/run_baselines_A.py \
  --backbones roma --mode pyramid_v2 --out results/baselines_A.csv
/venv/main/bin/python scripts/run_baselines_A.py \
  --backbones matchanything --mode pyramid_v2 --out results/baselines_A.csv
echo "V2 SWEEPS COMPLETE"
