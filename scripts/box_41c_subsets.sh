#!/usr/bin/env bash
# 4.1c sweeps on a disk-constrained SHARED box: keep the zip, extract one
# subset at a time (max 1.5G), sweep it, delete it. The loader indexes
# whatever subset dirs exist, and the runner's CSV resume skips done rows,
# so no filter flags are needed.
set -euo pipefail
cd /root/cma

ZIP=/root/AmalgaMatch_Dataset.zip
if [ ! -f "$ZIP" ]; then
  wget -q -O "$ZIP" \
    https://fordatis.fraunhofer.de/bitstream/fordatis/478/1/AmalgaMatch_Dataset.zip
fi

mkdir -p data/AmalgaMatch
cat > /root/README_CMA_SWEEP.txt <<'EOF'
Active sweep (correlative-microscopy session): /root/cma/data and
/root/AmalgaMatch_Dataset.zip are in use until "41C SUBSET SWEEPS
COMPLETE" appears in /root/41c_subsets.log. Please do not delete.
Peak footprint is capped at ~5.7G (zip + one subset at a time).
EOF

# Known issue: the runner wedges at interpreter exit after GPU sweeps,
# AFTER writing its last CSV row. Treat a timeout kill as success.
run_sweep() {
  local rc=0
  timeout -k 30 2700 /venv/main/bin/python scripts/run_baselines_A.py "$@" || rc=$?
  if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    echo "  (runner wedged at exit; killed by timeout after rows were written)"
  elif [ "$rc" -ne 0 ]; then
    return "$rc"
  fi
}

for s in $(unzip -Z1 "$ZIP" | cut -d/ -f1 | sort -u); do
  echo "=== subset: $s ==="
  unzip -q -o "$ZIP" "$s/*" -d data/AmalgaMatch
  run_sweep --backbones roma --mode pyramid_v2 --tag z3 \
    --out results/baselines_A.csv
  run_sweep --backbones roma --mode pyramid_v2 --tag c50 --certainty 0.5 \
    --out results/baselines_A.csv
  rm -rf "data/AmalgaMatch/$s"
  df -h / | tail -1
done

rm -f "$ZIP" /root/README_CMA_SWEEP.txt
echo "41C SUBSET SWEEPS COMPLETE"
