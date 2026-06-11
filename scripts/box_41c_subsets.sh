#!/usr/bin/env bash
# 4.1c sweeps on a disk-constrained SHARED box: zip and the one extracted
# subset both live in RAM-backed /dev/shm (31G), so the dataset uses zero
# overlay disk and the other tenant's growth can't evict it. The loader
# indexes whatever subset dirs exist behind the data/AmalgaMatch symlink,
# and the runner's CSV resume skips done rows, so no filter flags needed.
set -euo pipefail
cd /root/cma

ZIP=/dev/shm/AmalgaMatch_Dataset.zip
EXPECTED=4228037938
for attempt in 1 2 3; do
  [ "$(stat -c%s "$ZIP" 2>/dev/null || echo 0)" -eq "$EXPECTED" ] && break
  wget -c -q -O "$ZIP" \
    https://fordatis.fraunhofer.de/bitstream/fordatis/478/1/AmalgaMatch_Dataset.zip \
    || true
done
[ "$(stat -c%s "$ZIP")" -eq "$EXPECTED" ] || { echo "ZIP SIZE MISMATCH"; exit 1; }

mkdir -p /dev/shm/AmalgaMatch
rm -rf data/AmalgaMatch
mkdir -p data
ln -sfn /dev/shm/AmalgaMatch data/AmalgaMatch

cat > /root/README_CMA_SWEEP.txt <<'EOF'
Active sweep (correlative-microscopy session): /root/cma and
/dev/shm/AmalgaMatch* are in use until "41C SUBSET SWEEPS COMPLETE"
appears in /root/41c_subsets.log. Dataset lives in /dev/shm and uses
no overlay disk. Please do not delete.
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
  unzip -q -o "$ZIP" "$s/*" -d /dev/shm/AmalgaMatch
  run_sweep --backbones roma --mode pyramid_v2 --tag z3 \
    --out results/baselines_A.csv
  run_sweep --backbones roma --mode pyramid_v2 --tag c50 --certainty 0.5 \
    --out results/baselines_A.csv
  rm -rf "/dev/shm/AmalgaMatch/$s"
  df -h /dev/shm | tail -1
done

rm -rf "$ZIP" /dev/shm/AmalgaMatch /root/README_CMA_SWEEP.txt
echo "41C SUBSET SWEEPS COMPLETE"
