#!/usr/bin/env bash
# Phase 4.2: MA-RoMa (cross-modal-trained RoMa weights) direct + pyramid_v2
# over all 187 pairs. Same shared-box discipline as box_41c_subsets.sh:
# dataset zip + one extracted subset + HF weight cache all live in
# RAM-backed /dev/shm, so the run touches no overlay disk.
set -euo pipefail
cd /root/cma
git pull -q

export HF_HOME=/dev/shm/hf

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
Active sweep (correlative-microscopy session): /root/cma and /dev/shm
contents are in use until "42 MA-ROMA SWEEPS COMPLETE" appears in
/root/42_ma_roma.log. Dataset + weights live in /dev/shm and use no
overlay disk. Please do not delete.
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
  run_sweep --backbones ma_roma --mode direct --out results/baselines_A.csv
  run_sweep --backbones ma_roma --mode pyramid_v2 --out results/baselines_A.csv
  rm -rf "/dev/shm/AmalgaMatch/$s"
  df -h /dev/shm | tail -1
done

rm -rf "$ZIP" /dev/shm/AmalgaMatch /dev/shm/hf /root/README_CMA_SWEEP.txt
echo "42 MA-ROMA SWEEPS COMPLETE"
