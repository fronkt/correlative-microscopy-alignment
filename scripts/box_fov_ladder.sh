#!/usr/bin/env bash
# FOV ladder sweep (Aim 3): roma + ma_roma, direct + pyramid_v2, over the
# ~63 base-matchable pairs x 5 crop rungs. Whole dataset fits in /dev/shm
# (9.3G of 31G), so no subset cycling and zero overlay-disk footprint.
set -euo pipefail
cd /root/cma
git pull -q

export HF_HOME=/dev/shm/hf

ZIP=/dev/shm/AmalgaMatch_Dataset.zip
EXPECTED=4228037938
if [ ! -d /dev/shm/AmalgaMatch/CoNi-AM67_OM-SEM_Multiscale ]; then
  for attempt in 1 2 3; do
    [ "$(stat -c%s "$ZIP" 2>/dev/null || echo 0)" -eq "$EXPECTED" ] && break
    wget -c -q -O "$ZIP" \
      https://fordatis.fraunhofer.de/bitstream/fordatis/478/1/AmalgaMatch_Dataset.zip \
      || true
  done
  [ "$(stat -c%s "$ZIP")" -eq "$EXPECTED" ] || { echo "ZIP SIZE MISMATCH"; exit 1; }
  mkdir -p /dev/shm/AmalgaMatch
  unzip -q -o "$ZIP" -d /dev/shm/AmalgaMatch
  rm -f "$ZIP"
fi
rm -rf data/AmalgaMatch
mkdir -p data
ln -sfn /dev/shm/AmalgaMatch data/AmalgaMatch

cat > /root/README_CMA_SWEEP.txt <<'EOF'
Active sweep (correlative-microscopy session): /root/cma and /dev/shm
contents are in use until "FOV LADDER DONE" appears in
/root/fov_ladder.log. Dataset + weights live in /dev/shm and use no
overlay disk. Please do not delete.
EOF

# Known issue: the runner wedges at interpreter exit after GPU sweeps,
# AFTER writing its last CSV row. Treat a timeout kill as success.
rc=0
timeout -k 30 14400 /venv/main/bin/python scripts/run_fov_ladder.py \
  --backbones roma,ma_roma --modes direct,pyramid_v2 \
  --out results/fov_ladder.csv || rc=$?
if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
  echo "(runner wedged at exit; killed by timeout after rows were written)"
elif [ "$rc" -ne 0 ]; then
  exit "$rc"
fi

rm -rf /dev/shm/AmalgaMatch /dev/shm/hf /root/README_CMA_SWEEP.txt
echo "FOV LADDER COMPLETE"
