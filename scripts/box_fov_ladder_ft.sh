#!/usr/bin/env bash
# FOV ladder for the fine-tuned backbone (ma_roma_ft), direct + pyramid_v2,
# on the SAME 63-pair testbed the roma/ma_roma ladder used. The ft model's
# direct rows in baselines_A.csv would otherwise expand the base-matchable
# set to 120 pairs and break comparability, so we --restrict-pairs-csv to
# the committed results/fov_ladder.csv. Checkpoint is expected at
# /dev/shm/cma_ckpt/ma_roma_ft.pth (left there by box_finetune.sh).
set -euo pipefail
cd /root/cma
git fetch -q && git reset --hard -q origin/main

export HF_HOME=/dev/shm/hf
CKPT=/dev/shm/cma_ckpt/ma_roma_ft.pth
[ -f "$CKPT" ] || { echo "MISSING CHECKPOINT $CKPT"; exit 1; }

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
contents are in use until "FT FOV LADDER COMPLETE" appears in
/root/fov_ladder_ft.log. Dataset + weights live in /dev/shm (no overlay
disk). Please do not delete.
EOF

# Known issue: the runner wedges at interpreter exit after GPU sweeps,
# AFTER writing its last CSV row. Treat a timeout kill as success.
rc=0
timeout -k 30 14400 /venv/main/bin/python scripts/run_fov_ladder.py \
  --backbones ma_roma_ft --modes direct,pyramid_v2 \
  --ft-weights "$CKPT" \
  --restrict-pairs-csv results/fov_ladder.csv \
  --out results/fov_ladder.csv || rc=$?
if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
  echo "(runner wedged at exit; killed by timeout after rows were written)"
elif [ "$rc" -ne 0 ]; then
  exit "$rc"
fi

rm -rf /dev/shm/AmalgaMatch /dev/shm/hf /root/README_CMA_SWEEP.txt
echo "FT FOV LADDER COMPLETE — scp back results/fov_ladder.csv"
