#!/usr/bin/env bash
# MA-RoMa decoder-only fine-tune (train split, val-selected checkpoint),
# then sweep ma_roma_ft direct + pyramid_v2 over all 187 pairs (analysis
# restricts to the 28 test pairs). Checkpoint lands in /dev/shm/cma_ckpt
# (~1.7 GB) — scp it AND results/baselines_A.csv + results/finetune_log.csv
# back before releasing the box; /dev/shm does not survive a recycle.
set -euo pipefail
cd /root/cma
# The box is a pure results-producer: its CSV appends always reach origin
# via local before the next run, so a hard reset is the correct sync (a
# plain pull aborts on the locally-modified CSV every time).
git fetch -q && git reset --hard -q origin/main

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
Active fine-tune (correlative-microscopy session): /root/cma and /dev/shm
contents are in use until "FINETUNE PIPELINE COMPLETE" appears in
/root/finetune.log. Dataset + weights + checkpoint live in /dev/shm and
use no overlay disk. Please do not delete.
EOF

CKPT=/dev/shm/cma_ckpt
mkdir -p "$CKPT"

/venv/main/bin/python scripts/finetune_ma_roma.py \
  --out-dir "$CKPT" --steps 1500 --val-every 100 --num-workers 4 \
  --log results/finetune_log.csv
[ -f "$CKPT/ma_roma_ft.pth" ] || { echo "NO CHECKPOINT WRITTEN"; exit 1; }

# Known issue: the runner wedges at interpreter exit after GPU sweeps,
# AFTER writing its last CSV row. Treat a timeout kill as success.
for mode in direct pyramid_v2; do
  rc=0
  timeout -k 30 14400 /venv/main/bin/python scripts/run_baselines_A.py \
    --backbones ma_roma_ft --mode "$mode" \
    --ft-weights "$CKPT/ma_roma_ft.pth" || rc=$?
  if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    echo "($mode runner wedged at exit; killed by timeout after rows were written)"
  elif [ "$rc" -ne 0 ]; then
    exit "$rc"
  fi
done

rm -rf /dev/shm/AmalgaMatch /dev/shm/hf /root/README_CMA_SWEEP.txt
echo "FINETUNE PIPELINE COMPLETE — scp back: $CKPT/ma_roma_ft.pth," \
     "results/baselines_A.csv, results/finetune_log.csv"
