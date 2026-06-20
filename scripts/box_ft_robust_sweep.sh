#!/usr/bin/env bash
# Phase 9A — forgetting-robust fine-tune: L2-SP lambda sweep.
# Trains MA-RoMa decoder-only at each lambda (0 = plain ft control), logging
# per-modality val retention (C103 SEM<->LOM probe vs TEM gain probe). Keeps
# every checkpoint in /dev/shm/cma_ckpt (Phase 9B eval runs on the SAME box
# session, so we don't scp 445 MB x4); only the small per-lambda logs go back
# to origin via the local sync. Pick the winning lambda from those logs, then
# run scripts/box_ft_robust_eval.sh <lambda> for the headline sweep + ladder.
set -euo pipefail
cd /root/cma
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
Active forgetting-robust fine-tune (correlative-microscopy session):
/root/cma and /dev/shm are in use until "ROBUST SWEEP COMPLETE" appears in
/root/robust_sweep.log. Checkpoints in /dev/shm/cma_ckpt are needed by the
follow-up eval (box_ft_robust_eval.sh). Dataset/weights/ckpts live in
/dev/shm (no overlay disk). Please do not delete.
EOF

CKPT=/dev/shm/cma_ckpt
mkdir -p "$CKPT"

# lambda -> safe-name (0 is the plain-ft control: anchor none)
declare -A NAME=( [0]=l0 [0.01]=l0p01 [0.1]=l0p1 [1.0]=l1 )
for LAM in 0 0.01 0.1 1.0; do
  tag=${NAME[$LAM]}
  if [ "$LAM" = "0" ]; then AN=(--anchor none); else AN=(--anchor l2sp --anchor-lambda "$LAM"); fi
  echo "=== TRAIN lambda=$LAM ($tag) ==="
  /venv/main/bin/python scripts/finetune_ma_roma.py \
    --out-dir "$CKPT/$tag" --steps 1500 --val-every 100 --num-workers 4 \
    --log "results/finetune_robust_log_$tag.csv" "${AN[@]}"
  [ -f "$CKPT/$tag/ma_roma_ft.pth" ] || { echo "NO CKPT for lambda=$LAM"; exit 1; }
done

echo ""
echo "=== val retention summary (last row = step 1500; *saved* = best) ==="
for LAM in 0 0.01 0.1 1.0; do
  tag=${NAME[$LAM]}
  echo "-- lambda=$LAM ($tag): results/finetune_robust_log_$tag.csv"
  tail -n +1 "results/finetune_robust_log_$tag.csv"
done
echo "ROBUST SWEEP COMPLETE — checkpoints in $CKPT/<tag>/ma_roma_ft.pth; pick lambda from logs"
