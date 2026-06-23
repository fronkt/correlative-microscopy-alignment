#!/usr/bin/env bash
# Multi-seed expansion of the MA-RoMa decoder-only fine-tune (Tables 2/3/4).
#
# Re-review item: take the headline fine-tune / L2-SP variance tables from 3
# seeds to >=5. Trains the plain (lambda=0) and L2-SP (lambda=0.01) recipes at
# each seed in $SEEDS with the IDENTICAL recipe used for seeds 1-3 (AdamW 2e-5,
# 1500 steps, batch 4, num-workers 4, min-val-medED checkpoint selection), then
# evaluates each checkpoint on the 28-pair held-out test split (direct and/or
# pyramid_v2) with the paper protocol (TPS-refined ED). Per-(seed,config,mode)
# SR@{5,10,20}+medED is appended to results/seed_expand_summary.csv.
#
# Resumable: an existing checkpoint skips training; a (seed,config,mode) already
# in the summary skips eval. Run direct first (fast, widens Table 2/4), then
# re-run with MODES=pyramid_v2 to fill Table 3 on the cached checkpoints.
#
#   SEEDS="4 5 6 7 8"  MODES="direct"  bash scripts/box_seed_expand.sh
#   SEEDS="4 5 6 7 8"  MODES="pyramid_v2"  bash scripts/box_seed_expand.sh
set -euo pipefail
cd /root/cma
git fetch -q && git reset --hard -q origin/main

source /venv/main/bin/activate
export HF_HOME=/dev/shm/hf

SEEDS="${SEEDS:-4 5 6 7 8}"
MODES="${MODES:-direct pyramid_v2}"
CKPT_ROOT=/dev/shm/cma_ckpt
SUMMARY=results/seed_expand_summary.csv
mkdir -p "$CKPT_ROOT"

# ---- dataset into /dev/shm (overlay disk is only ~16G) ----
ZIP=/dev/shm/AmalgaMatch_Dataset.zip
EXPECTED=4228037938
if [ ! -d /dev/shm/AmalgaMatch/CoNi-AM67_OM-SEM_Multiscale ]; then
  for attempt in 1 2 3; do
    [ "$(stat -c%s "$ZIP" 2>/dev/null || echo 0)" -eq "$EXPECTED" ] && break
    wget -c -q -O "$ZIP" \
      https://fordatis.fraunhofer.de/bitstream/fordatis/478/1/AmalgaMatch_Dataset.zip || true
  done
  [ "$(stat -c%s "$ZIP")" -eq "$EXPECTED" ] || { echo "ZIP SIZE MISMATCH"; exit 1; }
  mkdir -p /dev/shm/AmalgaMatch
  unzip -q -o "$ZIP" -d /dev/shm/AmalgaMatch
  rm -f "$ZIP"
fi
rm -rf data/AmalgaMatch; mkdir -p data
ln -sfn /dev/shm/AmalgaMatch data/AmalgaMatch

cat > /root/README_CMA_SWEEP.txt <<'EOF'
Active multi-seed ft expansion (correlative-microscopy session): /root/cma and
/dev/shm in use until "SEED EXPAND COMPLETE" appears in /root/seed_expand.log.
Results accumulate in results/seed_expand_summary.csv. Please do not delete.
EOF

have_row() {  # have_row <seed> <config> <mode>
  [ -f "$SUMMARY" ] || return 1
  awk -F, -v s="$1" -v c="$2" -v m="$3" \
    'NR>1 && $1==s && $2==c && $4==m {f=1} END{exit !f}' "$SUMMARY"
}

# The GPU runner can wedge at interpreter exit AFTER writing its rows. Rather
# than wait out a long hard timeout, poll the eval CSV and kill once all 28
# test-pair rows are present.
run_eval_until_done() {  # run_eval_until_done <eval_csv> <hard_to> -- <cmd...>
  local ec="$1" hard="$2"; shift 2; shift
  rm -f "$ec"
  "$@" & local pid=$!
  local waited=0 have=0
  while kill -0 "$pid" 2>/dev/null; do
    have=0; [ -f "$ec" ] && have=$(($(wc -l < "$ec") - 1))
    if [ "$have" -ge 28 ]; then
      sleep 8; kill "$pid" 2>/dev/null || true; sleep 2
      kill -9 "$pid" 2>/dev/null || true
      echo "(eval done: $have/28 rows; killed wedged exit)"; return 0
    fi
    sleep 15; waited=$((waited + 15))
    if [ "$waited" -ge "$hard" ]; then
      kill -9 "$pid" 2>/dev/null || true
      echo "(hard timeout ${hard}s at $have/28 rows)"; return 0
    fi
  done
  return 0
}

train_eval() {  # train_eval <seed> <config> <lam> <anchor-args...>
  local seed="$1" config="$2" lam="$3"; shift 3
  local tag="s${seed}_${config}"
  local dir="$CKPT_ROOT/$tag" ck="$CKPT_ROOT/$tag/ma_roma_ft.pth"
  if [ ! -f "$ck" ]; then
    echo "=== TRAIN seed=$seed $config (lambda=$lam) ==="
    python scripts/finetune_ma_roma.py --seed "$seed" --out-dir "$dir" \
      --steps 1500 --val-every 100 --num-workers 4 \
      --log "results/ft_seed_log_${tag}.csv" "$@"
    [ -f "$ck" ] || { echo "NO CKPT for $tag"; exit 1; }
  else
    echo "=== ckpt exists for $tag, skip train ==="
  fi
  for mode in $MODES; do
    if have_row "$seed" "$config" "$mode"; then
      echo "=== have $tag/$mode, skip eval ==="; continue
    fi
    local ec="results/seed_eval_${tag}_${mode}.csv"
    local hard=1800; [ "$mode" = "pyramid_v2" ] && hard=3600
    echo "=== EVAL $tag mode=$mode ==="
    run_eval_until_done "$ec" "$hard" -- python scripts/run_baselines_A.py \
      --backbones ma_roma_ft --mode "$mode" --ft-weights "$ck" \
      --only-pairs results/split.json --only-pairs-key test --out "$ec"
    python scripts/seed_expand_metrics.py --eval-csv "$ec" --seed "$seed" \
      --config "$config" --lam "$lam" --mode "$mode" --summary "$SUMMARY"
  done
}

for seed in $SEEDS; do
  train_eval "$seed" plain 0    --anchor none
  train_eval "$seed" l2sp  0.01 --anchor l2sp --anchor-lambda 0.01
done

echo ""
echo "=== results/seed_expand_summary.csv ==="
cat "$SUMMARY"
rm -f /root/README_CMA_SWEEP.txt
echo "SEED EXPAND COMPLETE"
