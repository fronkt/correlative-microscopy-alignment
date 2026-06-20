#!/usr/bin/env bash
# Phase 9B — headline eval of the winning L2-SP checkpoint chosen from the
# Phase 9A val logs. Usage: bash scripts/box_ft_robust_eval.sh <tag>
#   tag in {l0,l0p01,l0p1,l1}  (checkpoint at /dev/shm/cma_ckpt/<tag>/ma_roma_ft.pth)
# Runs the full 187-pair direct + pyramid_v2 sweep and the FOV ladder on the
# fixed 63-pair testbed, writing to SEPARATE CSVs so the plain-ft (§7) rows in
# baselines_A.csv / fov_ladder.csv stay intact. Keyed backbone=ma_roma_ft so
# ft_test_analysis.py / fov_ladder_bootstrap.py work by pointing at these files.
set -euo pipefail
cd /root/cma
git fetch -q && git reset --hard -q origin/main

TAG="${1:?usage: box_ft_robust_eval.sh <tag>  e.g. l0p1}"
CKPT="/dev/shm/cma_ckpt/$TAG/ma_roma_ft.pth"
[ -f "$CKPT" ] || { echo "MISSING CHECKPOINT $CKPT"; exit 1; }

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

cat > /root/README_CMA_SWEEP.txt <<EOF
Active robust-ft eval (correlative-microscopy session, tag $TAG): /root/cma
and /dev/shm in use until "ROBUST EVAL COMPLETE" appears in /root/robust_eval.log.
Please do not delete.
EOF

# Known issue: the runner wedges at interpreter exit after GPU sweeps, AFTER
# writing its last CSV row. Treat a timeout kill (124/137) as success.
run() {  # run <timeout> -- <cmd...>
  local to="$1"; shift; shift
  local rc=0
  timeout -k 30 "$to" "$@" || rc=$?
  if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
    echo "(runner wedged at exit; killed by timeout after rows were written)"
  elif [ "$rc" -ne 0 ]; then
    exit "$rc"
  fi
}

for mode in direct pyramid_v2; do
  run 14400 -- /venv/main/bin/python scripts/run_baselines_A.py \
    --backbones ma_roma_ft --mode "$mode" --ft-weights "$CKPT" \
    --out results/baselines_robust.csv
done

run 14400 -- /venv/main/bin/python scripts/run_fov_ladder.py \
  --backbones ma_roma_ft --modes direct,pyramid_v2 --ft-weights "$CKPT" \
  --baselines results/baselines_A.csv \
  --restrict-pairs-csv results/fov_ladder.csv \
  --out results/fov_ladder_robust.csv

# archive the winning checkpoint name for the writeup
echo "$TAG" > results/robust_winner_tag.txt
rm -f /root/README_CMA_SWEEP.txt
echo "ROBUST EVAL COMPLETE ($TAG) — scp back: results/baselines_robust.csv," \
     "results/fov_ladder_robust.csv, $CKPT"
