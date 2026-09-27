#!/usr/bin/env bash
# CJSJ triage study: set up a fresh vast.ai box and generate all candidate registrations.
# Assumes the onstart command already began downloading the dataset to /root/AmalgaMatch_Dataset.zip.
# Run inside tmux:  tmux new-session -d -s tri 'bash /root/box_triage.sh > /root/tri.log 2>&1'
set -euo pipefail

cd /root
[ -d cma ] || git clone -q -b cjsj-tta https://github.com/fronkt/correlative-microscopy-alignment.git cma
cd /root/cma
git pull -q --ff-only || true
pip install -q -e ".[dev]" kornia romatch transformers pillow scipy scikit-image
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

ZIP=/root/AmalgaMatch_Dataset.zip
EXPECTED=4228037938
until [ "$(stat -c%s "$ZIP" 2>/dev/null || echo 0)" -eq "$EXPECTED" ]; do
  echo "waiting for dataset: $(stat -c%s "$ZIP" 2>/dev/null || echo 0)/$EXPECTED"; sleep 15
done
mkdir -p data/AmalgaMatch
python -m zipfile -e "$ZIP" data/AmalgaMatch
rm -f "$ZIP"
python -m pytest -q tests/test_triage.py

mkdir -p results/triage
# Two processes share the GPU; separate CSVs so appends never interleave.
python scripts/run_triage_candidates.py --pools gt,core,transform,control \
  --backbones sift,loftr,matchanything,roma --out results/triage/cand_a.csv > results/triage/run_a.log 2>&1 &
PA=$!
python scripts/run_triage_candidates.py --pools core,transform,control \
  --backbones ma_roma --out results/triage/cand_b.csv > results/triage/run_b.log 2>&1 &
PB=$!
wait $PA; echo "A exit $?"
wait $PB; echo "B exit $?"
echo "TRIAGE RUN COMPLETE"
