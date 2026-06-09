#!/bin/bash
# Run the FOV sweep for each cross-modal mode in turn.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate
for mode in gamma smooth edge stack; do
    echo "=== cross-modal mode: $mode ==="
    python -W ignore scripts/run_fov_sweep.py \
        --n 8 \
        --methods classical,pyramid,pyramid_loftr \
        --cross-modal "$mode" \
        --out "results/fov_sweep_${mode}.csv"
    echo
done
