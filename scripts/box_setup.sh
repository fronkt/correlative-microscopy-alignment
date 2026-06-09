#!/usr/bin/env bash
# One-shot setup for a vast.ai box (RTX 5090, /venv/main has torch preinstalled).
# Usage: bash box_setup.sh
set -euo pipefail

source /venv/main/bin/activate

cd /root/cma
uv pip install -q -e ".[dev]" kornia romatch transformers pillow

python - <<'EOF'
import torch, cma, kornia, romatch, transformers
print("torch", torch.__version__, "cuda ok:", torch.cuda.is_available())
print("imports ok")
EOF

# Wait for the dataset download if still running, then extract and drop the zip
ZIP=/workspace/AmalgaMatch_Dataset.zip
EXPECTED=4228037938
while [ "$(stat -c%s "$ZIP" 2>/dev/null || echo 0)" -lt "$EXPECTED" ]; do
  echo "waiting for download: $(stat -c%s "$ZIP" 2>/dev/null || echo 0)/$EXPECTED"
  sleep 20
done

echo "$(sha256sum "$ZIP")"
mkdir -p /root/cma/data/AmalgaMatch
unzip -q -o "$ZIP" -d /root/cma/data/AmalgaMatch
rm "$ZIP"
df -h / | tail -1

cd /root/cma && python -m pytest -q --tb=short
echo "SETUP COMPLETE"
