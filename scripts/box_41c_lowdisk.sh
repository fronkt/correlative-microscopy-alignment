#!/usr/bin/env bash
# Setup + 4.1c sweeps for a low-disk (16G) vast.ai box: the dataset zip is
# streamed straight into extraction so it never occupies disk.
set -euo pipefail

command -v bsdtar >/dev/null 2>&1 || {
  apt-get update -qq && apt-get install -y -qq libarchive-tools
}

if [ ! -d /root/cma ]; then
  git clone -q https://github.com/fronkt/correlative-microscopy-alignment.git /root/cma
fi
cd /root/cma

source /venv/main/bin/activate
uv pip install -q -e ".[dev]" kornia romatch transformers pillow

python - <<'EOF'
import torch, cma, kornia, romatch, transformers
print("torch", torch.__version__, "cuda ok:", torch.cuda.is_available())
print("imports ok")
EOF

if [ ! -d data/AmalgaMatch/CoNi-AM67_OM-SEM_Multiscale ]; then
  mkdir -p data/AmalgaMatch
  wget -q -O - \
    https://fordatis.fraunhofer.de/bitstream/fordatis/478/1/AmalgaMatch_Dataset.zip \
    | bsdtar -xf - -C data/AmalgaMatch
fi
df -h / | tail -1

python -m pytest -q --tb=short

# Resume logic in the runner skips rows already in the committed CSV.
python scripts/run_baselines_A.py \
  --backbones roma --mode pyramid_v2 --tag z3 --out results/baselines_A.csv
python scripts/run_baselines_A.py \
  --backbones roma --mode pyramid_v2 --tag c50 --certainty 0.5 \
  --out results/baselines_A.csv
echo "41C SWEEPS COMPLETE"
