#!/bin/bash
# Download AI2D dataset from HuggingFace
# Dataset: lmms-lab/ai2d

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data"

python -c "
from huggingface_hub import snapshot_download
snapshot_download('lmms-lab/ai2d', repo_type='dataset', local_dir='${DATA_DIR}')
print('Download complete. Data saved to: ${DATA_DIR}')
"
