#!/bin/bash
# Download HR-Bench dataset from HuggingFace (includes both 4K and 8K splits)
# Dataset: DreamMr/HR-Bench

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data"

python -c "
from huggingface_hub import snapshot_download
snapshot_download('DreamMr/HR-Bench', repo_type='dataset', local_dir='${DATA_DIR}')
print('Download complete. Data saved to: ${DATA_DIR}')
"
