#!/bin/bash
# Download RealWorldQA dataset from HuggingFace
# Dataset: xai-org/RealWorldQA

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data"

python -c "
from huggingface_hub import snapshot_download
snapshot_download('xai-org/RealWorldQA', repo_type='dataset', local_dir='${DATA_DIR}')
print('Download complete. Data saved to: ${DATA_DIR}')
"
