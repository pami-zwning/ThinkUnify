#!/bin/bash
# Download MMMU benchmark dataset from HuggingFace
# Dataset: MMMU/MMMU (30 subject configs)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data"

python -c "
from huggingface_hub import snapshot_download
snapshot_download('MMMU/MMMU', repo_type='dataset', local_dir='${DATA_DIR}')
print('Download complete. Data saved to: ${DATA_DIR}')
"
