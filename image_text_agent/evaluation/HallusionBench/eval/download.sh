#!/bin/bash
# Download HallusionBench dataset from HuggingFace
# Dataset: lmms-lab/HallusionBench

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data"

python -c "
from huggingface_hub import snapshot_download
snapshot_download('lmms-lab/HallusionBench', repo_type='dataset', local_dir='${DATA_DIR}')
print('Download complete. Data saved to: ${DATA_DIR}')
"
