#!/bin/bash
# Download ChartQA dataset from HuggingFace
# Dataset: HuggingFaceM4/ChartQA

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data"

python -c "
from huggingface_hub import snapshot_download
snapshot_download('HuggingFaceM4/ChartQA', repo_type='dataset', local_dir='${DATA_DIR}')
print('Download complete. Data saved to: ${DATA_DIR}')
"
