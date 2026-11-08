#!/bin/bash
# Download BLINK Benchmark from HuggingFace
# Dataset: BLINK-Benchmark/BLINK (14 subtasks)

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data"

python -c "
from huggingface_hub import snapshot_download
snapshot_download('BLINK-Benchmark/BLINK', repo_type='dataset', local_dir='${DATA_DIR}')
print('Download complete. Data saved to: ${DATA_DIR}')
"
