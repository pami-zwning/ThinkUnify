#!/bin/bash
# Download MathVerse dataset from HuggingFace
# Dataset: AI4Math/MathVerse

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../data"

python -c "
from huggingface_hub import snapshot_download
snapshot_download('AI4Math/MathVerse', repo_type='dataset', local_dir='${DATA_DIR}')
print('Download complete. Data saved to: ${DATA_DIR}')
"
