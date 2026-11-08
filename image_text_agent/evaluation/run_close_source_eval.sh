#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Close-source model evaluation for all 4 benchmarks
# Models: gemini-3-pro, gemini-3.1-pro, doubao-seed-2.0
# ============================================================

# --- Configuration ---
MODEL_KEY="${1:-gemini-3-pro}"   # Pass model key as first arg
WORKERS="${2:-16}"               # Pass worker count as second arg
RESULT_ROOT="./Results/${MODEL_KEY}"

echo "=== Close-source eval: ${MODEL_KEY} (workers=${WORKERS}) ==="

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EVAL_ROOT="${SCRIPT_DIR}"

# ============================================================
# 1) Uni-MMMU
# ============================================================
echo ""
echo ">>> [1/4] Uni-MMMU"
cd "${EVAL_ROOT}/Uni-MMMU"
python gemini_eval.py \
    --model_key "${MODEL_KEY}" \
    --dataset_root /mnt/afs/share_data/mashengnan/o3like/image_text_cot/benchmark/Uni-MMMU \
    --result_dir "${RESULT_ROOT}/Uni-MMMU" \
    --workers "${WORKERS}"

# Score
python eval_unimmmu.py \
    --run_root "${RESULT_ROOT}/Uni-MMMU" \
    --dataset_root /mnt/afs/share_data/mashengnan/o3like/image_text_cot/benchmark/Uni-MMMU \
    --modes direct \
    --workers "${WORKERS}" \
    --max_retries 2

# ============================================================
# 2) RealUnify
# ============================================================
echo ""
echo ">>> [2/4] RealUnify"
cd "${EVAL_ROOT}/RealUnify"
python gemini_eval.py \
    --model_key "${MODEL_KEY}" \
    --result_path "${RESULT_ROOT}/RealUnify" \
    --workers "${WORKERS}"

# Score
python score.py --input_file "${RESULT_ROOT}/RealUnify/Jsons/${MODEL_KEY}.json"

# ============================================================
# 3) MIRA
# ============================================================
echo ""
echo ">>> [3/4] MIRA"
cd "${EVAL_ROOT}/MIRA"
python gemini_eval.py \
    --model_key "${MODEL_KEY}" \
    --benchmark_dir /mnt/afs/share_data/mashengnan/o3like/visual_reasoning/MIRA \
    --output_dir "${RESULT_ROOT}/MIRA" \
    --workers "${WORKERS}"

# Score
python score.py -r "${RESULT_ROOT}/MIRA"

# ============================================================
# 4) VSP
# ============================================================
echo ""
echo ">>> [4/4] VSP"
cd "${EVAL_ROOT}/VSP"
python gemini_eval.py \
    --model_key "${MODEL_KEY}" \
    --data_path ./maze_task_main.jsonl \
    --result_path "${RESULT_ROOT}/VSP" \
    --workers "${WORKERS}" \
    --save_json

# Score (modify output path in eval.py before running)
# The eval.py reads from ./output/output_img/ by default.
# To score, either symlink or modify the check_answer_dir:
#   ln -sfn "${RESULT_ROOT}/VSP/${MODEL_KEY}" ./output/output_img
#   python eval.py

echo ""
echo "=== All benchmarks completed for ${MODEL_KEY} ==="
