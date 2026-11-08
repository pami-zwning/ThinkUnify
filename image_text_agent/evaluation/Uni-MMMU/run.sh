PROJECT_ROOT=/mnt/afs/mashengnan/o3like/Test_Code/image_text_agent #改成自己的路径
TOOLS_DIR="$PROJECT_ROOT/tools"
export PYTHONPATH="$TOOLS_DIR:$PYTHONPATH"

#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 1) TOOL-CALL RUN
# ============================================================

model_name=qwen_test

python qwen3vl_tool_call.py \
  --result_dir ./Results/${model_name} \
  --ip_list 10.119.18.176 \
  --workers 2 \
  --max_turns 10 \
  --save_images

# ============================================================
# 2) DIRECT RUN
# ============================================================
python qwen3vl_direct.py \
  --result_dir ./Results/${model_name} \
  --ip_list 10.119.18.176 \
  --workers 2

# ============================================================
# 3) EVAL (direct + tool_call)
# ============================================================
python eval_unimmmu.py \
  --run_root ./Results/${model_name} \
  --dataset_root /mnt/afs/share_data/mashengnan/o3like/image_text_cot/benchmark/Uni-MMMU \
  --modes direct tool_call \
  --workers 6 \
  --max_retries 2
