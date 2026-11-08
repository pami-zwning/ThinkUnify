#!/usr/bin/env bash
set -x

model_name=qwen3vl_base_notool
PROJECT_ROOT=/mnt/afs/nijingcheng/workspace/codes/benchmarks/image_text_agent #改成自己的路径
TOOLS_DIR="$PROJECT_ROOT/tools"

export PATH=/mnt/afs1/luojiapeng/miniconda3/bin:$PATH
source activate
conda activate /mnt/afs/nijingcheng/miniconda3/envs/thinkmorph_eval

export PYTHONPATH="$TOOLS_DIR:$PYTHONPATH"

export RESULT_PATH=$PROJECT_ROOT/evaluation/Results/${model_name}
export MODEL_NAME=$model_name
export IP_LIST="10.119.29.229"
export SAVE_INTERIM=TRUE
export MAX_TURNS=10
export NEED_SUMMARY=FALSE
export USE_TOOLS=FALSE

# VisPuzzle VStarBench BLINK_Jigsaw MMVP BLINK SAT_circular CV-Bench-2D CV-Bench-3D
cd $PROJECT_ROOT/evaluation/VLMEvalKit_Thinkmorph
python run.py \
--data BLINK_Jigsaw \
--model qwen3vl_base_notool \
--judge gpt-4.1 \
--judge-args '{"use_azure": true}' \
--work-dir ${RESULT_PATH}/base