#!/bin/bash
export OPENAI_API_KEY=${OPENAI_API_KEY:-YOUR_OPENAI_API_KEY}
PROJECT_ROOT=/mnt/afs/interleaved_data/ningzhiwei/code/ThinkMorph/image_text_agent
TOOLS_DIR="$PROJECT_ROOT/tools"
export PYTHONPATH="$TOOLS_DIR:$PYTHONPATH"
model_name='qwen3vl'
result_path=$PROJECT_ROOT/evaluation/Results/${model_name}
mkdir -p ${result_path}
workers=4
workdir=${PROJECT_ROOT}/evaluation/ChartQA/eval
cd ${workdir}
python test_benchmark.py \
    --model_name ${model_name} \
    --ip_list 10.119.30.87 10.119.30.64 10.119.30.71 10.119.30.57 \
    --result_path ${result_path} \
    --workers ${workers} \
    --max_turns 20 \
    --save_interim \
    2>&1 | tee ${result_path}/${model_name}_ChartQA.log
