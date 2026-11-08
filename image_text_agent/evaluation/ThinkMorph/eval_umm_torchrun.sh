#!/usr/bin/env bash
set -x
T=`date +%Y%m%d_%H%M%S`   # 获取时间戳

export PATH=/mnt/afs1/luojiapeng/miniconda3/bin:$PATH
source activate
conda activate /mnt/afs/nijingcheng/miniconda3/envs/thinkmorph_eval
which python

PROJECT_ROOT=/mnt/afs/tongwenwen1/workspace/interleave_thinking/image_text_agent # TODO 项目根目录，需要修改成自己的路径

export MODEL_PATH=/mnt/afs/tongwenwen1/workspace/interleave_thinking/ThinkMorph/RUN/d260118_debug_thinkmorph24k/ckpt/0004000   # TODO 评测的模型的路径，需要修改成自己的模型ckpt目录

model_name=d260118_debug_thinkmorph24k_ckpt_4000  # TODO 模型结果保存目录，需要修改成保留的目录

# VSP_maze_task_main_original VisPuzzle ChartQA_h_bar ChartQA_v_bar VStarBench BLINK_Jigsaw MMVP BLINK SAT_circular CV-Bench-2D CV-Bench-3D
task_name=$1
GPUS=$2

# 所有模型结果保留的根目录 ${PROJECT_ROOT}/evaluation/Results
export SAVE_DIR=${PROJECT_ROOT}/evaluation/Results/VLMEvalKit_Thinkmorph/${model_name}/${task_name}
export LMUData=/mnt/afs/omni_data/evaluation/interleave_data/LMUData

mkdir -p $SAVE_DIR
logpath=${SAVE_DIR}/log_${task_name}_${T}.log

cd ${PROJECT_ROOT}/evaluation/VLMEvalKit_Thinkmorph        # thinkmorph测评目录

echo "GPUS: ${GPUS}"

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-10086}
RANK=${RANK:-0} # srun env node rank
WORLD_SIZE=${WORLD_SIZE:-1} # srun env node num

torchrun --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nproc_per_node=$GPUS  run.py \
    --data  ${task_name} \
    --model thinkmorph \
    --judge gpt-4.1 \
    --judge-args '{"use_azure": true}' \
    --work-dir $SAVE_DIR \