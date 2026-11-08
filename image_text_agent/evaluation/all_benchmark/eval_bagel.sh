#!/usr/bin/env bash
set -x
T=`date +%Y%m%d_%H%M%S`   # 获取时间戳

# 参数格式: eval_bagel.sh <GPUS> <TASK1> [TASK2] [TASK3] ...
# 可选环境变量: ANCHOR_GEN=1
GPUS=$1
shift 1
task_names="$@"

NODES=1

# 直接使用 thinkmorph_eval 环境的 python，避免 PATH 优先级问题
PYTHON=/mnt/afs/nijingcheng/miniconda3/envs/thinkmorph_eval/bin/python
which $PYTHON
$PYTHON -c "import torch; print('torch:', torch.__version__); import flash_attn; print('flash_attn:', flash_attn.__version__)"

PROJECT_ROOT=/mnt/afs/ningzhiwei/code/ThinkMorph/image_text_agent # TODO 项目根目录，需要修改成自己的路径

STEPS=5000
model_name=recon_cfg_1_4_2_${STEPS}  # TODO 模型结果保存目录，需要修改成保留的目录
export MODEL_PATH=/mnt/afs/ningzhiwei/code/ThinkMorph/RUN/MODIFY_RECON_CFG_v1_4_2_20260301_001328/ckpt/0005000

work_dir=/mnt/afs/ningzhiwei/code/ThinkMorph/image_text_agent
# VSP_maze_task_main_original VisPuzzle ChartQA_h_bar ChartQA_v_bar VStarBench BLINK_Jigsaw MMVP BLINK SAT_circular CV-Bench-2D CV-Bench-3D
export PYTHONPATH=${PROJECT_ROOT}:$PYTHONPATH
export PYTHONPATH=${PROJECT_ROOT}/tools:$PYTHONPATH
export SAVE_DIR=${PROJECT_ROOT}/evaluation/Results/${model_name}/all_benchmark

# export MODEL_PATH=/mnt/afs/multimodal_ckpt/nijingcheng/thinkmorph/realunify_mt_v8/ckpt/000${STEPS}   # TODO 评测的模型的路径，需要修改成自己的模型ckpt目录

if (( $NODES <= 1 )); then
    MASTER_PORT=$((12000 + $RANDOM % 20000))
    export WORLD_SIZE=1
    export RANK=0
    export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
fi

mkdir -p $SAVE_DIR
logpath=${SAVE_DIR}/log_eval_${T}.log

STYLE_ARGS=""
if [ "${ANCHOR_GEN}" = "1" ]; then
    STYLE_ARGS="${STYLE_ARGS} --anchor_gen"
fi

cd ${PROJECT_ROOT}/evaluation/VLMEvalKit_Thinkmorph        # thinkmorph测评目录
$PYTHON -m torch.distributed.run --nnodes=$WORLD_SIZE --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT --nproc_per_node=$GPUS  \
    ${work_dir}/evaluation/all_benchmark/run_bagel.py \
    --model_name ${model_name} \
    --model_path ${MODEL_PATH} \
    --tasks ${task_names} \
    --result_path $SAVE_DIR \
    ${STYLE_ARGS}
