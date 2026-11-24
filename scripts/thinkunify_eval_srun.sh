

#!/bin/bash

set -x

# ================= General Understanding Benchmark Eval =================
# Pure understanding tasks: AI2D, ChartQA (no image generation).
# Mirrors scripts/thinkmorph_eval.sh structure. Accepts the same config
# env vars (MODEL_PATH override, ENABLE_QFORMER_DISTILL, etc.).

# 1. Core model path (override via env var MODEL_PATH when needed).
#    Point this at your trained checkpoint directory, e.g. .../ckpt/0010000
# MODEL_PATH="${MODEL_PATH:-YOUR_MODEL_PATH/ckpt/xxxstep}"
MODEL_PATH="${MODEL_PATH:-/apdcephfs_zwfy12/share_303204533/lukening/code/ThinkUnify/OUTPUT/model_ckpt_output/ckpt/0004000}"
BAGEL_MODEL_PATH="${BAGEL_MODEL_PATH:-/apdcephfs_zwfy12/share_303204533/lukening/model/BAGEL-7B-MoT}"
EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-/apdcephfs_cq5/share_300600172/lukening/data/ThinkUnify_eval}"
# 2. Base config: GPUs per node / node count / run timestamp
GPUS=${GPUS:-8}
NODES=${NODES:-1}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-${REPO_ROOT}/image_text_agent}"

# 4. Tasks (default: the three open-sourced benchmarks; override by passing as args)
ALL_TASKS="MIRA GenEval MathVista"
TASKS="${@:-${ALL_TASKS}}"

export ANCHOR_GEN=1
NUM_TIMESTEPS=${NUM_TIMESTEPS:-30}

# 6. CFG-Zero* + APG (generation-only, rarely relevant for understanding)


# 7. Q-Former (important for distilled understanding models), enabled in Train and disabled in Eval
export ENABLE_QFORMER_DISTILL=${ENABLE_QFORMER_DISTILL:-0}
export QFORMER_NUM_LAYERS=${QFORMER_NUM_LAYERS:-4}
export QFORMER_NUM_HEADS=${QFORMER_NUM_HEADS:-16}
export QFORMER_ALPHA=${QFORMER_ALPHA:-1.0}
export QFORMER_VERSION=${QFORMER_VERSION:-"v2"}
export QFORMER_NUM_CA_BLOCKS=${QFORMER_NUM_CA_BLOCKS:-4}
export QFORMER_NUM_SA_BLOCKS=${QFORMER_NUM_SA_BLOCKS:-9}

# ================= Auto parsing =================
CKPT_STEP=$(basename "${MODEL_PATH}")
EXP_NAME=$(basename "$(dirname "$(dirname "${MODEL_PATH}")")")
MODEL_NAME="${EXP_NAME}_step${CKPT_STEP}_EVAL"
# Exported so the child process (run_bagel.py / ThinkMorph.py) picks it up via
# os.environ.get('BAGEL_CONFIG_PATH'); otherwise it falls back to a placeholder.
export BAGEL_CONFIG_PATH="${BAGEL_MODEL_PATH}"
export EVAL_DATA_ROOT="${EVAL_DATA_ROOT}"

THINKUNIFY_ROOT=$(dirname "$PROJECT_ROOT")
SAVE_DIR="${THINKUNIFY_ROOT}/RUN/${MODEL_NAME}"
LOG_FILE="${THINKUNIFY_ROOT}/RUN/${MODEL_NAME}/eval_${MODEL_NAME}_${TIMESTAMP}.log"

echo "=========================================="
echo "Experiment Name : $EXP_NAME"
echo "Checkpoint Step : $CKPT_STEP"
echo "Tasks           : $TASKS"
echo "Q-Former        : ALPHA=${QFORMER_ALPHA}"
echo "Num Timesteps   : $NUM_TIMESTEPS"
echo "Model Name ID   : $MODEL_NAME"
echo "Save Directory  : $SAVE_DIR"
echo "Log File Path   : $LOG_FILE"
echo "=========================================="

mkdir -p "$SAVE_DIR"

# ================= Environment =================
# Path to the python interpreter of your evaluation conda env.
# PYTHON_EXE="${PYTHON_EXE:-thinkunify_eval_env}"
PYTHON_EXE="${PYTHON_EXE:-/apdcephfs_zwfy12/share_303204533/lukening/conda/envs/thinkunify-eval/bin/python}"
echo "Using Python: $PYTHON_EXE"

export PYTHONPATH=${PROJECT_ROOT}:$PYTHONPATH
export PYTHONPATH=${PROJECT_ROOT}/tools:$PYTHONPATH

# ================= NCCL timeout backstop =================
# Interleaved multi-round generation has highly uneven per-sample latency; a
# slow shard can stall the cross-benchmark dist.barrier(). Give NCCL a long
# timeout so the whole process group is not killed while a slow rank catches up.
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-18000}          # seconds (5h), matches run_bagel.py
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# ================= Efficiency profiling switch =================
# Set EVAL_PROFILE=1 to print per-task latency / throughput / peak GPU memory
# (per-rank and a global aggregate). Default 0 = off.
export EVAL_PROFILE=${EVAL_PROFILE:-1}

if (( $NODES <= 1 )); then
    MASTER_PORT=$((12000 + $RANDOM % 20000))
    # MASTER_PORT=${MASTER_PORT:-53078}
    export WORLD_SIZE=1
    export RANK=0
    export MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
fi

# ================= Build extra args =================
STYLE_ARGS=""
if [ "${ANCHOR_GEN}" = "1" ]; then
    STYLE_ARGS="${STYLE_ARGS} --anchor_gen"
fi

CFG_ARGS=""
if [ -n "${CFG_ZERO_INIT_STEPS}" ]; then
    CFG_ARGS="${CFG_ARGS} --cfg_zero_init_steps ${CFG_ZERO_INIT_STEPS}"
fi
if [ -n "${CFG_ZERO_OPTIMAL_RESCALE}" ]; then
    CFG_ARGS="${CFG_ARGS} --cfg_zero_optimal_rescale ${CFG_ZERO_OPTIMAL_RESCALE}"
fi
if [ -n "${CFG_GUIDANCE_MODE}" ]; then
    CFG_ARGS="${CFG_ARGS} --cfg_guidance_mode ${CFG_GUIDANCE_MODE}"
fi
if [ -n "${APG_TEXT_ETA}" ]; then
    CFG_ARGS="${CFG_ARGS} --apg_text_eta ${APG_TEXT_ETA}"
fi
if [ -n "${APG_IMG_ETA}" ]; then
    CFG_ARGS="${CFG_ARGS} --apg_img_eta ${APG_IMG_ETA}"
fi

QFORMER_ARGS=""
if [ -n "${QFORMER_NUM_LAYERS}" ]; then
    QFORMER_ARGS="${QFORMER_ARGS} --qformer_num_layers ${QFORMER_NUM_LAYERS}"
fi
if [ -n "${QFORMER_NUM_HEADS}" ]; then
    QFORMER_ARGS="${QFORMER_ARGS} --qformer_num_heads ${QFORMER_NUM_HEADS}"
fi
if [ -n "${QFORMER_ALPHA}" ]; then
    QFORMER_ARGS="${QFORMER_ARGS} --qformer_alpha ${QFORMER_ALPHA}"
fi
if [ -n "${QFORMER_VERSION}" ]; then
    QFORMER_ARGS="${QFORMER_ARGS} --qformer_version ${QFORMER_VERSION}"
fi
if [ -n "${QFORMER_NUM_CA_BLOCKS}" ]; then
    QFORMER_ARGS="${QFORMER_ARGS} --qformer_num_ca_blocks ${QFORMER_NUM_CA_BLOCKS}"
fi
if [ -n "${QFORMER_NUM_SA_BLOCKS}" ]; then
    QFORMER_ARGS="${QFORMER_ARGS} --qformer_num_sa_blocks ${QFORMER_NUM_SA_BLOCKS}"
fi
echo "Starting general understanding evaluation for tasks: $TASKS"

cd "${PROJECT_ROOT}/evaluation/VLMEvalKit_Thinkmorph" || { echo "Directory not found: VLMEvalKit_Thinkmorph"; exit 1; }

$PYTHON_EXE -m torch.distributed.run \
    --nnodes=$WORLD_SIZE \
    --node_rank=$RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --nproc_per_node=$GPUS \
    "${PROJECT_ROOT}/evaluation/all_benchmark/run_bagel.py" \
    --model_name "${MODEL_NAME}" \
    --model_path "${MODEL_PATH}" \
    --tasks ${TASKS} \
    --result_path "$SAVE_DIR" \
    --num_timesteps ${NUM_TIMESTEPS} \
    ${STYLE_ARGS} ${CFG_ARGS} ${QFORMER_ARGS} 2>&1 | tee "$LOG_FILE"

echo "All tasks finished."
echo "Results saved in: $SAVE_DIR"