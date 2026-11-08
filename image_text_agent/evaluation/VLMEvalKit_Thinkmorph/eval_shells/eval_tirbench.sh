#!/usr/bin/env bash
set -euo pipefail
set -x

cat > /etc/apt/sources.list << 'EOF'
deb http://th.archive.ubuntu.com/ubuntu jammy main restricted universe multiverse
deb http://th.archive.ubuntu.com/ubuntu jammy-updates main restricted universe multiverse
deb http://th.archive.ubuntu.com/ubuntu jammy-security main restricted universe multiverse
EOF

apt update
apt install libc6 -y
######################################
# Conda 环境
######################################
ENV_PATH="/mnt/afs/kongxiangli/.conda/envs/bagel"

export PATH=/mnt/afs1/luojiapeng/miniconda3/bin:$PATH
source /mnt/afs1/luojiapeng/miniconda3/etc/profile.d/conda.sh
conda activate ${ENV_PATH}

######################################
# 参数解析
######################################
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <DATASET> <GPUS>"
    exit 1
fi

DATASET=$1        # benchmark 名称
GPUS=$2           # 使用 GPU 数量

echo "=========================================="
echo "Running evaluation"
echo "Dataset : ${DATASET}"
echo "GPUs    : ${GPUS}"
echo "=========================================="

######################################
# GPU / Runtime 环境
######################################
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPUS-1)))
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

# NCCL（单进程 + 模型并行，禁用即可）
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

echo "[INFO] CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"

######################################
# 路径设置
######################################
ROOT_DIR=/mnt/afs/kongxiangli/workspace/ThinkMorph
THISDIR=$(dirname $(realpath $0))
CKPT_DIR=${ROOT_DIR}/RUN/bagel_tir_internal_math_insection_got_260119/ckpt/0005000
OUTPUT_DIR=${CKPT_DIR}/eval/${DATASET}

INPUT_JSON=/mnt/afs/kongxiangli/workspace/tool_call/vlmevalkit/TIR-Bench/TIR-Bench.json

mkdir -p ${OUTPUT_DIR}
mkdir -p ${OUTPUT_DIR}/images

######################################
# 日志
######################################
LOG_FILE=${OUTPUT_DIR}/eval.log
exec > >(tee -a ${LOG_FILE}) 2>&1

echo "[INFO] Output dir : ${OUTPUT_DIR}"
echo "[INFO] Checkpoint : ${CKPT_DIR}"
echo "[INFO] Input JSON : ${INPUT_JSON}"

######################################
# 启动评测（单进程 / 多 GPU / 模型并行）
######################################
python -u ${THISDIR}/infer_tir_bench.py \
    --dataset ${DATASET} \
    --input_json ${INPUT_JSON} \
    --output_json ${OUTPUT_DIR}/results.json \
    --output_dir ${OUTPUT_DIR}/images \
    --gpus ${GPUS}

######################################
# 完成
######################################
echo "=========================================="
echo "[DONE] Evaluation finished for ${DATASET}"
echo "Results saved to:"
echo "  ${OUTPUT_DIR}/results.json"
echo "=========================================="
