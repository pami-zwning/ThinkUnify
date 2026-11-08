#!/bin/bash
# Phase 4 Training Script: Q-Former VAE-to-Understanding Distillation
#
# Replaces SigLIP ViT encoder with a Q-Former that learns to extract
# understanding features from VAE latents via cross-attention distillation.
#
# Stage 1: Alpha gradual transition (ViT + Q-Former blended)
# Stage 2: ViT removal (alpha=1.0, Q-Former only) - happens automatically
#           in the last 20% of training steps

echo "starting Phase 4 training: Q-Former VAE-to-Understanding Distillation"
set -ex

# cat > /etc/apt/sources.list << 'EOF'
# deb http://th.archive.ubuntu.com/ubuntu jammy main restricted universe multiverse
# deb http://th.archive.ubuntu.com/ubuntu jammy-updates main restricted universe multiverse
# deb http://th.archive.ubuntu.com/ubuntu jammy-security main restricted universe multiverse
# EOF

# apt update
# apt install libc6 -y

############################
# Q-Former Configuration
############################
ENABLE_RECON_ALIGNMENT=${ENABLE_RECON_ALIGNMENT:-'True'}
RECON_GRADIENT_WEIGHT=${RECON_GRADIENT_WEIGHT:-'0.5'}
RECON_SPECTRAL_WEIGHT=${RECON_SPECTRAL_WEIGHT:-'0.1'}
ENABLE_QFORMER_DISTILL=${ENABLE_QFORMER_DISTILL:-'True'}
QFORMER_NUM_LAYERS=${QFORMER_NUM_LAYERS:-4}
QFORMER_NUM_HEADS=${QFORMER_NUM_HEADS:-16}
DISTILL_LOSS_WEIGHT=${DISTILL_LOSS_WEIGHT:-1.0}
QFORMER_VERSION=${QFORMER_VERSION:-'v2'}
QFORMER_NUM_CA_BLOCKS=${QFORMER_NUM_CA_BLOCKS:-4}
QFORMER_NUM_SA_BLOCKS=${QFORMER_NUM_SA_BLOCKS:-8}
QFORMER_SA_WARMUP_STEPS=${QFORMER_SA_WARMUP_STEPS:-0}

N_COPY=${N_COPY:-3}

############################
# SCFM Shortcut Distillation (optional)
############################
ENABLE_SCFM=${ENABLE_SCFM:-'True'}
SCFM_T_SKIP=${SCFM_T_SKIP:-2}
SCFM_SHORTCUT_RATIO=${SCFM_SHORTCUT_RATIO:-0.5}
SCFM_NUM_EULER_STEPS=${SCFM_NUM_EULER_STEPS:-50}
SCFM_SLOW_EMA_DECAY=${SCFM_SLOW_EMA_DECAY:-0.999}
SCFM_SHORTCUT_LOSS_WEIGHT=${SCFM_SHORTCUT_LOSS_WEIGHT:-1.0}
SCFM_LORA_RANK=${SCFM_LORA_RANK:-64}
SCFM_LORA_ALPHA=${SCFM_LORA_ALPHA:-128}

# DATA_YAML=${DATA_YAML:-'v1_4_2'}
# DATA_YAML=${DATA_YAML:-'v1_6_2'}
# DATA_YAML=${DATA_YAML:-'three_benchmark_v1_original'}
DATA_YAML=${DATA_YAML:-'three_benchmark_v1_af_amp'}
# DATA_YAML=${DATA_YAML:-'three_benchmark_v1_split10'}
# DATA_YAML=${DATA_YAML:-'three_benchmark_v1_split5'}
# DATA_YAML=${DATA_YAML:-'three_benchmark_v1_split2'}
# DATA_YAML=${DATA_YAML:-'three_benchmark_v2'}
# DATA_YAML=${DATA_YAML:-'three_benchmark_v2_S'}
T=$(date +%Y%m%d_%H%M%S)

# Auto-generate experiment tag from enabled features
EXP_TAG=""
[ "${ENABLE_RECON_ALIGNMENT}" = "True" ] && EXP_TAG="${EXP_TAG}_recon"
[ "${ENABLE_QFORMER_DISTILL}" = "True" ] && EXP_TAG="${EXP_TAG}_qformer"
[ "${N_COPY}" -gt 1 ] 2>/dev/null && EXP_TAG="${EXP_TAG}_${N_COPY}copy"
[ "${QFORMER_NUM_LAYERS}" -gt 4 ] 2>/dev/null && EXP_TAG="${EXP_TAG}_${QFORMER_NUM_LAYERS}layers"
[ "${ENABLE_SCFM}" = "True" ] && EXP_TAG="${EXP_TAG}_scfm"
EXP_TAG="${EXP_TAG:-_base}"  # fallback if nothing enabled
EXP_NAME="${DATA_YAML}_EMA_${EXP_TAG}"

############################
# 0) Environment Activation
############################
# Path to your conda environment (torch 2.5.1 / transformers 4.49 / flash-attn).
ENV_PATH=${ENV_PATH:-'YOUR_CONDA_ENV_PATH'}
export GPU_PER_NODE=${GPU_PER_NODE:-8}
export NNODES=${NNODES:-1}
export RANK=${RANK:-0}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}

shift $#

source /opt/conda/etc/profile.d/conda.sh
conda activate ${ENV_PATH}
# If you hit an nvjitlink load error, point LD_LIBRARY_PATH at your env's
# nvidia/nvjitlink/lib directory (optional):
# export LD_LIBRARY_PATH=${ENV_PATH}/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

which python3
python3 -V
which torchrun
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
############################
# 1) Code & Output Paths
############################
# Repo root (this cloned repository).
code_path=${CODE_PATH:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"}
# Base BAGEL-7B-MoT weights/config dir. See README "Model Download".
MODEL_PATH=${MODEL_PATH:-'YOUR_BAGEL_7B_MOT_PATH'}
# Checkpoint to resume/finetune from (e.g. BAGEL-7B-MoT).
resume_from=${RESUME_FROM:-'YOUR_RESUME_CKPT_PATH'}
CONFIG_PATH=${CONFIG_PATH:-"${code_path}/data/configs/${DATA_YAML}.yaml"}
# Checkpoints / model output (large; put on fast storage).
output_path=${OUTPUT_PATH:-"${code_path}/OUTPUT/${EXP_NAME}_${T}"}
# Wandb + log.txt (keep logs separate from large checkpoints if desired).
log_path=${LOG_PATH:-"${code_path}/OUTPUT/${EXP_NAME}_${T}"}

export PYTHONPATH=$code_path:$PYTHONPATH

############################
# NCCL / Distributed Timeouts
# （rank0 从 cephfs 加载 可能 > 10 分钟，默认 watchdog 会把其他 rank 打死）
############################
export TORCH_NCCL_BLOCKING_WAIT=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=7
# 打印 NCCL 调试信息到日志，后续排查用（嫌太吵可以删）
# export NCCL_DEBUG=INFO

############################
# 2) W&B Configuration
############################
export WANDB_MODE=offline
export WANDB_ERROR_REPORTING=false   # 禁用 sentry 错误上报（离线机器必须）
unset WANDB_API_KEY                  # 离线模式不需要 API key
USE_WANDB=true
export WANDB_PROJECT="bagel_zwning_adaln"
export WANDB_DIR="$log_path/wandb"
mkdir -p "$WANDB_DIR"

# ================= 核心修改位置 =================
# 将 wandb_name 设为 output_path 的最后一个文件夹名
# 使用 bash 字符串截取 ${output_path##*/} 
# ===============================================

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
python3 -m torch.distributed.run \
  --nnodes=$NNODES \
  --node_rank=$RANK \
  --nproc_per_node=$GPU_PER_NODE \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  $code_path/train/pretrain_unified_navit.py \
  --dataset_config_file $CONFIG_PATH \
  --layer_module Qwen2MoTDecoderLayer \
  --finetune_from_hf True \
  --finetune_from_ema True \
  --auto_resume True \
  --resume_from $resume_from \
  --results_dir $log_path \
  --checkpoint_dir $output_path/ckpt \
  --lr 1e-5 \
  --num_worker 4 \
  --max_latent_size 64  \
  --max_num_tokens 24576 \
  --expected_num_tokens 24576 \
  --mse_weight 4 \
  --ce_weight 1 \
  --total_steps 10000 \
  --save_every 2000 \
  --warmup_steps 500 \
  --model_path $MODEL_PATH \
  --num_shard $GPU_PER_NODE \
  --wandb_project "ThinkUnify-adaln-recon" \
  --wandb_name "${output_path##*/}" \
  --wandb_offline True \
  --resume_model_only True \
  --num_replicate $NNODES \
  --sharding_strategy FULL_SHARD \
  --enable_reconstruction_alignment $ENABLE_RECON_ALIGNMENT \
  --recon_gradient_weight $RECON_GRADIENT_WEIGHT \
  --recon_spectral_weight $RECON_SPECTRAL_WEIGHT \
  --enable_qformer_distill $ENABLE_QFORMER_DISTILL \
  --qformer_num_layers $QFORMER_NUM_LAYERS \
  --qformer_num_heads $QFORMER_NUM_HEADS \
  --distill_loss_weight $DISTILL_LOSS_WEIGHT \
  --qformer_version $QFORMER_VERSION \
  --qformer_num_ca_blocks $QFORMER_NUM_CA_BLOCKS \
  --qformer_num_sa_blocks $QFORMER_NUM_SA_BLOCKS \
  --qformer_sa_warmup_steps $QFORMER_SA_WARMUP_STEPS \
  --n_copy $N_COPY \
  --disable_ema True \
  --enable_scfm $ENABLE_SCFM \
  --scfm_t_skip $SCFM_T_SKIP \
  --scfm_shortcut_ratio $SCFM_SHORTCUT_RATIO \
  --scfm_num_euler_steps $SCFM_NUM_EULER_STEPS \
  --scfm_slow_ema_decay $SCFM_SLOW_EMA_DECAY \
  --scfm_shortcut_loss_weight $SCFM_SHORTCUT_LOSS_WEIGHT \
  --scfm_lora_rank $SCFM_LORA_RANK \
  --scfm_lora_alpha $SCFM_LORA_ALPHA