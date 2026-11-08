# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import functools
import gc
import os
import wandb
import yaml
from copy import deepcopy
from dataclasses import dataclass, field
from time import time

import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.utils.data import DataLoader
from transformers import HfArgumentParser, set_seed
from transformers.optimization import (
    get_constant_schedule_with_warmup,
    get_cosine_with_min_lr_schedule_with_warmup,
)

from data.dataset_base import DataConfig, PackedDataset, collate_wrapper
from data.data_utils import add_special_tokens
from modeling.autoencoder import load_ae
from modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM, SiglipVisionConfig, SiglipVisionModel
)
from modeling.qwen2 import Qwen2Tokenizer
from train.train_utils import create_logger, get_latest_ckpt
from train.fsdp_utils import (
    FSDPCheckpoint, FSDPConfig, grad_checkpoint_check_fn, fsdp_wrapper,
    fsdp_ema_setup, fsdp_ema_update,
)
from train.scfm_utils import (
    sample_scfm_timesteps,
    LoRAEMAManager,
    compute_teacher_trajectory,
)


@dataclass
class ModelArguments:
    model_path: str = field(
        default="hf/BAGEL-7B-MoT",
        metadata={"help": "Path of the pretrained BAGEL model."}
    )
    llm_path: str = field(
        default="hf/Qwen2.5-0.5B-Instruct/",
        metadata={"help": "Path or HuggingFace repo ID of the pretrained Qwen2-style language model."}
    )
    llm_qk_norm: bool = field(
        default=True,
        metadata={"help": "Enable QK LayerNorm (qk_norm) inside the attention blocks."}
    )
    tie_word_embeddings: bool = field(
        default=False,
        metadata={"help": "Share input and output word embeddings (tied embeddings)."}
    )
    layer_module: str = field(
        default="Qwen2MoTDecoderLayer",
        metadata={"help": "Python class name of the decoder layer to instantiate."}
    )
    vae_path: str = field(
        default="flux/vae/ae.safetensors",
        metadata={"help": "Path to the pretrained VAE checkpoint for latent-space image generation."}
    )
    vit_path: str = field(
        default="hf/siglip-so400m-14-980-flash-attn2-navit/",
        metadata={"help": "Path or repo ID of the SigLIP Vision Transformer used for image understanding."}
    )
    max_latent_size: int = field(
        default=32,
        metadata={"help": "Maximum latent grid size (patches per side) for the VAE latent tensor."}
    )
    latent_patch_size: int = field(
        default=2,
        metadata={"help": "Spatial size (in VAE pixels) covered by each latent patch."}
    )
    vit_patch_size: int = field(
        default=14,
        metadata={"help": "Patch size (pixels) for the Vision Transformer encoder."}
    )
    vit_max_num_patch_per_side: int = field(
        default=70,
        metadata={"help": "Maximum number of ViT patches along one image side after cropping / resize."}
    )
    connector_act: str = field(
        default="gelu_pytorch_tanh",
        metadata={"help": "Activation function used in the latent-to-text connector MLP."}
    )
    interpolate_pos: bool = field(
        default=False,
        metadata={"help": "Interpolate positional embeddings when image resolution differs from pre-training."}
    )
    vit_select_layer: int = field(
        default=-2,
        metadata={"help": "Which hidden layer of the ViT to take as the visual feature (negative = from the end)."}
    )
    vit_rope: bool = field(
        default=False,
        metadata={"help": "Replace ViT positional encodings with RoPE."}
    )

    text_cond_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "Probability of dropping text embeddings during training."}
    )
    vae_cond_dropout_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of dropping VAE latent inputs during training."}
    )
    vit_cond_dropout_prob: float = field(
        default=0.3,
        metadata={"help": "Probability of dropping ViT visual features during training."}
    )


@dataclass
class DataArguments:
    dataset_config_file: str = field(
        default="data/configs/example.yaml",
        metadata={"help": "YAML file specifying dataset groups, weights, and preprocessing rules."}
    )
    prefetch_factor: int = field(
        default=2,
        metadata={"help": "How many batches each DataLoader worker pre-loads in advance."}
    )
    num_workers: int = field(
        default=4,
        metadata={"help": "Number of background workers for the PyTorch DataLoader."}
    )
    max_num_tokens_per_sample: int = field(
        default=16384,
        metadata={"help": "Maximum tokens allowed in one raw sample; longer samples are skipped."}
    )
    max_num_tokens: int = field(
        default=36864,
        metadata={"help": "Hard limit on tokens in a packed batch; flush if adding a sample would exceed it."}
    )
    prefer_buffer_before: int = field(
        default=16384,
        metadata={"help": "While batch length is below this, pop from the overflow buffer before new sampling."}
    )
    max_buffer_size: int = field(
        default=50,
        metadata={"help": "Maximum number of oversized samples kept in the overflow buffer."}
    )
    data_seed: int = field(
        default=42,
        metadata={"help": "Seed used when shuffling / sampling data shards to ensure reproducibility."}
    )


@dataclass
class TrainingArguments:
    # --- modality switches ---
    visual_gen: bool = field(
        default=True,
        metadata={"help": "Train image generation branch."}
    )
    visual_und: bool = field(
        default=True,
        metadata={"help": "Train image understanding branch."}
    )
    use_stage_weighting: bool = field(
        default=False,
        metadata={"help": "Enable Multi-Stage Loss Weighting."}
    )

    # --- bookkeeping & logging ---
    results_dir: str = field(
        default="results",
        metadata={"help": "Root directory for logs."}
    )
    checkpoint_dir: str = field(
        default="results/checkpoints",
        metadata={"help": "Root directory for model checkpoints."}
    )
    wandb_project: str = field(
        default="bagel",
        metadata={"help": "Weights & Biases project name."}
    )
    wandb_name: str = field(
        default="run",
        metadata={"help": "Name shown in the Weights & Biases UI for this run."}
    )
    wandb_runid: str = field(
        default="0",
        metadata={"help": "Unique identifier to resume a previous W&B run, if desired."}
    )
    wandb_resume: str = field(
        default="allow",
        metadata={"help": "W&B resume mode: 'allow', 'must', or 'never'."}
    )
    wandb_offline: bool = field(
        default=False,
        metadata={"help": "Run W&B in offline mode (logs locally, sync later)."}
    )

    # --- reproducibility & resume ---
    global_seed: int = field(
        default=4396,
        metadata={"help": "Base random seed; actual seed is offset by rank for DDP."}
    )
    auto_resume: bool = field(
        default=False,
        metadata={"help": "Automatically pick up the latest checkpoint found in checkpoint_dir."}
    )
    resume_from: str = field(
        default=None,
        metadata={"help": "Explicit checkpoint path to resume from (overrides auto_resume)." }
    )
    resume_model_only: bool = field(
        default=False,
        metadata={"help": "Load only model weights, ignoring optimizer/scheduler states."}
    )
    finetune_from_ema: bool = field(
        default=False,
        metadata={"help": "When resume_model_only=True, load the EMA (exponential moving average) weights instead of raw weights."}
    )
    finetune_from_hf: bool = field(
        default=False,
        metadata={"help": "Whether finetune from HugginFace model."}
    )

    # --- reporting frequency ---
    log_every: int = field(
        default=10,
        metadata={"help": "Print / log every N training steps."}
    )
    save_every: int = field(
        default=2000,
        metadata={"help": "Save a checkpoint every N training steps."}
    )
    total_steps: int = field(
        default=500_000,
        metadata={"help": "Total number of optimizer steps to train for."}
    )

    # --- optimization & scheduler ---
    warmup_steps: int = field(
        default=2000,
        metadata={"help": "Linear warm-up steps before applying the main LR schedule."}
    )
    lr_scheduler: str = field(
        default="constant",
        metadata={"help": "Type of LR schedule: 'constant' or 'cosine'."}
    )
    lr: float = field(
        default=1e-4,
        metadata={"help": "Peak learning rate after warm-up."}
    )
    min_lr: float = field(
        default=1e-7,
        metadata={"help": "Minimum learning rate for cosine schedule (ignored for constant)."}
    )
    beta1: float = field(
        default=0.9,
        metadata={"help": "AdamW β₁ coefficient."}
    )
    beta2: float = field(
        default=0.95,
        metadata={"help": "AdamW β₂ coefficient."}
    )
    eps: float = field(
        default=1e-15,
        metadata={"help": "AdamW ε for numerical stability."}
    )
    ema: float = field(
        default=0.9999,
        metadata={"help": "Decay rate for the exponential moving average of model weights."}
    )
    disable_ema: bool = field(
        default=False,
        metadata={"help": "Disable EMA model to save CPU memory."}
    )
    max_grad_norm: int = field(
        default=1.0,
        metadata={"help": "Gradient clipping threshold (L2 norm)."}
    )
    timestep_shift: float = field(
        default=1.0,
        metadata={"help": "Shift applied to diffusion timestep indices (for latent prediction)."}
    )
    mse_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the image-reconstruction MSE loss term."}
    )
    ce_weight: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for the language cross-entropy loss term."}
    )
    ce_loss_reweighting: bool = field(
        default=False,
        metadata={"help": "Reweight CE loss by token importance (provided via ce_loss_weights)."}
    )
    expected_num_tokens: int = field(
        default=16768,
        metadata={"help": "Soft target token count; yield the batch once it reaches or exceeds this size."}
    )

    # --- distributed training / FSDP ---
    num_replicate: int = field(
        default=1,
        metadata={"help": "Number of model replicas per GPU rank for tensor parallelism."}
    )
    num_shard: int = field(
        default=8,
        metadata={"help": "Number of parameter shards when using FSDP HYBRID_SHARD."}
    )
    sharding_strategy: str = field(
        default="HYBRID_SHARD",
        metadata={"help": "FSDP sharding strategy: FULL_SHARD, SHARD_GRAD_OP, HYBRID_SHARD, etc."}
    )
    backward_prefetch: str = field(
        default="BACKWARD_PRE",
        metadata={"help": "FSDP backward prefetch strategy (BACKWARD_PRE or NO_PREFETCH)."}
    )
    cpu_offload: bool = field(
        default=False,
        metadata={"help": "Enable FSDP parameter offload to CPU."}
    )

    # --- module freezing ---
    freeze_llm: bool = field(
        default=False,
        metadata={"help": "Keep language-model weights fixed (no gradient updates)."}
    )
    freeze_vit: bool = field(
        default=False,
        metadata={"help": "Keep ViT weights fixed during training."}
    )
    freeze_vae: bool = field(
        default=True,
        metadata={"help": "Keep VAE weights fixed; only predict latents, don’t fine-tune encoder/decoder."}
    )
    freeze_und: bool = field(
        default=False,
        metadata={"help": "Freeze the visual understanding connector layers."}
    )
    copy_init_moe: bool = field(
        default=True,
        metadata={"help": "Duplicate initial MoE experts so each has identical initialisation."}
    )
    use_flex: bool = field(
        default=False,
        metadata={"help": "Enable FLEX (flash-ext friendly) packing algorithm for sequence data."}
    )

    anchor_und: bool = field(
        default=True,
        metadata={"help": "Use understanding branch (ViT) features as style anchor for adaptive normalization"}
    )

    anchor_gen: bool = field(
        default=False,
        metadata={"help": "Use generation branch (VAE) features as style anchor for adaptive normalization"}
    )

    # --- Reconstruction Alignment (ReAC style) ---
    enable_reconstruction_alignment: bool = field(
        default=False,
        metadata={"help": "Enable reconstruction alignment loss (ReAC style) for flow matching"}
    )

    recon_gradient_weight: float = field(
        default=0.5,
        metadata={"help": "Weight for spatial gradient loss component (texture continuity)"}
    )

    recon_spectral_weight: float = field(
        default=0.1,
        metadata={"help": "Weight for spectral (FFT) loss component (high-frequency texture)"}
    )

    n_copy: int = field(
        default=1,
        metadata={"help": "N-Copy multi-step training: replicate each gen image N times with different timesteps (default=1, no copy)"}
    )

    # --- Q-Former Distillation ---
    enable_qformer_distill: bool = field(
        default=False,
        metadata={"help": "Enable Q-Former VAE-to-Understanding distillation to replace ViT"}
    )
    qformer_num_layers: int = field(
        default=4,
        metadata={"help": "Number of cross-attention layers in Q-Former"}
    )
    qformer_num_heads: int = field(
        default=16,
        metadata={"help": "Number of attention heads in Q-Former"}
    )
    qformer_alpha: float = field(
        default=0.0,
        metadata={"help": "Q-Former blending alpha (0=ViT only, 1=Q-Former only). Use schedule in training script."}
    )
    distill_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for Q-Former distillation loss (MSE + cosine)"}
    )
    qformer_version: str = field(
        default="v2",
        metadata={"help": "Q-Former version: 'v2' (original) or 'v3' (SigLIP-aligned)"}
    )
    qformer_num_ca_blocks: int = field(
        default=4,
        metadata={"help": "v3 only: number of CA+SA blocks (first stage)"}
    )
    qformer_num_sa_blocks: int = field(
        default=8,
        metadata={"help": "v3 only: number of SA+SA blocks (second stage)"}
    )
    qformer_sa_warmup_steps: int = field(
        default=0,
        metadata={"help": "v3 only: freeze SA layers for first N steps (0=no warmup)"}
    )

    # --- SCFM Shortcut Distillation ---
    enable_scfm: bool = field(
        default=False,
        metadata={"help": "Enable SCFM shortcut distillation for fast image generation (4-8 steps)."}
    )
    scfm_start_step: int = field(
        default=0,
        metadata={"help": "Step at which SCFM kicks in. Before this step, normal full-param training. "
                          "0 means SCFM from the start (if enable_scfm=True)."}
    )
    scfm_t_skip: int = field(
        default=2,
        metadata={"help": "Number of teacher Euler sub-steps per shortcut."}
    )
    scfm_shortcut_ratio: float = field(
        default=0.2,
        metadata={"help": "Fraction of steps using normal velocity loss (rest use shortcut distillation)."}
    )
    scfm_num_euler_steps: int = field(
        default=50,
        metadata={"help": "Teacher Euler grid resolution."}
    )
    scfm_slow_ema_decay: float = field(
        default=0.999,
        metadata={"help": "Slow EMA decay for SCFM self-teaching."}
    )
    scfm_shortcut_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for shortcut MSE loss."}
    )
    scfm_lora_rank: int = field(
        default=64,
        metadata={"help": "LoRA rank for SCFM distillation."}
    )
    scfm_lora_alpha: int = field(
        default=128,
        metadata={"help": "LoRA alpha scaling for SCFM distillation."}
    )
    scfm_lora_target_modules: str = field(
        default="q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen,"
                "mlp_moe_gen.gate_proj,mlp_moe_gen.up_proj,mlp_moe_gen.down_proj",
        metadata={"help": "Comma-separated LoRA target module names for SCFM."},
    )



def main():
    assert torch.cuda.is_available()
    from datetime import timedelta
    dist.init_process_group("nccl", timeout=timedelta(hours=2))
    device = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(device)
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    try:
        os.makedirs(training_args.results_dir, exist_ok=True)
        os.makedirs(training_args.checkpoint_dir, exist_ok=True)
    except OSError:
        pass
    
    dist.barrier()

    # Setup logging:
    if dist.get_rank() == 0:
        logger = create_logger(training_args.results_dir, dist.get_rank())
        wandb.init(
            project=training_args.wandb_project, 
            id=f"{training_args.wandb_name}-run{training_args.wandb_runid}", 
            name=training_args.wandb_name, 
            resume=training_args.wandb_resume,
            mode="offline" if training_args.wandb_offline else "online"
        )
        wandb.config.update(training_args, allow_val_change=True)
        wandb.config.update(model_args)
        wandb.config.update(data_args)
    else:
        logger = create_logger(None, dist.get_rank())
    dist.barrier()
    logger.info(f'Training arguments {training_args}')
    logger.info(f'Model arguments {model_args}')
    logger.info(f'Data arguments {data_args}')

    # prepare auto resume logic:
    if training_args.auto_resume:
        resume_from = get_latest_ckpt(training_args.checkpoint_dir)
        if resume_from is None:
            resume_from = training_args.resume_from
            resume_model_only = training_args.resume_model_only
            if resume_model_only:
                finetune_from_ema = training_args.finetune_from_ema
            else:
                finetune_from_ema = False
        else:
            resume_model_only = False
            finetune_from_ema = False
    else:
        resume_from = training_args.resume_from
        resume_model_only = training_args.resume_model_only
        if resume_model_only:
            finetune_from_ema = training_args.finetune_from_ema
        else:
            finetune_from_ema = False

    # Set seed:
    seed = training_args.global_seed * dist.get_world_size() + dist.get_rank()
    set_seed(seed)

    # Setup model:
    if training_args.finetune_from_hf:
        llm_config = Qwen2Config.from_json_file(os.path.join(model_args.model_path, "llm_config.json"))
    else:
        llm_config = Qwen2Config.from_pretrained(model_args.llm_path)
    # logger.info(f"llm_config {llm_config}")
    llm_config.layer_module = model_args.layer_module
    llm_config.qk_norm = model_args.llm_qk_norm
    llm_config.tie_word_embeddings = model_args.tie_word_embeddings
    llm_config.freeze_und = training_args.freeze_und
    logger.info(f"llm_config {llm_config}")
    if training_args.finetune_from_hf:
        language_model = Qwen2ForCausalLM(llm_config)
    else:
        language_model = Qwen2ForCausalLM.from_pretrained(model_args.llm_path, config=llm_config)
    if training_args.copy_init_moe:
        language_model.init_moe()

    logger.info(f"llm model loaded finished!!")

    if training_args.visual_und:
        if training_args.finetune_from_hf:
            vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_args.model_path, "vit_config.json"))
        else:
            vit_config = SiglipVisionConfig.from_pretrained(model_args.vit_path)
        vit_config.num_hidden_layers = vit_config.num_hidden_layers + 1 + model_args.vit_select_layer
        vit_config.rope = model_args.vit_rope
        if training_args.finetune_from_hf:
            vit_model = SiglipVisionModel(vit_config)
        else:
            vit_model = SiglipVisionModel.from_pretrained(model_args.vit_path, config=vit_config)

    logger.info(f"vit model loaded finished!!")

    if training_args.visual_gen:
        vae_model, vae_config = load_ae(
            local_path=os.path.join(model_args.model_path, "ae.safetensors") 
            if training_args.finetune_from_hf else model_args.vae_path
        )

    logger.info(f"vae model loaded finished!!")

    config = BagelConfig(
        visual_gen=training_args.visual_gen,
        visual_und=training_args.visual_und,
        llm_config=llm_config,
        vit_config=vit_config if training_args.visual_und else None,
        vae_config=vae_config if training_args.visual_gen else None,
        latent_patch_size=model_args.latent_patch_size,
        max_latent_size=model_args.max_latent_size,
        vit_max_num_patch_per_side=model_args.vit_max_num_patch_per_side,
        connector_act=model_args.connector_act,
        interpolate_pos=model_args.interpolate_pos,
        timestep_shift=training_args.timestep_shift,
        anchor_und=training_args.anchor_und,
        anchor_gen=training_args.anchor_gen,
        enable_reconstruction_alignment=training_args.enable_reconstruction_alignment,
        recon_gradient_weight=training_args.recon_gradient_weight,
        recon_spectral_weight=training_args.recon_spectral_weight,
        enable_qformer_distill=training_args.enable_qformer_distill,
        qformer_num_layers=training_args.qformer_num_layers,
        qformer_num_heads=training_args.qformer_num_heads,
        qformer_alpha=training_args.qformer_alpha,
        qformer_version=training_args.qformer_version,
        qformer_num_ca_blocks=training_args.qformer_num_ca_blocks,
        qformer_num_sa_blocks=training_args.qformer_num_sa_blocks,
        qformer_sa_warmup_steps=training_args.qformer_sa_warmup_steps,
    )

    # === Debug Code ===
    # Setup packed dataloader
    # with open(data_args.dataset_config_file, "r") as stream:
    #     dataset_meta = yaml.safe_load(stream)
    # dataset_config = DataConfig(grouped_datasets=dataset_meta)
    # if training_args.visual_und:
    #     dataset_config.vit_patch_size = model_args.vit_patch_size
    #     dataset_config.max_num_patch_per_side = model_args.vit_max_num_patch_per_side
    # if training_args.visual_gen:
    #     vae_image_downsample = model_args.latent_patch_size * vae_config.downsample
    #     dataset_config.vae_image_downsample = vae_image_downsample
    #     dataset_config.max_latent_size = model_args.max_latent_size
    #     dataset_config.text_cond_dropout_prob = model_args.text_cond_dropout_prob
    #     dataset_config.vae_cond_dropout_prob = model_args.vae_cond_dropout_prob
    #     dataset_config.vit_cond_dropout_prob = model_args.vit_cond_dropout_prob
    # import pdb; pdb.set_trace()
    # train_dataset = PackedDataset(
    #     dataset_config,
    #     tokenizer=None,
    #     special_tokens={},
    #     local_rank=dist.get_rank(),
    #     world_size=dist.get_world_size(),
    #     num_workers=data_args.num_workers,
    #     expected_num_tokens=training_args.expected_num_tokens,
    #     max_num_tokens_per_sample=data_args.max_num_tokens_per_sample,
    #     max_num_tokens=data_args.max_num_tokens,
    #     max_buffer_size=data_args.max_buffer_size,
    #     prefer_buffer_before=data_args.prefer_buffer_before,
    #     interpolate_pos=model_args.interpolate_pos,
    #     use_flex=training_args.use_flex,
    #     data_status=None
    # )
    # === Debug Code ===

    model = Bagel(
        language_model, 
        vit_model if training_args.visual_und else None, 
        config
    )

    logger.info(f"bagel model initial finished!!")

    if training_args.visual_und:
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    # Setup tokenizer for model:
    tokenizer = Qwen2Tokenizer.from_pretrained(model_args.model_path if training_args.finetune_from_hf else model_args.llm_path)
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    logger.info(f"tokenize and vocab initial finished!!")

    # maybe freeze something:
    if training_args.freeze_vae and training_args.visual_gen:
        for param in vae_model.parameters():
            param.requires_grad = False
    if training_args.freeze_llm:
        model.language_model.eval()
        for param in model.language_model.parameters():
            param.requires_grad = False
    if training_args.freeze_vit and training_args.visual_und:
        model.vit_model.eval()
        for param in model.vit_model.parameters():
            param.requires_grad = False

    # Setup FSDP and load pretrained model:
    fsdp_config = FSDPConfig(
        sharding_strategy=training_args.sharding_strategy,
        backward_prefetch=training_args.backward_prefetch,
        cpu_offload=training_args.cpu_offload,
        num_replicate=training_args.num_replicate,
        num_shard=training_args.num_shard,
    )

    logger.info(f"fsdp config initial finished!!")

    # Load checkpoint into model FIRST, then deepcopy for ema.
    model, _ = FSDPCheckpoint.try_load_ckpt(
        resume_from, logger, model, ema_model=None, resume_from_ema=finetune_from_ema
    )
    import gc; gc.collect()

    # --- SCFM: two-phase training setup ---
    # Phase 1 (steps < scfm_start_step): full-parameter training with EMA
    # Phase 2 (steps >= scfm_start_step): SCFM shortcut distillation (teacher = current model, no_grad)
    # No LoRA needed — full-parameter training throughout, SCFM just adds shortcut loss in phase 2.
    if training_args.enable_scfm:
        logger.info(
            f"SCFM enabled: phase 1 (full-param) steps 0-{training_args.scfm_start_step}, "
            f"phase 2 (SCFM shortcut) steps {training_args.scfm_start_step}-{training_args.total_steps}"
        )

    if not training_args.disable_ema:
        logger.info("creating ema_model via deepcopy...")
        ema_model = deepcopy(model)
        logger.info("ema_model created.")
        ema_model = fsdp_ema_setup(ema_model, fsdp_config)
    else:
        ema_model = None

    fsdp_model = fsdp_wrapper(model, fsdp_config)

    logger.info(f"load fsdp and ema model finished!!")
    
    apply_activation_checkpointing(
        fsdp_model, 
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ), 
        check_fn=grad_checkpoint_check_fn
    )

    logger.info(f"apply_activation_checkpointing finished!!")

    if dist.get_rank() == 0:
        print(fsdp_model)
        for name, param in model.named_parameters():
            print(name, param.requires_grad)

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        fsdp_model.parameters(), 
        lr=training_args.lr, 
        betas=(training_args.beta1, training_args.beta2), 
        eps=training_args.eps, 
        weight_decay=0
    )
    if training_args.lr_scheduler == 'cosine':
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.total_steps,
            min_lr=training_args.min_lr,
        )
    elif training_args.lr_scheduler == 'constant':
        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=training_args.warmup_steps
        )
    else:
        raise ValueError

    # maybe resume optimizer, scheduler, and train_steps
    if resume_model_only:
        train_step = 0
        data_status = None
    else:
        optimizer, scheduler, train_step, data_status = FSDPCheckpoint.try_load_train_state(
            resume_from, optimizer, scheduler, fsdp_config, 
        )
    # Setup packed dataloader
    with open(data_args.dataset_config_file, "r") as stream:
        dataset_meta = yaml.safe_load(stream)
    dataset_config = DataConfig(grouped_datasets=dataset_meta, enable_qformer_distill=training_args.enable_qformer_distill)
    if training_args.visual_und:
        dataset_config.vit_patch_size = model_args.vit_patch_size
        dataset_config.max_num_patch_per_side = model_args.vit_max_num_patch_per_side
    if training_args.visual_gen:
        vae_image_downsample = model_args.latent_patch_size * vae_config.downsample
        dataset_config.vae_image_downsample = vae_image_downsample
        dataset_config.max_latent_size = model_args.max_latent_size
        dataset_config.text_cond_dropout_prob = model_args.text_cond_dropout_prob
        dataset_config.vae_cond_dropout_prob = model_args.vae_cond_dropout_prob
        dataset_config.vit_cond_dropout_prob = model_args.vit_cond_dropout_prob
        dataset_config.n_copy = training_args.n_copy
    train_dataset = PackedDataset(
        dataset_config,
        tokenizer=tokenizer,
        special_tokens=new_token_ids,
        local_rank=dist.get_rank(),
        world_size=dist.get_world_size(),
        num_workers=data_args.num_workers,
        expected_num_tokens=training_args.expected_num_tokens,
        max_num_tokens_per_sample=data_args.max_num_tokens_per_sample,
        max_num_tokens=data_args.max_num_tokens,
        max_buffer_size=data_args.max_buffer_size,
        prefer_buffer_before=data_args.prefer_buffer_before,
        interpolate_pos=model_args.interpolate_pos,
        use_flex=training_args.use_flex,
        data_status=data_status,
        use_stage_weighting=training_args.use_stage_weighting,
    )
    train_dataset.set_epoch(data_args.data_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1, # batch size is 1 packed dataset
        num_workers=data_args.num_workers,
        pin_memory=True,
        collate_fn=collate_wrapper(),
        drop_last=True,
        prefetch_factor=data_args.prefetch_factor,
    )

    # Prepare models for training:
    if training_args.visual_gen:
        vae_model.to(device).eval()
    fsdp_model.train()
    if ema_model is not None:
        ema_model.eval()

    # train loop
    start_time = time()
    logger.info(f"Training for {training_args.total_steps} steps, starting at {train_step}...")
    all_total_samples_appro = 0
    # SCFM shortcut scheduler accumulator (Bresenham-style, see loop body).
    # Seeded from train_step so the shortcut fraction is consistent across resumes.
    _scfm_active_start = max(train_step, training_args.scfm_start_step)
    scfm_shortcut_accum = max(0, train_step - _scfm_active_start) * training_args.scfm_shortcut_ratio
    for curr_step, data in enumerate(train_loader, start=train_step):
        if curr_step > training_args.total_steps:
            break
        data = data.cuda(device).to_dict()
        data_indexes = data.pop('batch_data_indexes', None)
        ce_loss_weights = data.pop('ce_loss_weights', None)
        stage_weights = data.pop('stage_weights', None)

        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            if training_args.visual_gen:
                with torch.no_grad():
                    data['padded_latent'] = vae_model.encode(data.pop('padded_images'))

            # Q-Former: compute alpha schedule (K/V now comes from packed_latent_clean in forward())
            if training_args.enable_qformer_distill:
                # Alpha schedule: linear ramp over total_steps
                # 0-50%: alpha 0->0.5, 50-80%: alpha 0.5->1.0, 80-100%: alpha=1.0
                progress = curr_step / max(training_args.total_steps, 1)
                if progress < 0.5:
                    alpha = 0.5 * (progress / 0.5)
                elif progress < 0.8:
                    alpha = 0.5 + 0.5 * ((progress - 0.5) / 0.3)
                else:
                    alpha = 1.0
                data['qformer_alpha'] = alpha

                # QFormerV3: SA freeze warmup — freeze for first N steps
                if (training_args.qformer_version == 'v3'
                    and training_args.qformer_sa_warmup_steps > 0):
                    if curr_step == 1:
                        # Freeze on first step
                        try:
                            fsdp_model.module.qformer.freeze_sa_layers()
                        except AttributeError:
                            fsdp_model.qformer.freeze_sa_layers()
                    elif curr_step == training_args.qformer_sa_warmup_steps:
                        # Unfreeze after warmup
                        try:
                            fsdp_model.module.qformer.unfreeze_sa_layers()
                        except AttributeError:
                            fsdp_model.qformer.unfreeze_sa_layers()
            # print(f"current data: {data.keys()}")

            # --- SCFM: decide shortcut vs normal step ---
            # Phase 1 (curr_step < scfm_start_step): normal full-param training
            # Phase 2 (curr_step >= scfm_start_step): SCFM shortcut with probability
            # CRITICAL: all ranks must make the same decision (FSDP requires symmetric forwards).
            # Use curr_step-based deterministic schedule instead of per-rank random.
            is_scfm_shortcut = False
            scfm_active = (
                training_args.enable_scfm
                and training_args.visual_gen
                and curr_step >= training_args.scfm_start_step
            )
            if scfm_active:
                # Deterministic scheduler whose long-run fraction of shortcut steps
                # equals shortcut_frac = 1 - non_shortcut_ratio exactly (no rounding
                # collapse). We accumulate shortcut_frac each active step and fire a
                # shortcut whenever the accumulator crosses an integer boundary
                # (Bresenham-style). E.g. shortcut_frac=0.8 → shortcut on 4 of every 5
                # steps; 0.2 → 1 of every 5; 1.0 → every step.
                shortcut_frac = training_args.scfm_shortcut_ratio
                prev_accum = scfm_shortcut_accum
                scfm_shortcut_accum += shortcut_frac
                is_scfm_shortcut = int(scfm_shortcut_accum) > int(prev_accum)

            if is_scfm_shortcut:
                loss_dict = _run_scfm_shortcut_step(
                    fsdp_model=fsdp_model, model=model,
                    data=data, training_args=training_args, device=device,
                )
            else:
                loss_dict = fsdp_model(**data)
            # print(f"loss_dict: {loss_dict}")

        _rk = dist.get_rank()
        loss = 0
        ce = loss_dict["ce"]
        if ce is not None:
            total_ce_tokens = torch.tensor(len(data['ce_loss_indexes']), device=device)
            dist.all_reduce(total_ce_tokens, op=dist.ReduceOp.SUM)
            if training_args.ce_loss_reweighting:
                ce = ce * ce_loss_weights
                total_ce_loss_weights = ce_loss_weights.sum()
                dist.all_reduce(total_ce_loss_weights, op=dist.ReduceOp.SUM)
                ce = ce.sum() * dist.get_world_size() / total_ce_loss_weights
            else:
                ce = ce.sum() * dist.get_world_size() / total_ce_tokens
            loss_dict["ce"] = ce.detach()
            loss = loss + ce * training_args.ce_weight
        else:
            assert not training_args.visual_und
            loss_dict["ce"] = torch.tensor(0, device=device)
            total_ce_tokens = torch.tensor(0, device=device)

        if training_args.visual_gen:
            mse = loss_dict["mse"]
            total_mse_tokens = torch.tensor(len(data['mse_loss_indexes']), device=device)
            dist.all_reduce(total_mse_tokens, op=dist.ReduceOp.SUM)
            mse = mse.mean(dim=-1).sum() * dist.get_world_size() / total_mse_tokens
            loss_dict["mse"] = mse.detach()
            mse_w = training_args.scfm_shortcut_loss_weight if is_scfm_shortcut else training_args.mse_weight
            loss = loss + mse * mse_w
        else:
            assert not training_args.visual_gen
            loss_dict["mse"] = torch.tensor(0, device=device)
            total_mse_tokens = torch.tensor(0, device=device)


        # Reconstruction alignment loss (directly added, no separate weight)
        reconstruction_alignment_loss = loss_dict.get("reconstruction_alignment_loss")
        if training_args.enable_reconstruction_alignment and reconstruction_alignment_loss is not None:
            if isinstance(reconstruction_alignment_loss, torch.Tensor):
                loss_dict["reconstruction_alignment_loss"] = reconstruction_alignment_loss.detach()
                loss = loss + reconstruction_alignment_loss

                # Flatten alignment metrics (recon_mse, recon_grad, recon_spec) into loss_dict for logging
                alignment_metrics = loss_dict.pop("alignment_metrics", {})
                for k, v in alignment_metrics.items():
                    loss_dict[k] = torch.tensor(v, device=device)
            else:
                loss_dict.pop("alignment_metrics", None)
                loss_dict["reconstruction_alignment_loss"] = torch.tensor(0, device=device)
        else:
            loss_dict.pop("alignment_metrics", None)
            loss_dict["reconstruction_alignment_loss"] = torch.tensor(0, device=device)

        # Q-Former distillation loss
        distill_loss = loss_dict.get("distill_loss")
        if training_args.enable_qformer_distill and distill_loss is not None:
            if isinstance(distill_loss, torch.Tensor):
                loss_dict["distill_loss"] = distill_loss.detach()
                loss = loss + distill_loss * training_args.distill_loss_weight
            else:
                loss_dict["distill_loss"] = torch.tensor(0, device=device)
        else:
            loss_dict["distill_loss"] = torch.tensor(0, device=device)

        # Q-Former robustness loss (on predicted x_hat_0)
        qformer_robust_loss = loss_dict.get("qformer_robust_loss")
        if qformer_robust_loss is not None and isinstance(qformer_robust_loss, torch.Tensor) and qformer_robust_loss.item() > 0:
            loss_dict["qformer_robust_loss"] = qformer_robust_loss.detach()
            loss = loss + qformer_robust_loss
        else:
            loss_dict["qformer_robust_loss"] = torch.tensor(0, device=device)

        logger.info(f"loss_dict: {loss_dict.items()}")

        _rk = dist.get_rank()
        optimizer.zero_grad()
        loss.backward()
        total_norm = fsdp_model.clip_grad_norm_(training_args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        if ema_model is not None:
            fsdp_ema_update(ema_model, fsdp_model, decay=training_args.ema)

        # Log loss values:
        if curr_step % training_args.log_every == 0:
            total_samples = torch.tensor(len(data['sample_lens']), device=device)
            dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
            all_total_samples_appro += total_samples.item() * training_args.log_every

            # Measure training speed:
            torch.cuda.synchronize()
            end_time = time()
            steps_per_sec = training_args.log_every / (end_time - start_time)
            message = f"(step={curr_step:07d}) "
            wandb_log = {}
            for key, value in loss_dict.items():
                if not isinstance(value, torch.Tensor):
                    continue
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(value.item(), device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                message += f"Train Loss {key}: {avg_loss:.4f}, "
                wandb_log[key] = avg_loss
            message += f"Train Steps/Sec: {steps_per_sec:.2f}, "
            logger.info(message)

            wandb_log['lr'] = optimizer.param_groups[0]['lr']
            wandb_log['total_mse_tokens'] = total_mse_tokens.item()
            wandb_log['total_ce_tokens'] = total_ce_tokens.item()
            wandb_log['total_norm'] = total_norm.item()
            wandb_log['total_samples'] = total_samples.item()
            wandb_log['all_total_samples_appro'] = all_total_samples_appro
            if training_args.enable_qformer_distill and 'qformer_alpha' in data:
                wandb_log['qformer_alpha'] = data['qformer_alpha']

            mem_allocated = torch.tensor(torch.cuda.max_memory_allocated() / 1024**2, device=device)
            dist.all_reduce(mem_allocated, op=dist.ReduceOp.MAX)
            wandb_log['mem_allocated'] = mem_allocated
            mem_cache = torch.tensor(torch.cuda.max_memory_reserved() / 1024**2, device=device)
            dist.all_reduce(mem_cache, op=dist.ReduceOp.MAX)
            wandb_log['mem_cache'] = mem_cache

            if dist.get_rank() == 0:
                wandb.log(wandb_log, step=curr_step)
            start_time = time()

        if data_status is None:
            data_status = {}
        for item in data_indexes:
            if item['dataset_name'] not in data_status.keys():
                data_status[item['dataset_name']] = {}
            data_status[item['dataset_name']][item['worker_id']] = item['data_indexes']

        if curr_step > 0 and curr_step % training_args.save_every == 0:
            # Free training tensors before checkpoint save to avoid OOM
            del loss, loss_dict, data
            gc.collect()
            torch.cuda.empty_cache()
            # current_data_status = {}
            # for item in data_indexes:
            #     if item['dataset_name'] not in current_data_status:
            #         current_data_status[item['dataset_name']] = {}
            #     current_data_status[item['dataset_name']][item['worker_id']] = item['data_indexes']
    
    # 使用current_data_status而不是累积的data_status
            
            if dist.get_rank() == 0:
                gather_list = [None] * dist.get_world_size()
            else:
                gather_list = None
            torch.cuda.empty_cache()
            gc.collect()
            dist.gather_object(data_status, gather_list, dst=0)

            FSDPCheckpoint.fsdp_save_ckpt(
                ckpt_dir=training_args.checkpoint_dir, 
                train_steps=curr_step, 
                model=fsdp_model, 
                ema_model=ema_model, 
                optimizer=optimizer, 
                scheduler=scheduler, 
                logger=logger,
                fsdp_config=fsdp_config,
                data_status=gather_list
            )
            del gather_list
            
            torch.cuda.empty_cache()

    logger.info("Done!")
    if dist.get_rank() == 0:
        wandb.finish()
    dist.destroy_process_group()


# ---------------------------------------------------------------------------
# SCFM shortcut step (called from training loop when enable_scfm=True)
# ---------------------------------------------------------------------------

def _run_scfm_shortcut_step(fsdp_model, model, data, training_args, device):
    """Run one SCFM shortcut distillation step (full-parameter, no LoRA).

    Uses the current model itself as teacher (under no_grad). All forward
    passes go through fsdp_model so FSDP AllGather is properly triggered.

    1. Patchify VAE latents → packed_latent_clean.
    2. Sample noise + variable-length shortcut timesteps.
    3. Teacher trajectory: t_skip Euler steps via fsdp_model with scfm_teacher_mode.
    4. Student forward via fsdp_model with scfm_mode + shortcut target.
    """
    # --- Patchify (same as Bagel.forward) ---
    padded_latent = data["padded_latent"]
    shapes = data["patchified_vae_latent_shapes"]
    p = model.latent_patch_size
    packed_pieces = []
    for latent, (h, w) in zip(padded_latent, shapes):
        latent = latent[:, :h * p, :w * p].reshape(model.latent_channel, h, p, w, p)
        latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * model.latent_channel)
        packed_pieces.append(latent)
    packed_latent_clean = torch.cat(packed_pieces, dim=0)

    # --- Sample noise + shortcut timesteps for LOSS images only ---
    # v_t from teacher comes from mse_loss_indexes (loss VAE tokens only).
    # wo_loss VAE tokens have t=0, so they stay as clean latent and are not part of the shortcut.
    noise = torch.randn_like(packed_latent_clean)
    vae_token_seqlens = data["vae_token_seqlens"]
    loss_vae_token_seqlens = data["loss_vae_token_seqlens"]
    num_loss_images = loss_vae_token_seqlens.shape[0]

    t_start_per_img, dt_teacher_per_img, _ = sample_scfm_timesteps(
        num_images=num_loss_images,
        t_skip=training_args.scfm_t_skip,
        num_euler_steps=training_args.scfm_num_euler_steps,
        timestep_shift=training_args.timestep_shift,
        device=packed_latent_clean.device,
    )

    # Broadcast per-loss-image → per-loss-token
    t_start_loss = torch.repeat_interleave(t_start_per_img, loss_vae_token_seqlens)
    dt_teacher_loss = torch.repeat_interleave(dt_teacher_per_img, loss_vae_token_seqlens, dim=0)

    # Build scfm_timesteps for ALL VAE tokens: 0 for wo_loss, t_start for loss
    total_vae_tokens = packed_latent_clean.shape[0]
    scfm_timesteps_all = torch.zeros(total_vae_tokens, device=packed_latent_clean.device)
    # Map loss token positions within the VAE token sequence
    # mse_loss_indexes contains positions in the FULL packed sequence;
    # packed_vae_token_indexes maps VAE-token-index → full-sequence-position.
    # We need a boolean mask over VAE tokens that says "is this a loss token?"
    packed_vae_idxs = data["packed_vae_token_indexes"]
    mse_idxs_set = set(data["mse_loss_indexes"].tolist())
    vae_is_loss = torch.tensor(
        [pos.item() in mse_idxs_set for pos in packed_vae_idxs],
        dtype=torch.bool, device=packed_latent_clean.device,
    )
    scfm_timesteps_all[vae_is_loss] = t_start_loss

    # Build x_t for ALL VAE tokens: noisy for loss tokens, clean for wo_loss (t=0)
    t_all = scfm_timesteps_all[:, None]
    x_t_all = (1.0 - t_all) * packed_latent_clean + t_all * noise  # t=0 → x_0

    # --- Teacher trajectory (no_grad, all forwards through fsdp_model) ---
    with torch.no_grad():
        x_t = x_t_all.clone()
        t_current_all = scfm_timesteps_all.clone()
        accumulated_displacement = None

        for step in range(training_args.scfm_t_skip):
            teacher_out = fsdp_model(
                **data,
                scfm_teacher_mode=True,
                scfm_x_t=x_t,
                scfm_timesteps=t_current_all,
            )
            v_t = teacher_out["scfm_velocity"]  # [N_loss_tokens, latent_dim]

            dt_sub = dt_teacher_loss[:, step:step + 1]  # [N_loss_tokens, 1]
            displacement = v_t * dt_sub

            if accumulated_displacement is None:
                accumulated_displacement = displacement
            else:
                accumulated_displacement = accumulated_displacement + displacement

            # Euler step: update only loss VAE tokens in the full x_t
            full_disp = torch.zeros_like(x_t)
            full_disp[vae_is_loss] = displacement
            x_t = x_t - full_disp
            # Update timesteps for loss tokens
            t_current_all[vae_is_loss] = t_current_all[vae_is_loss] - dt_teacher_loss[:, step]

        # Target velocity = total displacement / total shortcut dt
        dt_total = dt_teacher_loss.sum(dim=1, keepdim=True)  # [N_loss_tokens, 1]
        target_v = accumulated_displacement / dt_total.clamp(min=1e-8)
        target_v = target_v.detach()

    # --- Student forward with SCFM mode ---
    loss_dict = fsdp_model(
        **data,
        scfm_mode=True,
        scfm_x_t=x_t_all,
        scfm_timesteps=scfm_timesteps_all,
        scfm_target=target_v,
    )

    return loss_dict


if __name__ == "__main__":
    main()
