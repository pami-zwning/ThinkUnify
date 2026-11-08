# Copyright 2025 ThinkUnify. SCFM distillation training script.
# Adapted from pretrain_unified_navit.py for Shortcutting Flow Matching distillation.
#
# Key differences from base training:
#   1. LoRA applied to generation-branch modules (frozen base model)
#   2. Batch split: non_shortcut_ratio for normal velocity loss, rest for shortcut distillation
#   3. Dual EMA (slow + fast) for two-jump self-teaching
#   4. Teacher trajectory generation via multi-step Euler with EMA weights

import functools
import gc
import os
import random
import wandb
import yaml
from copy import deepcopy
from dataclasses import dataclass, field
from time import time
from typing import List

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
)
from train.scfm_utils import (
    sample_scfm_timesteps,
    LoRAEMAManager,
    compute_teacher_trajectory,
)


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

@dataclass
class ModelArguments:
    model_path: str = field(default="hf/BAGEL-7B-MoT")
    llm_path: str = field(default="hf/Qwen2.5-0.5B-Instruct/")
    llm_qk_norm: bool = field(default=True)
    tie_word_embeddings: bool = field(default=False)
    layer_module: str = field(default="Qwen2MoTDecoderLayer")
    vae_path: str = field(default="flux/vae/ae.safetensors")
    vit_path: str = field(default="hf/siglip-so400m-14-980-flash-attn2-navit/")
    max_latent_size: int = field(default=32)
    latent_patch_size: int = field(default=2)
    vit_patch_size: int = field(default=14)
    vit_max_num_patch_per_side: int = field(default=70)
    connector_act: str = field(default="gelu_pytorch_tanh")
    interpolate_pos: bool = field(default=False)
    vit_select_layer: int = field(default=-2)
    vit_rope: bool = field(default=False)
    text_cond_dropout_prob: float = field(default=0.1)
    vae_cond_dropout_prob: float = field(default=0.3)
    vit_cond_dropout_prob: float = field(default=0.3)


@dataclass
class DataArguments:
    dataset_config_file: str = field(default="data/configs/example.yaml")
    prefetch_factor: int = field(default=2)
    num_workers: int = field(default=4)
    max_num_tokens_per_sample: int = field(default=16384)
    max_num_tokens: int = field(default=36864)
    prefer_buffer_before: int = field(default=16384)
    max_buffer_size: int = field(default=50)
    data_seed: int = field(default=42)


@dataclass
class TrainingArguments:
    # --- modality ---
    visual_gen: bool = True
    visual_und: bool = True

    # --- bookkeeping ---
    results_dir: str = "results"
    checkpoint_dir: str = "results/checkpoints"
    wandb_project: str = "ThinkUnify-SCFM"
    wandb_name: str = "scfm-distill"
    wandb_runid: str = "0"
    wandb_resume: str = "allow"
    wandb_offline: bool = False

    # --- resume ---
    global_seed: int = 4396
    auto_resume: bool = False
    resume_from: str = None
    resume_model_only: bool = True
    finetune_from_ema: bool = True
    finetune_from_hf: bool = True

    # --- schedule ---
    log_every: int = 10
    save_every: int = 500
    total_steps: int = 5000
    warmup_steps: int = 200
    lr_scheduler: str = "constant"
    lr: float = 2e-5
    min_lr: float = 1e-7
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-15
    max_grad_norm: float = 1.0

    # --- loss weights ---
    timestep_shift: float = 1.0
    mse_weight: float = 4.0
    ce_weight: float = 1.0
    ce_loss_reweighting: bool = False
    expected_num_tokens: int = 16768

    # --- FSDP ---
    num_replicate: int = 1
    num_shard: int = 8
    sharding_strategy: str = "FULL_SHARD"
    backward_prefetch: str = "BACKWARD_PRE"
    cpu_offload: bool = False

    # --- freezing (mostly frozen for SCFM) ---
    freeze_vae: bool = True
    freeze_und: bool = False
    copy_init_moe: bool = True
    use_flex: bool = False
    anchor_und: bool = True
    anchor_gen: bool = False
    enable_reconstruction_alignment: bool = False
    recon_gradient_weight: float = 0.5
    recon_spectral_weight: float = 0.1
    n_copy: int = 1
    enable_qformer_distill: bool = False
    qformer_num_layers: int = 4
    qformer_num_heads: int = 16
    qformer_alpha: float = 0.0
    distill_loss_weight: float = 1.0

    # --- SCFM-specific ---
    t_skip: int = field(default=2, metadata={"help": "Number of teacher Euler sub-steps per shortcut."})
    non_shortcut_ratio: float = field(default=0.6, metadata={"help": "Fraction of steps using normal velocity loss."})
    num_euler_steps: int = field(default=50, metadata={"help": "Teacher Euler grid resolution."})
    slow_ema_decay: float = field(default=0.999, metadata={"help": "Slow EMA decay for teacher."})
    shortcut_loss_weight: float = field(default=1.0, metadata={"help": "Weight for shortcut MSE loss."})

    # --- LoRA ---
    lora_rank: int = field(default=64, metadata={"help": "LoRA rank."})
    lora_alpha: int = field(default=128, metadata={"help": "LoRA alpha scaling."})
    lora_dropout: float = field(default=0.0, metadata={"help": "LoRA dropout."})
    lora_target_modules: str = field(
        default="q_proj_moe_gen,k_proj_moe_gen,v_proj_moe_gen,o_proj_moe_gen,"
                "mlp_moe_gen.gate_proj,mlp_moe_gen.up_proj,mlp_moe_gen.down_proj",
        metadata={"help": "Comma-separated LoRA target module names."},
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    assert torch.cuda.is_available()
    dist.init_process_group("nccl")
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

    # Logging
    if dist.get_rank() == 0:
        logger = create_logger(training_args.results_dir, dist.get_rank())
        wandb.init(
            project=training_args.wandb_project,
            id=f"{training_args.wandb_name}-run{training_args.wandb_runid}",
            name=training_args.wandb_name,
            resume=training_args.wandb_resume,
            mode="offline" if training_args.wandb_offline else "online",
        )
        wandb.config.update(training_args, allow_val_change=True)
        wandb.config.update(model_args)
        wandb.config.update(data_args)
    else:
        logger = create_logger(None, dist.get_rank())
    dist.barrier()
    logger.info(f"SCFM Distillation | Training args: {training_args}")

    # Resume logic
    resume_from = training_args.resume_from
    resume_model_only = training_args.resume_model_only
    finetune_from_ema = training_args.finetune_from_ema if resume_model_only else False

    seed = training_args.global_seed * dist.get_world_size() + dist.get_rank()
    set_seed(seed)

    # ======================================================================
    # Model setup (same as pretrain_unified_navit.py)
    # ======================================================================
    if training_args.finetune_from_hf:
        llm_config = Qwen2Config.from_json_file(os.path.join(model_args.model_path, "llm_config.json"))
    else:
        llm_config = Qwen2Config.from_pretrained(model_args.llm_path)
    llm_config.layer_module = model_args.layer_module
    llm_config.qk_norm = model_args.llm_qk_norm
    llm_config.tie_word_embeddings = model_args.tie_word_embeddings
    llm_config.freeze_und = training_args.freeze_und

    if training_args.finetune_from_hf:
        language_model = Qwen2ForCausalLM(llm_config)
    else:
        language_model = Qwen2ForCausalLM.from_pretrained(model_args.llm_path, config=llm_config)
    if training_args.copy_init_moe:
        language_model.init_moe()
    logger.info("LLM loaded.")

    vit_model = None
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
    logger.info("ViT loaded.")

    vae_model = vae_config = None
    if training_args.visual_gen:
        vae_model, vae_config = load_ae(
            local_path=os.path.join(model_args.model_path, "ae.safetensors")
            if training_args.finetune_from_hf else model_args.vae_path
        )
    logger.info("VAE loaded.")

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
    )

    model = Bagel(language_model, vit_model, config)
    if training_args.visual_und:
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    tokenizer = Qwen2Tokenizer.from_pretrained(
        model_args.model_path if training_args.finetune_from_hf else model_args.llm_path
    )
    tokenizer, new_token_ids, num_new_tokens = add_special_tokens(tokenizer)
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    # Load pretrained checkpoint
    model, _ = FSDPCheckpoint.try_load_ckpt(
        resume_from, logger, model, ema_model=None, resume_from_ema=finetune_from_ema
    )
    gc.collect()
    logger.info("Checkpoint loaded.")

    # ======================================================================
    # LoRA: freeze base model, apply LoRA to gen-branch
    # ======================================================================
    for param in model.parameters():
        param.requires_grad = False

    from peft import LoraConfig, get_peft_model

    target_modules = [m.strip() for m in training_args.lora_target_modules.split(",")]
    lora_config = LoraConfig(
        r=training_args.lora_rank,
        lora_alpha=training_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=training_args.lora_dropout,
        bias="none",
        init_lora_weights="gaussian",
    )
    model.language_model = get_peft_model(model.language_model, lora_config)
    logger.info(f"LoRA applied. Target modules: {target_modules}")

    # Also unfreeze small projection layers that participate in flow matching
    for name, param in model.named_parameters():
        if any(k in name for k in ["vae2llm", "llm2vae", "time_embedder"]):
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # ======================================================================
    # FSDP wrapping
    # ======================================================================
    fsdp_config = FSDPConfig(
        sharding_strategy=training_args.sharding_strategy,
        backward_prefetch=training_args.backward_prefetch,
        cpu_offload=training_args.cpu_offload,
        num_replicate=training_args.num_replicate,
        num_shard=training_args.num_shard,
    )
    fsdp_model = fsdp_wrapper(model, fsdp_config)
    apply_activation_checkpointing(
        fsdp_model,
        checkpoint_wrapper_fn=functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT
        ),
        check_fn=grad_checkpoint_check_fn,
    )
    logger.info("FSDP wrapped with activation checkpointing.")

    if dist.get_rank() == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"  [trainable] {name}  {param.shape}")

    # ======================================================================
    # Dual EMA for SCFM self-teaching
    # ======================================================================
    lora_ema = LoRAEMAManager(fsdp_model, slow_decay=training_args.slow_ema_decay)
    logger.info(f"LoRA EMA initialized. Tracking {len(lora_ema._lora_names)} parameters.")

    # ======================================================================
    # Optimizer (only trainable params)
    # ======================================================================
    optimizer = torch.optim.AdamW(
        [p for p in fsdp_model.parameters() if p.requires_grad],
        lr=training_args.lr,
        betas=(training_args.beta1, training_args.beta2),
        eps=training_args.eps,
        weight_decay=0,
    )
    if training_args.lr_scheduler == "cosine":
        scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=training_args.total_steps,
            min_lr=training_args.min_lr,
        )
    else:
        scheduler = get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=training_args.warmup_steps
        )

    # ======================================================================
    # Dataloader
    # ======================================================================
    with open(data_args.dataset_config_file, "r") as f:
        dataset_meta = yaml.safe_load(f)
    dataset_config = DataConfig(
        grouped_datasets=dataset_meta,
        enable_qformer_distill=training_args.enable_qformer_distill,
    )
    if training_args.visual_und:
        dataset_config.vit_patch_size = model_args.vit_patch_size
        dataset_config.max_num_patch_per_side = model_args.vit_max_num_patch_per_side
    if training_args.visual_gen:
        dataset_config.vae_image_downsample = model_args.latent_patch_size * vae_config.downsample
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
        data_status=None,
    )
    train_dataset.set_epoch(data_args.data_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=data_args.num_workers,
        pin_memory=True,
        collate_fn=collate_wrapper(),
        drop_last=True,
        prefetch_factor=data_args.prefetch_factor,
    )

    # ======================================================================
    # Training loop
    # ======================================================================
    if training_args.visual_gen:
        vae_model.to(device).eval()
    fsdp_model.train()

    start_time = time()
    logger.info(f"SCFM training for {training_args.total_steps} steps ...")
    data_status = None

    for curr_step, data in enumerate(train_loader, start=0):
        if curr_step > training_args.total_steps:
            break
        data = data.cuda(device).to_dict()
        data_indexes = data.pop("batch_data_indexes", None)
        ce_loss_weights = data.pop("ce_loss_weights", None)
        data.pop("stage_weights", None)

        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            if training_args.visual_gen:
                with torch.no_grad():
                    data["padded_latent"] = vae_model.encode(data.pop("padded_images"))

            # Decide: normal velocity step or shortcut distillation step
            is_shortcut = random.random() > training_args.non_shortcut_ratio

            if not is_shortcut:
                # ---- Normal velocity loss (identical to base training) ----
                loss_dict = fsdp_model(**data)
            else:
                # ---- SCFM shortcut distillation step ----
                loss_dict = _scfm_shortcut_step(
                    fsdp_model=fsdp_model,
                    model=model,  # unwrapped, for calling helper methods
                    data=data,
                    lora_ema=lora_ema,
                    training_args=training_args,
                    device=device,
                )

        # ------------------------------------------------------------------
        # Loss aggregation (same as pretrain_unified_navit.py)
        # ------------------------------------------------------------------
        loss = 0
        ce = loss_dict.get("ce")
        if ce is not None:
            total_ce_tokens = torch.tensor(len(data["ce_loss_indexes"]), device=device)
            dist.all_reduce(total_ce_tokens, op=dist.ReduceOp.SUM)
            ce = ce.sum() * dist.get_world_size() / total_ce_tokens.clamp(min=1)
            loss_dict["ce"] = ce.detach()
            loss = loss + ce * training_args.ce_weight
        else:
            loss_dict["ce"] = torch.tensor(0, device=device)

        if training_args.visual_gen and loss_dict.get("mse") is not None:
            mse = loss_dict["mse"]
            total_mse_tokens = torch.tensor(len(data["mse_loss_indexes"]), device=device)
            dist.all_reduce(total_mse_tokens, op=dist.ReduceOp.SUM)
            mse = mse.mean(dim=-1).sum() * dist.get_world_size() / total_mse_tokens.clamp(min=1)
            loss_dict["mse"] = mse.detach()
            weight = training_args.shortcut_loss_weight if is_shortcut else training_args.mse_weight
            loss = loss + mse * weight
        else:
            loss_dict["mse"] = torch.tensor(0, device=device)

        # Backward + optimize
        optimizer.zero_grad()
        loss.backward()
        total_norm = fsdp_model.clip_grad_norm_(training_args.max_grad_norm)
        optimizer.step()
        scheduler.step()

        # Update dual EMA
        lora_ema.update(fsdp_model)

        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        if curr_step % training_args.log_every == 0:
            torch.cuda.synchronize()
            end_time = time()
            steps_per_sec = training_args.log_every / max(end_time - start_time, 1e-6)
            msg = f"(step={curr_step:06d}) mode={'shortcut' if is_shortcut else 'velocity'} "
            wandb_log = {"is_shortcut": float(is_shortcut)}
            for key, value in loss_dict.items():
                if not isinstance(value, torch.Tensor):
                    continue
                avg = torch.tensor(value.item(), device=device)
                dist.all_reduce(avg, op=dist.ReduceOp.SUM)
                avg = avg.item() / dist.get_world_size()
                msg += f"{key}: {avg:.4f}, "
                wandb_log[key] = avg
            msg += f"steps/s: {steps_per_sec:.2f}"
            logger.info(msg)
            wandb_log["lr"] = optimizer.param_groups[0]["lr"]
            wandb_log["total_norm"] = total_norm.item()
            if dist.get_rank() == 0:
                wandb.log(wandb_log, step=curr_step)
            start_time = time()

        # Data status tracking
        if data_status is None:
            data_status = {}
        if data_indexes is not None:
            for item in data_indexes:
                if item["dataset_name"] not in data_status:
                    data_status[item["dataset_name"]] = {}
                data_status[item["dataset_name"]][item["worker_id"]] = item["data_indexes"]

        # Checkpoint saving
        if curr_step > 0 and curr_step % training_args.save_every == 0:
            del loss, loss_dict
            gc.collect()
            torch.cuda.empty_cache()
            if dist.get_rank() == 0:
                gather_list = [None] * dist.get_world_size()
            else:
                gather_list = None
            dist.gather_object(data_status, gather_list, dst=0)
            FSDPCheckpoint.fsdp_save_ckpt(
                ckpt_dir=training_args.checkpoint_dir,
                train_steps=curr_step,
                model=fsdp_model,
                ema_model=None,  # No separate EMA model; dual EMA is LoRA-only
                optimizer=optimizer,
                scheduler=scheduler,
                logger=logger,
                fsdp_config=fsdp_config,
                data_status=gather_list,
            )
            del gather_list

    logger.info("SCFM distillation training complete.")


# ---------------------------------------------------------------------------
# SCFM shortcut step
# ---------------------------------------------------------------------------

def _scfm_shortcut_step(fsdp_model, model, data, lora_ema, training_args, device):
    """Run one SCFM shortcut distillation step.

    1. Patchify VAE latents to get packed_latent_clean.
    2. Sample noise + shortcut timesteps.
    3. Generate teacher trajectory target (no_grad, EMA weights).
    4. Student forward with shortcut target.
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

    # --- Sample noise + shortcut timesteps ---
    noise = torch.randn_like(packed_latent_clean)
    vae_token_seqlens = data["vae_token_seqlens"]
    num_images = len(shapes)

    t_start_per_img, dt_teacher_per_img, dt_shortcut_per_img = sample_scfm_timesteps(
        num_images=num_images,
        t_skip=training_args.t_skip,
        num_euler_steps=training_args.num_euler_steps,
        timestep_shift=training_args.timestep_shift,
        device=packed_latent_clean.device,
    )

    # Broadcast per-image timesteps to per-token
    t_start = torch.repeat_interleave(t_start_per_img, vae_token_seqlens)
    dt_teacher = torch.repeat_interleave(
        dt_teacher_per_img, vae_token_seqlens.unsqueeze(1).expand(-1, training_args.t_skip), dim=0
    )

    # --- Build packed_sequence_base (text + ViT, reusable across teacher steps) ---
    with torch.no_grad():
        packed_seq_base = model.prepare_sequence_base(
            sequence_length=data["sequence_length"],
            packed_text_ids=data["packed_text_ids"],
            packed_text_indexes=data["packed_text_indexes"],
            packed_vit_tokens=data.get("packed_vit_tokens"),
            packed_vit_token_indexes=data.get("packed_vit_token_indexes"),
            packed_vit_position_ids=data.get("packed_vit_position_ids"),
            vit_token_seqlens=data.get("vit_token_seqlens"),
        )

    # --- Build MoE routing inputs ---
    extra_inputs = {}
    if model.use_moe:
        packed_und_token_indexes = data["packed_text_indexes"]
        if data.get("packed_vit_token_indexes") is not None:
            packed_und_token_indexes = torch.cat(
                [data["packed_text_indexes"], data["packed_vit_token_indexes"]], dim=0
            )
        extra_inputs["packed_und_token_indexes"] = packed_und_token_indexes
        extra_inputs["packed_gen_token_indexes"] = data["packed_vae_token_indexes"]

    # --- Teacher trajectory (no_grad) ---
    with torch.no_grad():
        target_v = compute_teacher_trajectory(
            model=model,
            fsdp_model=fsdp_model,
            lora_ema=lora_ema,
            packed_seq_base=packed_seq_base,
            packed_latent_clean=packed_latent_clean,
            noise=noise,
            t_start_per_token=t_start,
            dt_teacher_per_token=dt_teacher,
            t_skip=training_args.t_skip,
            packed_vae_token_indexes=data["packed_vae_token_indexes"],
            packed_latent_position_ids=data["packed_latent_position_ids"],
            sample_lens=data["sample_lens"],
            packed_position_ids=data["packed_position_ids"],
            attention_mask=data.get("nested_attention_masks"),
            mse_loss_indexes=data["mse_loss_indexes"],
            **extra_inputs,
        )

    # --- Student forward with SCFM mode ---
    x_t_start = (1.0 - t_start[:, None]) * packed_latent_clean + t_start[:, None] * noise

    loss_dict = fsdp_model(
        **data,
        scfm_mode=True,
        scfm_x_t=x_t_start,
        scfm_timesteps=t_start,
        scfm_target=target_v,
    )

    return loss_dict


if __name__ == "__main__":
    main()
