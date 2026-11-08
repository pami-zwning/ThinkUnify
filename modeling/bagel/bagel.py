# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.attention.flex_attention import create_block_mask
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from data.data_utils import (
    create_sparse_mask,
    get_flattened_position_ids_extrapolate,
    get_flattened_position_ids_interpolate,
    patchify,
)
from .qwen2_navit import NaiveCache
from .modeling_utils import MLPconnector, TimestepEmbedder, PositionEmbedding
from .reconstruction_alignment import ReconstructionAlignmentLoss
from .cfg_distillation import CFGPredictor
from .qformer import QFormerBlock, compute_distillation_loss
from modeling.cache_utils.taylorseer import cache_init

from tqdm import tqdm

from transformers.utils import logging

logger = logging.get_logger(__name__)


def _apply_apg(v_cond, v_uncond, cfg_scale, eta):
    """Adaptive Projected Guidance (APG).

    Decomposes CFG update into parallel/perpendicular components relative to v_cond.
    Suppresses the parallel component (causes oversaturation) while preserving the
    perpendicular component (provides effective directional guidance).

    Args:
        v_cond: Conditional velocity prediction
        v_uncond: Unconditional velocity prediction
        cfg_scale: CFG guidance scale
        eta: Parallel component suppression factor (0=full suppression, 1=standard CFG)

    Returns:
        Guided velocity with reduced oversaturation
    """
    delta = v_cond - v_uncond

    # Project delta onto v_cond direction
    dot = (delta * v_cond).sum(dim=-1, keepdim=True)
    norm_sq = (v_cond * v_cond).sum(dim=-1, keepdim=True) + 1e-8
    delta_parallel = (dot / norm_sq) * v_cond
    delta_perp = delta - delta_parallel

    # Suppress parallel, preserve perpendicular
    delta_apg = eta * delta_parallel + delta_perp

    return v_uncond + cfg_scale * delta_apg

class StyleCrossAttentionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads=16, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.norm_q = nn.LayerNorm(hidden_size)
        self.norm_kv = nn.LayerNorm(hidden_size)
        
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True 
        )
        
        self.norm_ffn = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, query, key_value, attn_mask=None, key_padding_mask=None):
        """
        Args:
            query: (Batch, Target_Seq, Dim)
            key_value: (Batch, Source_Seq, Dim)
            attn_mask: (Target_Seq, Source_Seq) or (Batch*Heads, Target, Source)
            key_padding_mask: vanilla (Batch, Source_Seq) padding mask
        """
        residual = query
        q_norm = self.norm_q(query)
        kv_norm = self.norm_kv(key_value)

        attn_out, _ = self.attn(
            query=q_norm, 
            key=kv_norm, 
            value=kv_norm, 
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask 
        )
        
        query = residual + attn_out
        residual = query
        ffn_out = self.ffn(self.norm_ffn(query))
        query = residual + ffn_out
        return query

class BagelConfig(PretrainedConfig):
    def __init__(
        self,
        visual_gen=True,
        visual_und=True,
        llm_config=None,
        vit_config=None,
        vae_config=None,
        latent_patch_size=2,
        max_latent_size=32,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        interpolate_pos=False,
        timestep_shift=1.0,
        use_snr_loss=True,
        snr_gamma=5.0,
        anchor_und=True,
        anchor_gen=False,
        enable_reconstruction_alignment=False,
        recon_gradient_weight=0.5,
        recon_spectral_weight=0.1,
        # Q-Former distillation
        enable_qformer_distill=False,
        qformer_num_layers=4,
        qformer_num_heads=16,
        qformer_alpha=0.0,
        qformer_noise_sigma=0.0,
        qformer_robust_weight=0.0,
        # Q-Former version: "v2" (original) or "v3" (SigLIP-aligned)
        qformer_version="v2",
        qformer_num_ca_blocks=4,   # v3 only: number of CA+SA blocks
        qformer_num_sa_blocks=8,   # v3 only: number of SA+SA blocks
        qformer_sa_warmup_steps=0, # v3 only: freeze SA layers for first N steps
        **kwargs
    ):
        super().__init__(**kwargs)
        self.visual_gen = visual_gen
        self.visual_und = visual_und
        self.llm_config = llm_config
        self.vit_config = vit_config
        self.vae_config = vae_config
        self.latent_patch_size = latent_patch_size
        self.max_latent_size = max_latent_size
        self.vit_max_num_patch_per_side = vit_max_num_patch_per_side
        self.connector_act = connector_act
        self.interpolate_pos = interpolate_pos
        self.timestep_shift = timestep_shift
        self.use_snr_loss = use_snr_loss
        self.snr_gamma = snr_gamma
        self.anchor_und = anchor_und
        self.anchor_gen = anchor_gen
        self.enable_reconstruction_alignment = enable_reconstruction_alignment
        self.recon_gradient_weight = recon_gradient_weight
        self.recon_spectral_weight = recon_spectral_weight
        self.enable_qformer_distill = enable_qformer_distill
        self.qformer_num_layers = qformer_num_layers
        self.qformer_num_heads = qformer_num_heads
        self.qformer_alpha = qformer_alpha
        self.qformer_noise_sigma = qformer_noise_sigma
        self.qformer_robust_weight = qformer_robust_weight
        self.qformer_version = qformer_version
        self.qformer_num_ca_blocks = qformer_num_ca_blocks
        self.qformer_num_sa_blocks = qformer_num_sa_blocks
        self.qformer_sa_warmup_steps = qformer_sa_warmup_steps


class Bagel(PreTrainedModel):
    config_class = BagelConfig
    base_model_prefix = 'bagel'

    def __init__(self, language_model, vit_model, config: BagelConfig):
        super().__init__(config)
        self.language_model = language_model
        self.hidden_size = config.llm_config.hidden_size
        self.use_moe = "Mo" in config.llm_config.layer_module
        self.num_heads = config.llm_config.num_attention_heads

        if config.visual_gen:
            self.latent_patch_size = config.latent_patch_size
            self.timestep_shift = config.timestep_shift
            self.latent_downsample = config.vae_config.downsample * config.latent_patch_size
            self.max_latent_size = config.max_latent_size
            self.latent_channel = config.vae_config.z_channels
            self.patch_latent_dim = self.latent_patch_size ** 2 * self.latent_channel
            self.time_embedder = TimestepEmbedder(self.hidden_size)
            self.vae2llm = nn.Linear(self.patch_latent_dim, self.hidden_size)
            self.llm2vae = nn.Linear(self.hidden_size, self.patch_latent_dim)
            self.latent_pos_embed = PositionEmbedding(self.max_latent_size, self.hidden_size)

        if config.visual_und:
            self.vit_model = vit_model
            self.vit_patch_size = config.vit_config.patch_size
            self.vit_max_num_patch_per_side = config.vit_max_num_patch_per_side
            self.vit_hidden_size = config.vit_config.hidden_size
            self.connector = MLPconnector(self.vit_hidden_size, self.hidden_size, config.connector_act)
            self.vit_pos_embed = PositionEmbedding(self.vit_max_num_patch_per_side, self.hidden_size)

        # Q-Former distillation module
        if config.enable_qformer_distill:
            assert config.visual_und and config.visual_gen, \
                "Q-Former distillation requires both visual_und and visual_gen"
            if getattr(config, 'qformer_version', 'v2') == 'v3':
                from .qformer_v3 import QFormerV3
                # Get SigLIP config for creating structurally-identical SA layers
                siglip_config = vit_model.config if hasattr(vit_model, 'config') else None
                # If SiglipVisionModel wraps the transformer, get the inner config
                if siglip_config is None and hasattr(vit_model, 'vision_model'):
                    siglip_config = vit_model.vision_model.config
                # Internal Q-Former dim = SigLIP hidden_size (so we can load SigLIP weights).
                # Output dim = LLM hidden_size (so it fits into the sequence).
                self.qformer = QFormerV3(
                    hidden_size=siglip_config.hidden_size if siglip_config is not None else self.hidden_size,
                    vae_latent_dim=self.patch_latent_dim,
                    max_query_size=config.vit_max_num_patch_per_side ** 2,
                    num_ca_blocks=getattr(config, 'qformer_num_ca_blocks', 4),
                    num_sa_blocks=getattr(config, 'qformer_num_sa_blocks', 8),
                    siglip_config=siglip_config,
                    num_heads=config.qformer_num_heads,
                    vit_max_num_patch_per_side=config.vit_max_num_patch_per_side,
                    output_hidden_size=self.hidden_size,
                )
                # Load SigLIP encoder layer weights into QFormerV3 SA layers + CA FFN
                if hasattr(vit_model, 'vision_model'):
                    siglip_layers = vit_model.vision_model.encoder.layers
                else:
                    siglip_layers = vit_model.encoder.layers
                self.qformer.load_siglip_weights(siglip_layers)
            else:
                self.qformer = QFormerBlock(
                    hidden_size=self.hidden_size,
                    vae_latent_dim=self.patch_latent_dim,
                    max_query_size=config.vit_max_num_patch_per_side ** 2,
                    num_layers=config.qformer_num_layers,
                    num_heads=config.qformer_num_heads,
                    vit_max_num_patch_per_side=config.vit_max_num_patch_per_side,
                )
            self.qformer_alpha = config.qformer_alpha


        # Initialize reconstruction alignment loss if enabled
        if config.enable_reconstruction_alignment:
            self.reconstruction_alignment = ReconstructionAlignmentLoss(
                scale_factor=1.0,
                gradient_weight=config.recon_gradient_weight,
                spectral_weight=config.recon_spectral_weight,
            )

        if config.interpolate_pos:
            self.get_flattened_position_ids = get_flattened_position_ids_interpolate
        else:
            self.get_flattened_position_ids = get_flattened_position_ids_extrapolate

        self.config = config
        self._init_weights()

    def _init_weights(self):
        if self.config.visual_gen:
            nn.init.constant_(self.llm2vae.weight, 0)
            nn.init.constant_(self.llm2vae.bias, 0)

    def forward(
        self,
        sequence_length: int,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        sample_lens: List[int],
        packed_position_ids: torch.LongTensor,
        nested_attention_masks: List[torch.Tensor] = None,
        split_lens: List[int] = None,
        attn_modes: List[str] = None,
        # --- Visual Understanding Inputs ---
        ce_loss_indexes: Optional[torch.BoolTensor] = None,
        packed_label_ids: Optional[torch.LongTensor] = None,
        packed_vit_tokens: Optional[torch.Tensor] = None,
        packed_vit_token_indexes: Optional[torch.LongTensor] = None,
        packed_vit_position_ids: Optional[torch.LongTensor] = None,
        vit_token_seqlens: Optional[torch.IntTensor] = None,
        vit_cnt: Optional[List[int]] = None,
        vit_cnt_group: Optional[List[int]] = None,
        # --- Visual Generation Inputs ---
        padded_latent: Optional[torch.Tensor] = None,
        patchified_vae_latent_shapes: Optional[List[Tuple[int, int]]] = None,
        packed_latent_position_ids: Optional[torch.LongTensor] = None,
        packed_vae_token_indexes: Optional[torch.LongTensor] = None,
        packed_timesteps: Optional[torch.LongTensor] = None,
        mse_loss_indexes: Optional[torch.BoolTensor] = None,
        vae_token_seqlens: Optional[torch.IntTensor] = None,
        vae_idx: Optional[List[int]] = None,
        vae_idx_group: Optional[List[int]] = None,
        wo_loss_packed_vae_token_indexes: Optional[torch.LongTensor] = None,
        wo_loss_vae_token_seqlens: Optional[torch.IntTensor] = None,
        wo_loss_vae_idx: Optional[List[int]] = None,
        wo_loss_vae_idx_group: Optional[List[int]] = None,
        loss_vae_token_seqlens: Optional[torch.IntTensor] = None,
        loss_vae_idx: Optional[List[int]] = None,
        loss_vae_idx_group: Optional[List[int]] = None,
        stage_weights: Optional[torch.Tensor] = None,
        n_copy_group_ids: Optional[torch.LongTensor] = None,  # reserved: per-token copy group IDs for N-Copy monitoring
        # --- Q-Former Distillation ---
        qformer_alpha: Optional[float] = None,
        wo_loss_vae_image_ids: Optional[torch.LongTensor] = None,  # image_id per wo_loss VAE image
        vit_image_ids: Optional[torch.LongTensor] = None,  # image_id per surviving ViT image
        # --- SCFM Distillation Mode ---
        scfm_mode: bool = False,
        scfm_x_t: Optional[torch.Tensor] = None,          # pre-computed noisy latent
        scfm_timesteps: Optional[torch.Tensor] = None,     # pre-computed per-token timesteps (already warped)
        scfm_target: Optional[torch.Tensor] = None,        # teacher trajectory target velocity
        scfm_teacher_mode: bool = False,                  # if True, return raw velocity predictions only
    ) -> Dict[str, torch.Tensor]:

        # 1. Text Embeddings
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros(size=(sequence_length, self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        # 2. Attention Masks
        if nested_attention_masks is None:
            sparse_mask = create_sparse_mask(sample_lens, split_lens, attn_modes, packed_text_embedding.device)
            seqlen = sum(sample_lens)
            block_mask = create_block_mask(
                sparse_mask, B=1, H=self.num_heads, Q_LEN=seqlen, KV_LEN=seqlen, 
                device=packed_text_embedding.device, BLOCK_SIZE=128, _compile=True
            )
            attention_mask = block_mask
        else:
            attention_mask = nested_attention_masks

        # 3. Visual Understanding (SigLIP Encoder) & Style Extraction
        distill_loss = None
        qformer_mse_loss = None
        qformer_attn_loss = None
        packed_vit_feat = None
        vit_mid_features_list = []  # v3 only: intermediate SigLIP features for mid loss
        if self.config.visual_und:
            cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
            cu_seqlens = cu_seqlens.to(torch.int32)
            max_seqlen = torch.max(vit_token_seqlens).item()

            # For QFormerV3 we need intermediate SigLIP features (mid loss)
            # OPTIMIZATION: skip expensive mid-feature extraction during SCFM teacher
            # mode — the teacher only needs velocity predictions, not distillation losses.
            use_v3 = (
                getattr(self.config, 'qformer_version', 'v2') == 'v3'
                and self.config.enable_qformer_distill
                and not scfm_teacher_mode  # skip mid-feature extraction for SCFM teacher
            )
            if use_v3:
                # Use forward hooks to capture intermediate SigLIP features.
                # We MUST call self.vit_model() (not vit_transformer.embeddings()
                # or .encoder directly) because FSDP wraps SiglipVisionTransformer
                # and only AllGathers parameters when its forward() is invoked.
                # Directly accessing sub-modules would see sharded (size-0) params.
                num_ca = getattr(self.config, 'qformer_num_ca_blocks', 4)
                mid_layer_indices = set(range(num_ca))
                mid_feats_dict = {}

                # Get the encoder layers for hook registration
                vit_transformer = self.vit_model.vision_model if hasattr(self.vit_model, 'vision_model') else self.vit_model
                hooks = []
                for idx in mid_layer_indices:
                    layer = vit_transformer.encoder.layers[idx]
                    def _make_hook(layer_idx):
                        def _hook(module, input, output):
                            mid_feats_dict[layer_idx] = output.detach()
                        return _hook
                    hooks.append(layer.register_forward_hook(_make_hook(idx)))

                # Run normal forward (triggers FSDP AllGather properly)
                packed_vit_token_embed = self.vit_model(
                    packed_pixel_values=packed_vit_tokens,
                    packed_flattened_position_ids=packed_vit_position_ids,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                )

                # Remove hooks
                for h in hooks:
                    h.remove()

                # Store mid features in order
                vit_mid_features_list = [mid_feats_dict[i] for i in sorted(mid_feats_dict.keys())]
            else:
                packed_vit_token_embed = self.vit_model(
                    packed_pixel_values=packed_vit_tokens,
                    packed_flattened_position_ids=packed_vit_position_ids,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                )

            packed_vit_token_embed = self.connector(packed_vit_token_embed)
            vit_token_pos_emb = self.vit_pos_embed(packed_vit_position_ids)
            packed_vit_feat = packed_vit_token_embed + vit_token_pos_emb

            # Always place ViT features; step 4b may overwrite wo_loss positions
            # print('packed_vit_token_indexes',packed_vit_token_indexes)
            packed_sequence[packed_vit_token_indexes] = packed_vit_feat
            packed_vit_token_embed = packed_vit_feat

        # 4. Visual Generation (VAE / Flow Matching)
        time_feat = None

        if self.config.visual_gen:
            p = self.latent_patch_size
            packed_latent = []

            for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
                latent = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
                latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
                packed_latent.append(latent)
            
            packed_latent_clean = torch.cat(packed_latent, dim=0)

            # Flow Matching (with SCFM mode support)
            if (scfm_mode or scfm_teacher_mode) and scfm_x_t is not None:
                # SCFM distillation: use externally provided noisy latent and timesteps
                packed_latent = scfm_x_t
                packed_timesteps = scfm_timesteps
                noise = None  # not needed; target is provided externally
                packed_latent_noisy = packed_latent
            else:
                # Standard flow matching: sample noise and construct x_t
                noise = torch.randn_like(packed_latent_clean)
                packed_timesteps = torch.sigmoid(packed_timesteps)
                packed_timesteps = self.timestep_shift * packed_timesteps / (1 + (self.timestep_shift - 1) * packed_timesteps)
                packed_latent = (1 - packed_timesteps[:, None]) * packed_latent_clean + packed_timesteps[:, None] * noise
                packed_latent_noisy = packed_latent  # x_t before projection, needed for reconstruction alignment

            # Compute pure time features
            time_feat = self.time_embedder(packed_timesteps) 
            
            latent_token_pos_emb = self.latent_pos_embed(packed_latent_position_ids)
            packed_latent = self.vae2llm(packed_latent) + time_feat + latent_token_pos_emb
            
            packed_sequence[packed_vae_token_indexes] = packed_latent

        # 4b. Q-Former distillation: use wo_loss clean latent as K/V
        # Data loading guarantees samples have matching ViT<->VAE image_ids.
        # OPTIMIZATION: skip QFormer entirely during SCFM teacher mode — the teacher
        # forward only needs velocity predictions from the LLM, not QFormer distillation.
        if self.config.enable_qformer_distill and self.config.visual_gen and self.config.visual_und:

            rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
            alpha = qformer_alpha if qformer_alpha is not None else getattr(self, 'qformer_alpha', 0.0)

            # Extract wo_loss clean latent from packed_latent_clean
            # Cast to long for torch.isin (BFloat16 not supported under FSDP mixed precision)
            wo_loss_mask = torch.isin(
                packed_vae_token_indexes.long(), wo_loss_packed_vae_token_indexes.long()
            )
            packed_wo_loss_latent_clean = packed_latent_clean[wo_loss_mask]

            # Per-image mapping: find which ViT features correspond to which wo_loss VAE images.
            device = packed_wo_loss_latent_clean.device
            vit_seqlen_list = vit_token_seqlens.tolist()
            wo_loss_vae_seqlen_list = wo_loss_vae_token_seqlens.tolist()

            wo_loss_vit_seqlens_list = []
            wo_loss_vit_feat_indices = []
            valid_vae_img_indices = []

            vit_id_list = vit_image_ids.tolist()
            vae_id_list = wo_loss_vae_image_ids.tolist()

            vit_id_to_info = {}
            vit_offset = 0
            for vit_i, (vid, s) in enumerate(zip(vit_id_list, vit_seqlen_list)):
                vit_id_to_info[vid] = (vit_i, vit_offset, s)
                vit_offset += s

            for vae_i, vae_id in enumerate(vae_id_list):
                if vae_id in vit_id_to_info:
                    _, v_offset, v_seqlen = vit_id_to_info[vae_id]
                    wo_loss_vit_seqlens_list.append(v_seqlen)
                    wo_loss_vit_feat_indices.extend(range(v_offset, v_offset + v_seqlen))
                    valid_vae_img_indices.append(vae_i)

            # Guard: ViT dropout during pack_sequence may have dropped matching ViT images
            if len(wo_loss_vit_seqlens_list) == 0:
                logger.info(f"[rank{rank}] Q-Former skip: no matched pairs after ViT dropout")
            else:
                wo_loss_vit_seqlens_t = torch.tensor(wo_loss_vit_seqlens_list, dtype=vit_token_seqlens.dtype, device=device)
                wo_loss_vit_idx = torch.tensor(wo_loss_vit_feat_indices, dtype=torch.long, device=device)

                if len(valid_vae_img_indices) < len(wo_loss_vae_seqlen_list):
                    vae_splits = torch.split(packed_wo_loss_latent_clean, wo_loss_vae_seqlen_list)
                    filtered_vae_latent = torch.cat([vae_splits[i] for i in valid_vae_img_indices], dim=0)
                    filtered_vae_seqlens = torch.tensor(
                        [wo_loss_vae_seqlen_list[i] for i in valid_vae_img_indices],
                        dtype=wo_loss_vae_token_seqlens.dtype, device=device,
                    )
                else:
                    filtered_vae_latent = packed_wo_loss_latent_clean
                    filtered_vae_seqlens = wo_loss_vae_token_seqlens

                # ── Latent noise augmentation for Q-Former ──
                # During inference the Q-Former receives ODE-denoised latents that
                # carry small numerical errors from Euler integration and CFG/APG
                # guidance.  Adding light Gaussian noise during training simulates
                # this distribution shift so the Q-Former remains robust at test
                # time.  The noise scale (sigma) is annealed linearly with alpha:
                # when alpha is low the Q-Former output is blended with ViT
                # features (low impact), so we use less noise; when alpha → 1 the
                # Q-Former stands alone and needs to be fully robust.
                qformer_noise_sigma = getattr(self.config, 'qformer_noise_sigma', 0.0)
                if self.training and qformer_noise_sigma > 0:
                    sigma = qformer_noise_sigma * alpha  # anneal with alpha
                    noise = torch.randn_like(filtered_vae_latent) * sigma
                    filtered_vae_latent = filtered_vae_latent + noise

                wo_loss_vit_position_ids = packed_vit_position_ids[wo_loss_vit_idx] if packed_vit_position_ids is not None else None

                # ── Q-Former forward: dispatch v2 vs v3 ──
                use_v3 = getattr(self.config, 'qformer_version', 'v2') == 'v3'
                if use_v3:
                    # v3 needs RoPE tensors and returns mid_features
                    v3_rope = {}
                    vit_transformer = self.vit_model.vision_model if hasattr(self.vit_model, 'vision_model') else self.vit_model
                    if hasattr(vit_transformer, 'rope') and vit_transformer.config.rope and wo_loss_vit_position_ids is not None:
                        v3_rope = dict(
                            cos_h=vit_transformer.rope.cos_h[wo_loss_vit_position_ids],
                            sin_h=vit_transformer.rope.sin_h[wo_loss_vit_position_ids],
                            cos_w=vit_transformer.rope.cos_w[wo_loss_vit_position_ids],
                            sin_w=vit_transformer.rope.sin_w[wo_loss_vit_position_ids],
                        )
                    packed_qformer_feat, qf_mid_features, qformer_self_attn_weights, qformer_distill_feat = self.qformer.forward_packed(
                        packed_vae_latent=filtered_vae_latent,
                        vae_token_seqlens=filtered_vae_seqlens,
                        vit_token_seqlens=wo_loss_vit_seqlens_t,
                        packed_position_ids=wo_loss_vit_position_ids,
                        **v3_rope,
                    )
                else:
                    packed_qformer_feat, qformer_self_attn_weights = self.qformer.forward_packed(
                        packed_vae_latent=filtered_vae_latent,
                        vae_token_seqlens=filtered_vae_seqlens,
                        vit_token_seqlens=wo_loss_vit_seqlens_t,
                        packed_position_ids=wo_loss_vit_position_ids,
                    )
                    qf_mid_features = None
                    qformer_distill_feat = None

                wo_loss_vit_positions = packed_vit_token_indexes[wo_loss_vit_idx]
                # print('wo_loss_vit_positions: ', wo_loss_vit_positions)

                if alpha >= 1.0:
                    packed_sequence[wo_loss_vit_positions] = packed_qformer_feat
                else:
                    packed_wo_loss_vit_feat = packed_vit_feat[wo_loss_vit_idx]
                    blended = (1 - alpha) * packed_wo_loss_vit_feat + alpha * packed_qformer_feat
                    packed_sequence[wo_loss_vit_positions] = blended

                    distill_weight = max(0.0, 1.0 - alpha)

                    # ── Distillation loss: dispatch v2 vs v3 ──
                    if use_v3 and qf_mid_features is not None:
                        from .qformer_v3 import compute_distillation_loss_v3
                        # v3: match raw SigLIP features at SigLIP hidden dim (NOT after connector)
                        # Extract raw SigLIP mid features (pre-connector) for wo_loss_vit subset
                        wo_vit_mid_list = [
                            mf[wo_loss_vit_idx] for mf in vit_mid_features_list
                        ]
                        # Get raw SigLIP final feat (pre-connector) for wo_loss_vit subset
                        wo_vit_final_raw = vit_mid_features_list[-1][wo_loss_vit_idx] \
                            if vit_mid_features_list else packed_vit_feat[wo_loss_vit_idx]
                        distill_loss, qformer_mse_loss, qformer_attn_loss, _, _ = compute_distillation_loss_v3(
                            qformer_distill_feat, qf_mid_features, qformer_self_attn_weights,
                            wo_vit_final_raw, wo_vit_mid_list, wo_loss_vit_seqlens_t,
                        )
                    else:
                        if distill_weight > 0:
                            distill_loss, qformer_mse_loss, qformer_attn_loss = compute_distillation_loss(
                                packed_qformer_feat, packed_wo_loss_vit_feat,
                                wo_loss_vit_seqlens_t, qformer_self_attn_weights,
                            )
                            distill_loss = distill_loss * distill_weight
                            qformer_mse_loss = qformer_mse_loss * distill_weight
                            qformer_attn_loss = qformer_attn_loss * distill_weight

        # 5. MoE Routing Inputs
        extra_inputs = {}
        if self.use_moe:
            packed_und_token_indexes = packed_text_indexes
            if packed_vit_token_indexes is not None:
                packed_und_token_indexes=torch.cat([packed_text_indexes, packed_vit_token_indexes], dim=0)
            extra_inputs.update(
                packed_und_token_indexes=packed_und_token_indexes,
                packed_gen_token_indexes=packed_vae_token_indexes,
            )
        # 6. Main Forward Pass
        last_hidden_state = self.language_model(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_ids=packed_position_ids,
            **extra_inputs,
        )


        mse = None
        if self.config.visual_gen:
            packed_mse_preds = self.llm2vae(last_hidden_state[mse_loss_indexes])

            # SCFM teacher mode: return raw velocity predictions, skip loss computation
            if scfm_teacher_mode:
                return dict(
                    mse=None,
                    ce=None,
                    reconstruction_alignment_loss=torch.tensor(0.0, device=last_hidden_state.device),
                    distill_loss=None,
                    qformer_mse_loss=None,
                    qformer_attn_loss=None,
                    scfm_velocity=packed_mse_preds,  # raw velocity for teacher trajectory
                )

            # SCFM mode: use teacher trajectory target; otherwise standard velocity target
            if scfm_mode and scfm_target is not None:
                target = scfm_target  # already [N_loss_tokens, latent_dim]
            else:
                target = noise - packed_latent_clean

            has_mse = packed_timesteps > 0

            if has_mse.any():
                # scfm_target is already loss-token-sized; standard target needs filtering
                if scfm_mode and scfm_target is not None:
                    diff = packed_mse_preds - target
                else:
                    diff = packed_mse_preds - target[has_mse]
                squared_error = diff ** 2  # [N, Hidden_Dim]

                if self.config.use_snr_loss:
                    cur_t = packed_timesteps[has_mse]

                    snr = (1 - cur_t)**2 / (cur_t**2 + 1e-8)
                    gamma = self.config.snr_gamma

                    snr_clamped = torch.minimum(snr, torch.tensor(gamma, device=snr.device))

                    weights = snr_clamped / (1 + snr)
                    mse = squared_error * weights.unsqueeze(-1)

                else:
                    mse = squared_error
                # N-Copy note: when n_copy > 1, multiple copies contribute more MSE tokens,
                # but the training loop normalizes by total_mse_tokens, so the per-image
                # loss is naturally averaged across copies.
            else:
                mse = torch.tensor(0.0, device=last_hidden_state.device, requires_grad=True)

        ce = None
        if ce_loss_indexes is not None:
            packed_ce_preds = self.language_model.lm_head(last_hidden_state[ce_loss_indexes])
            ce = F.cross_entropy(packed_ce_preds, packed_label_ids, reduction="none")
            
            if stage_weights is not None:
                ce = ce * stage_weights

        # E. Reconstruction Alignment Loss (ReAC style)
        reconstruction_alignment_loss = torch.tensor(0.0, device=last_hidden_state.device)
        alignment_metrics = {}

        if self.config.enable_reconstruction_alignment and self.config.visual_gen:
            try:
                if packed_latent_clean is not None and mse_loss_indexes is not None:
                    # packed_latent_noisy/clean/timesteps cover ALL vae tokens (loss + wo_loss),
                    # but packed_mse_preds only covers loss tokens (via mse_loss_indexes).
                    # Filter to loss-only tokens using timestep > 0 (wo_loss images have t=0).
                    has_mse = packed_timesteps > 0
                    loss_latent_noisy = packed_latent_noisy[has_mse]
                    loss_latent_clean = packed_latent_clean[has_mse]
                    loss_timesteps = packed_timesteps[has_mse]

                    # Derive loss-only latent shapes: filter out wo_loss images (all t=0)
                    loss_latent_shapes = []
                    offset = 0
                    for (h, w) in patchified_vae_latent_shapes:
                        n = h * w
                        if packed_timesteps[offset].item() > 0:
                            loss_latent_shapes.append((h, w))
                        offset += n

                    reconstruction_alignment_loss = self.reconstruction_alignment(
                        v_pred=packed_mse_preds,
                        x_t=loss_latent_noisy,
                        x_0_gt=loss_latent_clean,
                        timesteps=loss_timesteps,
                        latent_shapes=loss_latent_shapes if loss_latent_shapes else None,
                        mask=None,
                    )
            except Exception as e:
                import traceback
                logger.warning(f"Failed to compute reconstruction alignment loss: {e}\n{traceback.format_exc()}")
                reconstruction_alignment_loss = torch.tensor(0.0, device=last_hidden_state.device)

        # F. Q-Former robustness loss on predicted x_hat_0
        # After the LLM forward, we have v_theta.  Compute x_hat_0 = x_t - t * v_theta
        # and run Q-Former on it to teach it to handle imperfect denoised latents
        # (closing the train-test distribution gap).
        qformer_robust_loss = torch.tensor(0.0, device=last_hidden_state.device)
        if (
            self.config.enable_qformer_distill
            and self.config.visual_gen
            and self.config.visual_und
            and getattr(self.config, 'qformer_robust_weight', 0.0) > 0
            and packed_vit_feat is not None
            and mse is not None
        ):
            try:
                alpha = qformer_alpha if qformer_alpha is not None else getattr(self, 'qformer_alpha', 0.0)
                if alpha > 0 and has_mse.any():
                    # Compute predicted clean latent from the velocity prediction
                    loss_timesteps = packed_timesteps[has_mse]
                    loss_latent_noisy = packed_latent_noisy[has_mse]
                    x_hat_0 = loss_latent_noisy - loss_timesteps.unsqueeze(-1) * packed_mse_preds.detach()

                    # Map x_hat_0 back to per-image segments matching wo_loss VAE images
                    # We need to match loss VAE images to their ViT counterparts
                    if wo_loss_packed_vae_token_indexes is not None and wo_loss_vae_image_ids is not None:
                        # Filter x_hat_0 to wo_loss images only (those with ViT pairs)
                        loss_vae_mask = torch.isin(
                            packed_vae_token_indexes[has_mse].long(),
                            wo_loss_packed_vae_token_indexes.long()
                        )
                        if loss_vae_mask.any():
                            x_hat_0_wo = x_hat_0[loss_vae_mask]

                            # Reuse the same ViT-VAE matching from step 4b
                            device = x_hat_0_wo.device
                            vit_seqlen_list = vit_token_seqlens.tolist()
                            wo_loss_vae_seqlen_list = wo_loss_vae_token_seqlens.tolist()
                            robust_vit_seqlens_list = []
                            robust_vit_feat_indices = []
                            robust_vae_img_indices = []

                            vit_id_list = vit_image_ids.tolist()
                            vae_id_list = wo_loss_vae_image_ids.tolist()
                            vit_id_to_info = {}
                            vit_offset = 0
                            for _, (vid, s) in enumerate(zip(vit_id_list, vit_seqlen_list)):
                                vit_id_to_info[vid] = (vit_offset, s)
                                vit_offset += s
                            for vae_i, vae_id in enumerate(vae_id_list):
                                if vae_id in vit_id_to_info:
                                    v_offset, v_seqlen = vit_id_to_info[vae_id]
                                    robust_vit_seqlens_list.append(v_seqlen)
                                    robust_vit_feat_indices.extend(range(v_offset, v_offset + v_seqlen))
                                    robust_vae_img_indices.append(vae_i)

                            if robust_vit_seqlens_list:
                                robust_vit_seqlens_t = torch.tensor(robust_vit_seqlens_list, dtype=vit_token_seqlens.dtype, device=device)
                                robust_vit_idx = torch.tensor(robust_vit_feat_indices, dtype=torch.long, device=device)

                                if len(robust_vae_img_indices) < len(wo_loss_vae_seqlen_list):
                                    vae_splits = torch.split(x_hat_0_wo, wo_loss_vae_seqlen_list)
                                    robust_vae_latent = torch.cat([vae_splits[i] for i in robust_vae_img_indices], dim=0)
                                    robust_vae_seqlens = torch.tensor(
                                        [wo_loss_vae_seqlen_list[i] for i in robust_vae_img_indices],
                                        dtype=wo_loss_vae_token_seqlens.dtype, device=device,
                                    )
                                else:
                                    robust_vae_latent = x_hat_0_wo
                                    robust_vae_seqlens = wo_loss_vae_token_seqlens

                                robust_vit_pos = packed_vit_position_ids[robust_vit_idx] if packed_vit_position_ids is not None else None

                                robust_qf_out = self.qformer.forward_packed(
                                    packed_vae_latent=robust_vae_latent,
                                    vae_token_seqlens=robust_vae_seqlens,
                                    vit_token_seqlens=robust_vit_seqlens_t,
                                    packed_position_ids=robust_vit_pos,
                                )
                                robust_qf_feat = robust_qf_out[0]
                                robust_vit_feat = packed_vit_feat[robust_vit_idx].detach()
                                # MSE + cosine loss (same as main distillation)
                                robust_mse = F.mse_loss(robust_qf_feat, robust_vit_feat)
                                cos_sim = F.cosine_similarity(robust_qf_feat, robust_vit_feat, dim=-1).mean()
                                qformer_robust_loss = (robust_mse + 1.0 - cos_sim) * self.config.qformer_robust_weight
            except Exception as e:
                import traceback
                logger.warning(f"Failed to compute Q-Former robust loss: {e}\n{traceback.format_exc()}")

        return dict(
            mse=mse,
            ce=ce,
            reconstruction_alignment_loss=reconstruction_alignment_loss,
            distill_loss=distill_loss,
            qformer_mse_loss=qformer_mse_loss,
            qformer_attn_loss=qformer_attn_loss,
            qformer_robust_loss=qformer_robust_loss,
        )


    # ------------------------------------------------------------------
    # SCFM Distillation Helpers
    # ------------------------------------------------------------------

    def prepare_sequence_base(
        self,
        sequence_length: int,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_vit_tokens: Optional[torch.Tensor] = None,
        packed_vit_token_indexes: Optional[torch.LongTensor] = None,
        packed_vit_position_ids: Optional[torch.LongTensor] = None,
        vit_token_seqlens: Optional[torch.IntTensor] = None,
    ) -> torch.Tensor:
        """Build packed_sequence with text + ViT embeddings, without VAE tokens.

        This is reusable across multiple teacher Euler steps in SCFM distillation,
        since only the VAE token positions change between steps.
        """
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros(
            size=(sequence_length, self.hidden_size)
        )
        packed_sequence[packed_text_indexes] = packed_text_embedding

        if self.config.visual_und and packed_vit_tokens is not None:
            cu_seqlens = torch.nn.functional.pad(
                torch.cumsum(vit_token_seqlens, dim=0), (1, 0)
            ).to(torch.int32)
            max_seqlen = torch.max(vit_token_seqlens).item()

            packed_vit_token_embed = self.vit_model(
                packed_pixel_values=packed_vit_tokens,
                packed_flattened_position_ids=packed_vit_position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            packed_vit_token_embed = self.connector(packed_vit_token_embed)
            vit_token_pos_emb = self.vit_pos_embed(packed_vit_position_ids)
            packed_vit_feat = packed_vit_token_embed + vit_token_pos_emb
            packed_sequence[packed_vit_token_indexes] = packed_vit_feat

        return packed_sequence

    def forward_scfm_teacher_step(
        self,
        packed_seq_base: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_latent_position_ids: torch.LongTensor,
        sample_lens: List[int],
        packed_position_ids: torch.LongTensor,
        attention_mask,
        mse_loss_indexes: torch.BoolTensor,
        **extra_inputs,
    ) -> torch.Tensor:
        """Single teacher Euler step for SCFM trajectory generation.

        Injects noisy latent x_t at the given timestep into the pre-built
        packed_sequence_base, runs the LLM forward, and extracts velocity
        prediction for loss-bearing VAE tokens.

        Args:
            packed_seq_base: packed sequence with text+ViT already placed.
            x_t: [N_vae_tokens, latent_dim] noisy latent at current timestep.
            timestep: [N_vae_tokens] per-token timestep values (already warped).
            ... remaining args for LLM forward.

        Returns:
            v_t: [N_loss_tokens, latent_dim] velocity prediction for loss tokens.
        """
        packed_sequence = packed_seq_base.clone()

        # Embed noisy latent: project + time + position
        time_feat = self.time_embedder(timestep)
        latent_pos_emb = self.latent_pos_embed(packed_latent_position_ids)
        packed_sequence[packed_vae_token_indexes] = (
            self.vae2llm(x_t) + time_feat + latent_pos_emb
        )

        # LLM forward
        last_hidden = self.language_model(
            packed_sequence=packed_sequence,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_ids=packed_position_ids,
            **extra_inputs,
        )

        # Extract velocity prediction for loss tokens only
        return self.llm2vae(last_hidden[mse_loss_indexes])


    def vit2vae_idx_convert(self, vit_cnt, vae_idx, vit_cnt_group, vae_idx_group):
        device = vit_cnt.device
        vit_list, vae_list = vit_cnt.tolist(), vae_idx.tolist()
        vit_indices = range(len(vit_list)) 

        valid_res, added_res = [], []
        v_ptr, a_ptr = 0, 0

        for v_len, a_len in zip(vit_cnt_group.tolist(), vae_idx_group.tolist()):
            
            curr_vals = vit_list[v_ptr : v_ptr + v_len]
            curr_idxs = vit_indices[v_ptr : v_ptr + v_len]
            lookup = dict(zip(reversed(curr_vals), reversed(curr_idxs)))
            

            curr_vae = vae_list[a_ptr : a_ptr + a_len]
            
            batch_added = [lookup.get(x, -1) if x != -1 else -1 for x in curr_vae]

            batch_valid = [x if idx != -1 else -1 for x, idx in zip(curr_vae, batch_added)]

            valid_res.extend(batch_valid)
            added_res.extend(batch_added)
            
            v_ptr += v_len
            a_ptr += a_len

        return torch.tensor(valid_res, device=device, dtype=torch.long), torch.tensor(added_res, device=device, dtype=torch.long)

    def prepare_prompts(self, curr_kvlens, curr_rope, prompts, tokenizer, new_token_ids):
        packed_text_ids = list()
        packed_text_position_ids = list()
        text_token_lens = list()
        packed_text_indexes = list()
        packed_key_value_indexes = list()

        curr = 0
        newlens, new_rope = list(), list()
        for prompt, curr_kvlen, curr_position_id in zip(prompts, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            text_ids = tokenizer.encode(prompt)
            text_ids = [new_token_ids['bos_token_id']] + text_ids + [new_token_ids['eos_token_id']]
            text_token_lens.append(len(text_ids))
            packed_text_ids.extend(text_ids)
            packed_text_position_ids.extend(range(curr_position_id, curr_position_id + len(text_ids)))
            packed_text_indexes.extend(range(curr, curr + len(text_ids)))
            newlens.append(curr_kvlen + len(text_ids))
            new_rope.append(curr_position_id + len(text_ids))
            curr += len(text_ids)

        generation_input = {
            "text_token_lens": torch.tensor(text_token_lens, dtype=torch.int),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_position_ids": torch.tensor(packed_text_position_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_text(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.IntTensor,
        packed_text_position_ids: torch.LongTensor,
        text_token_lens: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_text_embedding,
            query_lens=text_token_lens,
            packed_query_position_ids=packed_text_position_ids,
            packed_query_indexes=packed_text_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=True,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vit_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids):
        packed_vit_token_indexes = list()
        vit_token_seqlens, packed_vit_tokens, packed_vit_position_ids = list(), list(), list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        newlens, new_rope = list(), list()
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = transforms(image)
            vit_position_ids = self.get_flattened_position_ids(
                image_tensor.size(1), image_tensor.size(2), 
                self.vit_patch_size, 
                max_num_patches_per_side=self.vit_max_num_patch_per_side
            )
            vit_tokens = patchify(image_tensor, self.vit_patch_size)
            packed_vit_tokens.append(vit_tokens)
            num_img_tokens = vit_tokens.shape[0]
            packed_vit_position_ids.append(vit_position_ids)
            vit_token_seqlens.append(num_img_tokens)
            packed_vit_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "vit_token_seqlens": torch.tensor(vit_token_seqlens, dtype=torch.int),
            "packed_vit_tokens": torch.cat(packed_vit_tokens, dim=0),
            "packed_vit_position_ids": torch.cat(packed_vit_position_ids, dim=0),
            "packed_vit_token_indexes": torch.tensor(packed_vit_token_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_vit(
        self,
        past_key_values: NaiveCache,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_vit_tokens: torch.Tensor,
        packed_vit_token_indexes: torch.LongTensor,
        packed_vit_position_ids: torch.LongTensor,
        vit_token_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_clean_vae_latent: torch.Tensor = None,
        clean_vae_token_seqlens: torch.IntTensor = None,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        use_qformer_only = (
            self.config.enable_qformer_distill
            and getattr(self, 'qformer_alpha', 0.0) >= 1.0
            and packed_clean_vae_latent is not None
        )

        if use_qformer_only:
            # Stage 2 / inference: skip ViT, use Q-Former on denoised latent
            qformer_out = self.qformer.forward_packed(
                packed_vae_latent=packed_clean_vae_latent,
                vae_token_seqlens=clean_vae_token_seqlens,
                vit_token_seqlens=vit_token_seqlens,
                packed_position_ids=packed_vit_position_ids,
            )
            # v2 returns (feat, attn); v3 returns (feat, mid, attn, distill_feat)
            packed_qformer_feat = qformer_out[0]
            if packed_qformer_feat.dtype != packed_sequence.dtype:
                packed_qformer_feat = packed_qformer_feat.to(packed_sequence.dtype)
            packed_sequence[packed_vit_token_indexes] = packed_qformer_feat

        else:
            # Standard path: use ViT
            cu_seqlens = torch.nn.functional.pad(torch.cumsum(vit_token_seqlens, dim=0), (1, 0))
            cu_seqlens = cu_seqlens.to(torch.int32)
            max_seqlen = torch.max(vit_token_seqlens).item()
            packed_vit_token_embed = self.vit_model(
                packed_pixel_values=packed_vit_tokens,
                packed_flattened_position_ids=packed_vit_position_ids,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

            packed_vit_token_embed = self.connector(packed_vit_token_embed)
            pos_emb = self.vit_pos_embed(packed_vit_position_ids)
            packed_vit_token_embed = packed_vit_token_embed + pos_emb
            if packed_vit_token_embed.dtype != packed_sequence.dtype:
                packed_vit_token_embed = packed_vit_token_embed.to(packed_sequence.dtype)
            packed_sequence[packed_vit_token_indexes] = packed_vit_token_embed

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {"mode": "und"}

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            packed_key_value_indexes=packed_key_value_indexes,
            key_values_lens=key_values_lens,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vae_images(self, curr_kvlens, curr_rope, images, transforms, new_token_ids, timestep=0):
        patchified_vae_latent_shapes, packed_vae_position_ids = list(), list()
        packed_vae_token_indexes = list()
        packed_text_ids, packed_text_indexes = list(), list()
        packed_seqlens, packed_position_ids, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        _curr = curr = 0
        vae_image_tensors = list()
        newlens, new_rope = list(), list()
        for image, curr_kvlen, curr_position_id in zip(images, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            image_tensor = transforms(image)
            vae_image_tensors.append(image_tensor)
            vae_posiiton_ids = self.get_flattened_position_ids(
                image_tensor.size(1), image_tensor.size(2),
                self.latent_downsample, 
                max_num_patches_per_side=self.max_latent_size
            )
            packed_vae_position_ids.append(vae_posiiton_ids)
            H, W = image_tensor.shape[1:]
            h = H // self.latent_downsample
            w = W // self.latent_downsample
            patchified_vae_latent_shapes.append((h, w))

            num_img_tokens = w * h
            packed_vae_token_indexes.extend(range(_curr, _curr + num_img_tokens))
            packed_indexes.extend(range(curr, curr + num_img_tokens))
            curr += num_img_tokens
            _curr += num_img_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(_curr)
            packed_indexes.append(curr)
            curr += 1
            _curr += 1

            packed_position_ids.extend([curr_position_id] * (num_img_tokens + 2))
            packed_seqlens.append(num_img_tokens + 2)
            newlens.append(curr_kvlen + num_img_tokens + 2)
            new_rope.append(curr_position_id + 1)

        image_sizes = [item.shape for item in vae_image_tensors]
        max_image_size = [max(item) for item in list(zip(*image_sizes))]
        padded_images = torch.zeros(size=(len(vae_image_tensors), *max_image_size))
        for i, image_tensor in enumerate(vae_image_tensors):
            padded_images[i, :, :image_tensor.shape[1], :image_tensor.shape[2]] = image_tensor

        generation_input = {
            "padded_images": padded_images,
            "patchified_vae_latent_shapes": patchified_vae_latent_shapes,
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_timesteps": torch.tensor([timestep]),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
        }

        return generation_input, newlens, new_rope

    @torch.no_grad
    def forward_cache_update_vae(
        self,
        vae_model,
        past_key_values: NaiveCache,
        padded_images: torch.Tensor,
        patchified_vae_latent_shapes: List,
        packed_vae_position_ids: torch.LongTensor,
        packed_timesteps: torch.Tensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.Tensor,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        padded_latent = vae_model.encode(padded_images)

        p = self.latent_patch_size
        packed_latent = list()
        for latent, (h, w) in zip(padded_latent, patchified_vae_latent_shapes):
            latent = latent[:, :h * p, :w * p].reshape(self.latent_channel, h, p, w, p)
            latent = torch.einsum("chpwq->hwpqc", latent).reshape(-1, p * p * self.latent_channel)
            packed_latent.append(latent)
        packed_latent = torch.cat(packed_latent, dim=0)
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(packed_timesteps)
        packed_latent = self.vae2llm(packed_latent) + packed_timestep_embeds + packed_pos_embed
        if packed_latent.dtype != packed_sequence.dtype:
            packed_latent = packed_latent.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = packed_latent

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes
            }

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=True,
            is_causal=False,
            **extra_inputs,
        )
        past_key_values = output.past_key_values

        return past_key_values

    def prepare_vae_latent(self, curr_kvlens, curr_rope, image_sizes, new_token_ids):
        packed_text_ids, packed_text_indexes = list(), list()
        packed_vae_position_ids, packed_vae_token_indexes, packed_init_noises = list(), list(), list()
        packed_position_ids, packed_seqlens, packed_indexes = list(), list(), list()
        packed_key_value_indexes = list()

        query_curr = curr = 0
        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_text_ids.append(new_token_ids['start_of_image'])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            vae_posiiton_ids = self.get_flattened_position_ids(
                H, W,
                self.latent_downsample, 
                max_num_patches_per_side=self.max_latent_size
            )
            packed_vae_position_ids.append(vae_posiiton_ids)

            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_image_tokens = h * w
            packed_init_noises.append(
                torch.randn(num_image_tokens, self.latent_channel * self.latent_patch_size ** 2)
            )
            packed_vae_token_indexes.extend(range(query_curr, query_curr + num_image_tokens))
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens

            packed_text_ids.append(new_token_ids['end_of_image'])
            packed_text_indexes.append(query_curr)
            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))
            packed_seqlens.append(num_image_tokens + 2)

        generation_input = {
            "packed_text_ids": torch.tensor(packed_text_ids, dtype=torch.long),
            "packed_text_indexes": torch.tensor(packed_text_indexes, dtype=torch.long),
            "packed_init_noises": torch.cat(packed_init_noises, dim=0),
            "packed_vae_position_ids": torch.cat(packed_vae_position_ids, dim=0),
            "packed_vae_token_indexes": torch.tensor(packed_vae_token_indexes, dtype=torch.long),
            "packed_seqlens": torch.tensor(packed_seqlens, dtype=torch.int),
            "packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    def prepare_vae_latent_cfg(self, curr_kvlens, curr_rope, image_sizes):
        packed_position_ids, packed_indexes, packed_key_value_indexes = list(), list(), list()

        query_curr = curr = 0
        for (H, W), curr_kvlen, curr_position_id in zip(image_sizes, curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            curr += curr_kvlen

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            h, w = H // self.latent_downsample, W // self.latent_downsample
            num_image_tokens = h * w
            packed_indexes.extend(range(curr, curr + num_image_tokens))
            curr += num_image_tokens
            query_curr += num_image_tokens

            packed_indexes.append(curr)
            curr += 1
            query_curr += 1

            packed_position_ids.extend([curr_position_id] * (num_image_tokens + 2))

        generation_input = {
            "cfg_packed_position_ids": torch.tensor(packed_position_ids, dtype=torch.long),
            "cfg_key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "cfg_packed_query_indexes": torch.tensor(packed_indexes, dtype=torch.long),
            "cfg_packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    @torch.no_grad
    def generate_image(
        self,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_init_noises: torch.Tensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        past_key_values: NaiveCache,
        key_values_lens: torch.IntTensor,
        packed_key_value_indexes: torch.LongTensor,
        num_timesteps: int = 24,
        timestep_shift: float = 1.0,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        cfg_interval: Optional[Tuple[float, float]] = [0, 1],
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_key_values_lens: Optional[torch.IntTensor] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
        # cache_args
        enable_taylorseer=False,
        # CFG-Zero*: zero-init first N steps + optimal rescale
        cfg_zero_init_steps: int = 0,
        cfg_zero_optimal_rescale: float = 0.0,
        # APG: adaptive projected guidance
        cfg_guidance_mode: str = "renorm",  # "renorm" (existing) or "apg"
        apg_text_eta: float = 0.5,
        apg_img_eta: float = 0.5,
        # CFG Distillation: use lightweight predictor instead of 3-pass CFG
        cfg_predictor: Optional[CFGPredictor] = None,
    ):
        if enable_taylorseer:
            self.language_model.model.enable_taylorseer = True
            model_pred_cache_dic, model_pred_current = cache_init(self, num_timesteps)
            model_pred_text_cache_dic, model_pred_text_current = cache_init(self, num_timesteps)
            model_pred_img_cache_dic, model_pred_img_current = cache_init(self, num_timesteps)
        else:
            self.language_model.model.enable_taylorseer = False
            model_pred_cache_dic, model_pred_current = None, None
            model_pred_text_cache_dic, model_pred_text_current = None, None
            model_pred_img_cache_dic, model_pred_img_current = None, None
    
        x_t = packed_init_noises

        timesteps = torch.linspace(1, 0, num_timesteps, device=x_t.device)
        timesteps = timestep_shift * timesteps / (1 + (timestep_shift - 1) * timesteps)
        dts =  timesteps[:-1] - timesteps[1:]
        timesteps = timesteps[:-1]

        for i, t in tqdm(enumerate(timesteps), total=len(timesteps)):

            timestep = torch.tensor([t] * x_t.shape[0], device=x_t.device)
            if t > cfg_interval[0] and t <= cfg_interval[1]:
                cfg_text_scale_ = cfg_text_scale
                cfg_img_scale_ = cfg_img_scale
            else:
                cfg_text_scale_ = 1.0
                cfg_img_scale_ = 1.0
            v_t = self._forward_flow(
                x_t=x_t,
                timestep=timestep,
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_vae_position_ids=packed_vae_position_ids,
                packed_text_ids=packed_text_ids,
                packed_text_indexes=packed_text_indexes,
                packed_position_ids=packed_position_ids,
                packed_indexes=packed_indexes,
                packed_seqlens=packed_seqlens,
                key_values_lens=key_values_lens,
                past_key_values=past_key_values,
                packed_key_value_indexes=packed_key_value_indexes,
                cfg_renorm_min=cfg_renorm_min,
                cfg_renorm_type=cfg_renorm_type,
                # cfg_text
                cfg_text_scale=cfg_text_scale_,
                cfg_text_packed_position_ids=cfg_text_packed_position_ids,
                cfg_text_packed_query_indexes=cfg_text_packed_query_indexes,
                cfg_text_key_values_lens=cfg_text_key_values_lens,
                cfg_text_past_key_values=cfg_text_past_key_values,
                cfg_text_packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                # cfg_img
                cfg_img_scale=cfg_img_scale_,
                cfg_img_packed_position_ids=cfg_img_packed_position_ids,
                cfg_img_packed_query_indexes=cfg_img_packed_query_indexes,
                cfg_img_key_values_lens=cfg_img_key_values_lens,
                cfg_img_past_key_values=cfg_img_past_key_values,
                cfg_img_packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                cfg_type=cfg_type,
                # cache
                model_pred_cache_dic=model_pred_cache_dic,
                model_pred_current=model_pred_current,
                model_pred_text_cache_dic=model_pred_text_cache_dic,
                model_pred_text_current=model_pred_text_current,
                model_pred_img_cache_dic=model_pred_img_cache_dic,
                model_pred_img_current=model_pred_img_current,
                # CFG-Zero* optimal rescale + APG
                cfg_zero_optimal_rescale=cfg_zero_optimal_rescale,
                cfg_guidance_mode=cfg_guidance_mode,
                apg_text_eta=apg_text_eta,
                apg_img_eta=apg_img_eta,
                # CFG Distillation
                cfg_predictor=cfg_predictor,
            )

            # CFG-Zero*: zero out velocity for first N steps
            if i < cfg_zero_init_steps:
                v_t = torch.zeros_like(v_t)

            x_t = x_t - v_t.to(x_t.device) * dts[i] # velocity pointing from data to noise
        
        if enable_taylorseer:
            del model_pred_cache_dic, model_pred_current
            del model_pred_text_cache_dic, model_pred_text_current
            del model_pred_img_cache_dic, model_pred_img_current

        unpacked_latent = x_t.split((packed_seqlens - 2).tolist())
        return unpacked_latent

    @torch.no_grad
    def _forward_flow(
        self,
        x_t: torch.Tensor,
        timestep: torch.LongTensor,
        packed_vae_token_indexes: torch.LongTensor,
        packed_vae_position_ids: torch.LongTensor,
        packed_text_ids: torch.LongTensor,
        packed_text_indexes: torch.LongTensor,
        packed_indexes: torch.LongTensor,
        packed_position_ids: torch.LongTensor,
        packed_seqlens: torch.IntTensor,
        key_values_lens: torch.IntTensor,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        cfg_renorm_min: float = 0.0,
        cfg_renorm_type: str = "global",
        # cfg_text
        cfg_text_scale: float = 1.0,
        cfg_text_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_text_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_text_key_values_lens: Optional[torch.Tensor] = None,
        cfg_text_past_key_values: Optional[NaiveCache] = None,
        cfg_text_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        # cfg_img
        cfg_img_scale: float = 1.0,
        cfg_img_packed_position_ids: Optional[torch.LongTensor] = None,
        cfg_img_packed_query_indexes: Optional[torch.LongTensor] = None,
        cfg_img_key_values_lens: Optional[torch.Tensor] = None,
        cfg_img_past_key_values: Optional[NaiveCache] = None,
        cfg_img_packed_key_value_indexes: Optional[torch.LongTensor] = None,
        cfg_type: str = "parallel",
        # cache
        model_pred_cache_dic: Optional[Dict[str, Any]] = None,
        model_pred_current: Optional[int] = None,
        model_pred_text_cache_dic: Optional[Dict[str, Any]] = None,
        model_pred_text_current: Optional[int] = None,
        model_pred_img_cache_dic: Optional[Dict[str, Any]] = None,
        model_pred_img_current: Optional[int] = None,
        # CFG-Zero* optimal rescale + APG
        cfg_zero_optimal_rescale: float = 0.0,
        cfg_guidance_mode: str = "renorm",
        apg_text_eta: float = 0.5,
        apg_img_eta: float = 0.5,
        # CFG Distillation
        cfg_predictor: Optional[CFGPredictor] = None,
    ):
        packed_text_embedding = self.language_model.model.embed_tokens(packed_text_ids)
        packed_sequence = packed_text_embedding.new_zeros((sum(packed_seqlens), self.hidden_size))
        packed_sequence[packed_text_indexes] = packed_text_embedding

        assert timestep.unique().shape[0] == 1
        packed_pos_embed = self.latent_pos_embed(packed_vae_position_ids)
        packed_timestep_embeds = self.time_embedder(timestep)

        x_t = self.vae2llm(x_t) + packed_timestep_embeds + packed_pos_embed
        if x_t.dtype != packed_sequence.dtype:
            x_t = x_t.to(packed_sequence.dtype)
        packed_sequence[packed_vae_token_indexes] = x_t

        extra_inputs = {}
        if self.use_moe:
            extra_inputs = {
                "mode": "gen",
                "packed_vae_token_indexes": packed_vae_token_indexes,
                "packed_text_indexes": packed_text_indexes
            }

        if self.language_model.model.enable_taylorseer:
            self.language_model.model.cache_dic = model_pred_cache_dic
            self.language_model.model.current = model_pred_current

        output = self.language_model.forward_inference(
            packed_query_sequence=packed_sequence,
            query_lens=packed_seqlens,
            packed_query_position_ids=packed_position_ids,
            packed_query_indexes=packed_indexes,
            past_key_values=past_key_values,
            key_values_lens=key_values_lens,
            packed_key_value_indexes=packed_key_value_indexes,
            update_past_key_values=False,
            is_causal=False,
            **extra_inputs,
        )
        v_t = self.llm2vae(output.packed_query_sequence)
        v_t = v_t[packed_vae_token_indexes]

        # CFG Distillation: skip uncond forward passes and use predictor
        if cfg_predictor is not None and (cfg_text_scale > 1.0 or cfg_img_scale > 1.0):
            v_t = cfg_predictor(v_t, timestep)
            return v_t

        if cfg_text_scale > 1.0:
            if self.language_model.model.enable_taylorseer:
                self.language_model.model.cache_dic = model_pred_text_cache_dic
                self.language_model.model.current = model_pred_text_current
            cfg_text_output = self.language_model.forward_inference(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_text_packed_position_ids,
                packed_query_indexes=cfg_text_packed_query_indexes,
                past_key_values=cfg_text_past_key_values,
                key_values_lens=cfg_text_key_values_lens,
                packed_key_value_indexes=cfg_text_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            cfg_text_v_t = self.llm2vae(cfg_text_output.packed_query_sequence)
            cfg_text_v_t = cfg_text_v_t[packed_vae_token_indexes]

        if cfg_img_scale > 1.0:
            if self.language_model.model.enable_taylorseer:
                self.language_model.model.cache_dic = model_pred_img_cache_dic
                self.language_model.model.current = model_pred_img_current
            cfg_img_output = self.language_model.forward_inference(
                packed_query_sequence=packed_sequence,
                query_lens=packed_seqlens,
                packed_query_position_ids=cfg_img_packed_position_ids,
                packed_query_indexes=cfg_img_packed_query_indexes,
                past_key_values=cfg_img_past_key_values,
                key_values_lens=cfg_img_key_values_lens,
                packed_key_value_indexes=cfg_img_packed_key_value_indexes,
                update_past_key_values=False,
                is_causal=False,
                **extra_inputs,
            )
            cfg_img_v_t = self.llm2vae(cfg_img_output.packed_query_sequence)
            cfg_img_v_t = cfg_img_v_t[packed_vae_token_indexes]

        # Save v_cond before CFG for optimal rescale
        v_t_cond = v_t.clone()

        if cfg_text_scale > 1.0:
            if cfg_guidance_mode == "apg":
                # APG: Adaptive Projected Guidance
                v_t_text = _apply_apg(v_t, cfg_text_v_t, cfg_text_scale, apg_text_eta)
                if cfg_img_scale > 1.0:
                    v_t = _apply_apg(v_t_text, cfg_img_v_t, cfg_img_scale, apg_img_eta)
                else:
                    v_t = v_t_text
            else:
                # Existing renorm mode
                if cfg_renorm_type == "text_channel":
                    v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)
                    norm_v_t = torch.norm(v_t, dim=-1, keepdim=True)
                    norm_v_t_text_ = torch.norm(v_t_text_, dim=-1, keepdim=True)
                    scale = (norm_v_t / (norm_v_t_text_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                    v_t_text = v_t_text_ * scale
                    if cfg_img_scale > 1.0:
                        v_t = cfg_img_v_t + cfg_img_scale * (v_t_text - cfg_img_v_t)
                    else:
                        v_t = v_t_text
                else:
                    v_t_text_ = cfg_text_v_t + cfg_text_scale * (v_t - cfg_text_v_t)

                    if cfg_img_scale > 1.0:
                        v_t_ = cfg_img_v_t + cfg_img_scale * (v_t_text_ - cfg_img_v_t)
                    else:
                        v_t_ = v_t_text_

                    # NOTE norm is computed over all dimensions, thus currently only supports batch_size = 1 with navit
                    if cfg_renorm_type == "global":
                        norm_v_t = torch.norm(v_t_cond)
                        norm_v_t_ = torch.norm(v_t_)
                    elif cfg_renorm_type == "channel":
                        norm_v_t = torch.norm(v_t_cond, dim=-1, keepdim=True)
                        norm_v_t_ = torch.norm(v_t_, dim=-1, keepdim=True)
                    else:
                        raise NotImplementedError(f"{cfg_renorm_type} is not suppoprted")
                    scale = (norm_v_t / (norm_v_t_ + 1e-8)).clamp(min=cfg_renorm_min, max=1.0)
                    v_t = v_t_ * scale
        else:
            # No CFG
            pass

        # CFG-Zero* optimal rescale: blend guided velocity with conditional
        if cfg_zero_optimal_rescale > 0 and cfg_text_scale > 1.0:
            v_t = cfg_zero_optimal_rescale * v_t + (1 - cfg_zero_optimal_rescale) * v_t_cond

        return v_t

    def prepare_start_tokens(self, curr_kvlens, curr_rope, new_token_ids):
        packed_start_tokens, packed_key_value_indexes = list(), list()
        packed_query_position_ids = list()

        curr = 0
        for curr_kvlen, curr_position_id in zip(curr_kvlens, curr_rope):
            packed_key_value_indexes.extend(range(curr, curr + curr_kvlen))
            packed_start_tokens.append(new_token_ids['bos_token_id'])
            packed_query_position_ids.append(curr_position_id)
            curr += curr_kvlen

        generation_input = {
            "packed_start_tokens": torch.tensor(packed_start_tokens, dtype=torch.long),
            "packed_query_position_ids": torch.tensor(packed_query_position_ids, dtype=torch.long),
            "key_values_lens": torch.tensor(curr_kvlens, dtype=torch.int),
            "packed_key_value_indexes": torch.tensor(packed_key_value_indexes, dtype=torch.long),
        }

        return generation_input

    @torch.no_grad
    def generate_text(
        self,
        past_key_values: NaiveCache,
        packed_key_value_indexes: torch.LongTensor,
        key_values_lens: torch.IntTensor,
        packed_start_tokens: torch.LongTensor,
        packed_query_position_ids: torch.LongTensor,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
        end_token_id: int = None,
    ):
        step = 0
        generated_sequence = []
        curr_tokens = packed_start_tokens
        while step < max_length:
            generated_sequence.append(curr_tokens)
            packed_text_embedding = self.language_model.model.embed_tokens(curr_tokens)
            query_lens = torch.ones_like(curr_tokens)
            packed_query_indexes = torch.cumsum(key_values_lens, dim=0) + torch.arange(
                0, len(key_values_lens), 
                device=key_values_lens.device, 
                dtype=key_values_lens.dtype
            )

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] += i
            packed_key_value_indexes = torch.cat(uppacked, dim=0)

            extra_inputs = {}
            if self.use_moe:
                extra_inputs = {"mode": "und"}

            output = self.language_model.forward_inference(
                packed_query_sequence=packed_text_embedding,
                query_lens=query_lens,
                packed_query_position_ids=packed_query_position_ids,
                packed_query_indexes=packed_query_indexes,
                past_key_values=past_key_values,
                key_values_lens=key_values_lens,
                packed_key_value_indexes=packed_key_value_indexes,
                update_past_key_values=True,
                is_causal=True,
                **extra_inputs,
            )
            past_key_values = output.past_key_values
            packed_query_sequence = output.packed_query_sequence
            pred_logits = self.language_model.lm_head(packed_query_sequence)

            if do_sample:
                probs = nn.functional.softmax(pred_logits / temperature, dim=-1)
                curr_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                curr_tokens = torch.argmax(pred_logits, dim=-1)

            uppacked = list(packed_key_value_indexes.split(key_values_lens.tolist(), dim=0))
            for i in range(len(uppacked)):
                uppacked[i] = torch.cat(
                    [uppacked[i], torch.tensor([uppacked[i][-1] + 1], device=uppacked[i].device)], dim=0
                )
            packed_key_value_indexes = torch.cat(uppacked, dim=0)
            key_values_lens = key_values_lens + 1
            packed_query_position_ids = packed_query_position_ids + 1
            step += 1

            if end_token_id is not None and curr_tokens[0] == end_token_id: # only support batch=1
                break

        output_device = generated_sequence[0].device
        return torch.stack([i.to(output_device) for i in generated_sequence], dim=0)

    # for evaluation
    @torch.no_grad()
    def chat(
        self,
        tokenizer,
        new_token_ids,
        image_transform,
        images,
        prompt,
        max_length: int,
        do_sample: bool = False,
        temperature: float = 1.0,
    ):
        device = next(self.parameters()).device

        if isinstance(new_token_ids, dict):
            for k, v in new_token_ids.items():
                if torch.is_tensor(v):
                    new_token_ids[k] = v.to(device)
        elif torch.is_tensor(new_token_ids):
            new_token_ids = new_token_ids.to(device)

        # prefill
        past_key_values = NaiveCache(self.config.llm_config.num_hidden_layers)
        newlens = [0]
        new_rope = [0]
        # add images
        for image in images:
            generation_input, newlens, new_rope = self.prepare_vit_images(
                curr_kvlens=newlens,
                curr_rope=new_rope,
                images=[image],
                transforms=image_transform,
                new_token_ids=new_token_ids,
            )
            for k, v in generation_input.items():
                if torch.is_tensor(v):
                    generation_input[k] = v.to(device)
            with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
                past_key_values = self.forward_cache_update_vit(past_key_values, **generation_input)

        # add text
        generation_input, newlens, new_rope = self.prepare_prompts(
            curr_kvlens=newlens,
            curr_rope=new_rope,
            prompts=[prompt],
            tokenizer=tokenizer,
            new_token_ids=new_token_ids,
        )
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            past_key_values = self.forward_cache_update_text(past_key_values, **generation_input)

        # decode
        generation_input = self.prepare_start_tokens(newlens, new_rope, new_token_ids)
        for k, v in generation_input.items():
            if torch.is_tensor(v):
                generation_input[k] = v.to(device)
        with torch.amp.autocast("cuda", enabled=True, dtype=torch.bfloat16):
            unpacked_latent = self.generate_text(
                past_key_values=past_key_values,
                max_length=max_length,
                do_sample=do_sample,
                temperature=temperature,
                end_token_id=new_token_ids['eos_token_id'],
                **generation_input,
            )
        output = tokenizer.decode(unpacked_latent[:,0])
        output = output.split('<|im_end|>')[0].split('<|im_start|>')[1]

        return output