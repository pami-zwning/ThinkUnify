"""
Q-Former v3: SigLIP-Aligned VAE-to-Understanding Distillation

Architecture (from diagram):
  - First `num_ca_blocks` blocks: CrossAttn + SelfAttn (CA+SA)
    - CA: Q=queries, K/V=VAE latent; randomly initialized
    - SA: full SigLIP encoder layer (weights loaded from frozen SigLIP)
    - Mid loss after each block: attn_map loss vs SigLIP intermediate features

  - Remaining `num_sa_blocks` blocks: SelfAttn1 + SelfAttn2 (SA+SA)
    - Both SA layers loaded from consecutive SigLIP layers
    - No cross-attention (information already injected in CA+SA blocks)

  - End loss: attn_map + MSE loss vs SigLIP final output

  Total SigLIP layers used: num_ca_blocks + num_sa_blocks * 2
  Example: 4 CA+SA + 8 SA+SA = 4 + 16 = 20 SigLIP layers

Loss: total = w_mid * sum(mid_losses) + w_end * end_loss
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_varlen_func
except ImportError as e:
    raise RuntimeError("flash_attn required") from e

from .modeling_utils import PositionEmbedding
from .siglip_navit import SiglipEncoderLayer, SiglipVisionConfig


# ---------------------------------------------------------------------------
# Cross-Attention Layer (randomly initialized, takes VAE latent as K/V)
# ---------------------------------------------------------------------------

class CrossAttentionLayer(nn.Module):
    """Cross-attention: Q=queries, K/V=vae_latent. Pre-norm + FFN.

    The FFN (norm_ffn + ffn) can optionally be initialized from a SigLIP
    encoder layer via load_ffn_from_siglip(), so the nonlinear transform
    is already in SigLIP's feature space.
    """

    def __init__(self, hidden_size, num_heads, intermediate_size=None, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.norm_q = nn.LayerNorm(hidden_size)
        self.norm_kv = nn.LayerNorm(hidden_size)

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        # FFN: use intermediate_size if given (matches SigLIP), else mlp_ratio
        ffn_dim = intermediate_size if intermediate_size else int(hidden_size * mlp_ratio)
        self.norm_ffn = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, ffn_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(ffn_dim, hidden_size)
        self.ffn_drop = nn.Dropout(dropout)

    def load_ffn_from_siglip(self, siglip_layer):
        """Copy FFN weights (layer_norm2 + mlp) from a SigLIP encoder layer.

        SigLIP layer structure:
          layer_norm2 → mlp.fc1 → activation → mlp.fc2
        Our CA layer structure:
          norm_ffn → fc1 → act → fc2
        """
        self.norm_ffn.load_state_dict(siglip_layer.layer_norm2.state_dict())
        self.fc1.load_state_dict(siglip_layer.mlp.fc1.state_dict())
        self.fc2.load_state_dict(siglip_layer.mlp.fc2.state_dict())

    def forward(self, queries, kv, q_cu, kv_cu, max_q, max_kv):
        residual = queries
        q_norm = self.norm_q(queries)
        kv_norm = self.norm_kv(kv)

        q = self.q_proj(q_norm).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(kv_norm).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(kv_norm).reshape(-1, self.num_heads, self.head_dim)

        out = flash_attn_varlen_func(
            q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16),
            cu_seqlens_q=q_cu, cu_seqlens_k=kv_cu,
            max_seqlen_q=max_q, max_seqlen_k=max_kv,
            causal=False,
        )
        out = self.out_proj(out.reshape(-1, self.hidden_size))
        queries = residual + out

        # FFN (structure matches SigLIP: LN → fc1 → act → fc2)
        residual = queries
        h = self.norm_ffn(queries)
        h = self.act(self.fc1(h))
        h = self.ffn_drop(self.fc2(h))
        queries = residual + h
        return queries


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_cu_seqlens(seqlens, device):
    if not isinstance(device, torch.device):
        device = torch.device(device)
    if device.type != 'cuda':
        device = torch.device('cuda')
    cu = torch.zeros(len(seqlens) + 1, dtype=torch.int32, device=device)
    torch.cumsum(
        torch.tensor(seqlens, dtype=torch.int32, device=device),
        dim=0, out=cu[1:],
    )
    return cu


def _compute_per_image_attn_weights(q, k, cu_seqlens, q_seqlens, head_dim):
    """Compute per-image attention weights for distillation loss."""
    weights = []
    for i, sl in enumerate(q_seqlens):
        s, e = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        qi = q[s:e]
        ki = k[s:e]
        attn_score = torch.bmm(
            qi.permute(1, 0, 2), ki.permute(1, 2, 0),
        ) / (head_dim ** 0.5)
        attn_w = F.softmax(attn_score, dim=-1)
        weights.append(attn_w.mean(dim=0, keepdim=True))
    return weights


# ---------------------------------------------------------------------------
# Main Q-Former v3
# ---------------------------------------------------------------------------

class QFormerV3(nn.Module):
    """
    SigLIP-aligned Q-Former for VAE→understanding feature distillation.

    Args:
        hidden_size: must match SigLIP hidden_size
        vae_latent_dim: p*p*z_channels from VAE
        max_query_size: max ViT patch count
        num_ca_blocks: number of CA+SA blocks (first stage)
        num_sa_blocks: number of SA+SA blocks (second stage)
        siglip_config: SiglipVisionConfig for creating encoder layers
        vit_max_num_patch_per_side: for position embedding
    """

    def __init__(
        self,
        hidden_size: int,
        vae_latent_dim: int,
        max_query_size: int,
        num_ca_blocks: int = 4,
        num_sa_blocks: int = 8,
        siglip_config: SiglipVisionConfig = None,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        vit_max_num_patch_per_side: int = 70,
        output_hidden_size: int = None,
    ):
        super().__init__()
        # Internal working dim MUST match SigLIP hidden_size so we can
        # load SigLIP encoder layer weights as-is.
        if siglip_config is not None:
            siglip_hidden_size = siglip_config.hidden_size
            if hidden_size != siglip_hidden_size:
                print(
                    f"[QFormerV3] Overriding hidden_size={hidden_size} with "
                    f"siglip hidden_size={siglip_hidden_size} (required for weight loading)."
                )
                hidden_size = siglip_hidden_size
            # SigLIP attention expects its own num_heads
            if num_heads != siglip_config.num_attention_heads:
                print(
                    f"[QFormerV3] Overriding num_heads={num_heads} with "
                    f"siglip num_attention_heads={siglip_config.num_attention_heads}."
                )
                num_heads = siglip_config.num_attention_heads

        self.hidden_size = hidden_size
        self.num_ca_blocks = num_ca_blocks
        self.num_sa_blocks = num_sa_blocks
        self.num_heads = num_heads
        self.total_siglip_layers_needed = num_ca_blocks + num_sa_blocks * 2

        # Output dim (= LLM hidden size). If unspecified, keep equal to internal.
        self.output_hidden_size = output_hidden_size if output_hidden_size is not None else hidden_size

        # Learnable query embeddings
        self.query_embed = nn.Parameter(torch.zeros(max_query_size, hidden_size))
        nn.init.trunc_normal_(self.query_embed, std=0.02)

        # Position embedding (ViT-compatible)
        self.pos_embed = PositionEmbedding(vit_max_num_patch_per_side, hidden_size)

        # Project VAE latent to hidden_size for K/V
        self.kv_proj = nn.Linear(vae_latent_dim, hidden_size)

        # Stage 1: CA+SA blocks
        # CA layers: cross-attn randomly init, FFN will be loaded from SigLIP
        intermediate_size = siglip_config.intermediate_size if siglip_config else int(hidden_size * mlp_ratio)
        self.ca_layers = nn.ModuleList([
            CrossAttentionLayer(hidden_size, num_heads,
                                intermediate_size=intermediate_size, dropout=dropout)
            for _ in range(num_ca_blocks)
        ])
        # SA layers in stage 1: will be loaded from SigLIP layers 0..num_ca_blocks-1
        if siglip_config is not None:
            self.sa_layers_stage1 = nn.ModuleList([
                SiglipEncoderLayer(siglip_config)
                for _ in range(num_ca_blocks)
            ])
        else:
            self.sa_layers_stage1 = nn.ModuleList()

        # Stage 2: SA+SA blocks
        # Each block has 2 SA layers loaded from consecutive SigLIP layers
        if siglip_config is not None:
            self.sa_layers_stage2 = nn.ModuleList([
                SiglipEncoderLayer(siglip_config)
                for _ in range(num_sa_blocks * 2)
            ])
        else:
            self.sa_layers_stage2 = nn.ModuleList()

        # Output projection (small-init): internal siglip_dim -> LLM hidden dim
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, self.output_hidden_size)
        nn.init.trunc_normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    def load_siglip_weights(self, siglip_encoder_layers: nn.ModuleList):
        """Copy weights from frozen SigLIP encoder layers.

        Mapping:
          - CA layer i FFN ← SigLIP layer i (layer_norm2 + mlp)
          - SA layer i (stage 1) ← SigLIP layer i (full encoder layer)
          - SA layers (stage 2) ← SigLIP layers num_ca_blocks..num_ca_blocks+num_sa_blocks*2-1

        Args:
            siglip_encoder_layers: nn.ModuleList of SiglipEncoderLayer from frozen SigLIP
        """
        n_available = len(siglip_encoder_layers)
        n_needed = self.total_siglip_layers_needed
        if n_available < n_needed:
            print(f"[QFormerV3] Warning: SigLIP has {n_available} layers but need {n_needed}. "
                  f"Only loading first {n_available} layers.")

        siglip_idx = 0

        # Stage 1: SA layers + CA FFN
        for i in range(self.num_ca_blocks):
            if siglip_idx < n_available:
                # Load full SA layer from SigLIP
                self.sa_layers_stage1[i].load_state_dict(
                    siglip_encoder_layers[siglip_idx].state_dict()
                )
                # Also load FFN (layer_norm2 + mlp) into the CA layer
                self.ca_layers[i].load_ffn_from_siglip(
                    siglip_encoder_layers[siglip_idx]
                )
                siglip_idx += 1

        # Stage 2: SA+SA layers
        for i, sa_layer in enumerate(self.sa_layers_stage2):
            if siglip_idx < n_available:
                sa_layer.load_state_dict(siglip_encoder_layers[siglip_idx].state_dict())
                siglip_idx += 1

        print(f"[QFormerV3] Loaded {siglip_idx} SigLIP layers "
              f"({self.num_ca_blocks} CA+SA + {self.num_sa_blocks} SA+SA blocks). "
              f"CA FFN also initialized from SigLIP.")

    # ------------------------------------------------------------------
    # SA freeze/unfreeze for warmup (suggestion 1)
    # ------------------------------------------------------------------

    def freeze_sa_layers(self):
        """Freeze all SA layers (both stages). Only CA + kv_proj + query_embed train.

        Call at training start. After warmup steps, call unfreeze_sa_layers().
        This prevents CA's random-init outputs from corrupting the SigLIP SA weights
        during early training.
        """
        for sa in self.sa_layers_stage1:
            for p in sa.parameters():
                p.requires_grad = False
        for sa in self.sa_layers_stage2:
            for p in sa.parameters():
                p.requires_grad = False
        # Also freeze the FFN in CA layers since they come from SigLIP
        for ca in self.ca_layers:
            for name, p in ca.named_parameters():
                if 'fc1' in name or 'fc2' in name or 'norm_ffn' in name:
                    p.requires_grad = False
        print("[QFormerV3] SA layers + CA-FFN frozen (warmup mode)")

    def unfreeze_sa_layers(self):
        """Unfreeze all SA layers for full fine-tuning after warmup."""
        for sa in self.sa_layers_stage1:
            for p in sa.parameters():
                p.requires_grad = True
        for sa in self.sa_layers_stage2:
            for p in sa.parameters():
                p.requires_grad = True
        for ca in self.ca_layers:
            for name, p in ca.named_parameters():
                if 'fc1' in name or 'fc2' in name or 'norm_ffn' in name:
                    p.requires_grad = True
        print("[QFormerV3] SA layers + CA-FFN unfrozen (full fine-tuning)")

    def forward_packed(
        self,
        packed_vae_latent: torch.Tensor,
        vae_token_seqlens: torch.IntTensor,
        vit_token_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor = None,
        # RoPE tensors from SigLIP (needed for SA layers)
        cos_h: torch.Tensor = None,
        sin_h: torch.Tensor = None,
        cos_w: torch.Tensor = None,
        sin_w: torch.Tensor = None,
    ):
        """Batched packed forward.

        Args:
            packed_vae_latent: (Total_VAE, vae_latent_dim)
            vae_token_seqlens: (N_imgs,) VAE token count per image
            vit_token_seqlens: (N_imgs,) ViT token count per image (= query count)
            packed_position_ids: (Total_Q,) position IDs for RoPE
            cos_h, sin_h, cos_w, sin_w: RoPE tensors indexed by position_ids

        Returns:
            output: (Total_Q, hidden_size)
            mid_features: list of (Total_Q, hidden_size) after each CA+SA block
            per_image_attn: per-image attention weights from last SA layer
        """
        if packed_vae_latent.device.type != 'cuda':
            packed_vae_latent = packed_vae_latent.cuda()
        if packed_position_ids is not None and packed_position_ids.device.type != 'cuda':
            packed_position_ids = packed_position_ids.cuda()

        device = packed_vae_latent.device
        q_seqlens = vit_token_seqlens.tolist()
        kv_seqlens = vae_token_seqlens.tolist()
        head_dim = self.hidden_size // self.num_heads

        # 1) Build queries: (Total_Q, D)
        indices = torch.cat([torch.arange(ql, device=device) for ql in q_seqlens])
        queries = self.query_embed[indices]
        if packed_position_ids is not None:
            queries = queries + self.pos_embed(packed_position_ids)

        # 2) Build KV for cross-attention: (Total_KV, D)
        kv = self.kv_proj(packed_vae_latent)

        # 3) Build cu_seqlens
        q_cu = _build_cu_seqlens(q_seqlens, device)
        kv_cu = _build_cu_seqlens(kv_seqlens, device)
        max_q = max(q_seqlens)
        max_kv = max(kv_seqlens)

        # RoPE kwargs for SigLIP SA layers
        rope_kwargs = {}
        if cos_h is not None:
            rope_kwargs = dict(cos_h=cos_h, sin_h=sin_h, cos_w=cos_w, sin_w=sin_w)

        # 4) Stage 1: CA+SA blocks with mid features
        mid_features = []
        for ca_layer, sa_layer in zip(self.ca_layers, self.sa_layers_stage1):
            # Cross-attention: inject VAE information
            queries = ca_layer(queries, kv, q_cu, kv_cu, max_q, max_kv)
            # Self-attention: SigLIP-initialized processing
            queries = sa_layer(queries, q_cu, max_q, **rope_kwargs)
            # Save mid feature for mid loss
            mid_features.append(queries.clone())

        # 5) Stage 2: SA+SA blocks
        sa_q = sa_k = None
        for i in range(self.num_sa_blocks):
            sa1 = self.sa_layers_stage2[i * 2]
            sa2 = self.sa_layers_stage2[i * 2 + 1]
            queries = sa1(queries, q_cu, max_q, **rope_kwargs)
            is_last = (i == self.num_sa_blocks - 1)
            if is_last:
                # Capture Q/K from the last SA layer in one forward pass.
                # Avoids re-accessing submodule parameters after forward completes.
                queries, sa_q, sa_k = sa2(queries, q_cu, max_q, return_qk=True, **rope_kwargs)
            else:
                queries = sa2(queries, q_cu, max_q, **rope_kwargs)

        # 6) Compute attention weights from last SA layer for end loss
        if sa_q is not None and sa_k is not None:
            # Q/K captured from Stage 2 last layer
            per_image_attn = _compute_per_image_attn_weights(sa_q, sa_k, q_cu, q_seqlens, head_dim)
        elif len(self.sa_layers_stage2) == 0 and len(self.sa_layers_stage1) > 0:
            # Fallback: no stage-2 layers, use last stage-1 layer
            last_sa = self.sa_layers_stage1[-1]
            last_hidden = last_sa.layer_norm1(queries)
            sa_q = last_sa.self_attn.q_proj(last_hidden).reshape(-1, self.num_heads, head_dim)
            sa_k = last_sa.self_attn.k_proj(last_hidden).reshape(-1, self.num_heads, head_dim)
            per_image_attn = _compute_per_image_attn_weights(sa_q, sa_k, q_cu, q_seqlens, head_dim)
        else:
            per_image_attn = []

        # 7) Output projection
        output = self.output_proj(self.output_norm(queries))

        return output, mid_features, per_image_attn


# ---------------------------------------------------------------------------
# Distillation Loss: Mid Loss (attn_map) + End Loss (attn_map + MSE)
# ---------------------------------------------------------------------------

def _attention_map_loss(attn_weights_list, vit_feat):
    """KL divergence between Q-Former attn map and ViT feature similarity."""
    N = vit_feat.shape[0]
    if N < 2:
        return vit_feat.new_tensor(0.0)

    with torch.no_grad():
        vit_sim = torch.mm(vit_feat, vit_feat.t()) / (vit_feat.shape[-1] ** 0.5)
        target_attn = F.softmax(vit_sim, dim=-1)

    last_attn = attn_weights_list[-1].squeeze(0)
    if torch.isnan(last_attn).any() or last_attn.sum() == 0:
        return vit_feat.new_tensor(0.0)

    log_student = torch.log(last_attn.clamp(min=1e-8))
    return F.kl_div(log_student, target_attn, reduction='batchmean')


def compute_distillation_loss_v3(
    qformer_output: torch.Tensor,
    mid_features: list,
    per_image_attn: list,
    vit_final_feat: torch.Tensor,
    vit_mid_features: list,
    vit_token_seqlens: torch.IntTensor,
    w_mid: float = 0.5,
    w_end: float = 1.0,
):
    """
    Compute the combined distillation loss.

    Args:
        qformer_output: (Total_Q, D) final Q-Former output
        mid_features: list of (Total_Q, D) after each CA+SA block
        per_image_attn: list of (1, N_i, N_i) attn weights from last SA
        vit_final_feat: (Total_Q, D) frozen SigLIP final output (detached)
        vit_mid_features: list of (Total_Q, D) SigLIP intermediate features
        vit_token_seqlens: (N_imgs,) for per-image splitting
        w_mid, w_end: loss weights

    Returns:
        (total_loss, mid_loss, end_loss, mse_loss, attn_loss)
    """
    vit_final_feat = vit_final_feat.detach()
    seqlens = vit_token_seqlens.tolist()

    # --- Mid Loss: attn_map at each CA+SA block vs SigLIP mid features ---
    total_mid_loss = qformer_output.new_tensor(0.0)
    num_mid = min(len(mid_features), len(vit_mid_features))
    for layer_idx in range(num_mid):
        qf_mid = mid_features[layer_idx]
        vit_mid = vit_mid_features[layer_idx].detach()

        # Per-image attn map loss using the mid features as proxy
        q_splits = torch.split(qf_mid, seqlens)
        v_splits = torch.split(vit_mid, seqlens)
        for q_feat, v_feat in zip(q_splits, v_splits):
            if q_feat.shape[0] < 2:
                continue
            # Compute Q-Former self-similarity as attention proxy
            with torch.no_grad():
                vit_sim = torch.mm(v_feat, v_feat.t()) / (v_feat.shape[-1] ** 0.5)
                target = F.softmax(vit_sim, dim=-1)
            qf_sim = torch.mm(q_feat, q_feat.t()) / (q_feat.shape[-1] ** 0.5)
            student = F.softmax(qf_sim, dim=-1)
            log_student = torch.log(student.clamp(min=1e-8))
            total_mid_loss = total_mid_loss + F.kl_div(log_student, target, reduction='batchmean')

    mid_loss = total_mid_loss / max(num_mid * len(seqlens), 1)

    # --- End Loss: attn_map + MSE on final output vs SigLIP final ---
    # MSE loss
    mse_loss = F.mse_loss(qformer_output, vit_final_feat)

    # Cosine similarity loss
    cos_loss = 1.0 - F.cosine_similarity(qformer_output, vit_final_feat, dim=-1).mean()

    # Attention map loss (per-image)
    attn_loss = qformer_output.new_tensor(0.0)
    if per_image_attn:
        q_splits = torch.split(vit_final_feat, seqlens)
        for i, v_feat in enumerate(q_splits):
            if i < len(per_image_attn) and v_feat.shape[0] >= 2:
                attn_loss = attn_loss + _attention_map_loss([per_image_attn[i]], v_feat)
        attn_loss = attn_loss / max(len(q_splits), 1)

    end_loss = mse_loss + cos_loss + attn_loss

    total_loss = w_mid * mid_loss + w_end * end_loss

    return total_loss, mid_loss, end_loss, mse_loss, attn_loss


# ---------------------------------------------------------------------------
# Helper: extract SigLIP intermediate features
# ---------------------------------------------------------------------------

def extract_siglip_mid_features(
    siglip_encoder: nn.Module,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.IntTensor,
    max_seqlen: int,
    layer_indices: list,
    **rope_kwargs,
):
    """Run SigLIP encoder and return intermediate features at specified layers.

    Args:
        siglip_encoder: SiglipEncoder with .layers attribute
        hidden_states: (Total, D) input embeddings
        cu_seqlens, max_seqlen: for flash_attn
        layer_indices: list of layer indices to capture (e.g., [0,1,2,3])
        **rope_kwargs: cos_h, sin_h, cos_w, sin_w

    Returns:
        final_hidden: (Total, D) final encoder output
        mid_features: dict mapping layer_idx -> (Total, D)
    """
    mid_features = {}
    for idx, encoder_layer in enumerate(siglip_encoder.layers):
        hidden_states = encoder_layer(hidden_states, cu_seqlens, max_seqlen, **rope_kwargs)
        if idx in layer_indices:
            mid_features[idx] = hidden_states.detach()

    return hidden_states, mid_features
