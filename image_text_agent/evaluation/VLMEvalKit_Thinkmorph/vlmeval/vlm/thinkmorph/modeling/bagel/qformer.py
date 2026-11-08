"""
Q-Former: VAE-to-Understanding Distillation Module (v2)

Replaces SigLIP ViT encoder by learning to extract understanding features
from VAE latent tokens via cross-attention. Uses task-oriented distillation
with alpha-blended gradual transition from ViT to Q-Former.

v2 Improvements:
  - Hybrid Self+Cross Attention: self-attention among queries before
    cross-attention to VAE latents, allowing queries to exchange information
  - ViT-compatible positional embeddings: uses same 2D sincos pos embed
    as ViT (frozen), indexed by ViT position IDs
  - RKD + Attention Map distillation: relational knowledge distillation
    (pairwise distance alignment) + KL divergence on attention maps

Design:
  - Learnable query embeddings (pool size = vit_max_num_patch_per_side^2)
  - 2D sincos position embeddings (same as ViT pos_embed, frozen)
  - N-layer hybrid blocks: Self-Attention -> Cross-Attention -> FFN
  - Output projection to align with hidden_size (zero-initialized)

Training K/V source: generation branch's packed_latent_clean (wo_loss portion)
Inference K/V source: final denoised latent after flow matching
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_varlen_func
except ImportError as e:
    raise RuntimeError( "can not load flash_attn" ) from e

from .modeling_utils import PositionEmbedding


class QFormerSelfAttentionLayer(nn.Module):
    """Self-attention layer for query-query interaction (pre-norm)."""

    def __init__(self, hidden_size, num_heads, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, query, attn_mask=None):
        """
        Args:
            query: (B, N_q, D)
            attn_mask: (N_q, N_q) additive mask, 0=attend, -inf=block
        Returns:
            query: (B, N_q, D)
            attn_weights: (B, N_q, N_q) averaged across heads
        """
        residual = query
        q_norm = self.norm(query)
        attn_out, attn_weights = self.attn(
            query=q_norm, key=q_norm, value=q_norm,
            attn_mask=attn_mask,
            need_weights=True, average_attn_weights=True,
        )
        query = residual + attn_out
        return query, attn_weights


class QFormerCrossAttentionLayer(nn.Module):
    """Single cross-attention layer with pre-norm (LayerNorm) + FFN."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(hidden_size)
        self.norm_kv = nn.LayerNorm(hidden_size)

        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm_ffn = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, int(hidden_size * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(hidden_size * mlp_ratio), hidden_size),
            nn.Dropout(dropout),
        )

    def forward(self, query, kv, attn_mask=None, key_padding_mask=None):
        """
        Args:
            query: (B, N_q, D)
            kv: (B, N_kv, D)
            attn_mask: (N_q, N_kv) additive mask, 0=attend, -inf=block
            key_padding_mask: (B, N_kv) - True for padded positions
        """
        residual = query
        q_norm = self.norm_q(query)
        kv_norm = self.norm_kv(kv)
        attn_out, _ = self.attn(
            query=q_norm, key=kv_norm, value=kv_norm,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        query = residual + attn_out

        residual = query
        query = residual + self.ffn(self.norm_ffn(query))
        return query

# ---------------------------------------------------------------------------
# Helpers for batched Q-Former forward
# ---------------------------------------------------------------------------

def _build_cu_seqlens(seqlens, device):
    """Build cumulative sequence lengths for flash_attn_varlen_func.

    Args:
        seqlens: list[int] of per-image token counts
    Returns:
        cu_seqlens: (num_images+1,) int32 tensor on CUDA
    """
    # flash_attn requires cu_seqlens on CUDA; force it regardless of input device
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
    """Compute per-image attention weights from packed Q, K for distillation.

    Only called for the last self-attention layer. Each image's attention
    matrix is small (N_i x N_i), so total cost is negligible.

    Args:
        q: (Total, H, Hd) packed query tensor
        k: (Total, H, Hd) packed key tensor
        cu_seqlens: (num_images+1,) cumulative sequence lengths
        q_seqlens: list[int] of per-image query counts
        head_dim: head dimension for scaling

    Returns:
        list of (1, N_i, N_i) attention weight tensors (averaged over heads)
    """
    weights = []
    for i, sl in enumerate(q_seqlens):
        s, e = cu_seqlens[i].item(), cu_seqlens[i + 1].item()
        qi = q[s:e]  # (N_i, H, Hd)
        ki = k[s:e]  # (N_i, H, Hd)
        # (H, N_i, Hd) @ (H, Hd, N_i) -> (H, N_i, N_i)
        attn_score = torch.bmm(
            qi.permute(1, 0, 2),
            ki.permute(1, 2, 0),
        ) / (head_dim ** 0.5)
        attn_w = F.softmax(attn_score, dim=-1)  # (H, N_i, N_i)
        # Average over heads -> (1, N_i, N_i)
        weights.append(attn_w.mean(dim=0, keepdim=True))
    return weights


# Fallback dense mask helpers (used when flash_attn is not available)

def _build_self_attn_mask(seqlens, device, dtype):
    """Block-diagonal mask (Total_Q x Total_Q): 0 for attend, -inf to block."""
    total = sum(seqlens)
    mask = torch.full((total, total), float('-inf'), device=device, dtype=dtype)
    offset = 0
    for s in seqlens:
        mask[offset:offset + s, offset:offset + s] = 0.0
        offset += s
    return mask


def _build_cross_attn_mask(q_seqlens, kv_seqlens, device, dtype):
    """Rectangular block-diagonal mask (Total_Q x Total_KV)."""
    total_q, total_kv = sum(q_seqlens), sum(kv_seqlens)
    mask = torch.full((total_q, total_kv), float('-inf'), device=device, dtype=dtype)
    q_off, kv_off = 0, 0
    for ql, kvl in zip(q_seqlens, kv_seqlens):
        mask[q_off:q_off + ql, kv_off:kv_off + kvl] = 0.0
        q_off += ql
        kv_off += kvl
    return mask


def _split_attn_weights(layer_weights_list, q_seqlens):
    """Extract per-image attention sub-matrices from batched weights.

    Returns: list[list[tensor]] — per_image[i][layer] = (1, N_i, N_i)
    """
    offsets = [0]
    for s in q_seqlens:
        offsets.append(offsets[-1] + s)
    per_image = []
    for i in range(len(q_seqlens)):
        s, e = offsets[i], offsets[i + 1]
        per_image.append([w[:, s:e, s:e] for w in layer_weights_list])
    return per_image


class QFormerBlock(nn.Module):
    """
    Q-Former module that extracts understanding features from VAE latents.

    v2: Hybrid Self+Cross Attention with ViT-compatible positional embeddings.

    During training, query count is dynamic (= current ViT token count per image).
    During inference, query count is fixed.

    K/V is always clean VAE latent (timestep=0), so no timestep conditioning needed.
    """

    def __init__(
        self,
        hidden_size: int,
        vae_latent_dim: int,
        max_query_size: int,
        num_layers: int = 4,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        vit_max_num_patch_per_side: int = 70,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vae_latent_dim = vae_latent_dim
        self.max_query_size = max_query_size
        self.num_layers = num_layers

        # Learnable query embedding pool
        self.query_embed = nn.Parameter(
            torch.zeros(max_query_size, hidden_size)
        )
        nn.init.trunc_normal_(self.query_embed, std=0.02)

        # ViT-compatible 2D sincos positional embedding (frozen, same as vit_pos_embed)
        self.pos_embed = PositionEmbedding(vit_max_num_patch_per_side, hidden_size)

        # Project VAE latent dim to hidden_size for K/V
        self.kv_proj = nn.Linear(vae_latent_dim, hidden_size)

        # Hybrid Self+Cross attention layers
        self.self_attn_layers = nn.ModuleList([
            QFormerSelfAttentionLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.cross_attn_layers = nn.ModuleList([
            QFormerCrossAttentionLayer(
                hidden_size=hidden_size,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Output projection (small-init for stable training while avoiding NaN in RKD loss)
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, hidden_size)
        nn.init.trunc_normal_(self.output_proj.weight, std=0.02)
        nn.init.zeros_(self.output_proj.bias)

    def forward_single(
        self,
        vae_latent_tokens: torch.Tensor,
        num_queries: int,
        position_ids: torch.LongTensor = None,
    ):
        """Forward for a single image (no padding needed).

        Args:
            vae_latent_tokens: (N_kv, vae_latent_dim)
            num_queries: number of queries for this image
            position_ids: (num_queries,) ViT position IDs for pos embed

        Returns:
            output: (num_queries, hidden_size)
            self_attn_weights: list of (1, N_q, N_q) attention weights per layer
        """
        # Prepare queries with positional embedding: (1, N_q, D)
        queries = self.query_embed[:num_queries].unsqueeze(0)
        if position_ids is not None:
            pos_emb = self.pos_embed(position_ids).unsqueeze(0)
            queries = queries + pos_emb

        # Prepare K/V: (1, N_kv, D)
        kv = self.kv_proj(vae_latent_tokens).unsqueeze(0)

        self_attn_weights_list = []
        for self_layer, cross_layer in zip(self.self_attn_layers, self.cross_attn_layers):
            queries, self_attn_w = self_layer(queries)
            self_attn_weights_list.append(self_attn_w)
            queries = cross_layer(queries, kv)

        output = self.output_proj(self.output_norm(queries))
        return output.squeeze(0), self_attn_weights_list

    def forward_packed(
        self,
        packed_vae_latent: torch.Tensor,
        vae_token_seqlens: torch.IntTensor,
        vit_token_seqlens: torch.IntTensor,
        packed_position_ids: torch.LongTensor = None,
    ):
        """Batched packed forward for training.

        Uses flash_attn_varlen_func when available (O(N) memory, no masks).
        Falls back to dense block-diagonal masks otherwise.

        Args:
            packed_vae_latent: (Total_VAE_tokens, vae_latent_dim) - clean VAE latents
            vae_token_seqlens: (Num_Images,) - VAE token count per image
            vit_token_seqlens: (Num_Images,) - ViT token count per image (= query count)
            packed_position_ids: (Total_ViT_tokens,) - ViT position IDs for pos embed

        Returns:
            packed_output: (Total_ViT_tokens, hidden_size) - concatenated outputs
            all_self_attn_weights: list of (list of (1, N_i, N_i)) per image
        """
        # Ensure inputs are on CUDA (accelerate dispatch_model may defer GPU transfer)
        if packed_vae_latent.device.type != 'cuda':
            packed_vae_latent = packed_vae_latent.cuda()
        if packed_position_ids is not None and packed_position_ids.device.type != 'cuda':
            packed_position_ids = packed_position_ids.cuda()
        device = packed_vae_latent.device
        q_seqlens = vit_token_seqlens.tolist()
        kv_seqlens = vae_token_seqlens.tolist()
        D = self.hidden_size
        num_heads = self.self_attn_layers[0].attn.num_heads
        head_dim = D // num_heads

        # 1) Build queries: (Total_Q, D)
        indices = torch.cat([torch.arange(ql, device=device) for ql in q_seqlens])
        queries = self.query_embed[indices]
        if packed_position_ids is not None:
            queries = queries + self.pos_embed(packed_position_ids)

        # 2) Build KV: (Total_KV, D)
        kv = self.kv_proj(packed_vae_latent)

        # 3) Build cu_seqlens (once, reused across layers)
        q_cu = _build_cu_seqlens(q_seqlens, device)
        kv_cu = _build_cu_seqlens(kv_seqlens, device)
        max_q = max(q_seqlens)
        max_kv = max(kv_seqlens)

        # 4) Forward through all layers
        last_layer_attn_weights = None

        for layer_idx, (self_layer, cross_layer) in enumerate(
            zip(self.self_attn_layers, self.cross_attn_layers)
        ):
            is_last = (layer_idx == self.num_layers - 1)

            # --- Self-attention via flash_attn ---
            residual = queries
            q_norm = self_layer.norm(queries)

            sa_mha = self_layer.attn
            sa_qkv = F.linear(q_norm, sa_mha.in_proj_weight, sa_mha.in_proj_bias)
            sa_q, sa_k, sa_v = sa_qkv.reshape(-1, 3, num_heads, head_dim).unbind(1)

            sa_out = flash_attn_varlen_func(
                sa_q, sa_k, sa_v, q_cu, q_cu, max_q, max_q,
                dropout_p=sa_mha.dropout if self.training else 0.0,
            )
            sa_out = sa_mha.out_proj(sa_out.reshape(-1, D))
            queries = residual + sa_out

            # Compute attention weights for last layer (for distillation loss)
            if is_last:
                last_layer_attn_weights = _compute_per_image_attn_weights(
                    sa_q, sa_k, q_cu, q_seqlens, head_dim,
                )

            # --- Cross-attention via flash_attn ---
            residual = queries
            q_norm_ca = cross_layer.norm_q(queries)
            kv_norm_ca = cross_layer.norm_kv(kv)

            ca_mha = cross_layer.attn
            w_q, w_k, w_v = ca_mha.in_proj_weight.chunk(3, dim=0)
            if ca_mha.in_proj_bias is not None:
                b_q, b_k, b_v = ca_mha.in_proj_bias.chunk(3, dim=0)
            else:
                b_q = b_k = b_v = None

            ca_q = F.linear(q_norm_ca, w_q, b_q).reshape(-1, num_heads, head_dim)
            ca_k = F.linear(kv_norm_ca, w_k, b_k).reshape(-1, num_heads, head_dim)
            ca_v = F.linear(kv_norm_ca, w_v, b_v).reshape(-1, num_heads, head_dim)

            ca_out = flash_attn_varlen_func(
                ca_q, ca_k, ca_v, q_cu, kv_cu, max_q, max_kv,
                dropout_p=ca_mha.dropout if self.training else 0.0,
            )
            ca_out = ca_mha.out_proj(ca_out.reshape(-1, D))
            queries = residual + ca_out

            # FFN
            residual = queries
            queries = residual + cross_layer.ffn(cross_layer.norm_ffn(queries))

        # 5) Output projection
        output = self.output_proj(self.output_norm(queries))  # (Total_Q, D)

        # 6) Build per_image_attn in expected format: per_image[i] = [last_layer_w]
        per_image_attn = []
        if last_layer_attn_weights is not None:
            for w in last_layer_attn_weights:
                per_image_attn.append([w])

        return output, per_image_attn

# ---------------------------------------------------------------------------
# Distillation Losses: MSE + Attention Map (KL, last layer)
# ---------------------------------------------------------------------------


def _attention_map_loss(
    layer_attn_weights: list,
    vit_feat: torch.Tensor,
) -> torch.Tensor:
    """Attention Map distillation: KL divergence between Q-Former last-layer
    self-attention map and ViT feature similarity distribution.

    Uses only the last self-attention layer of Q-Former, which captures the
    most semantically meaningful token relationships after information has
    propagated through all layers.

    Args:
        layer_attn_weights: list of (1, N, N) self-attention weights per layer
        vit_feat: (N, D) ViT features (should be detached)
    """
    N = vit_feat.shape[0]
    if N < 2:
        return vit_feat.new_tensor(0.0)

    # Target: ViT feature similarity as soft attention distribution
    with torch.no_grad():
        vit_sim = torch.mm(vit_feat, vit_feat.t()) / (vit_feat.shape[-1] ** 0.5)
        target_attn = F.softmax(vit_sim, dim=-1)  # (N, N)

    # Use only the last layer's self-attention: (N, N)
    last_attn = layer_attn_weights[-1].squeeze(0)

    # Guard against degenerate attention (all zeros or NaN)
    if torch.isnan(last_attn).any() or last_attn.sum() == 0:
        return vit_feat.new_tensor(0.0)

    # KL(target || student): use log of student attention for KL computation
    log_student = torch.log(last_attn.clamp(min=1e-8))
    kl_loss = F.kl_div(log_student, target_attn, reduction='batchmean')

    return kl_loss

def compute_distillation_loss(
    qformer_feat: torch.Tensor,
    vit_feat: torch.Tensor,
    vit_token_seqlens: torch.IntTensor = None,
    self_attn_weights_per_image: list = None,
):
    """
    Compute MSE + Attention Map distillation loss.

    Args:
        qformer_feat: (N, D) Q-Former output features
        vit_feat: (N, D) ViT features (will be detached internally)
        vit_token_seqlens: (Num_Images,) token counts per image for per-image split
        self_attn_weights_per_image: list of (list of (1, N_i, N_i)) per image

    Returns:
        (distill_loss, mse_loss, attn_loss): tuple of scalar tensors
    """
    vit_feat = vit_feat.detach()

    # If no per-image split info, compute on the whole batch
    if vit_token_seqlens is None:
        mse_loss = F.mse_loss(qformer_feat, vit_feat)
        attn_loss = qformer_feat.new_tensor(0.0)
        if self_attn_weights_per_image is not None and len(self_attn_weights_per_image) > 0:
            attn_loss = _attention_map_loss(self_attn_weights_per_image[0], vit_feat)
        return mse_loss + attn_loss, mse_loss, attn_loss

    # Per-image computation
    q_splits = torch.split(qformer_feat, vit_token_seqlens.tolist())
    v_splits = torch.split(vit_feat, vit_token_seqlens.tolist())

    total_mse = qformer_feat.new_tensor(0.0)
    total_attn_map = qformer_feat.new_tensor(0.0)
    num_images = len(q_splits)

    for i, (q_feat, v_feat) in enumerate(zip(q_splits, v_splits)):
        total_mse = total_mse + F.mse_loss(q_feat, v_feat)

        if self_attn_weights_per_image is not None and i < len(self_attn_weights_per_image):
            total_attn_map = total_attn_map + _attention_map_loss(
                self_attn_weights_per_image[i], v_feat
            )

    mse_loss = total_mse / max(num_images, 1)
    attn_loss = total_attn_map / max(num_images, 1)

    return mse_loss + attn_loss, mse_loss, attn_loss
