# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
CFG Distillation Module

Trains a lightweight MLP predictor to approximate the CFG guidance delta:
    delta = v_guided - v_cond

During inference, instead of 3 forward passes (cond + text_uncond + img_uncond),
we do 1 forward pass (cond) and predict the guidance delta with the MLP:
    v_guided ≈ v_cond + cfg_predictor(v_cond, timestep)

This achieves ~3x inference speedup.

Two-stage approach:
1. Distillation training: Run the full 3-pass CFG to get v_guided, train MLP to predict delta
2. Inference: Use MLP to predict delta from v_cond alone

Usage in training:
    python train_cfg_distillation.py \
        --model_path /path/to/trained_model \
        --distill_data_path /path/to/distill_data \
        --output_dir /path/to/cfg_predictor
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CFGPredictor(nn.Module):
    """Lightweight MLP that predicts the CFG guidance delta.

    Input: concatenation of [v_cond, timestep_embedding]
    Output: predicted delta (v_guided - v_cond)

    Architecture: 3-layer MLP with residual connection and layer norm.
    Parameter count is ~2-3% of the main model, negligible overhead.
    """

    def __init__(self, latent_dim, time_embed_dim=256, hidden_dim=1024):
        super().__init__()
        self.time_embed_dim = time_embed_dim

        # Timestep embedding (shared with main model or standalone)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        input_dim = latent_dim + time_embed_dim
        self.norm_in = nn.LayerNorm(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Residual scaling, initialized to zero for stable training start
        self.residual_scale = nn.Parameter(torch.zeros(1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Zero-init the last layer for stable start
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, v_cond, timestep):
        """
        Args:
            v_cond: Conditional velocity prediction [N, latent_dim]
            timestep: Scalar timestep value [N, 1] or [N]

        Returns:
            Predicted guided velocity [N, latent_dim]
        """
        if timestep.dim() == 1:
            timestep = timestep.unsqueeze(-1)

        t_emb = self.time_mlp(timestep)  # [N, time_embed_dim]
        x = torch.cat([v_cond, t_emb], dim=-1)  # [N, latent_dim + time_embed_dim]
        x = self.norm_in(x)
        delta = self.net(x) * torch.tanh(self.residual_scale)
        return v_cond + delta


class CFGDistillationTrainer:
    """Collects (v_cond, v_guided, timestep) pairs and trains the CFG predictor."""

    def __init__(self, latent_dim, device='cuda', lr=1e-4,
                 time_embed_dim=256, hidden_dim=1024):
        self.predictor = CFGPredictor(
            latent_dim=latent_dim,
            time_embed_dim=time_embed_dim,
            hidden_dim=hidden_dim,
        ).to(device)
        self.optimizer = torch.optim.AdamW(
            self.predictor.parameters(), lr=lr, weight_decay=1e-4
        )
        self.device = device
        self.buffer = []
        self.max_buffer_size = 10000

    def collect_sample(self, v_cond, v_guided, timestep):
        """Collect a training sample from the full 3-pass CFG inference."""
        sample = {
            'v_cond': v_cond.detach().cpu(),
            'v_guided': v_guided.detach().cpu(),
            'timestep': timestep.detach().cpu(),
        }
        self.buffer.append(sample)
        if len(self.buffer) > self.max_buffer_size:
            self.buffer.pop(0)

    def train_step(self, batch_size=256):
        """Train on collected samples."""
        if len(self.buffer) < batch_size:
            return None

        import random
        batch = random.sample(self.buffer, batch_size)

        v_cond = torch.cat([s['v_cond'] for s in batch], dim=0).to(self.device)
        v_guided = torch.cat([s['v_guided'] for s in batch], dim=0).to(self.device)
        timestep = torch.cat([s['timestep'] for s in batch], dim=0).to(self.device)

        self.predictor.train()
        pred = self.predictor(v_cond, timestep)
        loss = F.mse_loss(pred, v_guided)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def save(self, path):
        torch.save({
            'predictor_state_dict': self.predictor.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.predictor.load_state_dict(ckpt['predictor_state_dict'])
        if 'optimizer_state_dict' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
