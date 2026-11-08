# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

"""
Reconstruction Alignment Loss Module (ReAC style)
Reference: Harmon (https://arxiv.org/abs/2509.07295)

Adapted for flow matching architecture:
  In flow matching, the model predicts velocity v_pred ≈ noise - x_0.
  Given noisy input x_t = (1-t)*x_0 + t*noise, we recover:
      x_0_pred = x_t - t * v_pred
  The reconstruction alignment loss ensures x_0_pred ≈ x_0_gt.

Loss components:
  1. Base MSE: per-token reconstruction error
  2. Spatial Gradient Loss: enforces local texture continuity on the 2D latent grid
  3. Spectral Loss: penalizes high-frequency reconstruction errors via FFT
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List


class ReconstructionAlignmentLoss(nn.Module):
    """
    Reconstruction Alignment Loss for flow matching with texture-level supervision.

    Three complementary loss components:
      - Base MSE: MSE(x_0_pred, x_0_gt) per token
      - Gradient loss: MSE on spatial gradients (horizontal + vertical) of 2D latent grid
      - Spectral loss: MSE weighted by frequency magnitude (high-freq emphasis)
    """

    def __init__(
        self,
        scale_factor: float = 1.0,
        gradient_weight: float = 0.5,
        spectral_weight: float = 0.1,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.gradient_weight = gradient_weight
        self.spectral_weight = spectral_weight

    def _compute_base_loss(
        self,
        x_0_pred: torch.Tensor,  # [N, D]
        x_0_gt: torch.Tensor,    # [N, D]
    ) -> torch.Tensor:
        """Per-token MSE, returns scalar."""
        return ((x_0_pred - x_0_gt) ** 2).mean()

    def _compute_gradient_loss(
        self,
        x_0_pred: torch.Tensor,  # [N, D] packed tokens for ONE image
        x_0_gt: torch.Tensor,    # [N, D]
        h: int,
        w: int,
    ) -> torch.Tensor:
        """
        Spatial gradient loss on a single image's 2D latent grid.

        Reshapes [h*w, D] → [h, w, D], computes horizontal/vertical diffs,
        then MSE between predicted and GT gradients.
        """
        pred_2d = x_0_pred.view(h, w, -1)  # [h, w, D]
        gt_2d = x_0_gt.view(h, w, -1)

        # Vertical gradients (along h)
        pred_grad_h = pred_2d[1:, :, :] - pred_2d[:-1, :, :]  # [h-1, w, D]
        gt_grad_h = gt_2d[1:, :, :] - gt_2d[:-1, :, :]

        # Horizontal gradients (along w)
        pred_grad_w = pred_2d[:, 1:, :] - pred_2d[:, :-1, :]  # [h, w-1, D]
        gt_grad_w = gt_2d[:, 1:, :] - gt_2d[:, :-1, :]

        loss_h = ((pred_grad_h - gt_grad_h) ** 2).mean()
        loss_w = ((pred_grad_w - gt_grad_w) ** 2).mean()

        return (loss_h + loss_w) * 0.5

    def _compute_spectral_loss(
        self,
        x_0_pred: torch.Tensor,  # [N, D] packed tokens for ONE image
        x_0_gt: torch.Tensor,    # [N, D]
        h: int,
        w: int,
    ) -> torch.Tensor:
        """
        Spectral loss via 2D FFT on the latent grid.

        Reshapes to [D, h, w], applies rfft2, computes MSE on frequency magnitudes.
        High-frequency components naturally have smaller magnitudes, so normalizing
        by GT magnitude emphasizes their relative accuracy → texture sharpness.
        """
        D = x_0_pred.shape[-1]
        pred_2d = x_0_pred.view(h, w, D).permute(2, 0, 1).float()  # [D, h, w]
        gt_2d = x_0_gt.view(h, w, D).permute(2, 0, 1).float()

        # 2D real FFT (requires float32)
        pred_fft = torch.fft.rfft2(pred_2d, norm="ortho")
        gt_fft = torch.fft.rfft2(gt_2d, norm="ortho")

        # MSE on complex magnitudes
        pred_mag = torch.abs(pred_fft)
        gt_mag = torch.abs(gt_fft)

        return ((pred_mag - gt_mag) ** 2).mean()

    def forward(
        self,
        v_pred: torch.Tensor,           # [N, D] velocity prediction from llm2vae
        x_t: torch.Tensor,              # [N, D] noisy latent input
        x_0_gt: torch.Tensor,           # [N, D] ground truth clean latent
        timesteps: torch.Tensor,         # [N] timestep for each token
        latent_shapes: Optional[List[Tuple[int, int]]] = None,  # per-image (h, w)
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute reconstruction alignment loss with texture-level components.

        Args:
            v_pred: velocity prediction [N, D]
            x_t: noisy latent x_t [N, D]
            x_0_gt: ground truth clean latent [N, D]
            timesteps: per-token timestep [N]
            latent_shapes: list of (h, w) per image, used for gradient/spectral loss.
                           If None, only base MSE is computed.
            mask: optional validity mask [N]
        """
        # Recover predicted clean latent
        t = timesteps.unsqueeze(-1)  # [N, 1]
        x_0_pred = x_t - t * v_pred

        # --- Texture losses (gradient + spectral) ---
        grad_loss = torch.tensor(0.0, device=x_0_pred.device)
        spec_loss = torch.tensor(0.0, device=x_0_pred.device)

        if latent_shapes is not None and (self.gradient_weight > 0 or self.spectral_weight > 0):
            # Split packed tokens back to per-image chunks
            chunk_sizes = [h * w for h, w in latent_shapes]
            if sum(chunk_sizes) == x_0_gt.shape[0]:
                pred_chunks = x_0_pred.split(chunk_sizes)
                gt_chunks = x_0_gt.split(chunk_sizes)

                grad_losses, spec_losses = [], []
                for pred_chunk, gt_chunk, (h, w) in zip(pred_chunks, gt_chunks, latent_shapes):
                    if h < 2 or w < 2:
                        continue
                    if self.gradient_weight > 0:
                        grad_losses.append(self._compute_gradient_loss(pred_chunk, gt_chunk, h, w))
                    if self.spectral_weight > 0:
                        spec_losses.append(self._compute_spectral_loss(pred_chunk, gt_chunk, h, w))

                if grad_losses:
                    grad_loss = torch.stack(grad_losses).mean()
                if spec_losses:
                    spec_loss = torch.stack(spec_losses).mean()

        # --- Combine ---
        total_loss = self.gradient_weight * grad_loss + self.spectral_weight * spec_loss
        total_loss = total_loss * self.scale_factor

        # # --- Metrics ---
        # metrics = {
        #     'recon_grad': grad_loss.item(),
        #     'recon_spec': spec_loss.item(),
        # }

        return total_loss
