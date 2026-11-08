# SCFM distillation utilities.
# Adapted from "Shortcutting Pre-trained Flow Matching Diffusion Models
# is Almost Free Lunch" (arXiv:2510.17858)

import math
import random
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import (
    FullStateDictConfig,
    StateDictType,
)


# ---------------------------------------------------------------------------
# Timestep Sampling
# ---------------------------------------------------------------------------

def sample_scfm_timesteps(
    num_images: int,
    t_skip: int,
    num_euler_steps: int,
    timestep_shift: float,
    device: torch.device,
    teacher_ratio: float = 0.4,
    teacher_min_timesteps: int = 32,
    teacher_max_timesteps: int = 32,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Sample starting timesteps and compute teacher/student dt for SCFM.

    Uses variable-length trajectory sampling so the model learns shortcuts
    at multiple scales, enabling 4-8 step inference (not just one fixed scale).

    For each image, we pick one of two sampling modes:
      - Teacher mode (teacher_ratio fraction): long trajectory
        (teacher_min..teacher_max steps), shortcut covers t_skip sub-steps.
        Model learns fine-grained shortcuts.
      - Self-teaching mode (1-teacher_ratio fraction): short trajectory
        (randomly from powers of 2: 2, 4, 8, ..., up to teacher_min).
        Model learns large shortcuts for few-step inference.

    When tts=4 and t_skip=2, the Euler grid has 4 steps. Shortcut covers
    2 of those 4 steps → at inference, the model can do 2-step generation.
    When tts=8, shortcut covers 2/8 → 4-step inference.

    Args:
        num_images: number of VAE images in the packed batch.
        t_skip: number of teacher Euler sub-steps per shortcut (e.g. 2).
        num_euler_steps: max Euler steps for teacher grid (e.g. 50).
        timestep_shift: timestep warping parameter.
        device: target device.
        teacher_ratio: fraction of images using long (teacher) trajectories.
        teacher_min_timesteps: min steps for teacher trajectory.
        teacher_max_timesteps: max steps for teacher trajectory.

    Returns:
        t_start:     [num_images]          starting timestep (warped, in [0,1]).
        dt_teacher:  [num_images, t_skip]  per-substep dt for teacher Euler.
        dt_shortcut: [num_images]          total dt the student must cover.
    """
    teacher_min_timesteps = max(teacher_min_timesteps, t_skip)

    t_start = torch.empty(num_images)
    dt_teacher = torch.empty(num_images, t_skip)
    dt_shortcut = torch.empty(num_images)

    for i in range(num_images):
        # Choose trajectory length (tts = total timesteps in the Euler grid)
        if random.random() < teacher_ratio:
            # Teacher mode: long trajectory for fine-grained shortcuts
            candidates = list(range(teacher_min_timesteps, teacher_max_timesteps + t_skip, t_skip))
            tts = random.choice(candidates) if candidates else teacher_min_timesteps
        else:
            # Self-teaching mode: short trajectory (powers of 2) for large shortcuts
            # e.g. if teacher_min=32 → power=5 → choices: 4, 8, 16 (filtered by t_skip)
            power = int(math.log2(teacher_min_timesteps))
            choices = [1 << p for p in range(1, power) if (1 << p) >= t_skip]
            tts = random.choice(choices) if choices else t_skip

        # Build Euler grid for this trajectory length
        t_grid = torch.linspace(1.0, 0.0, tts + 1)
        # Apply timestep warping
        t_grid = timestep_shift * t_grid / (1.0 + (timestep_shift - 1.0) * t_grid)
        t_grid[0] = 1.0 - 1e-5  # avoid pure noise

        # Pick a random starting index within this grid
        max_start_idx = tts - t_skip
        idx = random.randint(0, max_start_idx)

        t_start[i] = t_grid[idx]
        for j in range(t_skip):
            dt_teacher[i, j] = t_grid[idx + j] - t_grid[idx + j + 1]
        dt_shortcut[i] = t_grid[idx] - t_grid[idx + t_skip]

    return t_start.to(device), dt_teacher.to(device), dt_shortcut.to(device)


# ---------------------------------------------------------------------------
# LoRA EMA Manager
# ---------------------------------------------------------------------------

class LoRAEMAManager:
    """Dual EMA for LoRA parameters (SCFM self-teaching).

    Maintains two exponential moving averages of the trainable LoRA weights:
      - slow_ema (decay=0.999): stable teacher signal for the second jump.
      - fast_ema (decay=0, i.e. copy of current): self-teaching for the first jump.

    Parameters are stored as plain dicts of full (non-sharded) tensors so
    that they can be swapped into the FSDP-wrapped model during teacher
    trajectory generation.
    """

    def __init__(self, fsdp_model: FSDP, slow_decay: float = 0.999):
        self.slow_decay = slow_decay
        self._lora_names: List[str] = []
        self.slow_ema: Dict[str, Tensor] = {}
        self.fast_ema: Dict[str, Tensor] = {}

        # Collect LoRA parameter names
        for name, param in fsdp_model.named_parameters():
            if "lora_" in name and param.requires_grad:
                self._lora_names.append(name)

        # Initialize from current model state
        self._gather_and_init(fsdp_model)

    def _gather_and_init(self, fsdp_model: FSDP):
        """Gather full LoRA params from FSDP and initialize both EMAs."""
        with FSDP.state_dict_type(
            fsdp_model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=False, offload_to_cpu=True),
        ):
            state = fsdp_model.state_dict()
            for name in self._lora_names:
                if name in state:
                    self.slow_ema[name] = state[name].clone()
                    self.fast_ema[name] = state[name].clone()

    @torch.no_grad()
    def update(self, fsdp_model: FSDP):
        """Update both EMAs from current LoRA parameters. Call after optimizer.step()."""
        with FSDP.state_dict_type(
            fsdp_model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=False, offload_to_cpu=True),
        ):
            state = fsdp_model.state_dict()
            for name in self._lora_names:
                if name not in state:
                    continue
                current = state[name]
                # slow EMA: exponential moving average
                self.slow_ema[name].mul_(self.slow_decay).add_(
                    current, alpha=1.0 - self.slow_decay
                )
                # fast EMA: just copy current (decay=0)
                self.fast_ema[name].copy_(current)

    def get_teacher_params(
        self, step_in_trajectory: int, t_skip: int
    ) -> Dict[str, Tensor]:
        """Two-jump scheme: first half → fast_ema, second half → slow_ema."""
        if step_in_trajectory < t_skip // 2:
            return self.fast_ema
        else:
            return self.slow_ema

    @contextmanager
    def swap_weights(self, fsdp_model: FSDP, lora_params_dict: Dict[str, Tensor]):
        """Context manager: swap LoRA weights to given values, restore on exit.

        Uses FSDP.summon_full_params with writeback=True so that in-place
        parameter modifications are reflected back into the sharded state.
        """
        saved: Dict[str, Tensor] = {}
        with FSDP.summon_full_params(fsdp_model, writeback=True):
            for name, param in fsdp_model.named_parameters():
                if name in lora_params_dict:
                    saved[name] = param.data.clone()
                    param.data.copy_(lora_params_dict[name].to(param.device, param.dtype))
            try:
                yield
            finally:
                for name, param in fsdp_model.named_parameters():
                    if name in saved:
                        param.data.copy_(saved[name])


# ---------------------------------------------------------------------------
# Teacher Trajectory Generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_teacher_trajectory(
    model,  # Bagel model (unwrapped)
    fsdp_model: FSDP,
    lora_ema: LoRAEMAManager,
    packed_seq_base: Tensor,
    packed_latent_clean: Tensor,
    noise: Tensor,
    t_start_per_token: Tensor,
    dt_teacher_per_token: Tensor,
    t_skip: int,
    # data plumbing (forwarded to model.forward_scfm_teacher_step)
    packed_vae_token_indexes: Tensor,
    packed_latent_position_ids: Tensor,
    sample_lens: List[int],
    packed_position_ids: Tensor,
    attention_mask,
    mse_loss_indexes: Tensor,
    **extra_inputs,
) -> Tensor:
    """Run t_skip teacher Euler steps and return the target shortcut velocity.

    Args:
        model: unwrapped Bagel model (for calling forward_scfm_teacher_step).
        fsdp_model: FSDP-wrapped model (for weight swapping).
        lora_ema: dual EMA manager.
        packed_seq_base: packed sequence with text+ViT already placed.
        packed_latent_clean: clean VAE latent x_0.
        noise: Gaussian noise epsilon.
        t_start_per_token: [N_vae_tokens] starting timestep per token.
        dt_teacher_per_token: [N_vae_tokens, t_skip] per-substep dt.
        t_skip: number of teacher sub-steps.
        ... data plumbing args for LLM forward.

    Returns:
        target_velocity: [N_loss_tokens, latent_dim] teacher target for student.
    """
    # x_t at start
    x_t = (
        (1.0 - t_start_per_token[:, None]) * packed_latent_clean
        + t_start_per_token[:, None] * noise
    )
    t_current = t_start_per_token.clone()
    accumulated_displacement = None

    for step in range(t_skip):
        teacher_params = lora_ema.get_teacher_params(step, t_skip)

        with lora_ema.swap_weights(fsdp_model, teacher_params):
            v_t = model.forward_scfm_teacher_step(
                packed_seq_base=packed_seq_base,
                x_t=x_t,
                timestep=t_current,
                packed_vae_token_indexes=packed_vae_token_indexes,
                packed_latent_position_ids=packed_latent_position_ids,
                sample_lens=sample_lens,
                packed_position_ids=packed_position_ids,
                attention_mask=attention_mask,
                mse_loss_indexes=mse_loss_indexes,
                **extra_inputs,
            )

        # dt for this sub-step (only for loss tokens = mse_loss_indexes)
        # dt_teacher_per_token is for ALL vae tokens; v_t is for loss tokens only.
        # We need the dt for loss tokens. Since packed_timesteps > 0 selects loss tokens,
        # and dt_teacher is per-image (broadcast to tokens), we extract loss-token dt.
        has_loss = t_start_per_token > 0  # loss tokens have t > 0
        dt_sub = dt_teacher_per_token[has_loss, step : step + 1]

        displacement = v_t * dt_sub
        if accumulated_displacement is None:
            accumulated_displacement = displacement
        else:
            accumulated_displacement = accumulated_displacement + displacement

        # Euler step on ALL vae tokens (not just loss tokens)
        # Build full-size displacement and apply
        full_displacement = torch.zeros_like(x_t)
        full_displacement[has_loss] = displacement
        x_t = x_t - full_displacement
        t_current = t_current - dt_teacher_per_token[:, step]

    # Target velocity = total displacement / total dt
    dt_shortcut = dt_teacher_per_token[has_loss].sum(dim=1, keepdim=True)
    target_velocity = accumulated_displacement / dt_shortcut.clamp(min=1e-8)

    return target_velocity.detach()
