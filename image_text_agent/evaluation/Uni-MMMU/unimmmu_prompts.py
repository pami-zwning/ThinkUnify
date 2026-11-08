# -*- coding: utf-8 -*-

"""
Prompts for Uni-MMMU (geometry / jigsaw / maze)

- Tool-call version:
  - geometry: SINGLE user prompt that requests both (draw auxiliary lines) + (solve)
  - jigsaw/maze: unified prompts (answer format aligned to our eval)

- Direct version kept for your other script if needed.
"""

# ========== Direct mode system prompt (tool-call mode uses SYSTEM_ROLE_PROMPT from desc ==========
DIRECT_SYSTEM_PROMPT = """You are a careful multimodal assistant.
Follow the user instructions and output the final answer wrapped in <answer>...</answer>.
Do not output tool calls."""

# ========== Jigsaw ==========
# Inputs order: [ref_2x2.png, cand0.png, cand1.png]
JIGSAW_USER_PROMPT = """You will be given 3 images in this exact order:
1) Reference 2×2 image with the bottom-right cell missing
2) Candidate 0 patch
3) Candidate 1 patch

Decide which candidate correctly completes the missing bottom-right cell.

Output ONLY:
<answer>0</answer>
or
<answer>1</answer>
"""

# ========== Maze ==========
MAZE_USER_PROMPT = """Analyze the provided maze image. Your task is to find the path from the start dot to the goal region.
(Notes: in some mazes the start dot may appear blue instead of red; the goal is green.)

Requirements:
1) Output the solution strictly as a continuous sequence of directional moves.
2) Use 'R' for Right, 'L' for Left, 'U' for Up, and 'D' for Down.
3) The path must not cross any walls.

Provide your final answer wrapped in <answer> </answer>, e.g. <answer>LRUD</answer>.
"""

# ========== Geometry (tool-call, single-turn user prompt) ==========
GEOMETRY_TOOLCALL_OVERLAY_PREFIX = """You are a geometry diagram editor.
On the attached base figure, overlay the auxiliary lines EXACTLY as specified below.
- Add overlays only; do not move/erase the original objects or labels.
- Keep labels (A, B, C, …) unchanged and clearly visible.
- Draw clean, visible lines.
Use tools when needed."""

GEOMETRY_TOOLCALL_REASONING_PREFIX = """Then solve the geometry problem using the drawn auxiliary lines.
Give a concise, logically ordered solution (≤150 words).
End with the final result wrapped in <answer>...</answer>."""

GEOMETRY_REASONING_DIRECT_PREFIX = """You are a geometry solver.
Solve the problem concisely (≤150 words) and output the final result wrapped in <answer>...</answer>."""
