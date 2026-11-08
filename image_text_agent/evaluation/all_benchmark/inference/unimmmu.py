from io import text_encoding
import os
import re
from PIL import Image
from string import Template

"""
设计原则:
1. 假设不需要官方的Prompt, 因为Bagel现在还不支持, 待后续推进
2. 如果遇到测试时的指令跟随问题, 请仔细检查构造的数据和官方的Prompt是否匹配
"""
PROMPT_TEMPLATE_MAZE = """You are a precise maze solver.

SEMANTICS (for all mazes)
- Black squares: walls (impassable)
- White squares: path (walkable)
- Blue dot: start (the agent)
- Green rectangular frame: goal (reaching any white cell inside the green frame counts as success)
- Legal moves: up, down, left, right only. One cell per step; no diagonals, no jumps; never cross walls.

OUTPUT FORMAT (STRICT)
1) MULTI-IMAGE MODE — generate a SEQUENCE OF SEPARATE IMAGES, one per move:
   - Each output image must depict the maze state AFTER applying exactly one legal move.
   - Do NOT include the initial (pre-move) state.
   - Keep palette/layout/scale identical to the input; only the blue dot moves.
   - The number of returned images MUST equal the number of moves in the final answer (see step 2).
   - Absolutely FORBIDDEN: any collage/montage/spritesheet/grid/multi-panel/side-by-side/stacked images; no arrows, captions, or overlays; no GIFs/animations/video.

2) After all step images, emit EXACTLY ONE LINE containing ONLY the final move (Use 'R' for Right, 'L' for Left, 'U' for Up, and 'D' for Down), wrapped as:
   <answer>LRUD</answer>


NO EXTRAS
- No tools, no OCR, no explanations, and no text other than the single <answer>...</answer> line.
- Do not restate the instructions or the condition.

REMINDERS
- Decide the full path first, then emit the image sequence (one image per move), then the single <answer> line.
- One move per image; images must be separate files/parts, not stitched together in any way.
"""

PROMPT_TEMPLATE_SLIDING = """You are a precise sliding puzzle solver.

TASK
- You will be given two images: an INITIAL state and a FINAL state of a 3x3 sliding puzzle.
- The goal is to find the sequence of moves to transform the INITIAL state into the FINAL state.

SEMANTICS
- The puzzle is a 3x3 grid with 8 colored tiles and one empty space.
- The RED square represents the EMPTY space.
- A "move" consists of sliding an adjacent colored tile INTO the empty (red) space.
- Moves are named by the direction the COLORED TILE moves. For example, if the blue tile is directly above the red space, moving the blue tile down into the red space's position is a "down" move.
- Legal moves: up, down, left, right only. One tile per step.

OUTPUT FORMAT (STRICT)
1) MULTI-IMAGE MODE — generate a SEQUENCE OF SEPARATE IMAGES, one per move:
   - Each output image must depict the puzzle state AFTER applying exactly one legal move.
   - Do NOT include the initial (pre-move) state.
   - Keep the visual style identical to the inputs; only tile positions change.
   - The number of returned images MUST equal the number of moves in the final answer (see step 2).
   - Absolutely FORBIDDEN: collage/montage/grid/stacked images; no arrows, captions, or overlays; no GIFs/animations/video.

2) After all step images, emit EXACTLY ONE LINE containing ONLY the final move (Use 'R' for Right, 'L' for Left, 'U' for Up, and 'D' for Down), wrapped as:
   <answer>LRUD</answer>


NO EXTRAS
- No tools, no explanations, and no text other than the single <answer>…</answer> line.
- Do not restate the instructions.

REMINDERS
- First decide the full path, then emit the image sequence (one image per move), then the single <answer> line.
- One move per image; images must be separate files/parts, not stitched.

After the single <answer>…</answer> line, output nothing else.
"""

PROMPT_JIGSAW = """
You are a unified vision-language model. You will be given:
(1) a 2×2 reference image with the bottom-right cell hidden, and
(2) two candidate patch images (“Candidate 0” and “Candidate 1”).

Your job:
- For each candidate, synthesize a completed 2×2 image by placing that candidate EXACTLY into the bottom-right cell. Keep the other three cells pixel-identical to the reference (no filtering, no re-rendering). If sizes differ, only scale the candidate to fit that quadrant; do NOT rotate, mirror, or alter colors.
- Compare the two completed results and decide which candidate yields the correct completion.

Output EXACTLY the following, in order:

1) A single image with Candidate 0 placed in the bottom-right cell

2) A single image with Candidate 1 placed in the bottom-right cell


3) analysis comparing seam continuity, color/texture gradient, structural alignment, and global semantics

4) Output ONLY:\n<answer>0</answer>\nor\n<answer>1</answer>\n

Hard constraints:
- Deterministic, single outcome. No hedging, no multiple possibilities.
- No meta talk about prompts, models, or pipelines.
- Do not restate the task as the answer; reason from visual evidence.
- The only edits allowed are pasting the candidate into the bottom-right cell and necessary size matching for that cell. All other pixels must remain identical to the reference.

Inputs :
"""

PROMPT_GEO = Template("""You are a geometry diagram editor and solver.

TASK ORDER:
1) OVERLAY: On the attached base figure, overlay the auxiliary lines EXACTLY as specified below.
   - Add overlays only; do not move/erase the original objects or labels.
   - Keep labels (A, B, C, …) unchanged and clearly visible.
   - Draw clean, visible lines.

2) REASONING: Give a concise, logically ordered solution or proof (≤150 words), using the constructed auxiliary lines.
   - Keep math tokens (△, ∠, √, π, °) unchanged.
   - Reference elements by their labels.

3) FINISHING:
   - For calculation problems, end with:  **Final answer: <VALUE>**.
   - For proving problems, end with:     **Conclusion: <STATEMENT>**.

PROBLEM:
$PROBLEM_TEXT

AUXILIARY LINES TO DRAW (English; follow exactly and draw these first):
$AUX_EN
""")

def message_to_step(item):
    assert item["messages"][0]["role"] == "user"
    content = item["messages"][0]["content"]
    
    messages = []
    for value in content:
        if value["type"] == "text":
            messages.append(dict(type='text', value=value["text"]))
        elif value["type"] == "image_url":
            messages.append(dict(type='image', value=value["image_url"]["url"]))
        else:
            raise ValueError(f"{value}")
    return messages

def to_seq_input(messages):

    input_list = []
    for m in messages:
        if m["type"] == "text":
            input_list.append(m["value"])
        elif m["type"] == "image":
            input_list.append(Image.open(m["value"]).convert("RGB"))
    return input_list

def to_geometry_steps(item, direct=True):
    if direct:
        return message_to_step(item)
    else:
        k = 1
        messages, messages_list = [], []
        prompt = PROMPT_GEO.safe_substitute(
            PROBLEM_TEXT=item["reasoning_qs"],
            AUX_EN=item["auxiliary_qs"],
        )
        messages.append(dict(type="text", value=prompt))
        messages.append(dict(type="text", value="BASE FIGURE: The following image is the original diagram."))
        messages.append(dict(type="image", value=item["reasoning_imgs"][0]))
        return messages, k

def to_jigsaw_steps(item, direct=True):
    # NOTE: 该任务需要分别将两个候选图填入初始图, 并二选一
    # 如果构造了CoT数据, 那么直接infer即可, 否则需要构造duoz
    if direct:
        return message_to_step(item)
    else:
        k = 2
        messages, messages_list = [], []
        messages.append(dict(type="text", value=PROMPT_JIGSAW))
        messages.append(dict(type="text", value="REFERENCE_2x2:"))
        messages.append(dict(type="image", value=item["reasoning_imgs"][0]))
        messages.append(dict(type="text", value="CANDIDATE_0:"))
        messages.append(dict(type="image", value=item["reasoning_imgs"][1]))
        messages.append(dict(type="text", value="CANDIDATE_1:"))
        messages.append(dict(type="image", value=item["reasoning_imgs"][2]))
        return messages, k

def to_maze_steps(item, direct=True):
    if direct:
        return message_to_step(item)
    else:
        k = len(item["gt_text"])
        messages, messages_list = [], []
        messages.append(dict(type="text", value=PROMPT_TEMPLATE_MAZE))
        messages.append(dict(type="image", value=item["reasoning_imgs"][0]))
        return messages, k

def to_sliding_steps(item, direct=True):
    if direct:
        return message_to_step(item)
    else:
        k = len(item["gt_text"])
        messages, messages_list = [], []
        messages.append(dict(type="text", value=PROMPT_TEMPLATE_SLIDING))
        messages.append(dict(type="text", value="NEW TASK: Initial state."))
        messages.append(dict(type="image", value=item["reasoning_imgs"][0]))
        messages.append(dict(type="text", value="NEW TASK: Final state is exactly the same as example"))
        messages.append(dict(type="image", value=item["reasoning_imgs"][1]))
        return messages, k

# step-wise的推理流程
def bagel_pipeline_direct(vlm, item):
    if item["task"] == "geometry":
        messages = to_geometry_steps(item, direct=True)
    elif item["task"] == "jigsaw":
        messages = to_jigsaw_steps(item, direct=True)
    elif item["task"] == "maze":
        messages = to_maze_steps(item, direct=True)
    elif item["task"] == "sliding":
        messages = to_sliding_steps(item, direct=True)
    else:
        return None, None, None
    max_rounds = int(os.environ.get("MAX_ROUNDS", 3))
    final_output, image_list = vlm.internal_infer(
        messages, understanding_output=False, think=True, max_rounds=max_rounds)
    return final_output, image_list, None           # Final output, Generated image, Mid output

def bagel_pipeline_step(vlm, item):
    if item["task"] == "geometry":
        current_messages, k = to_geometry_steps(item, direct=False)
    elif item["task"] == "jigsaw":
        current_messages, k = to_jigsaw_steps(item, direct=False)
    elif item["task"] == "maze":
        current_messages, k = to_maze_steps(item, direct=False)
    elif item["task"] == "sliding":
        current_messages, k = to_sliding_steps(item, direct=False)
    else:
        return None, None, None

    # Maze/Sliding
    
    mid_output = []
    for i in range(k):
        if item["task"] == "maze" or item["task"] == "sliding":
            current_messages.append(dict(type="text", value=f'Now planing for step {i}, Please output a sentence in the form: "Next, move one step up/down/left/right.'))
            final_output, image_list = vlm.internal_infer(
                to_seq_input(current_messages), understanding_output=True, think=False, postprocess=False)
            current_messages.append(dict(type="text", value=final_output))
            current_messages.append(dict(type="text", value=f"Now, generate the image for step {i}. "))
            mid_output.append(final_output)

        if item["task"] == "jigsaw":
            current_messages.append(dict(type="text", value=f"Output ONLY item ({i+1}): a single image with Candidate {i} placed in the bottom-right cell. No text."))
            mid_output.append(current_messages[-1]["value"])
        if item["task"] == "geometry":
            assert k == 1

            current_messages.append(dict(type="text", value=(
                    "STEP 1 — OVERLAY now: Output EXACTLY ONE image of the base figure "
                    "with the auxiliary lines overlaid as specified. "
                    "Do not change existing objects/labels, do not add text, no captions."
                )))
            mid_output.append(current_messages[-1]["value"])

        final_output, image_list = vlm.internal_infer(
            to_seq_input(current_messages), understanding_output=False, think=True, postprocess=False)
        if len(image_list) == 0:
            return None, None, None
        edited_image = image_list[-1]
        mid_output.append(edited_image)

        if item["task"] == "jigsaw":
            current_messages.append(dict(type="text", value=f"COMPLETED WITH CANDIDATE {i}:"))
        if item["task"] == "geometry":
            current_messages.append(dict(type="text", value=f"OVERLAY RESULT (reference for reasoning):"))
        current_messages.append(dict(type="image", value=edited_image))

    if item["task"] == "maze" or item["task"] == "sliding":
        current_messages.append(dict(type="text", value=("After the images, emit EXACTLY ONE LINE containing ONLY the final move list "
                                "as <answer>...</answer>. No other text.")))
    elif item["task"] == "jigsaw":
        current_messages.append(dict(type="text", value="After the images, emit EXACTLY ONE LINE containing ONLY <answer>0</answer>\nor\n<answer>1</answer>\n"))
    elif item["task"] == "geometry":
        current_messages.append(dict(type="text", value=(
            "STEP 2 — REASONING now: Provide ONLY the concise solution/proof (≤150 words), "
            "using the auxiliary lines. Keep math tokens (△, ∠, √, π, °) unchanged. "
            "For calculation problems, end with '**Final answer: <VALUE>**'. "
            "For proving problems, end with '**Conclusion: <STATEMENT>**'. "
            "Output TEXT ONLY (no images, no markdown images) wrapped in <answer>...</answer>."
        )))

    final_output, _ = vlm.internal_infer(
            to_seq_input(current_messages), understanding_output=True, think=False, postprocess=False)
    mid_output.append(final_output)
    return final_output, None, mid_output