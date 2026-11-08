"""GenEval inference module for ThinkMorph.

Benchmark: GenEval (NeurIPS 2023) — 553 compositional T2I prompts across 6 categories.

Official evaluation format (required by evaluate_images.py):
  {imagedir}/
    00000/
      metadata.jsonl   ← single JSON line from evaluation_metadata.jsonl
      samples/
        0.png
        1.png
        ...            ← IMAGES_PER_PROMPT images (default 4)
    00001/
      ...

This module generates images into that exact structure so that
calculate_score/geneval.py can call the official Mask2Former evaluator.

Sub-categories: single_object, two_object, counting, colors, position, color_attribution
"""

import os
import json

# Absolute path to the cloned GenEval repo
GENEVAL_REPO = os.path.join(
    os.path.dirname(__file__),
    "..", "calculate_score", "geneval",
)
EVALUATION_METADATA_PATH = os.path.join(GENEVAL_REPO, "prompts", "evaluation_metadata.jsonl")

IMAGES_PER_PROMPT = 1  # GenEval standard: 4 images per prompt


def _load_evaluation_metadata():
    """Load all 553 metadata entries from the official GenEval prompts file."""
    entries = []
    with open(EVALUATION_METADATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def prepare_geneval_jsonl(output_path, image_cache_dir=None):
    """Build JSONL for run_bagel.py from the official GenEval evaluation_metadata.jsonl.

    Each of the 553 prompts is expanded to IMAGES_PER_PROMPT records, each with
    a unique hash_uid encoding (prompt_idx, img_idx) so that bagel_pipeline can
    save images into the correct official directory structure.

    Args:
        output_path: path to write the JSONL consumed by run_bagel.py
        image_cache_dir: unused (kept for API compatibility with other tasks)
    """
    if os.path.exists(output_path):
        return

    metadata_entries = _load_evaluation_metadata()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for prompt_idx, meta in enumerate(metadata_entries):
            prompt = meta["prompt"]
            tag = meta["tag"]
            for img_idx in range(IMAGES_PER_PROMPT):
                record = {
                    # Encode both indices so bagel_pipeline can reconstruct the path
                    "hash_uid": f"geneval_{prompt_idx:05d}_img{img_idx}",
                    "prompt": prompt,
                    "prompt_idx": prompt_idx,
                    "img_idx": img_idx,
                    "category": tag,
                    "metadata": meta,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    total = len(metadata_entries) * IMAGES_PER_PROMPT
    print(
        f"[GenEval] Prepared {total} entries "
        f"({len(metadata_entries)} prompts × {IMAGES_PER_PROMPT}) at {output_path}"
    )


def bagel_pipeline(vlm, item):
    """Generate one image for a GenEval prompt.

    Saves the image into the official directory structure:
      {vlm.save_dir}/{prompt_idx:05d}/samples/{img_idx}.png

    Also writes metadata.jsonl the first time (img_idx == 0).

    Returns (None, [img_path], None) — text output irrelevant for T2I.
    """
    prompt = item["prompt"]
    prompt_idx = item["prompt_idx"]
    img_idx = item["img_idx"]
    metadata = item.get("metadata", {"tag": item.get("category", "unknown"), "prompt": prompt})

    # Build official directory structure under vlm.save_dir
    sample_dir = os.path.join(vlm.save_dir, f"{prompt_idx:05d}", "samples")
    os.makedirs(sample_dir, exist_ok=True)

    # Write metadata.jsonl once (on first image for this prompt)
    meta_path = os.path.join(vlm.save_dir, f"{prompt_idx:05d}", "metadata.jsonl")
    if not os.path.exists(meta_path):
        with open(meta_path, "w", encoding="utf-8") as mf:
            mf.write(json.dumps(metadata) + "\n")

    img_save_path = os.path.join(sample_dir, f"{img_idx}.png")

    # Temporarily redirect vlm.save_dir so internal_infer saves here
    original_save_dir = vlm.save_dir

    messages = [dict(type="text", value=prompt)]

    # Set a unique seed per image to ensure diversity across the 4 images per prompt.
    # seed = prompt_idx * 4 + img_idx ensures each (prompt, image) pair is unique and
    # reproducible across runs.
    import torch, random, numpy as np
    seed = prompt_idx * IMAGES_PER_PROMPT + img_idx
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    text_output, image_list = vlm.internal_infer(
        messages,
        understanding_output=False,
        think=False,
        max_rounds=1,
        gen_only=True,
    )

    vlm.save_dir = original_save_dir

    # Move / rename the generated image to the required path
    if image_list:
        generated_path = image_list[0]
        if generated_path and os.path.exists(generated_path) and generated_path != img_save_path:
            import shutil
            shutil.move(generated_path, img_save_path)
        image_list = [img_save_path]

    return text_output, image_list, None
