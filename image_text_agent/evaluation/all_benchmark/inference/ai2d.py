"""AI2D inference module for ThinkMorph.

Dataset: lmms-lab/ai2d (test split, 3088 samples)
Fields: image (PIL), question (str), options (list[4]), answer (0-based index as str)

This module provides:
  - prepare_ai2d_jsonl(): generate JSONL format from HF dataset (one-time prep)
  - bagel_pipeline(): per-sample inference callable used by run_bagel.py
"""

import os
import json
from PIL import Image


LETTER_LABELS = ["A", "B", "C", "D"]
INDEX_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}


def prepare_ai2d_jsonl(output_path, image_cache_dir):
    """Load lmms-lab/ai2d and write a JSONL with the schema run_bagel.py expects.

    Each line has: hash_uid, image (file path), question, options, gold_answer.
    Images are cached to disk under image_cache_dir on first run.
    """
    if os.path.exists(output_path):
        return

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "AI2D benchmark requires the `datasets` package. "
            "Install with: pip install datasets"
        ) from e

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(image_cache_dir, exist_ok=True)

    print(f"[AI2D] Loading lmms-lab/ai2d (test split)...")
    dataset = load_dataset("lmms-lab/ai2d", split="test")
    print(f"[AI2D] Loaded {len(dataset)} samples. Writing to {output_path}...")

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, sample in enumerate(dataset):
            pil_image = sample["image"]
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            img_path = os.path.join(image_cache_dir, f"ai2d_{idx}.png")
            if not os.path.exists(img_path):
                pil_image.save(img_path)

            answer_idx = int(sample["answer"])
            record = {
                "hash_uid": f"ai2d_{idx}",
                "image": img_path,
                "question": sample["question"],
                "options": list(sample["options"]),
                "gold_answer": INDEX_TO_LETTER[answer_idx],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[AI2D] Prepared JSONL at {output_path}")


def format_ai2d_prompt(question, options):
    choice_str = ""
    for i, opt in enumerate(options):
        letter = LETTER_LABELS[i]
        choice_str += f"\n({letter}) {opt}"
    return (
        f"{question}{choice_str}\n\n"
        "Please analyze the diagram carefully, conduct step-by-step reasoning, "
        "and provide the final answer as a single letter (A, B, C, or D) "
        "inside <answer></answer> tags."
    )


def data_to_messages(item):
    """Convert an AI2D item into the ThinkMorph message list format."""
    prompt = format_ai2d_prompt(item["question"], item["options"])
    return [
        dict(type="image", value=item["image"]),
        dict(type="text", value=prompt),
    ]


def bagel_pipeline(vlm, item):
    """Single-sample inference for AI2D: text-only answer, no image generation."""
    messages = data_to_messages(item)
    final_output, _ = vlm.internal_infer(
        messages,
        understanding_output=False,
        think=True,
        max_rounds=3,
    )
    return final_output, [], None
