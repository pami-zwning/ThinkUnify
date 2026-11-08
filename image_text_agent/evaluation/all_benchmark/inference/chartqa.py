"""ChartQA inference module for ThinkMorph.

Dataset: HuggingFaceM4/ChartQA (test split, 2500 samples)
Fields: image (PIL), query (str), label (list[str]), human_or_machine (0=human, 1=machine)

This module provides:
  - prepare_chartqa_jsonl(): generate JSONL format from HF dataset (one-time prep)
  - bagel_pipeline(): per-sample inference callable used by run_bagel.py
"""

import os
import json
from PIL import Image


def prepare_chartqa_jsonl(output_path, image_cache_dir):
    """Load HuggingFaceM4/ChartQA and write a JSONL with the schema run_bagel.py expects.

    Each line has: hash_uid, image (file path), query, label, human_or_machine.
    Images are cached to disk under image_cache_dir on first run.
    """
    if os.path.exists(output_path):
        return

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "ChartQA benchmark requires the `datasets` package. "
            "Install with: pip install datasets"
        ) from e

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(image_cache_dir, exist_ok=True)

    print(f"[ChartQA] Loading HuggingFaceM4/ChartQA (test split)...")
    dataset = load_dataset("HuggingFaceM4/ChartQA", split="test")
    print(f"[ChartQA] Loaded {len(dataset)} samples. Writing to {output_path}...")

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, sample in enumerate(dataset):
            pil_image = sample["image"]
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            img_path = os.path.join(image_cache_dir, f"chartqa_{idx}.png")
            if not os.path.exists(img_path):
                pil_image.save(img_path)

            # ChartQA label is list of strings
            label = sample["label"]
            if isinstance(label, str):
                label = [label]

            record = {
                "hash_uid": f"chartqa_{idx}",
                "image": img_path,
                "query": sample["query"],
                "label": label,
                "human_or_machine": int(sample.get("human_or_machine", -1)),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[ChartQA] Prepared JSONL at {output_path}")


def format_chartqa_prompt(query):
    return (
        f"{query}\n\n"
        "Please analyze the chart carefully, conduct step-by-step reasoning, "
        "and provide the final answer inside <answer></answer> tags."
    )


def data_to_messages(item):
    """Convert a ChartQA item into the ThinkMorph message list format."""
    prompt = format_chartqa_prompt(item["query"])
    return [
        dict(type="image", value=item["image"]),
        dict(type="text", value=prompt),
    ]


def bagel_pipeline(vlm, item):
    """Single-sample inference for ChartQA: text-only answer, no image generation."""
    messages = data_to_messages(item)
    final_output, _ = vlm.internal_infer(
        messages,
        understanding_output=False,
        think=True,
        max_rounds=3,
    )
    return final_output, [], None
