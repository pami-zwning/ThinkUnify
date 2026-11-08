"""HR-Bench inference module (shared for HR-Bench-4K and HR-Bench-8K).

Dataset: DreamMr/HR-Bench, splits: hrbench_4k (800) / hrbench_8k (800)
Fields: question, answer (letter A-D), A/B/C/D (options), category, image (base64)
"""

import os
import json
import base64
import io
from PIL import Image
from ._bench_utils import format_mc_prompt, save_pil_image


def _decode_base64_image(b64_str):
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))


def prepare_hrbench_jsonl(output_path, image_cache_dir, split):
    """split: 'hrbench_4k' or 'hrbench_8k'."""
    if os.path.exists(output_path):
        return
    from datasets import load_dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(image_cache_dir, exist_ok=True)

    tag = split.replace("hrbench_", "")
    print(f"[HR-Bench-{tag.upper()}] Loading DreamMr/HR-Bench ({split})...")
    dataset = load_dataset("DreamMr/HR-Bench", split=split)
    print(f"[HR-Bench-{tag.upper()}] Loaded {len(dataset)} samples.")

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, sample in enumerate(dataset):
            img_path = os.path.join(image_cache_dir, f"hrbench_{tag}_{idx}.png")
            if not os.path.exists(img_path):
                pil = _decode_base64_image(sample["image"])
                save_pil_image(pil, img_path)
            record = {
                "hash_uid": f"hrbench_{tag}_{idx}",
                "image": img_path,
                "question": sample["question"],
                "options": [sample["A"], sample["B"], sample["C"], sample["D"]],
                "gold_answer": sample["answer"],
                "category": sample.get("category", ""),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[HR-Bench-{tag.upper()}] Prepared {output_path}")


def data_to_messages(item):
    prompt = format_mc_prompt(item["question"], item["options"], image_desc="high-resolution image")
    return [
        dict(type="image", value=item["image"]),
        dict(type="text", value=prompt),
    ]


def bagel_pipeline(vlm, item):
    messages = data_to_messages(item)
    final_output, _ = vlm.internal_infer(messages, understanding_output=False, think=True, max_rounds=3)
    return final_output, [], None
