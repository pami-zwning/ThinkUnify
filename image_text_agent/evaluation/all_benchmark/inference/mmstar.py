"""MMStar inference module.

Dataset: Lin-Chen/MMStar, split=val (1500 samples)
Fields: index, question (with inline choices), answer (A-D), category, l2_category, image
"""

import os, json
from ._bench_utils import format_openended_prompt, save_pil_image


def prepare_mmstar_jsonl(output_path, image_cache_dir):
    if os.path.exists(output_path):
        return
    from datasets import load_dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(image_cache_dir, exist_ok=True)

    print("[MMStar] Loading Lin-Chen/MMStar (val)...")
    ds = load_dataset("Lin-Chen/MMStar", split="val")
    print(f"[MMStar] Loaded {len(ds)} samples.")

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, s in enumerate(ds):
            img_path = os.path.join(image_cache_dir, f"mmstar_{idx}.png")
            save_pil_image(s["image"], img_path)
            record = {
                "hash_uid": f"mmstar_{s.get('index', idx)}",
                "image": img_path,
                "question": s["question"],  # contains inline (A)...(B)...(C)...(D)...
                "gold_answer": s["answer"],
                "category": s.get("category", ""),
                "l2_category": s.get("l2_category", ""),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[MMStar] Prepared {output_path}")


def data_to_messages(item):
    # Question already includes inline choices; just add <answer> tag instruction.
    prompt = (
        f"{item['question']}\n\n"
        "Please provide the final answer as a single letter (A, B, C, or D) "
        "inside <answer></answer> tags."
    )
    return [dict(type="image", value=item["image"]), dict(type="text", value=prompt)]


def bagel_pipeline(vlm, item):
    messages = data_to_messages(item)
    out, _ = vlm.internal_infer(messages, understanding_output=False, think=True, max_rounds=3)
    return out, [], None
