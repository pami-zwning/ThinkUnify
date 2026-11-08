"""RealWorldQA inference module.

Dataset: xai-org/RealWorldQA, split=test (765 samples)
Fields: question (with inline choices), answer (full text of correct option), image
"""

import os, json
from ._bench_utils import format_openended_prompt, save_pil_image


def prepare_realworldqa_jsonl(output_path, image_cache_dir):
    if os.path.exists(output_path):
        return
    from datasets import load_dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(image_cache_dir, exist_ok=True)

    print("[RealWorldQA] Loading xai-org/RealWorldQA (test)...")
    ds = load_dataset("xai-org/RealWorldQA", split="test")
    print(f"[RealWorldQA] Loaded {len(ds)} samples.")

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, s in enumerate(ds):
            img_path = os.path.join(image_cache_dir, f"rwqa_{idx}.png")
            save_pil_image(s["image"], img_path)
            record = {
                "hash_uid": f"rwqa_{idx}",
                "image": img_path,
                "question": s["question"],
                "gold_answer": str(s["answer"]).strip(),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[RealWorldQA] Prepared {output_path}")


def data_to_messages(item):
    prompt = format_openended_prompt(item["question"])
    return [dict(type="image", value=item["image"]), dict(type="text", value=prompt)]


def bagel_pipeline(vlm, item):
    messages = data_to_messages(item)
    out, _ = vlm.internal_infer(messages, understanding_output=False, think=True, max_rounds=3)
    return out, [], None
