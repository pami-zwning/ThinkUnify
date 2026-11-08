"""MathVista inference module.

Dataset: AI4Math/MathVista, split=testmini (1000 samples)
Fields: pid, question_type (multi_choice/free_form), answer_type, answer, query, choices, decoded_image
"""

import os, json
from PIL import Image
from ._bench_utils import format_mc_prompt, format_openended_prompt, save_pil_image, LETTER_LABELS


def prepare_mathvista_jsonl(output_path, image_cache_dir):
    if os.path.exists(output_path):
        return
    from datasets import load_dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(image_cache_dir, exist_ok=True)

    print("[MathVista] Loading AI4Math/MathVista (testmini)...")
    dataset = load_dataset("AI4Math/MathVista", split="testmini")
    print(f"[MathVista] Loaded {len(dataset)} samples.")

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, s in enumerate(dataset):
            img_path = os.path.join(image_cache_dir, f"mathvista_{idx}.png")
            save_pil_image(s["decoded_image"], img_path)
            choices = s.get("choices") or []
            if isinstance(choices, str):
                import ast
                try:
                    choices = ast.literal_eval(choices)
                except Exception:
                    choices = []
            record = {
                "hash_uid": f"mathvista_{s.get('pid', idx)}",
                "image": img_path,
                "question": s["query"],
                "question_type": s.get("question_type", "free_form"),
                "answer_type": s.get("answer_type", "text"),
                "gold_answer": str(s["answer"]),
                "choices": choices,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[MathVista] Prepared {output_path}")


def data_to_messages(item):
    q = item["question"]
    choices = item.get("choices") or []
    if item.get("question_type") == "multi_choice" and choices:
        prompt = format_mc_prompt(q, choices, image_desc="image")
    else:
        prompt = format_openended_prompt(q, image_desc="image")
    return [dict(type="image", value=item["image"]), dict(type="text", value=prompt)]


def bagel_pipeline(vlm, item):
    messages = data_to_messages(item)
    out, _ = vlm.internal_infer(messages, understanding_output=False, think=True, max_rounds=3)
    return out, [], None
