"""MMMU inference module.

Dataset: MMMU/MMMU, split=validation
Fields: id, subject, question_type (multiple-choice/open), question (with <image_N> refs),
        options (list-str), answer, image_1..image_7 (PIL, may be None)
"""

import os, json, ast
from PIL import Image
from ._bench_utils import format_mc_prompt, format_openended_prompt, save_pil_image


MMMU_SUBJECTS = None  # loaded lazily


def _collect_images(sample, image_cache_dir, hash_uid):
    image_paths = []
    for i in range(1, 8):
        key = f"image_{i}"
        if key in sample and sample[key] is not None:
            p = os.path.join(image_cache_dir, f"{hash_uid}_img{i}.png")
            save_pil_image(sample[key], p)
            image_paths.append(p)
    return image_paths


def prepare_mmmu_jsonl(output_path, image_cache_dir):
    if os.path.exists(output_path):
        return
    from datasets import load_dataset, get_dataset_config_names
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(image_cache_dir, exist_ok=True)

    print("[MMMU] Loading MMMU/MMMU (validation) — all 30 subjects...")
    subjects = get_dataset_config_names("MMMU/MMMU")
    records = []
    for subj in subjects:
        try:
            ds = load_dataset("MMMU/MMMU", subj, split="validation")
        except Exception as e:
            print(f"[MMMU] skip {subj}: {e}")
            continue
        for idx, sample in enumerate(ds):
            hash_uid = f"mmmu_{subj}_{sample.get('id', idx)}"
            image_paths = _collect_images(sample, image_cache_dir, hash_uid)
            if not image_paths:
                continue  # MMMU requires at least one image for Bagel
            options = sample.get("options", "[]")
            if isinstance(options, str):
                try:
                    options = ast.literal_eval(options)
                except Exception:
                    options = []
            records.append({
                "hash_uid": hash_uid,
                "subject": subj,
                "images": image_paths,
                "question": sample["question"],
                "question_type": sample.get("question_type", "open"),
                "options": options,
                "gold_answer": str(sample["answer"]),
            })

    with open(output_path, "w", encoding="utf-8") as f:
        for rec in records:
            # Also set `image` to the first path for run_bagel.py compatibility.
            rec["image"] = rec["images"][0]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[MMMU] Prepared {output_path} ({len(records)} samples across {len(subjects)} subjects).")


def _format_question(question):
    """Replace <image_N> tags with [Image N] markers since images are passed separately."""
    import re
    return re.sub(r"<image_(\d+)>", r"[Image \1]", question)


def data_to_messages(item):
    q = _format_question(item["question"])
    if item.get("question_type") == "multiple-choice" and item.get("options"):
        prompt = format_mc_prompt(q, item["options"], image_desc="images")
    else:
        prompt = format_openended_prompt(q, image_desc="images")

    messages = [dict(type="image", value=p) for p in item["images"]]
    messages.append(dict(type="text", value=prompt))
    return messages


def bagel_pipeline(vlm, item):
    messages = data_to_messages(item)
    out, _ = vlm.internal_infer(messages, understanding_output=False, think=True, max_rounds=3)
    return out, [], None
