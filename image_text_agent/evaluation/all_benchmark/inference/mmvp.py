"""MMVP inference module for ThinkMorph.

Dataset: MMVP/MMVP (HuggingFace)
  - Images: 300 images in the HF dataset (``image`` column, PIL)
  - Questions: separate ``Questions.csv`` in the HF repo with columns
    Index, Question, Options, Correct Answer

Task: visual pattern matching — 150 image pairs where CLIP-similar images
      have different correct answers. Each pair has 2 yes/no questions.
Metric: *pair accuracy* — both questions in a pair must be correct.

Provides:
  - prepare_mmvp_jsonl(): download images + Questions.csv, merge, write JSONL
  - bagel_pipeline(): per-sample inference for run_bagel.py
"""

import os
import io
import json
import csv
from PIL import Image


def prepare_mmvp_jsonl(output_path, image_cache_dir):
    """Load MMVP images and Questions.csv from HuggingFace, merge, write JSONL."""
    if os.path.exists(output_path):
        return

    try:
        from datasets import load_dataset
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        raise ImportError(
            "MMVP requires `datasets` and `huggingface_hub`. "
            "Install with: pip install datasets huggingface_hub"
        ) from e

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(image_cache_dir, exist_ok=True)

    # 1. Load images from HF dataset
    print("[MMVP] Loading MMVP/MMVP images...")
    try:
        img_dataset = load_dataset("MMVP/MMVP", split="test")
    except ValueError:
        img_dataset = load_dataset("MMVP/MMVP", split="train")
    print(f"[MMVP] Loaded {len(img_dataset)} images.")

    # Save images to cache (indexed 1-300 to match Questions.csv Index)
    for idx in range(len(img_dataset)):
        img_path = os.path.join(image_cache_dir, f"{idx + 1}.png")
        if not os.path.exists(img_path):
            img_field = img_dataset[idx]["image"]
            if isinstance(img_field, Image.Image):
                pil_image = img_field
            elif isinstance(img_field, dict) and img_field.get("bytes"):
                pil_image = Image.open(io.BytesIO(img_field["bytes"]))
            elif isinstance(img_field, dict) and img_field.get("path"):
                pil_image = Image.open(img_field["path"])
            else:
                print(f"[MMVP] Warning: cannot decode image {idx}")
                continue
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            pil_image.save(img_path)

    # 2. Download Questions.csv from HF repo
    print("[MMVP] Downloading Questions.csv...")
    csv_path = hf_hub_download(
        repo_id="MMVP/MMVP",
        filename="Questions.csv",
        repo_type="dataset",
    )

    # 3. Parse CSV and merge with images
    questions = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(row)
    print(f"[MMVP] Loaded {len(questions)} questions from CSV.")

    # 4. Write JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for q_idx, row in enumerate(questions):
            photo_id = int(row.get("Index", q_idx + 1))
            img_path = os.path.join(image_cache_dir, f"{photo_id}.png")
            if not os.path.exists(img_path):
                print(f"[MMVP] Warning: image {photo_id}.png not found, skipping")
                continue

            question_text = row.get("Question", "")
            options_text = row.get("Options", "")
            correct = row.get("Correct Answer", "").strip()

            # Normalize answer: "(a)" -> "A", "(b)" -> "B"
            gold = ""
            if correct.lower().startswith("(a"):
                gold = "A"
            elif correct.lower().startswith("(b"):
                gold = "B"
            elif correct.upper() in ("A", "B", "YES", "NO"):
                gold = correct.upper()

            # Pair ID: questions come in consecutive pairs (1-2, 3-4, ...)
            pair_id = q_idx // 2
            question_idx = q_idx % 2

            full_question = f"{question_text} {options_text}".strip()

            record = {
                "hash_uid": f"mmvp_{q_idx}",
                "image": img_path,
                "question": full_question,
                "gold_answer": gold,
                "pair_id": pair_id,
                "question_idx": question_idx,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[MMVP] Prepared {len(questions)} entries at {output_path}")


def data_to_messages(item):
    prompt = (
        item["question"] + "\n\n"
        "Answer with the option letter (a) or (b). "
        "Provide your final answer inside <answer></answer> tags."
    )
    return [
        dict(type="image", value=item["image"]),
        dict(type="text", value=prompt),
    ]


def bagel_pipeline(vlm, item):
    """Single-sample inference for MMVP: understanding only."""
    messages = data_to_messages(item)
    final_output, _ = vlm.internal_infer(
        messages,
        understanding_output=False,
        think=True,
        max_rounds=3,
    )
    return final_output, [], None
