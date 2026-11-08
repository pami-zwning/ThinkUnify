"""V*Bench inference module for ThinkMorph.

Dataset: craigwu/vstar_bench (test split, 191 samples)
Schema:  image (str, relative path), text (str, full question+options),
         label (str, A/B/C/D), category (str), question_id (str)

The ``text`` field already contains both the question and the formatted
options, so we pass it directly as the prompt.

Provides:
  - prepare_vstar_jsonl(): generate JSONL from HF dataset (one-time)
  - bagel_pipeline(): per-sample inference for run_bagel.py
"""

import os
import io
import json
from PIL import Image
from huggingface_hub import hf_hub_download


def prepare_vstar_jsonl(output_path, image_cache_dir):
    """Load V*Bench from HuggingFace and write JSONL for run_bagel.py."""
    if os.path.exists(output_path):
        return

    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "V*Bench requires the `datasets` package. "
            "Install with: pip install datasets"
        ) from e

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(image_cache_dir, exist_ok=True)

    print("[V*Bench] Loading craigwu/vstar_bench (test split)...")
    dataset = load_dataset("craigwu/vstar_bench", split="test")
    print(f"[V*Bench] Loaded {len(dataset)} samples. Writing to {output_path}...")

    # Download the full image archive from the HF repo so we can resolve
    # relative paths like "direct_attributes/sa_4690.jpg".
    # The images live under the dataset repo as loose files.
    repo_id = "craigwu/vstar_bench"

    with open(output_path, "w", encoding="utf-8") as f:
        for idx, sample in enumerate(dataset):
            img_path = os.path.join(image_cache_dir, f"vstar_{idx}.png")
            if not os.path.exists(img_path):
                img_field = sample["image"]
                pil_image = None

                # 1. Already a PIL Image (decoded by HF)
                if isinstance(img_field, Image.Image):
                    pil_image = img_field
                # 2. Dict with bytes/path
                elif isinstance(img_field, dict):
                    if img_field.get("bytes"):
                        pil_image = Image.open(io.BytesIO(img_field["bytes"]))
                    elif img_field.get("path") and os.path.exists(img_field["path"]):
                        pil_image = Image.open(img_field["path"])
                # 3. String path — download from HF repo
                elif isinstance(img_field, str):
                    try:
                        local = hf_hub_download(
                            repo_id=repo_id,
                            filename=img_field,
                            repo_type="dataset",
                        )
                        pil_image = Image.open(local)
                    except Exception as e:
                        print(f"[V*Bench] Warning: could not download {img_field}: {e}")

                if pil_image is not None:
                    if pil_image.mode != "RGB":
                        pil_image = pil_image.convert("RGB")
                    pil_image.save(img_path)
                else:
                    print(f"[V*Bench] Warning: skipping sample {idx}, image not resolved")
                    continue

            # VStarBench fields:
            #   text  = full question + options (e.g. "What is...? (A) x (B) y ...")
            #   label = correct answer letter (A/B/C/D)
            record = {
                "hash_uid": f"vstar_{idx}",
                "image": img_path,
                "question": sample.get("text", ""),
                "options": [],  # options are embedded in the text field
                "gold_answer": sample.get("label", "").strip().upper(),
                "category": sample.get("category", ""),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[V*Bench] Prepared JSONL at {output_path}")


def data_to_messages(item):
    """The question field already contains the full prompt with options."""
    prompt = (
        item["question"] + "\n\n"
        "Look carefully at the image details. "
        "Provide your answer as a single letter "
        "inside <answer></answer> tags."
    )
    return [
        dict(type="image", value=item["image"]),
        dict(type="text", value=prompt),
    ]


def bagel_pipeline(vlm, item):
    """Single-sample inference for V*Bench: understanding only."""
    messages = data_to_messages(item)
    final_output, _ = vlm.internal_infer(
        messages,
        understanding_output=False,
        think=True,
        max_rounds=3,
    )
    return final_output, [], None
