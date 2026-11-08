"""T2I-CompBench inference module for ThinkMorph.

Benchmark: T2I-CompBench (NeurIPS 2023) — compositional T2I prompts
           across multiple sub-categories.

Data format: plain-text files, one prompt per line, named by category:
  color_val.txt, shape_val.txt, texture_val.txt, spatial_val.txt,
  non-spatial_val.txt, complex_val.txt

Data source priority:
  1. Local JSONL (if already exists, skip)
  2. Local dataset directory (T2I-CompBench_dataset/ with *_val.txt files)
  3. HuggingFace download (needs internet, fallback)

Provides:
  - prepare_t2icompbench_jsonl(): build JSONL from local txt files or HF
  - bagel_pipeline(): per-sample T2I generation via internal_infer
"""

import os
import json


T2ICOMP_CATEGORIES = [
    "color_binding",
    "shape_binding",
    "texture_binding",
    "spatial_relationships",
    "non_spatial_relationships",
    "complex_compositions",
]

# Map txt filename prefix → normalized category
_FILE_TO_CATEGORY = {
    "color": "color_binding",
    "shape": "shape_binding",
    "texture": "texture_binding",
    "spatial": "spatial_relationships",
    "non-spatial": "non_spatial_relationships",
    "complex": "complex_compositions",
}


def _find_local_dataset_dir():
    """Search for the T2I-CompBench dataset directory with *_val.txt files."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data", "T2I-CompBench_dataset"),
        os.path.join(os.path.dirname(__file__), "..", "data", "T2I-CompBench_dataset"),
        "/mnt/afs/share_data/T2I-CompBench_dataset",
        "/mnt/afs/interleaved_data/ningzhiwei/T2I-CompBench_dataset",
    ]
    for c in candidates:
        p = os.path.normpath(c)
        if os.path.isdir(p) and any(f.endswith("_val.txt") for f in os.listdir(p)):
            return p
    return None


def _find_local_jsonl():
    """Search for a pre-built JSONL prompts file."""
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data", "t2icompbench_prompts.jsonl"),
        os.path.join(os.path.dirname(__file__), "..", "data", "t2icompbench_prompts.jsonl"),
        "/mnt/afs/share_data/t2icompbench/prompts.jsonl",
    ]
    for c in candidates:
        p = os.path.normpath(c)
        if os.path.isfile(p):
            return p
    return None


def _load_from_txt_dir(dataset_dir):
    """Load prompts from *_val.txt files in T2I-CompBench_dataset/ directory."""
    prompts = []
    for prefix, category in _FILE_TO_CATEGORY.items():
        txt_path = os.path.join(dataset_dir, f"{prefix}_val.txt")
        if not os.path.isfile(txt_path):
            print(f"[T2I-CompBench] Warning: {txt_path} not found, skipping category {category}")
            continue
        with open(txt_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append({"prompt": line, "category": category})
    return prompts


def prepare_t2icompbench_jsonl(output_path):
    """Build T2I-CompBench JSONL from local txt files or HuggingFace."""
    if os.path.exists(output_path):
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    prompts = None

    # 1. Try local dataset directory (T2I-CompBench_dataset/ with *_val.txt)
    dataset_dir = os.path.join(os.path.dirname(output_path),'T2I-CompBench_dataset')
    if dataset_dir:
        print(f"[T2I-CompBench] Loading from local directory: {dataset_dir}")
        prompts = _load_from_txt_dir(dataset_dir)
        if prompts:
            print(f"[T2I-CompBench] Loaded {len(prompts)} prompts from {len(_FILE_TO_CATEGORY)} category files.")


    # Write JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, sample in enumerate(prompts):
            prompt = sample.get("prompt", sample.get("caption", sample.get("text", "")))
            category = sample.get("category", "unknown")

            record = {
                "hash_uid": f"t2icomp_{idx}",
                "prompt": prompt,
                "category": category,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Print per-category stats
    from collections import Counter
    cat_counts = Counter(json.loads(line)["category"] for line in open(output_path) if line.strip())
    print(f"[T2I-CompBench] Prepared {len(prompts)} entries at {output_path}")
    for cat, cnt in sorted(cat_counts.items()):
        print(f"  {cat}: {cnt}")


def bagel_pipeline(vlm, item):
    """Generate one image from a text prompt (no input image)."""
    prompt = item["prompt"]
    messages = [dict(type="text", value=prompt)]

    # Set a unique reproducible seed per sample
    import torch, random, numpy as np, hashlib
    seed = int(hashlib.md5(item.get("hash_uid", prompt).encode()).hexdigest(), 16) % (2**31)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Pure T2I: gen_only=True forces direct image gen without text generation
    text_output, image_list = vlm.internal_infer(
        messages,
        understanding_output=False,
        think=False,
        max_rounds=1,
        gen_only=True,
    )
    return text_output, image_list, None
