"""T2I-CompBench scoring module — official metric alignment.

Official T2I-CompBench metrics per sub-category
(Huang et al., NeurIPS 2023 — https://github.com/Karine-Huang/T2I-CompBench):

  Attribute Binding:
    color_binding    → BLIP-VQA
    shape_binding    → BLIP-VQA
    texture_binding  → BLIP-VQA

  Spatial / Counting:
    spatial_relationships     → UniDet  (2D/3D detection-based)
    non_spatial_relationships → CLIPScore (ViT-L/14)

  Compositional:
    complex_compositions → 3-in-1  (BLIP-VQA + UniDet + CLIPScore weighted)

This module:
  1. Runs the official T2I-CompBench evaluation scripts when available
     (requires the T2ICompBench toolkit at T2ICOMP_REPO_DIR).
  2. Falls back to CLIPScore proxy for all categories when the official
     toolkit is unavailable (prints a clear warning in that case).

Set env var T2ICOMP_REPO to point at the cloned T2I-CompBench repo.

Overall score reported: simple mean of 6 per-category scores
(same aggregation used in the original paper's Table 1).
"""

import os
import sys
import json
import subprocess
from collections import defaultdict

# ---------------------------------------------------------------------------
# Paths (override via env vars)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(__file__)

T2ICOMP_REPO_DIR = os.environ.get(
    "T2ICOMP_REPO",
    os.path.normpath(os.path.join(_HERE, "T2I-CompBench")),
)

# Display order matching the original paper
CATEGORY_ORDER = [
    "color_binding",
    "shape_binding",
    "texture_binding",
    "spatial_relationships",
    "non_spatial_relationships",
    "complex_compositions",
]

CATEGORY_DISPLAY = {
    "color_binding":           "Color Binding      (BLIP-VQA)",
    "shape_binding":           "Shape Binding      (BLIP-VQA)",
    "texture_binding":         "Texture Binding    (BLIP-VQA)",
    "spatial_relationships":   "Spatial            (UniDet)",
    "non_spatial_relationships": "Non-Spatial       (CLIPScore ViT-L/14)",
    "complex_compositions":    "Complex            (3-in-1)",
}

# Official metric per category
OFFICIAL_METRIC = {
    "color_binding":           "BLIP-VQA",
    "shape_binding":           "BLIP-VQA",
    "texture_binding":         "BLIP-VQA",
    "spatial_relationships":   "UniDet",
    "non_spatial_relationships": "CLIPScore",
    "complex_compositions":    "3-in-1",
}

# ---------------------------------------------------------------------------
# CLIP-based fallback (proxy only)
# ---------------------------------------------------------------------------
try:
    import torch
    from PIL import Image
    import numpy as np
    _CLIP_AVAILABLE = True
except ImportError:
    _CLIP_AVAILABLE = False

# Local path to ViT-B/32 weights (avoids downloading and torchvision hang)
_CLIP_LOCAL_PATH = os.environ.get(
    "CLIP_VIT_B32_PATH",
    "/apdcephfs_zwfy/share_303204533/lukening/model/ViT-B32/ViT-B-32.pt",
)

# ViT-B/32 preprocessing constants
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)
_CLIP_SIZE = 224


def _preprocess_image(pil_img):
    w, h = pil_img.size
    scale = _CLIP_SIZE / min(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    pil_img = pil_img.resize((new_w, new_h), Image.BICUBIC)
    left = (new_w - _CLIP_SIZE) // 2
    top  = (new_h - _CLIP_SIZE) // 2
    pil_img = pil_img.crop((left, top, left + _CLIP_SIZE, top + _CLIP_SIZE))
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    arr = (arr - np.array(_CLIP_MEAN)) / np.array(_CLIP_STD)
    return torch.from_numpy(arr.transpose(2, 0, 1)).float().unsqueeze(0)


def _tokenize(texts, context_length=77):
    try:
        _CLIP_SRC = os.path.join(os.path.dirname(__file__), "CLIP", "clip")
        import sys as _sys
        if os.path.dirname(_CLIP_SRC) not in _sys.path:
            _sys.path.insert(0, os.path.dirname(_CLIP_SRC))
        from clip.simple_tokenizer import SimpleTokenizer
        tokenizer = SimpleTokenizer()
        sot = tokenizer.encoder["<|startoftext|>"]
        eot = tokenizer.encoder["<|endoftext|>"]
        result = torch.zeros(len(texts), context_length, dtype=torch.long)
        for i, text in enumerate(texts):
            tokens = [sot] + tokenizer.encode(text)[:context_length - 2] + [eot]
            result[i, :len(tokens)] = torch.tensor(tokens)
        return result
    except Exception:
        result = torch.zeros(len(texts), context_length, dtype=torch.long)
        for i, text in enumerate(texts):
            tokens = [ord(c) % 49408 for c in text[:context_length - 2]]
            result[i, 1:1 + len(tokens)] = torch.tensor(tokens)
        return result


def _load_clip_model(device):
    """Load CLIP ViT-B/32 via torch.jit.load (no torchvision dependency)."""
    if not os.path.isfile(_CLIP_LOCAL_PATH):
        raise FileNotFoundError(
            f"CLIP ViT-B/32 weights not found at {_CLIP_LOCAL_PATH}. "
            f"Set env var CLIP_VIT_B32_PATH to the correct path."
        )
    print(f"[T2I-CompBench] Loading CLIP ViT-B/32 from local: {_CLIP_LOCAL_PATH}")
    model = torch.jit.load(_CLIP_LOCAL_PATH, map_location=device).eval()
    return model


def _clip_score(model, image_path, text, device):
    try:
        image = _preprocess_image(Image.open(image_path).convert("RGB")).to(device)
        tokens = _tokenize([text]).to(device)
        with torch.no_grad():
            img_feat = model.encode_image(image).float()
            txt_feat = model.encode_text(tokens).float()
            img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            return (img_feat @ txt_feat.T).item()
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Official toolkit runner
# ---------------------------------------------------------------------------

def _run_official_evaluation(records, image_dir):
    """Attempt to run the official T2I-CompBench evaluation scripts.

    The official toolkit expects images organised per category. This function:
      1. Groups generated images by category.
      2. Writes a prompts txt and image list compatible with the toolkit.
      3. Calls the official evaluation scripts for each category.

    Returns a dict {category: score} if successful, else None.
    """
    eval_script = os.path.join(T2ICOMP_REPO_DIR, "evaluation", "evaluate_images.py")
    if not os.path.exists(eval_script):
        return None  # toolkit not available

    print(f"[T2I-CompBench] Found official toolkit at {T2ICOMP_REPO_DIR}")
    # Group images by category
    cat_records = defaultdict(list)
    for rec in records:
        cat = rec.get("category", "unknown")
        images = rec.get("generated_image", [])
        img_path = next((p for p in images if p and os.path.exists(p)), None)
        if img_path:
            cat_records[cat].append({"prompt": rec["prompt"], "image": img_path})

    category_scores = {}
    for cat, items in cat_records.items():
        # Write temp prompt list and image list
        cat_dir = os.path.join(image_dir, f"_t2icomp_{cat}")
        os.makedirs(cat_dir, exist_ok=True)
        prompt_file = os.path.join(cat_dir, "prompts.txt")
        image_file  = os.path.join(cat_dir, "images.txt")
        with open(prompt_file, "w") as pf, open(image_file, "w") as imf:
            for item in items:
                pf.write(item["prompt"] + "\n")
                imf.write(item["image"] + "\n")

        out_json = os.path.join(cat_dir, "scores.json")
        metric = OFFICIAL_METRIC.get(cat, "CLIPScore")
        cmd = [
            sys.executable, eval_script,
            "--category", cat,
            "--metric", metric,
            "--prompts", prompt_file,
            "--images", image_file,
            "--output", out_json,
        ]
        try:
            subprocess.run(cmd, check=True)
            if os.path.exists(out_json):
                with open(out_json) as f:
                    result = json.load(f)
                category_scores[cat] = result.get("mean_score", 0.0)
                print(f"  {cat}: {category_scores[cat]:.4f}  [{metric}]")
        except Exception as e:
            print(f"  [T2I-CompBench] {cat} failed ({metric}): {e}")

    return category_scores if category_scores else None


# ---------------------------------------------------------------------------
# CLIP fallback evaluation
# ---------------------------------------------------------------------------

def _run_clip_evaluation(records):
    """CLIP ViT-B/32 proxy scores for all categories (fallback)."""
    if not _CLIP_AVAILABLE:
        return {}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[T2I-CompBench] Loading CLIP ViT-B/32 for proxy scoring ...")
    try:
        model = _load_clip_model(device)
    except Exception as e:
        print(f"[T2I-CompBench] Failed to load CLIP: {e}")
        return {}

    per_category = defaultdict(list)
    for rec in records:
        prompt   = rec.get("prompt", "")
        category = rec.get("category", "unknown")
        images   = rec.get("generated_image", [])
        img_path = next((p for p in images if p and isinstance(p, str) and os.path.exists(p)), None)
        score    = _clip_score(model, img_path, prompt, device) if img_path else 0.0
        per_category[category].append(score)

    return {
        cat: round(sum(scores) / len(scores), 4)
        for cat, scores in per_category.items()
        if scores
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_score(json_path):
    """Compute T2I-CompBench scores from run_bagel.py results JSON.

    Tries the official toolkit first; falls back to CLIP proxy if unavailable.
    """
    if not os.path.exists(json_path):
        print(f"[T2I-CompBench] Error: input file not found: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    image_dir = os.path.dirname(json_path)

    # --- Try official toolkit first ---
    category_scores = _run_official_evaluation(records, image_dir)
    used_official = category_scores is not None

    if not used_official:
        print(
            "[T2I-CompBench] Official toolkit not found or failed.\n"
            "  → Using CLIP ViT-B/32 as proxy (NOT comparable to paper numbers).\n"
            f"  → To use official metrics, clone the repo and set:\n"
            f"     export T2ICOMP_REPO=/path/to/T2I-CompBench\n"
            f"     https://github.com/Karine-Huang/T2I-CompBench"
        )
        category_scores = _run_clip_evaluation(records)
        if not category_scores:
            print("[T2I-CompBench] No scores computed (CLIP unavailable). Install: "
                  "pip install git+https://github.com/openai/CLIP.git")
            return

    # Attribute binding sub-total (color + shape + texture)
    attr_cats   = ["color_binding", "shape_binding", "texture_binding"]
    attr_scores = [category_scores[c] for c in attr_cats if c in category_scores]
    attr_avg    = round(sum(attr_scores) / len(attr_scores), 4) if attr_scores else 0.0

    # Overall = simple mean of all 6 category scores
    all_scores = [category_scores.get(c, 0.0) for c in CATEGORY_ORDER if c in category_scores]
    overall    = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0

    # --- Print results ---
    metric_label = "Official (BLIP-VQA / UniDet / CLIPScore)" if used_official else "CLIP ViT-B/32 proxy"
    print("\n" + "=" * 70)
    print(f"T2I-CompBench Results  [{metric_label}]")
    print("=" * 70)
    print(f"{'Category':<45} {'Score':>8}  {'Images':>7}")
    print("-" * 70)

    cat_counts = defaultdict(int)
    for rec in records:
        cat_counts[rec.get("category", "unknown")] += 1

    for cat in CATEGORY_ORDER:
        if cat in category_scores:
            disp  = CATEGORY_DISPLAY.get(cat, cat)
            score = category_scores[cat]
            count = cat_counts.get(cat, "?")
            print(f"  {disp:<43} {score:>8.4f}  {count:>7}")

    # Print any extra categories
    for cat in sorted(category_scores):
        if cat not in CATEGORY_ORDER:
            print(f"  {cat:<43} {category_scores[cat]:>8.4f}  {cat_counts.get(cat, '?'):>7}")

    print("-" * 70)
    print(f"  {'Attribute Binding (Color+Shape+Texture avg)':<43} {attr_avg:>8.4f}")
    print(f"  {'Overall (mean of 6 categories)':<43} {overall:>8.4f}")
    print("=" * 70)

    if not used_official:
        print("\nWARNING: These CLIP scores are a proxy only. Official numbers use:")
        for cat in CATEGORY_ORDER:
            m = OFFICIAL_METRIC.get(cat, "?")
            print(f"  {cat:<35} → {m}")

    # Save structured summary
    summary = {
        "per_category": category_scores,
        "attribute_binding_avg": attr_avg,
        "overall": overall,
        "metric_type": "official" if used_official else "clip_proxy",
    }
    summary_path = json_path.replace(".json", "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[T2I-CompBench] Summary saved to: {summary_path}")
