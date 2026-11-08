"""GenEval scoring module.

Evaluation strategy (priority order):
  1. Official Mask2Former pipeline (evaluate_images.py + summary_scores.py)
     - Requires: mmdet, mmcv, open_clip, clip_benchmark in GENEVAL_PYTHON env
     - Metric: binary correct/incorrect per image via object detection
     - Per-tag accuracy = fraction of prompts with >=1 correct image out of 4
     - Overall = simple mean of 6 tag accuracies
  2. CLIP ViT-B/32 proxy (fallback when official pipeline unavailable)
     - Metric: CLIP cosine similarity, averaged per tag then overall
     - NOTE: not comparable to official numbers, reported as proxy only

Required for official eval:
  - MASK2FORMER_CKPT   : path to .pth checkpoint
  - GENEVAL_REPO       : path to cloned djghosh13/geneval repo
  - MMDETECTION_REPO   : path to cloned open-mmlab/mmdetection (for config .py)
  - GENEVAL_PYTHON     : python executable with mmdet/mmcv/open_clip/clip_benchmark

GenEval image directory layout (produced by inference/geneval.py):
  {task_image_dir}/
    00000/metadata.jsonl + samples/0.png ... 3.png
    00001/...
    ...
"""

import os
import sys
import json
import subprocess
from collections import defaultdict

# ---------------------------------------------------------------------------
# Paths (all overridable via env vars)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(__file__)

GENEVAL_REPO_DIR = os.environ.get(
    "GENEVAL_REPO",
    os.path.normpath(os.path.join(_HERE, "geneval")),
)
MASK2FORMER_CKPT_PATH = os.environ.get(
    "MASK2FORMER_CKPT",
    "/apdcephfs_zwfy/share_303204533/lukening/model/mask2former/"
    "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth",
)
MMDETECTION_REPO_DIR = os.environ.get(
    "MMDETECTION_REPO",
    "/apdcephfs_cq5/share_300600172/lukening/code/mmdetection",
)
MASK2FORMER_CONFIG_PATH = os.environ.get(
    "MASK2FORMER_CONFIG",
    os.path.join(
        MMDETECTION_REPO_DIR,
        "configs", "mask2former",
        "mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.py",
    ),
)
EVALUATE_SCRIPT = os.path.join(GENEVAL_REPO_DIR, "evaluation", "evaluate_images.py")
SUMMARY_SCRIPT  = os.path.join(GENEVAL_REPO_DIR, "evaluation", "summary_scores.py")

GENEVAL_CATEGORIES = [
    "single_object", "two_object", "counting",
    "colors", "position", "color_attribution",
]


def _find_image_dir(json_path):
    """run_bagel.py stores images under dirname(json_path) in 00000/…/samples/."""
    return os.path.dirname(json_path)


# ---------------------------------------------------------------------------
# Official Mask2Former evaluation
# ---------------------------------------------------------------------------

def _check_official_env(python_exe):
    """Return (ok, reason) — whether python_exe has all required packages."""
    if not python_exe or not python_exe.strip():
        return False, "GENEVAL_PYTHON is empty"
    if not os.path.isfile(python_exe):
        return False, f"GENEVAL_PYTHON not found: {python_exe!r}"
    for script in [EVALUATE_SCRIPT, SUMMARY_SCRIPT]:
        if not os.path.exists(script):
            return False, f"GenEval script not found: {script}"
    if not os.path.exists(MASK2FORMER_CKPT_PATH):
        return False, f"Mask2Former checkpoint not found: {MASK2FORMER_CKPT_PATH}"
    if not os.path.exists(MASK2FORMER_CONFIG_PATH):
        return False, f"Mask2Former config not found: {MASK2FORMER_CONFIG_PATH}"
    check = subprocess.run(
        [python_exe, "-c", "import mmdet, mmcv, open_clip, clip_benchmark"],
        capture_output=True, text=True,
    )
    if check.returncode != 0:
        return False, f"Missing packages in {python_exe}: mmdet/mmcv/open_clip/clip_benchmark"
    return True, "ok"


def _run_official(image_dir, python_exe):
    """Run evaluate_images.py + summary_scores.py. Returns summary text or None."""
    mask2former_dir = os.path.dirname(MASK2FORMER_CKPT_PATH)
    results_jsonl = os.path.join(image_dir, "geneval_results.jsonl")

    print("\n[GenEval] Step 1/2 — Running Mask2Former evaluation ...")
    print(f"          python      : {python_exe}")
    print(f"          imagedir    : {image_dir}")
    print(f"          model-path  : {mask2former_dir}")
    print(f"          model-config: {MASK2FORMER_CONFIG_PATH}")

    cmd_eval = [
        python_exe, EVALUATE_SCRIPT,
        image_dir,
        "--outfile", results_jsonl,
        "--model-path", mask2former_dir,
        "--model-config", MASK2FORMER_CONFIG_PATH,
    ]
    try:
        subprocess.run(cmd_eval, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[GenEval] evaluate_images.py failed (exit {e.returncode})")
        return None

    if not os.path.exists(results_jsonl):
        print(f"[GenEval] evaluate_images.py produced no output file")
        return None

    print("\n[GenEval] Step 2/2 — Computing summary scores ...")
    try:
        result = subprocess.run(
            [python_exe, SUMMARY_SCRIPT, results_jsonl],
            check=True, capture_output=True, text=True,
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"[GenEval] summary_scores.py failed: {e}")
        return None


def _print_official_results(summary_text, image_dir):
    print("\n" + "=" * 65)
    print("GenEval Official Results  (Mask2Former + CLIP-ViT-L/14)")
    print("=" * 65)
    print(summary_text)
    print("  Metric: per-tag accuracy (prompt correct if ≥1/4 images pass)")
    print("  Overall = simple mean of 6 tag accuracies")
    print("=" * 65)

    summary_path = os.path.join(image_dir, "geneval_summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text)
    print(f"[GenEval] Summary saved to: {summary_path}")

    # Parse into structured JSON
    import re
    tag_scores, overall = {}, None
    for line in summary_text.splitlines():
        m = re.match(r"(\w+)\s*=\s*([\d.]+)%", line.strip())
        if m:
            tag_scores[m.group(1)] = round(float(m.group(2)) / 100.0, 4)
        m2 = re.match(r"Overall score.*?:\s*([\d.]+)", line.strip())
        if m2:
            overall = float(m2.group(1))
    if overall is None and tag_scores:
        overall = round(sum(tag_scores.values()) / len(tag_scores), 4)

    structured = {
        "per_tag": tag_scores,
        "overall": overall,
        "metric": "official_mask2former",
    }
    json_path = os.path.join(image_dir, "geneval_summary.json")
    with open(json_path, "w") as f:
        json.dump(structured, f, indent=2)
    print(f"[GenEval] Structured summary saved to: {json_path}")


# ---------------------------------------------------------------------------
# CLIP-based fallback
# Loads ViT-B/32 directly via torch.jit.load to avoid torchvision dependency
# that causes hangs in some environments.
# ---------------------------------------------------------------------------

try:
    import torch
    from PIL import Image
    import numpy as np
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

# Local path to ViT-B/32 JIT weights
CLIP_LOCAL_PATH = os.environ.get(
    "CLIP_VIT_B32_PATH",
    "/apdcephfs_zwfy/share_303204533/lukening/model/ViT-B32/ViT-B-32.pt",
)

# ViT-B/32 image preprocessing constants (from CLIP paper)
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)
_CLIP_SIZE = 224


def _preprocess_image(pil_img):
    """Replicate CLIP's image preprocessing without torchvision."""
    # Resize shortest side to 224, center crop to 224×224
    w, h = pil_img.size
    scale = _CLIP_SIZE / min(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    pil_img = pil_img.resize((new_w, new_h), Image.BICUBIC)
    left = (new_w - _CLIP_SIZE) // 2
    top  = (new_h - _CLIP_SIZE) // 2
    pil_img = pil_img.crop((left, top, left + _CLIP_SIZE, top + _CLIP_SIZE))
    # To float tensor [3, H, W], normalize
    arr = np.array(pil_img, dtype=np.float32) / 255.0
    arr = (arr - np.array(_CLIP_MEAN)) / np.array(_CLIP_STD)
    tensor = torch.from_numpy(arr.transpose(2, 0, 1)).float()  # [3, 224, 224]
    return tensor.unsqueeze(0)  # [1, 3, 224, 224]


def _tokenize(texts, context_length=77):
    """Simple BPE tokenizer using CLIP's vocabulary (via the clip source tree)."""
    try:
        # Try using the CLIP simple_tokenizer if available
        _CLIP_SRC = os.path.join(
            os.path.dirname(__file__),
            "CLIP", "clip",
        )
        import sys as _sys
        if _CLIP_SRC not in _sys.path:
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
        # Fallback: character-level ASCII tokens (rough proxy, less accurate)
        result = torch.zeros(len(texts), context_length, dtype=torch.long)
        for i, text in enumerate(texts):
            tokens = [ord(c) % 49408 for c in text[:context_length - 2]]
            result[i, 1:1 + len(tokens)] = torch.tensor(tokens)
        return result


def _load_clip_model(device):
    """Load CLIP ViT-B/32 from local .pt file via torch.jit.load."""
    if not os.path.isfile(CLIP_LOCAL_PATH):
        raise FileNotFoundError(
            f"CLIP ViT-B/32 weights not found at {CLIP_LOCAL_PATH}. "
            f"Set env var CLIP_VIT_B32_PATH to the correct path."
        )
    print(f"[GenEval] Loading CLIP ViT-B/32 from local: {CLIP_LOCAL_PATH}")
    model = torch.jit.load(CLIP_LOCAL_PATH, map_location=device).eval()
    return model


def _clip_score(model, image_path, text, device):
    """Compute CLIP cosine similarity between image and text."""
    try:
        img_tensor = _preprocess_image(
            Image.open(image_path).convert("RGB")
        ).to(device)
        text_tokens = _tokenize([text]).to(device)
        with torch.no_grad():
            img_feat  = model.encode_image(img_tensor).float()
            txt_feat  = model.encode_text(text_tokens).float()
            img_feat  = img_feat / img_feat.norm(dim=-1, keepdim=True)
            txt_feat  = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
            return (img_feat @ txt_feat.T).item()
    except Exception:
        return 0.0


def _run_clip_fallback(json_path, image_dir):
    """CLIP ViT-B/32 proxy: load results JSON, score each image, report per-tag."""
    if not _TORCH_AVAILABLE:
        print("[GenEval] CLIP not available. Install: pip install git+https://github.com/openai/CLIP.git")
        return

    if not _TORCH_AVAILABLE:
        print("[GenEval] torch/PIL not available, cannot run CLIP fallback.")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = _load_clip_model(device)
    except Exception as e:
        print(f"[GenEval] Failed to load CLIP model: {e}")
        return

    # Group by (prompt_idx, tag) — take max score over all images per prompt
    # to approximate the "any image correct" logic of the official eval
    prompt_best = defaultdict(lambda: defaultdict(float))  # {tag: {prompt_idx: best_score}}

    for rec in records:
        prompt = rec.get("prompt", "")
        tag = rec.get("category", "unknown")
        prompt_idx = rec.get("prompt_idx", rec.get("hash_uid", ""))
        images = rec.get("generated_image", [])

        img_path = next(
            (p for p in images if p and isinstance(p, str) and os.path.exists(p)),
            None,
        )
        # Also look in the official directory structure
        if img_path is None:
            img_idx = rec.get("img_idx", 0)
            candidate = os.path.join(
                image_dir, f"{int(str(prompt_idx).split('_')[-1]) if '_' in str(prompt_idx) else prompt_idx:05d}",
                "samples", f"{img_idx}.png",
            )
            if os.path.exists(candidate):
                img_path = candidate

        score = _clip_score(model, img_path, prompt, device) if img_path else 0.0
        print(f'Current prompt {prompt} tag {tag} score is :{score}')
        # Keep best score per prompt (analogous to "any image correct")
        if score > prompt_best[tag][prompt_idx]:
            prompt_best[tag][prompt_idx] = score

    # Per-tag: average of per-prompt best scores
    tag_scores = {}
    for tag in GENEVAL_CATEGORIES:
        if tag in prompt_best:
            scores = list(prompt_best[tag].values())
            tag_scores[tag] = round(sum(scores) / len(scores), 4)

    valid = [v for v in tag_scores.values() if v > 0]
    overall = round(sum(valid) / len(valid), 4) if valid else 0.0

    print("\n" + "=" * 65)
    print("GenEval Results  [CLIP ViT-B/32 proxy — NOT official metric]")
    print("=" * 65)
    print(f"  {'Category':<25} {'CLIP Score':>10}  {'N Prompts':>10}")
    print("  " + "-" * 50)
    for tag in GENEVAL_CATEGORIES:
        if tag in tag_scores:
            n = len(prompt_best[tag])
            print(f"  {tag:<25} {tag_scores[tag]:>10.4f}  {n:>10}")
    print("  " + "-" * 50)
    print(f"  {'Overall (mean of tags)':<25} {overall:>10.4f}")
    print("=" * 65)
    print("WARNING: CLIP scores are a proxy only. Official GenEval uses")
    print("         Mask2Former detection. Install mmdet/mmcv to get")
    print("         comparable numbers.")
    print("=" * 65)

    summary = {
        "per_tag": tag_scores,
        "overall": overall,
        "metric": "clip_proxy_vit_b32",
    }
    summary_path = os.path.join(image_dir, "geneval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[GenEval] Summary saved to: {summary_path}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def calculate_score(json_path):
    """Score GenEval results: official Mask2Former first, CLIP fallback on failure."""
    if not os.path.exists(json_path):
        print(f"[GenEval] Input file not found: {json_path}")
        return

    image_dir = _find_image_dir(json_path)

    # Verify generated images exist
    subdirs = [
        d for d in os.listdir(image_dir)
        if os.path.isdir(os.path.join(image_dir, d)) and d.isdigit()
    ]
    if not subdirs:
        print(f"[GenEval] No numbered subdirectories in {image_dir}. Images may not have been generated.")
        return
    print(f"[GenEval] Found {len(subdirs)} prompt directories in {image_dir}")

    # --- Attempt official Mask2Former evaluation ---
    python_exe = os.environ.get("GENEVAL_PYTHON", sys.executable)
    if not python_exe or not python_exe.strip():
        python_exe = sys.executable

    ok, reason = _check_official_env(python_exe)
    if ok:
        summary_text = _run_official(image_dir, python_exe)
        if summary_text is not None:
            _print_official_results(summary_text, image_dir)
            return
        print("[GenEval] Official evaluation failed at runtime. Falling back to CLIP proxy ...")
    else:
        print(f"[GenEval] Official evaluation unavailable: {reason}")
        print(f"[GenEval] Falling back to CLIP ViT-B/32 proxy scoring ...")

    # --- Fallback: CLIP proxy ---
    _run_clip_fallback(json_path, image_dir)
