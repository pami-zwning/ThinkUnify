import sys
sys.path.append("/mnt/afs/kongxiangli/workspace/ThinkMorph/")

import os
import json
import argparse
import random
import numpy as np
from pathlib import Path
from io import BytesIO
from copy import deepcopy
from typing import Dict, List, Optional

import requests
from PIL import Image

import torch
from accelerate import (
    infer_auto_device_map,
    load_checkpoint_and_dispatch,
    init_empty_weights,
)

from data.transforms import ImageTransform
from data.data_utils import add_special_tokens
from modeling.bagel import (
    BagelConfig,
    Bagel,
    Qwen2Config,
    Qwen2ForCausalLM,
    SiglipVisionConfig,
    SiglipVisionModel,
)
from modeling.qwen2 import Qwen2Tokenizer
from modeling.autoencoder import load_ae
from inferencer import InterleaveInferencer

# ======================================================
# Argument parsing
# ======================================================

def parse_args():
    parser = argparse.ArgumentParser("TIR-Bench Inference")

    parser.add_argument("--dataset", type=str, required=True,
                        help="Benchmark name (e.g. TIR-Bench)")
    parser.add_argument("--input_json", type=str, required=True,
                        help="Input benchmark json")
    parser.add_argument("--output_json", type=str, required=True,
                        help="Output result json")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save generated images")
    parser.add_argument("--gpus", type=int, default=1,
                        help="Number of GPUs (controlled by CUDA_VISIBLE_DEVICES)")

    return parser.parse_args()

# ======================================================
# Model loading
# ======================================================

MODEL_PATH = "/mnt/afs/omni_ckpt/BAGEL-7B-MoT"
TRAINING_CKPT = "/mnt/afs/kongxiangli/workspace/ThinkMorph/RUN/bagel_tir_internal_math_insection_got_260119/ckpt/0005000"

print("[INFO] Loading configs...")

llm_config = Qwen2Config.from_json_file(os.path.join(MODEL_PATH, "llm_config.json"))
llm_config.qk_norm = True
llm_config.tie_word_embeddings = False
llm_config.layer_module = "Qwen2MoTDecoderLayer"

vit_config = SiglipVisionConfig.from_json_file(os.path.join(MODEL_PATH, "vit_config.json"))
vit_config.rope = False
vit_config.num_hidden_layers -= 1

vae_model, vae_config = load_ae(
    local_path=os.path.join(MODEL_PATH, "ae.safetensors")
)

bagel_config = BagelConfig(
    visual_gen=True,
    visual_und=True,
    llm_config=llm_config,
    vit_config=vit_config,
    vae_config=vae_config,
    vit_max_num_patch_per_side=70,
    connector_act="gelu_pytorch_tanh",
    latent_patch_size=2,
    max_latent_size=64,
)

with init_empty_weights():
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model = SiglipVisionModel(vit_config)
    model = Bagel(language_model, vit_model, bagel_config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(
        vit_config, meta=True
    )

tokenizer = Qwen2Tokenizer.from_pretrained(MODEL_PATH)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

vae_transform = ImageTransform(1024, 512, 16)
vit_transform = ImageTransform(980, 224, 14)

max_mem = "40GiB"

device_map = infer_auto_device_map(
    model,
    max_memory={i: max_mem for i in range(torch.cuda.device_count())},
    no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
)

same_device_modules = [
    "language_model.model.embed_tokens",
    "time_embedder",
    "latent_pos_embed",
    "vae2llm",
    "llm2vae",
    "connector",
    "vit_pos_embed",
]

first_device = device_map.get(same_device_modules[0], "cuda:0")
for k in same_device_modules:
    device_map[k] = first_device

print("[INFO] Loading checkpoint...")
model = load_checkpoint_and_dispatch(
    model,
    checkpoint=os.path.join(TRAINING_CKPT, "model.safetensors"),
    device_map=device_map,
    dtype=torch.bfloat16,
    offload_buffers=True,
    force_hooks=True,
    offload_folder="/tmp/offload",
)

model.eval()
print("[INFO] Model loaded")

inferencer = InterleaveInferencer(
    model=model,
    vae_model=vae_model,
    tokenizer=tokenizer,
    vae_transform=vae_transform,
    vit_transform=vit_transform,
    new_token_ids=new_token_ids,
)

# ======================================================
# Reproducibility
# ======================================================

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ======================================================
# Inference hyperparameters
# ======================================================

INFERENCE_HYPER = dict(
    max_think_token_n=4096,
    do_sample=True,
    text_temperature=0.3,
    cfg_text_scale=4.0,
    cfg_img_scale=2.0,
    cfg_interval=[0.0, 1.0],
    timestep_shift=3.0,
    num_timesteps=50,
    cfg_renorm_min=0.0,
    cfg_renorm_type="text_channel",
)

# ======================================================
# Utilities
# ======================================================

def load_image(path: str) -> Optional[Image.Image]:
    try:
        if path.startswith("http"):
            r = requests.get(path, timeout=10)
            return Image.open(BytesIO(r.content)).convert("RGB")
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"[WARN] Failed to load image {path}: {e}")
        return None


def combine_images_for_comparison(img1, img2, sep=5):
    w1, h1 = img1.size
    w2, h2 = img2.size
    h = max(h1, h2)

    canvas = Image.new("RGB", (w1 + sep + w2, h), (240, 240, 240))
    canvas.paste(img1, (0, 0))
    canvas.paste(Image.new("RGB", (sep, h), (0, 0, 0)), (w1, 0))
    canvas.paste(img2, (w1 + sep, 0))
    return canvas


# ======================================================
# Core processing
# ======================================================

def process_single_item(item: Dict, output_dir: str):
    task_id = str(item.get("id"))
    prompt = item.get("prompt", "")
    task = item.get("task", "unknown")

    images = []
    image_paths = []

    for i in (1, 2):
        k = f"image_{i}"
        if k in item and item[k]:
            p = os.path.join(
                os.path.dirname(args.input_json),
                item[k],
            )
            img = load_image(p)
            if img:
                images.append(img)
                image_paths.append(p)

    if not images:
        return None

    if len(images) == 2:
        main_image = (
            combine_images_for_comparison(images[0], images[1])
            if task == "spot_difference"
            else images[0]
        )
        main_image.save(os.path.join(output_dir, f"{task_id}_combined.png"))
    else:
        main_image = images[0]

    with torch.no_grad():
        outputs = inferencer(
            image=main_image,
            text=prompt,
            understanding_output=False,
            think=True,
            **INFERENCE_HYPER,
        )

    cot = []
    final_answer = None
    gen_imgs = []

    for o in outputs:
        if isinstance(o, str):
            cot.append(o.strip())
            final_answer = o
        elif isinstance(o, Image.Image):
            p = os.path.join(output_dir, f"{task_id}_gen_{len(gen_imgs)}.png")
            o.save(p)
            gen_imgs.append(p)

    return {
        "id": task_id,
        "task": task,
        "prompt": prompt,
        "cot_process": cot,
        "final_answer": final_answer,
        "generated_images": gen_imgs,
        "input_images": image_paths,
        "ground_truth": item.get("answer"),
    }


def process_json_file(input_json, output_json, output_dir):
    processed = set()
    results = []

    if os.path.exists(output_json):
        with open(output_json, "r") as f:
            results = json.load(f)
        processed = {str(r["id"]) for r in results}

    with open(input_json, "r") as f:
        data = json.load(f)
    data = [item for item in data if item['task'] == 'math']
    print(f"[INFO] Loaded {len(data)} items")

    for idx, item in enumerate(data):
        tid = str(item.get("id"))
        if tid in processed:
            continue

        print(f"[{idx+1}/{len(data)}] Processing {tid}")
        res = process_single_item(item, output_dir)
        if res:
            results.append(res)
            with open(output_json, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)

    print("[DONE] All items processed")

# ======================================================
# Main
# ======================================================

def main():
    global args
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    print("=" * 60)
    print(f"Dataset     : {args.dataset}")
    print(f"Input JSON  : {args.input_json}")
    print(f"Output JSON : {args.output_json}")
    print(f"Image Dir   : {args.output_dir}")
    print(f"GPUs        : {args.gpus}")
    print("=" * 60)

    process_json_file(
        args.input_json,
        args.output_json,
        args.output_dir,
    )

if __name__ == "__main__":
    main()
