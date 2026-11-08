#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uni-MMMU benchmark evaluation for close-source models (Gemini, etc.)

Direct inference only (no tool calls).
Results are saved in the same format as qwen3vl_direct.py for compatibility
with eval_unimmmu.py scoring.

Usage:
    python gemini_eval.py \
        --model_key gemini-3-pro \
        --dataset_root /path/to/Uni-MMMU \
        --result_dir ./Results/gemini-3-pro \
        --workers 4
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

# Add parent dir to path for shared client
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from close_source_client import create_client, list_models

from unimmmu_prompts import (
    JIGSAW_USER_PROMPT,
    MAZE_USER_PROMPT,
    GEOMETRY_REASONING_DIRECT_PREFIX,
)


# ── Utils ────────────────────────────────────────────────────────────────────

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def atomic_write_json(obj: Any, path: Path) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)

def safe_id(s: str) -> str:
    return str(s).replace("/", "_")

def is_done(sample_dir: Path) -> bool:
    p = sample_dir / "result.json"
    if not p.exists():
        return False
    try:
        x = json.loads(p.read_text(encoding="utf-8"))
        return x.get("status") == "ok"
    except Exception:
        return False


# ── Single sample ────────────────────────────────────────────────────────────

def run_one_sample(
    *,
    model_key: str,
    task: str,
    sample: Dict[str, Any],
    out_root: Path,
    image_max_size: int,
    force_rerun: bool,
) -> Tuple[str, str]:
    sid = safe_id(sample.get("id"))
    sample_dir = out_root / "direct" / task / sid
    ensure_dir(sample_dir)

    if (not force_rerun) and is_done(sample_dir):
        return task, sid

    # Build prompt and image list
    if task == "jigsaw":
        prompt = JIGSAW_USER_PROMPT
        image_paths = list(sample["reasoning_imgs"])
    elif task == "maze":
        prompt = MAZE_USER_PROMPT
        image_paths = list(sample["reasoning_imgs"])
    elif task == "geometry":
        reasoning_qs = sample.get("reasoning_qs", "")
        prompt = f"{GEOMETRY_REASONING_DIRECT_PREFIX}\n\nPROBLEM:\n{reasoning_qs}\n"
        image_paths = list(sample["reasoning_imgs"][:1])
    else:
        raise ValueError(f"Unknown task: {task}")

    # Filter to existing images
    valid_images = [p for p in image_paths if Path(p).exists()]

    # Call close-source model
    client = create_client(model_key)
    start_t = time.time()
    status = "error"
    final_text = ""
    err_msg = None

    try:
        final_text = client.generate(
            prompt=prompt,
            image_paths=valid_images,
            image_max_size=image_max_size,
            max_tokens=16384,
            temperature=0.5,
            top_p=0.25,
            thinking_budget=8192,
        )
        if final_text.startswith("[ERROR]"):
            status = "error"
            err_msg = final_text
        else:
            status = "ok"
    except Exception as e:
        err_msg = repr(e)
        status = "error"
        final_text = f"[ERROR] {err_msg}"

    dur = time.time() - start_t

    result = {
        "task": task,
        "id": sid,
        "status": status,
        "model": model_key,
        "gt_text": sample.get("gt_text"),
        "reasoning_qs": sample.get("reasoning_qs"),
        "auxiliary_qs": sample.get("auxiliary_qs"),
        "prompt_used": prompt,
        "input_images": valid_images,
        "model_response": final_text,
        "error": err_msg,
        "runtime_sec": round(dur, 4),
        "ts": time.time(),
    }
    atomic_write_json(result, sample_dir / "result.json")

    return task, sid


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Uni-MMMU eval for close-source models")
    ap.add_argument("--model_key", type=str, required=True,
                    choices=list_models(),
                    help="Close-source model key")
    ap.add_argument("--dataset_root", type=str,
                    default="/mnt/afs/share_data/mashengnan/o3like/image_text_cot/benchmark/Uni-MMMU")
    ap.add_argument("--result_dir", type=str, required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--image_max_size", type=int, default=0,
                    help="Resize images to fit within this box (0 = no resize)")
    ap.add_argument("--force_rerun", action="store_true", default=False)
    ap.add_argument("--max_samples", type=int, default=0)
    args = ap.parse_args()

    out_root = Path(args.result_dir)
    dataset_root = Path(args.dataset_root)
    ensure_dir(out_root)

    for task in ["geometry", "jigsaw", "maze"]:
        data = read_json(dataset_root / task / "dataset.json")
        if args.max_samples and args.max_samples > 0:
            data = data[:args.max_samples]

        # Filter to pending jobs
        jobs = []
        for s in data:
            sid = safe_id(s.get("id"))
            sample_dir = out_root / "direct" / task / sid
            if args.force_rerun or (not is_done(sample_dir)):
                jobs.append(s)

        print(f"[direct/{task}] total={len(data)} todo={len(jobs)} workers={args.workers}")
        if not jobs:
            continue

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [
                ex.submit(
                    run_one_sample,
                    model_key=args.model_key,
                    task=task,
                    sample=s,
                    out_root=out_root,
                    image_max_size=args.image_max_size,
                    force_rerun=args.force_rerun,
                )
                for s in jobs
            ]
            for _ in tqdm(as_completed(futs), total=len(futs), desc=f"run_{task}"):
                pass

    print("DONE.")


if __name__ == "__main__":
    main()
