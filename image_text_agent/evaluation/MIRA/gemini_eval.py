#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MIRA benchmark evaluation for close-source models (Gemini, etc.)

Direct inference only (text_cot scenario, no tool calls).
Results saved as .jsonl files compatible with score.py.

Usage:
    python gemini_eval.py \
        --model_key gemini-3-pro \
        --benchmark_dir /path/to/MIRA \
        --output_dir ./Results/gemini-3-pro \
        --workers 4
"""

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

# Add parent dir to path for shared client
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from close_source_client import create_client, list_models


# ── Utils ────────────────────────────────────────────────────────────────────

FILE_LOCKS = {}


def get_target_images(task_dir: Path, data_item: dict, scenario: str) -> List[str]:
    """Find task-related image paths."""
    imgs = []
    base_name = Path(data_item.get("image_path", "")).stem
    if not base_name:
        return imgs

    # Find main image
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        p = task_dir / "image" / (base_name + ext)
        if p.exists():
            imgs.append(str(p))
            break

    # If visual_cot, find auxiliary images
    if scenario == "visual_cot":
        cot_dir = task_dir / "cot"
        candidates = list(cot_dir.glob(f"{base_name}.*")) + list(cot_dir.glob(f"{base_name}_*"))
        imgs.extend(sorted([
            str(p) for p in candidates
            if p.suffix.lower() in [".png", ".jpg", ".jpeg"]
        ]))

    return imgs


# ── Worker ───────────────────────────────────────────────────────────────────

def run_single_task(
    model_key: str,
    model_label: str,
    data_item: dict,
    task_dir: Path,
    output_dir: Path,
) -> None:
    """Run inference on a single MIRA task item."""
    scenario = "text_cot"

    # Build prompt
    prompt = (
        f"Question: {data_item.get('question', '')}\n\n"
        "Please first conduct step-by-step reasoning, and then provide the final answer. "
        "The final answer is placed in <answer></answer>."
    )

    # Get images
    img_paths = get_target_images(task_dir, data_item, scenario)

    # Call model
    client = create_client(model_key)
    try:
        resp = client.generate(
            prompt=prompt,
            image_paths=img_paths if img_paths else None,
            max_tokens=16384,
            temperature=0.5,
            top_p=0.25,
            thinking_budget=8192,
        )
    except Exception as e:
        resp = f"[ERROR] {repr(e)}"

    # Build result
    res = {
        "uid": data_item.get("uid"),
        "scenario": scenario,
        "model_name": model_label,
        "response": resp,
        "original_data": data_item,
    }

    # Write to file (with lock)
    out_file = output_dir / f"{model_label}_{task_dir.name}.jsonl"
    lock = FILE_LOCKS.setdefault(str(out_file), threading.Lock())
    with lock:
        with open(out_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            f.flush()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MIRA eval for close-source models")
    parser.add_argument("--model_key", type=str, required=True,
                        choices=list_models(),
                        help="Close-source model key")
    parser.add_argument("--model_label", type=str, default="",
                        help="Label for output files (defaults to model_key)")
    parser.add_argument("--benchmark_dir", type=str, required=True,
                        help="Path to MIRA benchmark root")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    model_label = args.model_label or args.model_key
    bench_root = Path(args.benchmark_dir)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Resume: scan already completed UIDs
    finished_uids = set()
    for f in out_root.glob(f"{model_label}_*.jsonl"):
        try:
            with open(f, "r", encoding="utf-8") as fr:
                for line in fr:
                    if line.strip():
                        finished_uids.add(json.loads(line).get("uid"))
        except Exception:
            pass
    print(f"Skip already completed: {len(finished_uids)}")

    # Scan tasks
    tasks = []
    for f in bench_root.rglob("*.jsonl"):
        with open(f, "r", encoding="utf-8") as fr:
            for line in fr:
                try:
                    item = json.loads(line)
                    if item.get("uid") not in finished_uids:
                        tasks.append((item, f.parent))
                except Exception:
                    continue

    if not tasks:
        print("All tasks completed.")
        return

    print(f"Running {len(tasks)} tasks (workers={args.workers})...")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                run_single_task,
                args.model_key,
                model_label,
                item,
                task_dir,
                out_root,
            )
            for item, task_dir in tasks
        ]
        for _ in tqdm(as_completed(futures), total=len(tasks), desc=model_label):
            pass

    print("Done.")


if __name__ == "__main__":
    main()
