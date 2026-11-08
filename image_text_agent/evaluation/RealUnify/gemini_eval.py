#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RealUnify benchmark evaluation for close-source models (Gemini, etc.)

Direct inference only. Results saved in the same JSON format as eval.py
for compatibility with score.py / eval_understanding.py.

Usage:
    python gemini_eval.py \
        --model_key gemini-3-pro \
        --result_path ./Results \
        --workers 4
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm

# Add parent dir to path for shared client
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from close_source_client import create_client, list_models


# ── Config ───────────────────────────────────────────────────────────────────

DATASET_ROOT = "/mnt/afs/share_data/mashengnan/o3like/image_text_cot/benchmark/RealUnify"
INPUT_FILE = os.path.join(DATASET_ROOT, "GEU_step.json")


# ── Utils ────────────────────────────────────────────────────────────────────

def refine_prompt(prompt_text: str) -> str:
    """
    Clean up evaluation prompt:
    1. If "Question:" exists, strip everything before it.
    2. Remove trailing "The best answer is:".
    """
    if "Question:" in prompt_text:
        prompt_text = prompt_text.split("Question:", 1)[-1].strip()
    prompt_text = re.sub(r"The best answer is:\s*$", "", prompt_text, flags=re.IGNORECASE).strip()
    return prompt_text


# ── Worker ───────────────────────────────────────────────────────────────────

def worker_process(item: Dict[str, Any], model_key: str) -> Dict[str, Any]:
    """Process a single RealUnify item."""
    # Fix image path
    input_img = item.get("input_image", "")
    item["input_image"] = input_img.replace(
        "/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Data/RealUnify",
        DATASET_ROOT,
    )

    # No image editing for close-source eval
    item["edit_image"] = ""

    # Collect image paths
    image_paths = [item["input_image"]]

    # Build prompt
    prompt = refine_prompt(item.get("evaluation_prompt", ""))

    # Call model
    client = create_client(model_key)
    try:
        response_text = client.generate(
            prompt=prompt,
            image_paths=image_paths,
            max_tokens=16384,
            temperature=0.5,
            top_p=0.25,
            thinking_budget=8192,
        )
        if response_text.startswith("[ERROR]"):
            item["model_response"] = response_text
        else:
            item["model_response"] = response_text
    except Exception as e:
        item["model_response"] = f"[ERROR] {repr(e)}"

    item["response"] = item["model_response"]
    item["mid_process"] = []

    return item


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RealUnify eval for close-source models")
    parser.add_argument("--model_key", type=str, required=True,
                        choices=list_models(),
                        help="Close-source model key")
    parser.add_argument("--input_file", type=str, default=INPUT_FILE,
                        help="Path to GEU_step.json")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Root output directory")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    # Load data
    if not os.path.exists(args.input_file):
        print(f"Error: {args.input_file} not found.")
        return

    with open(args.input_file, "r", encoding="utf-8") as f:
        data_list = json.load(f)

    print(f"Loaded {len(data_list)} tasks from {args.input_file}")

    # Prepare output path
    save_path = os.path.join(args.result_path, "Jsons", f"{args.model_key}.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Resume: load existing results
    existing_results = []
    processed_ids = set()
    if os.path.exists(save_path):
        try:
            with open(save_path, "r", encoding="utf-8") as f:
                existing_results = json.load(f)
            # Filter out errored items for retry
            existing_results = [
                item for item in existing_results
                if "ERROR" not in item.get("model_response", "")
            ]
            processed_ids = {
                item.get("id", item.get("input_image", ""))
                for item in existing_results
            }
            print(f"Resuming: {len(existing_results)} already completed")
        except json.JSONDecodeError:
            existing_results = []

    # Filter pending
    remaining = [
        item for item in data_list
        if item.get("id", item.get("input_image", "")) not in processed_ids
    ]
    print(f"Remaining to process: {len(remaining)}")

    if not remaining:
        print("All done.")
        return

    # Execute with thread pool
    results = list(existing_results)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(worker_process, item, args.model_key): item
            for item in remaining
        }
        count = 0
        for future in tqdm(as_completed(futures), total=len(remaining), desc=args.model_key):
            result = future.result()
            results.append(result)
            count += 1
            # Periodic save
            if count % 10 == 0:
                with open(save_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

    # Final save
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Done. Results saved to {save_path}")


if __name__ == "__main__":
    main()
