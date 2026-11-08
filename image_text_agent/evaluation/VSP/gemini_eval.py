#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VSP (Visual Spatial Planning) benchmark evaluation for close-source models.

Direct inference: sends maze image + prompt, saves response text to file.
The existing eval.py can then score the results.

Usage:
    python gemini_eval.py \
        --model_key gemini-3-pro \
        --data_path ./maze_task_main.jsonl \
        --result_path ./Results \
        --workers 4
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any

from tqdm import tqdm

# Add parent dir to path for shared client
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from close_source_client import create_client, list_models


# ── Default maze prompt (same as save_info.py) ──────────────────────────────

DEFAULT_MAZE_PROMPT = """You are a maze solver. Your goal is to guide a player from the start to the goal on a grid map while avoiding holes.

Generate the complete, optimized sequence of instructions to guide the robot.
Commands are: R (Right), L (Left), U (Up), D (Down). Do not include pauses or diagonal movements.
Provide your solution as a sequence of moves wrapped in <answer> </answer>, such as <answer>LRUD</answer>.
Please generate action plan for the following maze:
"""


# ── Utils ────────────────────────────────────────────────────────────────────

def read_jsonl_file(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ── Worker ───────────────────────────────────────────────────────────────────

def worker_process(item: Dict[str, Any], model_key: str, result_root: str) -> Dict[str, Any]:
    """Process a single VSP maze task."""
    image_path = item["image_path"]
    prompt = item.get("prompt", DEFAULT_MAZE_PROMPT)

    # Override save_path to use result_root
    level = item["level"]
    curr_id = item["curr_id"]
    save_path = os.path.join(result_root, f"level{level}", f"{curr_id}.txt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Call model
    client = create_client(model_key)
    try:
        response_text = client.generate(
            prompt=prompt,
            image_paths=[image_path],
            max_tokens=16384,
            temperature=0.5,
            top_p=0.25,
            thinking_budget=8192,
        )
    except Exception as e:
        response_text = f"[ERROR] {repr(e)}"

    # Save response
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(response_text)

    item["model_response"] = response_text
    item["save_path"] = save_path
    return item


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="VSP eval for close-source models")
    parser.add_argument("--model_key", type=str, required=True,
                        choices=list_models(),
                        help="Close-source model key")
    parser.add_argument("--data_path", type=str, default="./maze_task_main.jsonl",
                        help="Path to maze_task_main.jsonl")
    parser.add_argument("--result_path", type=str, required=True,
                        help="Root output directory for response text files")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--save_json", action="store_true", default=False,
                        help="Also save all results as a single JSON file")
    args = parser.parse_args()

    # Output directory
    output_root = os.path.join(args.result_path, args.model_key)
    os.makedirs(output_root, exist_ok=True)

    # Load data
    if not os.path.exists(args.data_path):
        print(f"Error: {args.data_path} not found.")
        return

    datas = read_jsonl_file(args.data_path)
    print(f"Loaded {len(datas)} tasks from {args.data_path}")

    # Resume: skip already completed
    remaining = []
    for item in datas:
        save_path = os.path.join(output_root, f"level{item['level']}", f"{item['curr_id']}.txt")
        if os.path.exists(save_path):
            with open(save_path, "r") as f:
                content = f.read().strip()
            if content and "ERROR" not in content:
                continue
        remaining.append(item)

    print(f"Already completed: {len(datas) - len(remaining)}")
    print(f"Remaining to process: {len(remaining)}")

    if not remaining:
        print("All done.")
        return

    # Execute
    all_results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(worker_process, item, args.model_key, output_root): item
            for item in remaining
        }
        for future in tqdm(as_completed(futures), total=len(remaining), desc=args.model_key):
            result = future.result()
            all_results.append(result)

    # Optionally save JSON summary
    if args.save_json:
        json_path = os.path.join(args.result_path, f"{args.model_key}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"JSON results saved to {json_path}")

    print("Done. Run eval.py to compute accuracy.")


if __name__ == "__main__":
    main()
