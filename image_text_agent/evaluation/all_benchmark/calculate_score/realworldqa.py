"""RealWorldQA scoring.

Metric: case-insensitive exact match of extracted answer vs gold.
"""

import os
import json
import re


def _extract_answer(response):
    if not response:
        return ""
    # 1. Try <answer>...</answer> tag first
    m = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # 2. Fall back to content after the last </think> tag
    parts = re.split(r"</think>", response, flags=re.IGNORECASE)
    if len(parts) > 1:
        after_think = parts[-1].strip()
        if after_think:
            return after_think
    # 3. Otherwise return the full response
    return response.strip()


def calculate_score(json_path):
    if not os.path.exists(json_path):
        print(f"[RealWorldQA] Error: Input file not found: {json_path}")
        return
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    total = 0
    correct = 0
    for rec in records:
        gold = (rec.get("gold_answer", "") or "").strip()
        if not gold:
            continue
        pred = _extract_answer(rec.get("model_response", "") or "")
        total += 1
        if pred.lower() == gold.lower():
            correct += 1

    acc = correct / total * 100 if total > 0 else 0.0

    print("=" * 60)
    print("RealWorldQA Evaluation (Case-Insensitive Exact Match)")
    print("=" * 60)
    print(f"Total:   {total}")
    print(f"Overall: {correct}/{total} = {acc:.2f}%")
    print("=" * 60)

    summary = {"total": total, "correct": correct, "accuracy": round(acc, 2)}
    summary_path = json_path.replace(".json", "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")
