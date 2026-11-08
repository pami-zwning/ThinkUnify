"""MMVP scoring module.

Metric: *pair accuracy* — both questions in a pair must be answered correctly.
Also reports per-question accuracy for reference.
"""

import os
import json
import re
from collections import defaultdict


def extract_answer(response):
    """Extract (a)/(b) or A/B from a model response.

    Priority:
      1. <answer>...</answer> tag content
      2. Content after the last </think> tag
      3. Full response
    Then match (a)/(b) or A/B pattern.
    """
    if not response:
        return ""

    # 1. Try <answer>...</answer> tag first
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
    if match:
        candidate = match.group(1).strip()
    else:
        # 2. Fall back to content after the last </think> tag
        parts = re.split(r"</think>", response, flags=re.IGNORECASE)
        if len(parts) > 1 and parts[-1].strip():
            candidate = parts[-1].strip()
        else:
            candidate = response.strip()

    # Match (a)/(b) style
    ab_match = re.search(r"\(([aAbB])\)", candidate)
    if ab_match:
        return ab_match.group(1).upper()

    # Match standalone A/B
    letter_match = re.search(r"\b([aAbB])\b", candidate)
    if letter_match:
        return letter_match.group(1).upper()

    # Last resort: find last A/B in full response
    all_matches = re.findall(r"\b([aAbB])\b", response)
    return all_matches[-1].upper() if all_matches else ""


def evaluate_records(records):
    # Per-question accuracy
    q_total = 0
    q_correct = 0
    # Group by pair_id for pair accuracy
    pairs = defaultdict(dict)

    for rec in records:
        pred = extract_answer(rec.get("model_response", "") or "")
        gold = (rec.get("gold_answer", "") or "").strip().upper()
        if not gold:
            continue

        q_total += 1
        hit = pred == gold
        if hit:
            q_correct += 1

        pair_id = rec.get("pair_id", None)
        q_idx = rec.get("question_idx", 0)
        if pair_id is not None:
            pairs[pair_id][q_idx] = hit

    # Pair accuracy: both questions in a pair must be correct
    pair_total = len(pairs)
    pair_correct = sum(1 for p in pairs.values() if all(p.values()) and len(p) >= 2)

    return {
        "question_total": q_total,
        "question_correct": q_correct,
        "question_accuracy": round(q_correct / q_total * 100, 2) if q_total > 0 else 0.0,
        "pair_total": pair_total,
        "pair_correct": pair_correct,
        "pair_accuracy": round(pair_correct / pair_total * 100, 2) if pair_total > 0 else 0.0,
    }


def calculate_score(json_path):
    if not os.path.exists(json_path):
        print(f"[MMVP] Error: Input file not found: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    metrics = evaluate_records(records)

    print("=" * 60)
    print("MMVP Evaluation Results")
    print("=" * 60)
    print(f"Per-question: {metrics['question_correct']}/{metrics['question_total']} = {metrics['question_accuracy']:.2f}%")
    print(f"Pair accuracy: {metrics['pair_correct']}/{metrics['pair_total']} = {metrics['pair_accuracy']:.2f}%")
    print("=" * 60)

    summary_path = json_path.replace(".json", "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MMVP pair-accuracy scoring")
    parser.add_argument("--input_file", type=str, required=True)
    args = parser.parse_args()
    calculate_score(args.input_file)
