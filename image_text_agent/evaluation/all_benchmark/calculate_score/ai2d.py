"""AI2D scoring module.

Metric: exact-letter match (A/B/C/D) between predicted answer and gold.
Prefers content inside <answer>...</answer> tags; falls back to scanning the
full response for the first A-D letter.
"""

import os
import json
import re


def extract_letter(response):
    """Extract A/B/C/D from a model response.

    Priority:
      1. <answer>...</answer> tag content (first A-D letter inside)
      2. Last A-D letter anywhere in the response
    """
    if not response:
        return ""

    # 1. Look inside <answer> tags first
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
    if match:
        candidate = match.group(1).strip()
        letter_match = re.search(r"[A-D]", candidate, re.IGNORECASE)
        if letter_match:
            return letter_match.group(0).upper()

    # 2. Extract last A-D letter in the full response (case insensitive)
    all_matches = re.findall(r"[A-D]", response, re.IGNORECASE)
    if all_matches:
        return all_matches[-1].upper()

    return ""


def evaluate_records(records):
    total = 0
    correct = 0
    for rec in records:
        pred = extract_letter(rec.get("model_response", "") or "")
        gold = (rec.get("gold_answer", "") or "").strip().upper()
        if not gold:
            continue
        total += 1
        if pred == gold:
            correct += 1
    return {
        "total": total,
        "correct": correct,
        "accuracy": (correct / total * 100) if total > 0 else 0.0,
    }


def calculate_score(json_path):
    if not os.path.exists(json_path):
        print(f"[AI2D] Error: Input file not found: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    metrics = evaluate_records(records)

    print("=" * 60)
    print("AI2D Evaluation Results (Exact Letter Match)")
    print("=" * 60)
    print(f"Total samples:    {metrics['total']}")
    print(
        f"Overall accuracy: {metrics['correct']}/{metrics['total']} = "
        f"{metrics['accuracy']:.2f}%"
    )
    print("=" * 60)

    # Save summary next to the input file
    summary_path = json_path.replace(".json", "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI2D exact-match scoring")
    parser.add_argument("--input_file", type=str, required=True)
    args = parser.parse_args()
    calculate_score(args.input_file)
