"""V*Bench scoring module.

Metric: exact-letter match (A/B/C/D).
Also reports per-category accuracy when the 'category' field is present.
"""

import os
import json
import re
from collections import defaultdict


def extract_letter(response):
    if not response:
        return ""
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
    candidate = match.group(1).strip() if match else response.strip()
    letter_match = re.search(r"[A-Z]", candidate, re.IGNORECASE)
    if letter_match:
        return letter_match.group(0).upper()
    return ""


def evaluate_records(records):
    total = 0
    correct = 0
    per_cat = defaultdict(lambda: {"total": 0, "correct": 0})

    for rec in records:
        pred = extract_letter(rec.get("model_response", "") or "")
        gold = (rec.get("gold_answer", "") or "").strip().upper()
        if not gold:
            continue
        total += 1
        hit = pred == gold
        if hit:
            correct += 1
        cat = rec.get("category", "overall")
        per_cat[cat]["total"] += 1
        if hit:
            per_cat[cat]["correct"] += 1

    per_cat_acc = {}
    for cat, v in per_cat.items():
        per_cat_acc[cat] = round(v["correct"] / v["total"] * 100, 2) if v["total"] > 0 else 0.0

    return {
        "total": total,
        "correct": correct,
        "accuracy": round(correct / total * 100, 2) if total > 0 else 0.0,
        "per_category": per_cat_acc,
    }


def calculate_score(json_path):
    if not os.path.exists(json_path):
        print(f"[V*Bench] Error: Input file not found: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    metrics = evaluate_records(records)

    print("=" * 60)
    print("V*Bench Evaluation Results (Exact Letter Match)")
    print("=" * 60)
    print(f"Total samples:    {metrics['total']}")
    print(f"Overall accuracy: {metrics['correct']}/{metrics['total']} = {metrics['accuracy']:.2f}%")
    if metrics.get("per_category"):
        print("-" * 40)
        for cat, acc in sorted(metrics["per_category"].items()):
            print(f"  {cat}: {acc:.2f}%")
    print("=" * 60)

    summary_path = json_path.replace(".json", "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="V*Bench scoring")
    parser.add_argument("--input_file", type=str, required=True)
    args = parser.parse_args()
    calculate_score(args.input_file)
