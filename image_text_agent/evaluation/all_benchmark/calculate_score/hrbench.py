"""HR-Bench scoring (shared for 4K and 8K).

Metric: exact letter match (A/B/C/D) vs gold.
Breakdown: per-category (e.g. single / cross).
"""

import os
import json
import re
from collections import defaultdict


def _extract_letter(response):
    if not response:
        return ""
    m = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
    candidate = m.group(1).strip() if m else response.strip()
    lm = re.search(r"[A-D]", candidate, re.IGNORECASE)
    return lm.group(0).upper() if lm else ""


def _evaluate(records):
    total = 0
    correct = 0
    cat_total = defaultdict(int)
    cat_correct = defaultdict(int)
    for rec in records:
        pred = _extract_letter(rec.get("model_response", "") or "")
        gold = (rec.get("gold_answer", "") or "").strip().upper()
        if not gold:
            continue
        total += 1
        cat = rec.get("category", "all")
        cat_total[cat] += 1
        if pred == gold:
            correct += 1
            cat_correct[cat] += 1
    return total, correct, cat_total, cat_correct


def _report(name, json_path):
    if not os.path.exists(json_path):
        print(f"[{name}] Error: Input file not found: {json_path}")
        return
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    total, correct, cat_total, cat_correct = _evaluate(records)
    acc = (correct / total * 100) if total > 0 else 0.0

    print("=" * 60)
    print(f"{name} Evaluation (Exact Letter Match)")
    print("=" * 60)
    print(f"Total:   {total}")
    print(f"Overall: {correct}/{total} = {acc:.2f}%")
    for cat in sorted(cat_total.keys()):
        cacc = (cat_correct[cat] / cat_total[cat] * 100) if cat_total[cat] > 0 else 0.0
        print(f"  [{cat}]: {cat_correct[cat]}/{cat_total[cat]} = {cacc:.2f}%")
    print("=" * 60)

    summary = {
        "total": total,
        "correct": correct,
        "accuracy": round(acc, 2),
        "per_category": {
            c: {
                "total": cat_total[c],
                "correct": cat_correct[c],
                "accuracy": round((cat_correct[c] / cat_total[c] * 100) if cat_total[c] > 0 else 0, 2),
            }
            for c in cat_total
        },
    }
    summary_path = json_path.replace(".json", "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")


def calculate_score_4k(json_path):
    _report("HR-Bench-4K", json_path)


def calculate_score_8k(json_path):
    _report("HR-Bench-8K", json_path)


# Default entry used by dispatcher (works for either).
calculate_score = calculate_score_4k
