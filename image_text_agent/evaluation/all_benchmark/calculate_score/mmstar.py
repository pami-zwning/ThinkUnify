"""MMStar scoring.

Metric: exact letter match (A/B/C/D).
Breakdown: per-category and per-l2_category.
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


def calculate_score(json_path):
    if not os.path.exists(json_path):
        print(f"[MMStar] Error: Input file not found: {json_path}")
        return
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    total = 0
    correct = 0
    by_cat = defaultdict(lambda: [0, 0])
    by_l2 = defaultdict(lambda: [0, 0])

    for rec in records:
        gold = (rec.get("gold_answer", "") or "").strip().upper()
        if not gold:
            continue
        pred = _extract_letter(rec.get("model_response", "") or "")
        total += 1
        is_correct = pred == gold
        if is_correct:
            correct += 1
        cat = rec.get("category", "unknown")
        l2 = rec.get("l2_category", "unknown")
        by_cat[cat][1] += 1
        by_l2[l2][1] += 1
        if is_correct:
            by_cat[cat][0] += 1
            by_l2[l2][0] += 1

    acc = correct / total * 100 if total > 0 else 0.0

    print("=" * 60)
    print("MMStar Evaluation (Exact Letter Match)")
    print("=" * 60)
    print(f"Total:   {total}")
    print(f"Overall: {correct}/{total} = {acc:.2f}%")
    for cat in sorted(by_cat.keys()):
        c, t = by_cat[cat]
        print(f"  [{cat}]: {c}/{t} = {(c/t*100 if t else 0):.2f}%")
    print("=" * 60)

    summary = {
        "total": total, "correct": correct, "accuracy": round(acc, 2),
        "per_category": {k: {"correct": v[0], "total": v[1]} for k, v in by_cat.items()},
        "per_l2_category": {k: {"correct": v[0], "total": v[1]} for k, v in by_l2.items()},
    }
    summary_path = json_path.replace(".json", "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")
