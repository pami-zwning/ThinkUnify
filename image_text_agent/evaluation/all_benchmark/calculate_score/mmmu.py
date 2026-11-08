"""MMMU scoring.

Metric: exact letter match for MC, case-insensitive exact match for open-ended.
Breakdown: per-subject and per-macro-category.
"""

import os
import json
import re
from collections import defaultdict

MACRO_CATEGORIES = {
    "Art_and_Design": ["Art", "Art_Theory", "Design", "Music"],
    "Business": ["Accounting", "Economics", "Finance", "Manage", "Marketing"],
    "Science": ["Biology", "Chemistry", "Geography", "Math", "Physics"],
    "Health_and_Medicine": ["Basic_Medical_Science", "Clinical_Medicine", "Diagnostics_and_Laboratory_Medicine", "Pharmacy", "Public_Health"],
    "Humanities_and_Social_Science": ["History", "Literature", "Psychology", "Sociology"],
    "Tech_and_Engineering": ["Agriculture", "Architecture_and_Engineering", "Computer_Science", "Electronics", "Energy_and_Power", "Materials", "Mechanical_Engineering"],
}

SUBJECT_TO_MACRO = {}
for macro, subjects in MACRO_CATEGORIES.items():
    for s in subjects:
        SUBJECT_TO_MACRO[s.lower()] = macro


def _get_candidate(response):
    """Get candidate text: prefer <answer> tag, then content after last </think>."""
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


def _extract_letter(response, num_choices=4):
    if not response:
        return ""
    candidate = _get_candidate(response)
    valid = "ABCDEFGH"[:num_choices]
    lm = re.search(f"[{valid}]", candidate, re.IGNORECASE)
    if lm:
        return lm.group(0).upper()
    # If no letter found in candidate, search last letter in full response
    all_matches = re.findall(f"[{valid}]", response, re.IGNORECASE)
    return all_matches[-1].upper() if all_matches else ""


def _extract_answer_tag(response):
    return _get_candidate(response)


def _judge(rec):
    pred_raw = rec.get("model_response", "") or ""
    gold = (rec.get("gold_answer", "") or "").strip()
    qtype = rec.get("question_type", "open")
    options = rec.get("options") or []

    if qtype == "multiple-choice" and options:
        pred_letter = _extract_letter(pred_raw, num_choices=len(options))
        return pred_letter == gold.upper()

    # Open-ended: case-insensitive match
    pred = _extract_answer_tag(pred_raw)
    return pred.strip().lower() == gold.lower()


def calculate_score(json_path):
    if not os.path.exists(json_path):
        print(f"[MMMU] Error: Input file not found: {json_path}")
        return
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    total = 0
    correct = 0
    by_subject = defaultdict(lambda: [0, 0])
    by_macro = defaultdict(lambda: [0, 0])

    for rec in records:
        if not rec.get("gold_answer"):
            continue
        total += 1
        is_correct = _judge(rec)
        if is_correct:
            correct += 1
        subj = rec.get("subject", "unknown")
        by_subject[subj][1] += 1
        if is_correct:
            by_subject[subj][0] += 1
        macro = SUBJECT_TO_MACRO.get(subj.lower(), "Other")
        by_macro[macro][1] += 1
        if is_correct:
            by_macro[macro][0] += 1

    acc = correct / total * 100 if total > 0 else 0.0

    # Macro-average: average of per-subject accuracies
    subj_accs = []
    for s, (c, t) in by_subject.items():
        if t > 0:
            subj_accs.append(c / t * 100)
    macro_avg = sum(subj_accs) / len(subj_accs) if subj_accs else 0.0

    print("=" * 60)
    print("MMMU Evaluation")
    print("=" * 60)
    print(f"Total:        {total}")
    print(f"Overall:      {correct}/{total} = {acc:.2f}%")
    print(f"Macro-avg:    {macro_avg:.2f}%")
    for macro in sorted(by_macro.keys()):
        c, t = by_macro[macro]
        print(f"  [{macro}]: {c}/{t} = {(c/t*100 if t else 0):.2f}%")
    print("=" * 60)

    summary = {
        "total": total, "correct": correct, "accuracy": round(acc, 2),
        "macro_average": round(macro_avg, 2),
        "per_macro": {k: {"correct": v[0], "total": v[1]} for k, v in by_macro.items()},
        "per_subject": {k: {"correct": v[0], "total": v[1]} for k, v in by_subject.items()},
    }
    summary_path = json_path.replace(".json", "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")
