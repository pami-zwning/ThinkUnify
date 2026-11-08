"""
MathVista Evaluation Script
Computes accuracy overall and by task category.

Evaluation rules:
- multi_choice: exact match of extracted letter vs gold answer
- free_form:
  - float: abs difference < 1e-3 or relative error < 5%
  - integer: exact match after parsing as int
  - text: case-insensitive exact match after stripping
"""

import json
import re
import argparse
from collections import defaultdict


def normalize_float(s):
    """Try to parse a string as float, handling common formats."""
    if s is None:
        return None
    s = str(s).strip()
    # Remove trailing % and divide by 100
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except ValueError:
            pass
    # Remove commas (e.g., "1,234.5")
    s = s.replace(",", "")
    # Remove leading $ or other currency symbols
    s = re.sub(r'^[$]', '', s)
    try:
        return float(s)
    except ValueError:
        return None


def normalize_int(s):
    """Try to parse a string as integer."""
    if s is None:
        return None
    s = str(s).strip()
    s = s.replace(",", "")
    s = re.sub(r'^[$]', '', s)
    try:
        return int(float(s))
    except (ValueError, OverflowError):
        return None


def is_correct(prediction, gold_answer, question_type, answer_type, choices=None):
    """
    Determine if a prediction matches the gold answer.
    For multi_choice: prediction is a letter (A/B/C...), gold_answer may be
    either a letter or the answer text. We try both comparisons.
    """
    if not prediction or prediction.startswith("[ERROR]"):
        return False

    pred = str(prediction).strip()
    gold = str(gold_answer).strip()

    if question_type == "multi_choice":
        # Direct match (both are letters, or both are text)
        if pred.upper() == gold.upper():
            return True
        # pred is a letter, gold is answer text → map letter to text via choices
        if choices and len(pred) == 1 and pred.upper() in "ABCDE":
            idx = ord(pred.upper()) - ord('A')
            if 0 <= idx < len(choices):
                pred_text = str(choices[idx]).strip()
                if pred_text.lower() == gold.lower():
                    return True
        # pred is answer text, gold is a letter → reverse map
        if choices and len(gold) == 1 and gold.upper() in "ABCDE":
            idx = ord(gold.upper()) - ord('A')
            if 0 <= idx < len(choices):
                gold_text = str(choices[idx]).strip()
                if pred.lower() == gold_text.lower():
                    return True
        return False

    # free_form
    if answer_type == "float":
        pred_f = normalize_float(pred)
        gold_f = normalize_float(gold)
        if pred_f is None or gold_f is None:
            return False
        # Absolute tolerance
        if abs(pred_f - gold_f) < 1e-3:
            return True
        # Relative tolerance (5%)
        if gold_f != 0 and abs(pred_f - gold_f) / abs(gold_f) < 0.05:
            return True
        return False

    elif answer_type == "integer":
        pred_i = normalize_int(pred)
        gold_i = normalize_int(gold)
        if pred_i is None or gold_i is None:
            return False
        return pred_i == gold_i

    else:
        # text: case-insensitive exact match after stripping
        return pred.lower().strip() == gold.lower().strip()


def compute_accuracy(results):
    """
    Compute accuracy from a list of result dicts.
    Each dict must have: prediction, gold_answer, question_type, answer_type.
    Returns a summary dict.
    """
    total = 0
    correct = 0
    by_question_type = defaultdict(lambda: {"correct": 0, "total": 0})
    by_answer_type = defaultdict(lambda: {"correct": 0, "total": 0})

    for r in results:
        pred = r.get("prediction", "")
        gold = r.get("gold_answer", r.get("answer", ""))
        q_type = r.get("question_type", "")
        a_type = r.get("answer_type", "")
        choices = r.get("choices", None)

        total += 1
        match = is_correct(pred, gold, q_type, a_type, choices=choices)

        if match:
            correct += 1

        by_question_type[q_type]["total"] += 1
        by_answer_type[a_type]["total"] += 1
        if match:
            by_question_type[q_type]["correct"] += 1
            by_answer_type[a_type]["correct"] += 1

    # Compute accuracy percentages
    for d in [by_question_type, by_answer_type]:
        for k, v in d.items():
            v["accuracy"] = v["correct"] / v["total"] if v["total"] > 0 else 0.0

    return {
        "overall": correct / total if total > 0 else 0.0,
        "correct": correct,
        "total": total,
        "by_question_type": dict(by_question_type),
        "by_answer_type": dict(by_answer_type),
    }


def compute_accuracy_by_task(results):
    """
    Compute accuracy grouped by metadata.task (if available in the dataset).
    Falls back to question_type grouping if task metadata is not present.
    """
    by_task = defaultdict(lambda: {"correct": 0, "total": 0})

    for r in results:
        pred = r.get("prediction", "")
        gold = r.get("gold_answer", r.get("answer", ""))
        q_type = r.get("question_type", "")
        a_type = r.get("answer_type", "")
        # task category from metadata (may not exist in all results)
        task = r.get("task", r.get("metadata", {}).get("task", "unknown") if isinstance(r.get("metadata"), dict) else "unknown")

        match = is_correct(pred, gold, q_type, a_type)

        by_task[task]["total"] += 1
        if match:
            by_task[task]["correct"] += 1

    for k, v in by_task.items():
        v["accuracy"] = v["correct"] / v["total"] if v["total"] > 0 else 0.0

    return dict(by_task)


def main():
    parser = argparse.ArgumentParser(description="MathVista Evaluation")
    parser.add_argument("--result_file", type=str, required=True,
                        help="Path to the prediction JSON file")
    parser.add_argument("--output_file", type=str, default="",
                        help="Path to save evaluation report (optional)")
    args = parser.parse_args()

    # Load predictions
    with open(args.result_file, "r") as f:
        results = json.load(f)

    print(f"Loaded {len(results)} predictions from {args.result_file}")
    print("=" * 60)

    # Overall accuracy
    acc = compute_accuracy(results)
    print(f"Overall Accuracy: {acc['overall']:.2%} ({acc['correct']}/{acc['total']})")
    print()

    # By question type
    print("Accuracy by Question Type:")
    print("-" * 40)
    for qt, info in sorted(acc["by_question_type"].items()):
        print(f"  {qt:20s}: {info['accuracy']:.2%} ({info['correct']}/{info['total']})")
    print()

    # By answer type
    print("Accuracy by Answer Type:")
    print("-" * 40)
    for at, info in sorted(acc["by_answer_type"].items()):
        print(f"  {at:20s}: {info['accuracy']:.2%} ({info['correct']}/{info['total']})")
    print()

    # By task category
    by_task = compute_accuracy_by_task(results)
    if by_task and not (len(by_task) == 1 and "unknown" in by_task):
        print("Accuracy by Task Category:")
        print("-" * 40)
        for task, info in sorted(by_task.items(), key=lambda x: x[0]):
            print(f"  {task:30s}: {info['accuracy']:.2%} ({info['correct']}/{info['total']})")
        print()

    # Save report if requested
    if args.output_file:
        report = {
            "overall": acc,
            "by_task": by_task,
        }
        with open(args.output_file, "w") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Report saved to {args.output_file}")


if __name__ == "__main__":
    main()
