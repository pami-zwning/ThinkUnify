"""
ChartQA Evaluation Script
Relaxed accuracy metric:
  - Numeric: correct if relative error <= 5%
  - String: case-insensitive exact match after strip
Reports: overall, human subset, machine subset accuracy
"""

import json
import re
import argparse


def is_numeric(s):
    """Check if a string can be parsed as a float."""
    try:
        # Remove commas, %, $ signs for numeric parsing
        cleaned = s.replace(",", "").replace("%", "").replace("$", "").strip()
        float(cleaned)
        return True
    except (ValueError, TypeError):
        return False


def parse_numeric(s):
    """Parse a string to float, handling commas, %, $ signs."""
    cleaned = s.replace(",", "").replace("%", "").replace("$", "").strip()
    return float(cleaned)


def relaxed_accuracy(prediction, ground_truths):
    """
    ChartQA relaxed accuracy:
    - If both are numeric: correct if relative error <= 5%
    - Otherwise: case-insensitive exact match after strip
    Returns True if prediction matches any of the ground truths.
    """
    pred = prediction.strip()

    for gt in ground_truths:
        gt = gt.strip()

        if is_numeric(pred) and is_numeric(gt):
            pred_val = parse_numeric(pred)
            gt_val = parse_numeric(gt)
            # Handle division by zero: if gt is 0, require exact match
            if gt_val == 0:
                if pred_val == 0:
                    return True
            else:
                if abs(pred_val - gt_val) / abs(gt_val) <= 0.05:
                    return True
        else:
            if pred.lower() == gt.lower():
                return True

    return False


def main():
    parser = argparse.ArgumentParser(description="Evaluate ChartQA predictions")
    parser.add_argument(
        "--result_file",
        type=str,
        required=True,
        help="Path to the JSONL result file from test_benchmark.py",
    )
    args = parser.parse_args()

    # Load results (JSON list format)
    with open(args.result_file, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not records:
        print("No records found in result file.")
        return

    # Evaluate
    total_correct = 0
    human_correct = 0
    machine_correct = 0
    total_count = 0
    human_count = 0
    machine_count = 0

    for rec in records:
        pred = rec.get("predicted_answer", "")
        label = rec.get("label", [])
        hom = rec.get("human_or_machine", -1)  # 0=human, 1=machine

        correct = relaxed_accuracy(pred, label)
        total_count += 1
        if correct:
            total_correct += 1

        if hom == 0:
            human_count += 1
            if correct:
                human_correct += 1
        elif hom == 1:
            machine_count += 1
            if correct:
                machine_correct += 1

    # Report
    print("=" * 60)
    print("ChartQA Evaluation Results (Relaxed Accuracy)")
    print("=" * 60)
    print(f"Total samples:   {total_count}")
    print(
        f"Overall accuracy: {total_correct}/{total_count} = "
        f"{total_correct / total_count * 100:.2f}%"
    )
    print()
    if human_count > 0:
        print(
            f"Human subset:    {human_correct}/{human_count} = "
            f"{human_correct / human_count * 100:.2f}%"
        )
    else:
        print("Human subset:    N/A (no human samples found)")
    if machine_count > 0:
        print(
            f"Machine subset:  {machine_correct}/{machine_count} = "
            f"{machine_correct / machine_count * 100:.2f}%"
        )
    else:
        print("Machine subset:  N/A (no machine samples found)")
    print("=" * 60)

    # Also save summary JSON
    summary = {
        "total_count": total_count,
        "total_correct": total_correct,
        "overall_accuracy": round(total_correct / total_count * 100, 2),
        "human_count": human_count,
        "human_correct": human_correct,
        "human_accuracy": round(human_correct / human_count * 100, 2) if human_count > 0 else None,
        "machine_count": machine_count,
        "machine_correct": machine_correct,
        "machine_accuracy": round(machine_correct / machine_count * 100, 2) if machine_count > 0 else None,
    }
    summary_path = args.result_file.replace(".json", "_eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
