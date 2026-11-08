"""
RealWorldQA Evaluation Script
Metric: case-insensitive exact match after stripping whitespace.
Reports overall accuracy.
"""

import json
import argparse


def main():
    parser = argparse.ArgumentParser(description="Evaluate RealWorldQA predictions")
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

    # Evaluate: case-insensitive exact match after stripping
    total_count = 0
    total_correct = 0

    for rec in records:
        pred = rec.get("predicted_answer", "").strip().lower()
        gold = rec.get("gold_answer", "").strip().lower()
        total_count += 1
        if pred == gold:
            total_correct += 1

    # Report
    print("=" * 60)
    print("RealWorldQA Evaluation Results (Case-Insensitive Exact Match)")
    print("=" * 60)
    print(f"Total samples:    {total_count}")
    print(
        f"Overall accuracy: {total_correct}/{total_count} = "
        f"{total_correct / total_count * 100:.2f}%"
    )
    print("=" * 60)

    # Save summary JSON
    summary = {
        "total_count": total_count,
        "total_correct": total_correct,
        "overall_accuracy": round(total_correct / total_count * 100, 2),
    }
    summary_path = args.result_file.replace(".jsonl", "_eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
