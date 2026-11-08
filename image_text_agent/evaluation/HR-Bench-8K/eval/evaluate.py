"""
HR-Bench evaluation metrics.
Computes accuracy overall and per category (single/cross, aka FSP/FCP).
"""

import json
import argparse
from collections import defaultdict


def compute_accuracy(results):
    by_category = defaultdict(lambda: {"correct": 0, "total": 0})
    total_correct = 0
    total = 0

    for r in results:
        cat = r.get("category", "unknown")
        by_category[cat]["total"] += 1
        total += 1
        if r.get("correct", False):
            by_category[cat]["correct"] += 1
            total_correct += 1

    for cat in by_category:
        info = by_category[cat]
        info["accuracy"] = info["correct"] / info["total"] if info["total"] > 0 else 0.0

    overall = total_correct / total if total > 0 else 0.0

    return {
        "overall": overall,
        "total_correct": total_correct,
        "total": total,
        "by_category": dict(by_category),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate HR-Bench predictions")
    parser.add_argument("--result_file", type=str, required=True)
    args = parser.parse_args()

    with open(args.result_file, "r") as f:
        results = json.load(f)

    print(f"Loaded {len(results)} results from {args.result_file}")

    report = compute_accuracy(results)

    print(f"\nOverall Accuracy: {report['overall']:.2%} ({report['total_correct']}/{report['total']})")
    print("\nPer-category:")
    for cat in sorted(report["by_category"]):
        info = report["by_category"][cat]
        label = "FSP (Fine-grained Single-instance)" if cat == "single" else \
                "FCP (Fine-grained Cross-instance)" if cat == "cross" else cat
        print(f"  {label}: {info['accuracy']:.2%} ({info['correct']}/{info['total']})")

    summary_path = args.result_file.replace(".json", "_eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
