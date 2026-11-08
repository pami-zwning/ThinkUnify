"""
MathVerse evaluation metrics.
Computes accuracy overall, per problem_version, per subject, per subfield.
"""

import json
import argparse
from collections import defaultdict


def compute_accuracy(results):
    total = 0
    correct = 0
    by_version = defaultdict(lambda: {"correct": 0, "total": 0})
    by_subject = defaultdict(lambda: {"correct": 0, "total": 0})
    by_subfield = defaultdict(lambda: {"correct": 0, "total": 0})

    for r in results:
        pred = r.get("prediction", "").strip().upper()
        gold = r.get("gold_answer", "").strip().upper()
        version = r.get("problem_version", "unknown")
        subject = r.get("subject", "unknown")
        subfield = r.get("subfield", "unknown")

        is_correct = pred == gold if pred and gold else False

        total += 1
        if is_correct:
            correct += 1

        by_version[version]["total"] += 1
        by_subject[subject]["total"] += 1
        by_subfield[subfield]["total"] += 1
        if is_correct:
            by_version[version]["correct"] += 1
            by_subject[subject]["correct"] += 1
            by_subfield[subfield]["correct"] += 1

    for d in [by_version, by_subject, by_subfield]:
        for k, v in d.items():
            v["accuracy"] = v["correct"] / v["total"] if v["total"] > 0 else 0.0

    return {
        "overall": correct / total if total > 0 else 0.0,
        "total_correct": correct,
        "total": total,
        "by_version": dict(by_version),
        "by_subject": dict(by_subject),
        "by_subfield": dict(by_subfield),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate MathVerse predictions")
    parser.add_argument("--result_file", type=str, required=True)
    args = parser.parse_args()

    with open(args.result_file, "r") as f:
        results = json.load(f)

    print(f"Loaded {len(results)} results from {args.result_file}")

    report = compute_accuracy(results)

    print(f"\nOverall Accuracy: {report['overall']:.2%} ({report['total_correct']}/{report['total']})")

    print("\nPer Problem Version:")
    for v in sorted(report["by_version"]):
        info = report["by_version"][v]
        print(f"  {v:25s}: {info['accuracy']:.2%} ({info['correct']}/{info['total']})")

    print("\nPer Subject:")
    for s in sorted(report["by_subject"]):
        info = report["by_subject"][s]
        print(f"  {s:25s}: {info['accuracy']:.2%} ({info['correct']}/{info['total']})")

    print("\nPer Subfield:")
    for s in sorted(report["by_subfield"]):
        info = report["by_subfield"][s]
        print(f"  {s:25s}: {info['accuracy']:.2%} ({info['correct']}/{info['total']})")

    summary_path = args.result_file.replace(".json", "_eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
