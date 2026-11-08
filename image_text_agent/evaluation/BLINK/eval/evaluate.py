"""
BLINK Benchmark evaluation metrics.

Computes per-subtask accuracy and macro-averaged overall accuracy.
Can be run standalone or imported by test_benchmark.py.
"""

import json
import argparse
from collections import defaultdict


SUBTASKS = [
    'Art_Style', 'Counting', 'Forensic_Detection', 'Functional_Correspondence',
    'IQ_Test', 'Jigsaw', 'Multi-view_Reasoning', 'Object_Localization',
    'Relative_Depth', 'Relative_Reflectance', 'Semantic_Correspondence',
    'Spatial_Relation', 'Visual_Correspondence', 'Visual_Similarity',
]


def compute_blink_accuracy(results, split_filter=None):
    """
    Compute per-task and overall accuracy for BLINK results.

    Args:
        results: list of result dicts, each with 'task', 'split', 'gold_answer', 'prediction'
        split_filter: if set, only evaluate on this split ('val' or 'test')

    Returns:
        dict with 'by_task', 'average', 'total_correct', 'total_samples'
    """
    by_task = defaultdict(lambda: {"correct": 0, "total": 0})

    for r in results:
        if split_filter and r.get("split") != split_filter:
            continue

        task = r.get("task", "unknown")
        by_task[task]["total"] += 1

        gold = r.get("gold_answer", "")
        pred = r.get("prediction", "")

        if gold and pred and gold.strip() == pred.strip():
            by_task[task]["correct"] += 1

    # Compute accuracy per task
    for task in by_task:
        total = by_task[task]["total"]
        correct = by_task[task]["correct"]
        by_task[task]["accuracy"] = correct / total if total > 0 else 0.0

    # Macro-average over all tasks that have samples
    task_accs = [info["accuracy"] for info in by_task.values() if info["total"] > 0]
    average = sum(task_accs) / len(task_accs) if task_accs else 0.0

    total_correct = sum(info["correct"] for info in by_task.values())
    total_samples = sum(info["total"] for info in by_task.values())

    return {
        "by_task": dict(by_task),
        "average": average,
        "total_correct": total_correct,
        "total_samples": total_samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate BLINK predictions")
    parser.add_argument("--result_file", type=str, required=True,
                        help="Path to the JSON result file from test_benchmark.py")
    parser.add_argument("--split", type=str, default=None, choices=["val", "test"],
                        help="Filter by split (default: all)")
    args = parser.parse_args()

    with open(args.result_file, "r") as f:
        results = json.load(f)

    print(f"Loaded {len(results)} results from {args.result_file}")
    if args.split:
        print(f"Filtering to split: {args.split}")

    report = compute_blink_accuracy(results, split_filter=args.split)

    print("\n" + "=" * 65)
    print(f"  BLINK Benchmark Results" + (f" ({args.split} split)" if args.split else ""))
    print("=" * 65)
    print(f"  {'Subtask':<35s} {'Correct':>7s} {'Total':>7s} {'Acc':>8s}")
    print("-" * 65)

    for task in SUBTASKS:
        if task in report["by_task"]:
            info = report["by_task"][task]
            print(f"  {task:<35s} {info['correct']:>7d} {info['total']:>7d} {info['accuracy']:>7.2%}")

    # Any tasks not in standard list
    for task in sorted(report["by_task"].keys()):
        if task not in SUBTASKS:
            info = report["by_task"][task]
            print(f"  {task:<35s} {info['correct']:>7d} {info['total']:>7d} {info['accuracy']:>7.2%}")

    print("-" * 65)
    print(f"  {'Macro Average':<35s} {report['total_correct']:>7d} {report['total_samples']:>7d} {report['average']:>7.2%}")
    print("=" * 65)

    # Save summary JSON
    summary_path = args.result_file.replace(".json", "_eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
