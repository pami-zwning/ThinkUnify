"""
MMStar evaluation metrics:
  - Overall accuracy
  - Per-category accuracy (6 categories)
  - Per l2_category accuracy
"""

import json
import argparse
from collections import defaultdict


def compute_metrics(results):
    """Compute overall, per-category, and per-l2_category accuracy."""

    total = len(results)
    if total == 0:
        print("No results to evaluate.")
        return

    correct = sum(1 for r in results if r.get("correct", False))
    overall_acc = correct / total

    print("=" * 60)
    print("MMStar Results")
    print("=" * 60)
    print(f"  Total samples    : {total}")
    print(f"  Overall accuracy : {correct}/{total} = {100*overall_acc:.2f}%")
    print("=" * 60)

    # Per-category accuracy
    cat_correct = defaultdict(int)
    cat_total = defaultdict(int)
    for r in results:
        cat = r["category"]
        cat_total[cat] += 1
        if r.get("correct", False):
            cat_correct[cat] += 1

    print("\nPer-category accuracy:")
    for cat in sorted(cat_total.keys()):
        acc = cat_correct[cat] / cat_total[cat]
        print(f"  {cat}: {cat_correct[cat]}/{cat_total[cat]} = {100*acc:.2f}%")

    # Per l2_category accuracy
    l2_correct = defaultdict(int)
    l2_total = defaultdict(int)
    for r in results:
        l2 = r["l2_category"]
        l2_total[l2] += 1
        if r.get("correct", False):
            l2_correct[l2] += 1

    print("\nPer l2_category accuracy:")
    for l2 in sorted(l2_total.keys()):
        acc = l2_correct[l2] / l2_total[l2]
        print(f"  {l2}: {l2_correct[l2]}/{l2_total[l2]} = {100*acc:.2f}%")

    return {"overall_accuracy": overall_acc}


def main():
    parser = argparse.ArgumentParser(description="Evaluate MMStar results")
    parser.add_argument("--result_file", type=str, required=True,
                        help="Path to the JSON result file from test_benchmark.py")
    args = parser.parse_args()

    with open(args.result_file, "r") as f:
        results = json.load(f)

    compute_metrics(results)


if __name__ == "__main__":
    main()
