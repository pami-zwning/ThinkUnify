"""
HallusionBench evaluation metrics:
  - qAcc: question-level accuracy (per-question match)
  - fAcc: figure-level accuracy (ALL questions for a figure must be correct)
  - aAcc: harmonic mean of qAcc and fAcc
"""

import json
import argparse
from collections import defaultdict


def compute_metrics(results):
    """Compute qAcc, fAcc, and aAcc from result records."""

    total = len(results)
    if total == 0:
        print("No results to evaluate.")
        return

    # ---- qAcc: question-level accuracy ----
    correct_q = sum(1 for r in results if r.get("correct", False))
    qAcc = correct_q / total

    # ---- fAcc: figure-level accuracy ----
    # Group by (category, subcategory, set_id, figure_id)
    figure_groups = defaultdict(list)
    for r in results:
        key = (r["category"], r["subcategory"], str(r["set_id"]), str(r["figure_id"]))
        figure_groups[key].append(r.get("correct", False))

    total_figures = len(figure_groups)
    correct_figures = sum(1 for qs in figure_groups.values() if all(qs))
    fAcc = correct_figures / total_figures if total_figures > 0 else 0.0

    # ---- aAcc: harmonic mean ----
    if qAcc > 0 and fAcc > 0:
        aAcc = (2 * qAcc * fAcc) / (qAcc + fAcc)
    else:
        aAcc = 0.0

    print("=" * 60)
    print("HallusionBench Results")
    print("=" * 60)
    print(f"  Total questions : {total}")
    print(f"  Total figures   : {total_figures}")
    print(f"  qAcc (question) : {correct_q}/{total} = {100*qAcc:.2f}%")
    print(f"  fAcc (figure)   : {correct_figures}/{total_figures} = {100*fAcc:.2f}%")
    print(f"  aAcc (harmonic) : {100*aAcc:.2f}%")
    print("=" * 60)

    # Per-category breakdown
    cat_correct = defaultdict(int)
    cat_total = defaultdict(int)
    for r in results:
        cat = r["category"]
        cat_total[cat] += 1
        if r.get("correct", False):
            cat_correct[cat] += 1

    print("\nPer-category qAcc:")
    for cat in sorted(cat_total.keys()):
        acc = cat_correct[cat] / cat_total[cat]
        print(f"  {cat}: {cat_correct[cat]}/{cat_total[cat]} = {100*acc:.2f}%")

    return {"qAcc": qAcc, "fAcc": fAcc, "aAcc": aAcc}


def main():
    parser = argparse.ArgumentParser(description="Evaluate HallusionBench results")
    parser.add_argument("--result_file", type=str, required=True,
                        help="Path to the JSON result file from test_benchmark.py")
    args = parser.parse_args()

    with open(args.result_file, "r") as f:
        results = json.load(f)

    compute_metrics(results)


if __name__ == "__main__":
    main()
