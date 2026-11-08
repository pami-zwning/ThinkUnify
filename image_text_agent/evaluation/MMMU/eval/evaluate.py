#!/usr/bin/env python3
"""
MMMU Benchmark evaluation: compute accuracy by macro category and overall.
Reads per-sample result JSONs produced by test_benchmark.py.

Usage:
    python evaluate.py --result_dir /path/to/Results/qwen3vl/MMMU
    python evaluate.py --result_file /path/to/Results/qwen3vl/MMMU/qwen3vl_MMMU_results.json
"""

import argparse
import json
import os
from collections import defaultdict

# ── Subject -> Macro Category mapping ────────────────────────────────

CATEGORY_MAP = {
    'Art & Design': [
        'Art', 'Art_Theory', 'Design', 'Music',
    ],
    'Business': [
        'Accounting', 'Economics', 'Finance', 'Manage', 'Marketing',
    ],
    'Science': [
        'Biology', 'Chemistry', 'Geography', 'Math', 'Physics',
    ],
    'Health & Medicine': [
        'Basic_Medical_Science', 'Clinical_Medicine',
        'Diagnostics_and_Laboratory_Medicine', 'Pharmacy', 'Public_Health',
    ],
    'Humanities & Social Science': [
        'History', 'Literature', 'Psychology', 'Sociology',
    ],
    'Tech & Engineering': [
        'Agriculture', 'Architecture_and_Engineering', 'Computer_Science',
        'Electronics', 'Energy_and_Power', 'Materials', 'Mechanical_Engineering',
    ],
}

# Build reverse map: subject -> category
SUBJECT_TO_CATEGORY = {}
for category, subjects in CATEGORY_MAP.items():
    for subj in subjects:
        SUBJECT_TO_CATEGORY[subj] = category


def load_results(result_dir=None, result_file=None):
    """Load all per-sample results from a directory of JSONs or a single combined file."""
    results = []
    if result_file and os.path.exists(result_file):
        with open(result_file, 'r') as f:
            data = json.load(f)
        if isinstance(data, list):
            results = data
        else:
            results = [data]
    elif result_dir and os.path.isdir(result_dir):
        per_sample_dir = os.path.join(result_dir, "per_sample")
        search_dir = per_sample_dir if os.path.isdir(per_sample_dir) else result_dir
        for fname in os.listdir(search_dir):
            if fname.endswith('.json'):
                fpath = os.path.join(search_dir, fname)
                try:
                    with open(fpath, 'r') as f:
                        results.append(json.load(f))
                except Exception:
                    pass
    return results


def evaluate(results):
    """Compute per-subject, per-category, and overall accuracy."""
    # Per-subject stats
    subject_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    for r in results:
        subj = r.get('subject', 'unknown')
        gold = r.get('gold_answer', '').strip().upper()
        pred = r.get('predicted', '').strip().upper()
        subject_stats[subj]['total'] += 1
        if pred == gold:
            subject_stats[subj]['correct'] += 1

    # Per-category stats (macro average over subjects within category)
    category_stats = {}
    for category, subjects in CATEGORY_MAP.items():
        cat_correct = 0
        cat_total = 0
        subject_accs = []
        for subj in subjects:
            s = subject_stats.get(subj)
            if s and s['total'] > 0:
                cat_correct += s['correct']
                cat_total += s['total']
                subject_accs.append(s['correct'] / s['total'])
        category_stats[category] = {
            'correct': cat_correct,
            'total': cat_total,
            'accuracy': cat_correct / cat_total if cat_total > 0 else 0.0,
            'macro_accuracy': sum(subject_accs) / len(subject_accs) if subject_accs else 0.0,
        }

    # Overall
    total_correct = sum(s['correct'] for s in subject_stats.values())
    total_count = sum(s['total'] for s in subject_stats.values())
    overall_accuracy = total_correct / total_count if total_count > 0 else 0.0

    # Macro average over all subjects
    all_subject_accs = []
    for subj, s in subject_stats.items():
        if s['total'] > 0:
            all_subject_accs.append(s['correct'] / s['total'])
    macro_overall = sum(all_subject_accs) / len(all_subject_accs) if all_subject_accs else 0.0

    return subject_stats, category_stats, overall_accuracy, macro_overall, total_correct, total_count


def main():
    parser = argparse.ArgumentParser(description="MMMU Benchmark Evaluation")
    parser.add_argument("--result_dir", type=str, default="",
                        help="Directory containing per-sample result JSONs or combined result.")
    parser.add_argument("--result_file", type=str, default="",
                        help="Path to a single combined results JSON file.")
    parser.add_argument("--output", type=str, default="",
                        help="Path to save evaluation summary JSON.")
    args = parser.parse_args()

    results = load_results(result_dir=args.result_dir, result_file=args.result_file)
    if not results:
        print("No results found. Provide --result_dir or --result_file.")
        return

    subject_stats, category_stats, overall_acc, macro_acc, total_correct, total_count = evaluate(results)

    # Print results
    print("=" * 70)
    print("MMMU Benchmark Evaluation Results")
    print("=" * 70)

    print(f"\nTotal samples: {total_count}")
    print(f"Overall accuracy: {total_correct}/{total_count} = {overall_acc * 100:.2f}%")
    print(f"Macro-average accuracy (over subjects): {macro_acc * 100:.2f}%")

    print("\n" + "-" * 70)
    print("Per-Category Results:")
    print("-" * 70)
    for category in CATEGORY_MAP:
        cs = category_stats[category]
        print(f"  {category:35s}  {cs['correct']:4d}/{cs['total']:<4d}  "
              f"Acc={cs['accuracy']*100:6.2f}%  "
              f"Macro={cs['macro_accuracy']*100:6.2f}%")

    print("\n" + "-" * 70)
    print("Per-Subject Results:")
    print("-" * 70)
    for category, subjects in CATEGORY_MAP.items():
        print(f"\n  [{category}]")
        for subj in sorted(subjects):
            s = subject_stats.get(subj, {'correct': 0, 'total': 0})
            acc = s['correct'] / s['total'] * 100 if s['total'] > 0 else 0.0
            print(f"    {subj:45s}  {s['correct']:3d}/{s['total']:<3d}  {acc:6.2f}%")

    # Save summary
    summary = {
        'total_samples': total_count,
        'overall_accuracy': round(overall_acc * 100, 2),
        'macro_accuracy': round(macro_acc * 100, 2),
        'per_category': {
            cat: {
                'accuracy': round(cs['accuracy'] * 100, 2),
                'macro_accuracy': round(cs['macro_accuracy'] * 100, 2),
                'correct': cs['correct'],
                'total': cs['total'],
            }
            for cat, cs in category_stats.items()
        },
        'per_subject': {
            subj: {
                'accuracy': round(s['correct'] / s['total'] * 100, 2) if s['total'] > 0 else 0.0,
                'correct': s['correct'],
                'total': s['total'],
            }
            for subj, s in subject_stats.items()
        },
    }

    output_path = args.output
    if not output_path and args.result_dir:
        output_path = os.path.join(args.result_dir, "MMMU_eval_summary.json")
    elif not output_path and args.result_file:
        output_path = args.result_file.replace('.json', '_eval_summary.json')

    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"\nSummary saved to: {output_path}")


if __name__ == "__main__":
    main()
