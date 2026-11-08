"""ChartQA scoring module.

Metric: relaxed accuracy
  - Numeric answers: correct if relative error <= 5%
  - String answers: case-insensitive exact match after strip
Breakdown: overall / human subset / machine subset (human_or_machine flag).
"""

import os
import json
import re


def extract_answer(response):
    """Extract short answer from model response.

    Priority:
      1. <answer>...</answer> tag content
      2. Everything after the LAST </think> tag
      3. Full response stripped
    """
    if not response:
        return ""

    # 1. 如果有 <answer> 标签，优先提取
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # 2. 【核心修复】以 </think> 作为菜刀切分字符串，永远拿最后一块（即最后一个 </think> 后面的内容）
    if "</think>" in response:
        ans = response.split("</think>")[-1].strip()
        
        # 去掉模型偶尔在数字或单词结尾加的句号，防止误判
        if ans.endswith('.'):
            ans = ans[:-1]
            
        return ans

    # 3. 如果什么标签都没有，返回原文
    return response.strip()


def is_numeric(s):
    try:
        cleaned = s.replace(",", "").replace("%", "").replace("$", "").strip()
        float(cleaned)
        return True
    except (ValueError, TypeError):
        return False


def parse_numeric(s):
    cleaned = s.replace(",", "").replace("%", "").replace("$", "").strip()
    return float(cleaned)


def relaxed_accuracy(prediction, ground_truths):
    """Return True if prediction matches any ground truth under the relaxed metric."""
    pred = (prediction or "").strip()
    if not pred:
        return False

    for gt in ground_truths:
        gt = str(gt).strip()
        if is_numeric(pred) and is_numeric(gt):
            pred_val = parse_numeric(pred)
            gt_val = parse_numeric(gt)
            if gt_val == 0:
                if pred_val == 0:
                    return True
            else:
                if pred_val == gt_val:
                    return True
        else:
            if pred.lower() in gt.lower() or gt.lower() in pred.lower():
                return True
    return False


def evaluate_records(records):
    total = 0
    total_correct = 0
    human_count = 0
    human_correct = 0
    machine_count = 0
    machine_correct = 0

    for rec in records:
        pred = extract_answer(rec.get("model_response", "") or "")
        print(f'the answer is {pred}')
        label = rec.get("label", [])
        if isinstance(label, str):
            label = [label]
        hom = rec.get("human_or_machine", -1)
        print(f'the label is {label}')

        correct = relaxed_accuracy(pred, label)
        print(f'the correct is {correct}')
        total += 1
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

    return {
        "total": total,
        "total_correct": total_correct,
        "overall_accuracy": (total_correct / total * 100) if total > 0 else 0.0,
        "human_count": human_count,
        "human_correct": human_correct,
        "human_accuracy": (human_correct / human_count * 100) if human_count > 0 else None,
        "machine_count": machine_count,
        "machine_correct": machine_correct,
        "machine_accuracy": (machine_correct / machine_count * 100) if machine_count > 0 else None,
    }


def calculate_score(json_path):
    if not os.path.exists(json_path):
        print(f"[ChartQA] Error: Input file not found: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    m = evaluate_records(records)

    print("=" * 60)
    print("ChartQA Evaluation Results (Relaxed Accuracy)")
    print("=" * 60)
    print(f"Total samples:    {m['total']}")
    print(
        f"Overall accuracy: {m['total_correct']}/{m['total']} = "
        f"{m['overall_accuracy']:.2f}%"
    )
    if m["human_count"] > 0:
        print(
            f"Human subset:     {m['human_correct']}/{m['human_count']} = "
            f"{m['human_accuracy']:.2f}%"
        )
    if m["machine_count"] > 0:
        print(
            f"Machine subset:   {m['machine_correct']}/{m['machine_count']} = "
            f"{m['machine_accuracy']:.2f}%"
        )
    print("=" * 60)

    summary_path = json_path.replace(".json", "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ChartQA relaxed-accuracy scoring")
    parser.add_argument("--input_file", type=str, required=True)
    args = parser.parse_args()
    calculate_score(args.input_file)
