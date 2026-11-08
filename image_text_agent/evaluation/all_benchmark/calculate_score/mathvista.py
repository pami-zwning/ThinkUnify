"""MathVista scoring.

Multi-metric:
  - multi_choice: extract letter, exact match
  - free_form float: relative tolerance 5% (or absolute 1e-3)
  - free_form integer: exact match after parsing
  - free_form text: case-insensitive exact match
"""

import os
import json
import re
from collections import defaultdict


def _extract_answer_tag(response):
    if not response:
        return ""
    # 1. Try <answer>...</answer> tag first
    m = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # 2. Fall back to content after the last </think> tag
    parts = re.split(r"</think>", response, flags=re.IGNORECASE)
    if len(parts) > 1:
        after_think = parts[-1].strip()
        if after_think:
            return after_think
        # 3. If nothing after </think>, extract the last number or letter before </think>
        before_think = parts[-2].strip()
        # Try to find the last number or standalone letter (A-D) before </think>
        tokens = re.findall(r"-?\d+(?:\.\d+)?|(?<!\w)[A-Ha-h](?!\w)", before_think)
        if tokens:
            return tokens[-1].upper() if re.match(r"^[A-Ha-h]$", tokens[-1]) else tokens[-1]
        # Otherwise return the last non-empty line
        lines = [l.strip() for l in before_think.splitlines() if l.strip()]
        return lines[-1] if lines else before_think
    # 4. Otherwise return the full response
    return response.strip()


def _extract_letter(response, num_choices=4):
    candidate = _extract_answer_tag(response)
    valid = "ABCDEFGH"[:num_choices]
    m = re.search(f"[{valid}]", candidate, re.IGNORECASE)
    return m.group(0).upper() if m else ""


def _is_numeric(s):
    try:
        float(str(s).replace(",", "").replace("%", "").replace("$", "").strip())
        return True
    except (ValueError, TypeError):
        return False


def _parse_numeric(s):
    return float(str(s).replace(",", "").replace("%", "").replace("$", "").strip())


def _judge(rec):
    pred_raw = rec.get("model_response", "") or ""
    pred = _extract_answer_tag(pred_raw)
    gold = (rec.get("gold_answer", "") or "").strip()
    qtype = rec.get("question_type", "free_form")
    atype = rec.get("answer_type", "text")
    choices = rec.get("choices") or []

    if qtype == "multi_choice" and choices:
        # Gold is the textual choice; map to letter index
        try:
            gold_idx = choices.index(gold)
            gold_letter = "ABCDEFGH"[gold_idx]
        except ValueError:
            # Gold may already be a letter
            gold_letter = gold[:1].upper()
        pred_letter = _extract_letter(pred_raw, num_choices=len(choices))
        return pred_letter == gold_letter

    if atype in ("float", "integer") and _is_numeric(pred) and _is_numeric(gold):
        p, g = _parse_numeric(pred), _parse_numeric(gold)
        if abs(p - g) <= 1e-3:
            return True
        if g == 0:
            return p == 0
        return abs(p - g) / abs(g) <= 0.05

    return pred.strip().lower() == gold.lower()


def calculate_score(json_path):
    if not os.path.exists(json_path):
        print(f"[MathVista] Error: Input file not found: {json_path}")
        return
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    total = 0
    correct = 0
    by_qtype = defaultdict(lambda: [0, 0])  # [correct, total]
    by_atype = defaultdict(lambda: [0, 0])

    for rec in records:
        if not rec.get("gold_answer"):
            continue
        total += 1
        is_correct = _judge(rec)
        if is_correct:
            correct += 1
        qt = rec.get("question_type", "unknown")
        at = rec.get("answer_type", "unknown")
        by_qtype[qt][1] += 1
        by_atype[at][1] += 1
        if is_correct:
            by_qtype[qt][0] += 1
            by_atype[at][0] += 1

    acc = correct / total * 100 if total > 0 else 0.0

    print("=" * 60)
    print("MathVista Evaluation")
    print("=" * 60)
    print(f"Total:   {total}")
    print(f"Overall: {correct}/{total} = {acc:.2f}%")
    for qt, (c, t) in sorted(by_qtype.items()):
        print(f"  [qtype={qt}]: {c}/{t} = {(c/t*100 if t else 0):.2f}%")
    for at, (c, t) in sorted(by_atype.items()):
        print(f"  [atype={at}]: {c}/{t} = {(c/t*100 if t else 0):.2f}%")
    print("=" * 60)

    summary = {
        "total": total,
        "correct": correct,
        "accuracy": round(acc, 2),
        "by_question_type": {k: {"correct": v[0], "total": v[1]} for k, v in by_qtype.items()},
        "by_answer_type": {k: {"correct": v[0], "total": v[1]} for k, v in by_atype.items()},
    }
    summary_path = json_path.replace(".json", "_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")
