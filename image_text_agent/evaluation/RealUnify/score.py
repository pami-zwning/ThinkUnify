import os
import json
import argparse
import sys
import re
from contextlib import redirect_stdout


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def load_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return [json.loads(line) for line in file]


def save_jsonl(data, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as file:
        for item in data:
            file.write(json.dumps(item, ensure_ascii=False) + "\n")


def extract_characters_regex(response):
    response = response.strip()
    answer_prefixes = [
        "The best answer is",
        "The correct answer is",
        "The answer is",
        "The answer",
        "The best option is",
        "The correct option is",
        "Best answer:",
        "Best option:",
        "Answer",
        "Answer is",
    ]
    for answer_prefix in answer_prefixes:
        response = response.replace(answer_prefix, "")

    if len(response.split()) > 10 and not re.search("[ABCD]", response):
        return ""

    matches = re.search(r"[ABCD]", response)
    if matches is None:
        return ""
    return matches[0]


def cal_multiple_choice_acc(input_file):
    try:
        if input_file.endswith(".json"):
            data = load_json(input_file)
        elif input_file.endswith(".jsonl"):
            data = load_jsonl(input_file)

        task = []
        task_correct_num = {}

        for item in data:
            model_response = item["response"]
            if type(model_response) == list:
                model_response = model_response[0]
            filter_resp = extract_characters_regex(model_response)
            task_type = item["task_type"]
            if task_type not in task:
                task.append(task_type)
            if filter_resp == item["answer"]:
                task_correct_num[task_type] = task_correct_num.get(task_type, 0) + 1

        print("===" * 20)
        for task_type in task:
            print("---" * 20)
            correct_num = task_correct_num.get(task_type, 0)
            acc = correct_num / 100.0
            print(f"Task: {task_type}, Acc: {acc:.4f} ({correct_num}/100)")
        total_correct_num = sum(task_correct_num.values())
        total_acc = total_correct_num / (100.0 * len(task)) if task else 0
        print("---" * 20)
        print(f"Total Acc: {total_acc:.4f} ({total_correct_num}/{100 * len(task)})")

    except Exception as e:
        print(f"Error processing for model: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Accuracy")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input JSON file')

    args = parser.parse_args()

    # 2. 检查文件是否存在 (可选的安全检查)
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)

    # 3. 直接调用函数
    print(f"Processing file: {args.input_file}")
    cal_multiple_choice_acc(args.input_file)

    print("Finished...")