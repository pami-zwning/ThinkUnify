import os
import json
import argparse
import sys
import re
from collections import defaultdict

# ==========================================
# 核心工具函数
# ==========================================

def extract_answer_from_response(response):
    """
    从模型回复中提取答案。
    优先匹配 <answer> 标签内容。
    """
    if not response:
        return ""
    
    # 1. 尝试匹配 <answer>...</answer>
    # re.DOTALL 让 . 可以匹配换行符
    match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    
    candidate = ""
    if match:
        candidate = match.group(1).strip()
    else:
        # 如果没有标签，回退到原来的逻辑，或者视为空
        # 这里为了稳健，如果找不到标签，尝试在整个文本最后找单个字母
        # 但针对你的新数据格式，强烈建议依赖标签
        candidate = response.strip()

    # 2. 从候选文本中提取 A, B, C, D
    # 这一步是为了处理 <answer>Option A</answer> 或 <answer>A.</answer> 这种情况
    # 查找第一个出现的 A-D 字母
    choice_match = re.search(r"[A-D]", candidate, re.IGNORECASE)
    
    if choice_match:
        return choice_match.group(0).upper()
    
    return ""

def load_json_or_jsonl(file_path):
    """通用加载函数"""
    with open(file_path, "r", encoding="utf-8") as f:
        if file_path.endswith(".jsonl"):
            return [json.loads(line) for line in f]
        else:
            return json.load(f)

# ==========================================
# 评测逻辑封装
# ==========================================

def evaluate_json_data(data_list):
    """
    接收数据列表，计算得分。
    
    Args:
        data_list (list): 包含评测数据的列表
        
    Returns:
        dict: 统计结果，结构为 { task_type: {'correct': int, 'total': int, 'acc': float} }
    """
    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for item in data_list:
        # 兼容处理：有些字段可能叫 response，有些叫 model_response
        raw_response = item.get("model_response", item.get("response", ""))
        
        # 处理 list 类型的 response (虽然你的新格式是 string，但保留兼容性)
        if isinstance(raw_response, list) and len(raw_response) > 0:
            raw_response = raw_response[0]
        
        # 提取预测答案
        pred_answer = extract_answer_from_response(str(raw_response))
        
        # 获取标准答案
        gt_answer = item.get("answer", "").strip().upper()
        task_type = item.get("task_type", "unknown")
        
        # 统计
        stats[task_type]["total"] += 1
        if pred_answer == gt_answer:
            stats[task_type]["correct"] += 1
            
    # 计算准确率并整理输出格式
    final_results = {}
    for task, metrics in stats.items():
        total = metrics["total"]
        correct = metrics["correct"]
        acc = (correct / total) if total > 0 else 0.0
        final_results[task] = {
            "correct": correct,
            "total": total,
            "acc": acc
        }
        
    return final_results

def print_report(results):
    """
    格式化打印评测报告
    """
    print("\n" + "="*60)
    print(f"{'Task Type':<40} | {'Acc':<10} | {'Count'}")
    print("-" * 60)
    
    total_correct = 0
    total_count = 0
    
    sorted_tasks = sorted(results.keys())
    
    for task in sorted_tasks:
        metrics = results[task]
        acc_percent = metrics['acc'] * 100
        print(f"{task:<40} | {acc_percent:>6.2f}%    | ({metrics['correct']}/{metrics['total']})")
        
        total_correct += metrics['correct']
        total_count += metrics['total']
        
    print("-" * 60)
    overall_acc = (total_correct / total_count * 100) if total_count > 0 else 0.0
    print(f"{'Overall':<40} | {overall_acc:>6.2f}%    | ({total_correct}/{total_count})")
    print("="*60 + "\n")

# ==========================================
# 命令行入口
# ==========================================

def calculate_score(json_path):
    if not os.path.exists(json_path):
        print(f"Error: Input file not found: {json_path}")
        return

    with open(json_path, "r") as f:
        my_data = json.load(f)

    # 2. 调用评测函数
    # 这会返回一个字典，你可以用它来做后续处理（比如存入数据库，或者画图）
    metrics = evaluate_json_data(my_data)

    # 3. 打印漂亮的报告
    print_report(metrics)

    # 4. 获取具体数值示例
    reconstruction_acc = metrics.get('mental_reconstruction', {}).get('acc', 0)
    print(f"Got specific accuracy: {reconstruction_acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate Multiple Choice Accuracy")
    parser.add_argument('--input_file', type=str, required=True, help='Path to the input JSON/JSONL file')

    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}")
        sys.exit(1)

    print(f"Loading data from: {args.input_file}")
    try:
        # 1. 加载数据
        data = load_json_or_jsonl(args.input_file)
        
        # 2. 计算得分
        results = evaluate_json_data(data)
        
        # 3. 打印报告
        print_report(results)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()