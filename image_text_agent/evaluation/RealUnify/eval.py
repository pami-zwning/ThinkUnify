import json
import os
from tqdm import tqdm  # 建议安装 tqdm 用于显示进度条: pip install tqdm

import argparse
import sys
import random
from openai import OpenAI
import re
# 引入多线程模块
from concurrent.futures import ThreadPoolExecutor

import torch
import transformers
import torchvision.transforms as T
from PIL import Image
from transformers import set_seed
from torchvision.transforms.functional import InterpolationMode

# 设置环境变量
def setup_environment():
    """设置PYTHONPATH环境变量"""
    tir_path = "../TIR-Bench"
    if tir_path not in sys.path:
        sys.path.insert(0, tir_path)
    
    # 也设置环境变量
    current_pythonpath = os.environ.get('PYTHONPATH', '')
    if tir_path not in current_pythonpath:
        if current_pythonpath:
            os.environ['PYTHONPATH'] = f"{tir_path}:{current_pythonpath}"
        else:
            os.environ['PYTHONPATH'] = tir_path
    
    print(f"✅ 已设置PYTHONPATH: {os.environ.get('PYTHONPATH')}")

setup_environment()

from qwen3vl_eval import model_response_func, clean_message_history

dataset_root = "/mnt/afs/share_data/mashengnan/o3like/image_text_cot/benchmark/RealUnify"
# ==========================================
# Part 1: LLM/Model Interface Definitions
# (仅定义接口和注释，不包含具体实现)
# ==========================================

# ==========================================
# Part 2: Main Processing Logic (GEU Task)
# ==========================================

def refine_prompt(prompt_text):
    """
    修改 Prompt：
    1. 如果存在 "Question:"，则去掉其之前的所有内容（包括 Question 本身）。
    2. 去掉结尾处的 "The best answer is:"。
    """
    if "Question:" in prompt_text:
        prompt_text = prompt_text.split("Question:", 1)[-1].strip()
    
    prompt_text = re.sub(r"The best answer is:\s*$", "", prompt_text, flags=re.IGNORECASE).strip()
    
    return prompt_text

def worker_process(input_args):
    """
    处理单个 Item 的完整逻辑：包括路径替换、模型推理、结果回填
    """
    item, client_urls, args = input_args
    
    # --- 1. Image Manipulation (Path logic moved here) ---
    input_img = item.get('input_image')
    # 路径替换逻辑
    item["input_image"] = input_img.replace("/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Data/RealUnify", dataset_root)
    
    generated_image_path = None
    if generated_image_path:
        item['edit_image'] = generated_image_path
    else:
        item['edit_image'] = ""

    # --- 2. Model Inference ---
    # 随机选择一个 API URL
    client_url = random.choice(client_urls)
    client = OpenAI(api_key="EMPTY", base_url=client_url)
    try:
        model_name = client.models.list().data[0].id
    except:
        model_name = "default-model"
    
    image_path_list = [item["input_image"]]
    if item["edit_image"]:
        image_path_list.append(item["image_2"]) # 注意：这里原代码使用了 item["image_2"]，请确保数据中包含此字段

    prompt = refine_prompt(item["evaluation_prompt"])

    try:
        result_text, full_history, all_images_list = model_response_func(client, model_name, prompt, image_path_list, args)    
        
        if result_text:
            item["model_response"] = result_text
        else:
            item["model_response"] = "[ERROR] Empty Response"
        
        if full_history:
            item["mid_process"] = clean_message_history(full_history)
        else:
            item["mid_process"] = []

        if args.save_interim:
            item["all_images_list"] = all_images_list
            
        # 回填到顶层字段（兼容原代码逻辑）
        item['response'] = item["model_response"]
        
    except Exception as e:
        print(f"Error processing item {item.get('id', 'unknown')}: {e}")
        item["model_response"] = f"[ERROR] Exception: {str(e)}"
        item['response'] = f"[ERROR] Exception: {str(e)}"
        item["mid_process"] = []

    return item

def process_geu_task(input_json_path, output_json_path, client_urls, args):
    # 1. 读取原始数据
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    
    print(f"Loaded {len(data_list)} tasks from {input_json_path}")
    print(f"Starting execution with {args.workers} workers...")

    # 2. 准备任务参数列表
    # 将每个任务所需的参数打包，以便传递给线程池
    task_args_list = [(item, client_urls, args) for item in data_list]

    # 3. 多线程处理
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # 使用 executor.map 可以保持结果顺序与输入顺序一致
        # tqdm 包装迭代器以显示进度
        results_iterator = executor.map(worker_process, task_args_list)
        
        # 将迭代器转换为列表以触发执行并显示进度
        results = list(tqdm(results_iterator, total=len(data_list), desc="Processing GEU Tasks"))

    # 4. 保存结果文件
    output_dir = os.path.dirname(output_json_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(output_json_path, 'w', encoding='utf-8') as f:
        # results 已经是处理完的 item 列表
        json.dump(results, f, indent=4, ensure_ascii=False)
        
    print(f"Processing complete. Results saved to {output_json_path}")
    print("Don't forget to update 'eval/eval_understanding.py' with the new model name and result file path.")

# ==========================================
# Execution Entry
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='测评的模型名字')
    parser.add_argument('--ip_list', type=str, nargs='+', help='vllm api 列表')
    parser.add_argument('--save_interim', action='store_true', default=False, help='是否需要保存过程中的中间结果文件。')
    parser.add_argument('--result_path', type=str, help='保存测评结果路径')
    parser.add_argument('--workers', type=int, default=8, help='并发数 (默认为8)')
    parser.add_argument('--max_turns', type=int, help='最大toolcall轮数')
    parser.add_argument('--need_summary', action='store_true', default=False, help='是否需要强制给出一个选项。')
    parser.add_argument('--use_tools', action='store_true', default=False, help='是否需要调用工具。')

    args = parser.parse_args()
    
    if not args.workers:
        args.workers = 1 # 防止为 None

    # 准备数据
    client_urls = [f"http://{host_ip}:8000/v1" for host_ip in args.ip_list]
    save_path = os.path.join(args.result_path, "Jsons", f'{args.model_name}.json')
    
    # 确保保存路径存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if args.save_interim:
        image_save_dir = os.path.join(args.result_path, "Images", args.model_name)
        args.image_save_dir = image_save_dir
        os.makedirs(image_save_dir, exist_ok=True)

    # 定义输入输出文件路径
    INPUT_FILE = "/mnt/afs/share_data/mashengnan/o3like/image_text_cot/benchmark/RealUnify/GEU_step.json"
    OUTPUT_FILE = save_path 

    # 检查文件是否存在
    if os.path.exists(INPUT_FILE):
        process_geu_task(INPUT_FILE, OUTPUT_FILE, client_urls, args)
    else:
        print(f"Error: {INPUT_FILE} not found.")