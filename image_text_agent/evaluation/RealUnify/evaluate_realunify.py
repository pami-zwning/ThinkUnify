import json
import os
from tqdm import tqdm  # 建议安装 tqdm 用于显示进度条: pip install tqdm

import argparse
import sys
import random
from openai import OpenAI
import re

import torch
import transformers
import torchvision.transforms as T
from PIL import Image
from transformers import set_seed
from torchvision.transforms.functional import InterpolationMode

# 设置环境变量
def setup_environment():
    """设置PYTHONPATH环境变量"""
    tir_path = "/mnt/afs/nijingcheng/workspace/codes/benchmarks/image_text_agent/evaluation/TIR-Bench"
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

dataset_root = "/mnt/afs/share_data/nijingcheng/RealUnify"
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
    # 步骤 1: 寻找 Question: 并截取其后的内容
    # 如果没找到 Question:，则保留原状
    if "Question:" in prompt_text:
        # 使用 split 分割并取最后一部分，去掉前面的说明文字
        prompt_text = prompt_text.split("Question:", 1)[-1].strip()
    
    # 步骤 2: 去掉结尾的 "The best answer is:"
    # 使用 re.sub 确保只匹配末尾，并忽略可能存在的空格
    prompt_text = re.sub(r"The best answer is:\s*$", "", prompt_text, flags=re.IGNORECASE).strip()
    
    return prompt_text

def process_geu_task(input_json_path, output_json_path, client_urls, args):
    # 1. 读取原始数据
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data_list = json.load(f)
    
    print(f"Loaded {len(data_list)} tasks from {input_json_path}")

    # 2. 遍历列表处理任务
    # 使用 tqdm 显示进度
    for item in tqdm(data_list, desc="Processing GEU Tasks"):
        # if item["task_type"] != "cognitive_navigation" or "maze" not in item.get('input_image'):
        #     continue
        
        # --- Step 1: Image Manipulation (Editing) ---
        input_img = item.get('input_image')
        edit_prompt = item.get('edit_prompt')
        
        # 调用编辑模型接口
        # 注意：实际运行时需确保 model_image_editing 返回有效的路径字符串
        # input_img = os.path.join(dataset_root, input_img.lstrip('/'))
        item["input_image"] = input_img.replace("/ytech_m2v5_hdd/workspace/kling_mm/shiyang06/Data/RealUnify", dataset_root)
        generated_image_path = None
        
        # 将生成的图片路径保存到 'edit_image' 字段
        # 为了演示代码流程通畅，若函数未实现返回 None，这里做一个防御性处理
        if generated_image_path:
            item['edit_image'] = generated_image_path
        else:
            # 仅作测试用：如果没有实际生成图片，这里可能需要根据你的逻辑填入占位符
            item['edit_image'] = "" 
        # --- Step 2: Visual Understanding ---        
        # 只有当存在编辑图片路径时才进行理解（或者根据需求即使为空也尝试）
        input_args = (item, client_urls, args)
        model_response = worker_process(input_args)
        
        # 将模型回复保存到 'response' 字段
        item['response'] = model_response["model_response"] if model_response else ""
        item['mid_process'] = model_response["mid_process"] if model_response else ""

    # 3. 保存结果文件
    # 确保输出目录存在
    output_dir = os.path.dirname(output_json_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=4, ensure_ascii=False)
        
    print(f"Processing complete. Results saved to {output_json_path}")
    print("Don't forget to update 'eval/eval_understanding.py' with the new model name and result file path.")

def worker_process(input_args):
    item, client_urls, args = input_args
    client_url = random.choice(client_urls)
    
    client = OpenAI(api_key="EMPTY", base_url=client_url)
    try:
        model_name = client.models.list().data[0].id
    except:
        model_name = "default-model"
    
    image_path_list = [item["input_image"]]
    if item["edit_image"]:
        image_path_list.append(item["image_2"])
    # prompt = item["evaluation_prompt"]
    prompt = refine_prompt(item["evaluation_prompt"])
    print(prompt)

    result_text, full_history, all_images_list = model_response_func(client, model_name, prompt, image_path_list, None, args)    
    if result_text:
        item["model_response"] = result_text
    else:
        item["model_response"] = "[ERROR] Empty Response"

    print(item["model_response"])
    
    if full_history:
        item["mid_process"] = clean_message_history(full_history)
    else:
        item["mid_process"] = []

    if args.save_interim:
        item["all_images_list"] = all_images_list

    return item

# ==========================================
# Execution Entry
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='测评的模型名字')
    parser.add_argument('--ip_list', type=str, nargs='+',help='vllm api 列表')
    parser.add_argument('--save_interim', action='store_true', default=False, help='是否需要保存过程中的中间结果文件。')
    parser.add_argument('--result_path', type=str, help='保存测评结果路径')
    parser.add_argument('--workers', type=int, help='并发数')
    parser.add_argument('--max_turns', type=int, help='最大toolcall轮数')
    parser.add_argument('--need_summary', action='store_true', default=False, help='是否需要强制给出一个选项。')

    args = parser.parse_args()
    

    # 准备数据
    client_urls = [f"http://{host_ip}:8000/v1" for host_ip in args.ip_list]
    save_path = os.path.join(args.result_path, "Jsons", f'{args.model_name}.json')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if args.save_interim:
        image_save_dir = os.path.join(args.result_path, "Images", args.model_name)
        args.image_save_dir = image_save_dir
        os.makedirs(image_save_dir, exist_ok=True)

    # 定义输入输出文件路径
    INPUT_FILE = "/mnt/afs/share_data/nijingcheng/RealUnify/GEU_step.json"
    OUTPUT_FILE = save_path  # 建议保存为新文件，避免直接覆盖原文件

    # 检查文件是否存在
    if os.path.exists(INPUT_FILE):
        process_geu_task(INPUT_FILE, OUTPUT_FILE, client_urls, args)
    else:
        print(f"Error: {INPUT_FILE} not found.")
