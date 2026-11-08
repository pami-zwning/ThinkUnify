import os
import base64
from binascii import Error

from openai import OpenAI
from typing import Optional, Dict, Any, Union, List
import logging
import matplotlib.pyplot as plt
from PIL import Image
import io
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
# import cairosvg
import tempfile
from tqdm import tqdm
import re
import concurrent.futures
import json
import os
from openai import AzureOpenAI
from demo_prompts import demo_prompt_instrument, demo_prompt_refcoco, demo_prompt_contrast, demo_prompt_jigsaw, demo_prompt_color, demo_prompt_maze, demo_prompt_math, demo_prompt_symbolic, demo_prompt_word_search, demo_prompt_rotation_game, demo_prompt_spot_difference, demo_prompt_visual_search

from mimetypes import guess_type
import uuid
from tqdm import tqdm
import argparse

key = 'YOUR_OPENAI_API_KEY'
client = AzureOpenAI(
    azure_endpoint = "https://duomotai.openai.azure.com/",
    api_key=key,
    api_version="2025-01-01-preview"
)

def create_test_prompt(demo_prompt, query, response):
    demo_prompt = demo_prompt.strip()
    test_prompt = f"Question: {query}\n\nModel response: {response}"
    full_prompt = f"{demo_prompt}\n\n{test_prompt}\n\nExtracted answer: "
    return full_prompt


def get_response_text(model, text):

    patience = 100
    while patience > 0:
        patience -= 1
        try:
            # client.chat.completions.create
            response = client.chat.completions.create(
                model=model,
                # use model in list [gpt-4o, gpt-4o-mini, o1, o3-mini, o3, o4-mini, gpt-4.1, gpt-4.1-mini, gpt-4.1-nano]
                messages=[{"role": "system", "content": [
                    {
                        "type": "text",
                        "text": " Assistant is a large language model trained by OpenAI."
                    }
                ]}, {"role": "user", "content": [
                    {
                        "type": "text",
                        "text": text
                    }
                ]}]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(e)


def api_call_worker(api_name, full_prompt):
    return get_response_text(api_name, full_prompt)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)
    parser.add_argument('--api_name', type=str)
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--workers', type=int, default=1)

    args = parser.parse_args()

    path = os.path.join(args.input_path, "Jsons", f"{args.model_name}.json")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"loading {path} file!")

    tasks_to_run = []
    ocr_items = []
    aa =[]
    for item in tqdm(data):
        if 'task' in item.keys():
            task = item['task']
        else:
            task = item['category']
        if task=='ocr':
            ocr_items.append(item)
            continue

        if task=='maze':
            demo_prompt = demo_prompt_maze
        elif task=='color':
            demo_prompt = demo_prompt_color
        elif task=='math':
            demo_prompt = demo_prompt_math
        elif task=='word_search':
            demo_prompt = demo_prompt_word_search
        elif task == 'jigsaw':
            demo_prompt = demo_prompt_jigsaw
        elif task=='rotation_game':
            demo_prompt = demo_prompt_rotation_game
        elif task=='refcoco':
            demo_prompt = demo_prompt_refcoco
        elif task=='spot_difference':
            demo_prompt = demo_prompt_spot_difference
        elif task=='visual_search':
            demo_prompt = demo_prompt_visual_search
        elif task=='symbolic':
            demo_prompt = demo_prompt_symbolic
        elif task=='contrast':
            demo_prompt = demo_prompt_contrast
        elif task=='instrument':
            demo_prompt = demo_prompt_instrument
        else:
            raise Exception("Incorrect task name.")
        # if 'extracted_answer' in item.keys():
        #     continue
        text = item['prompt']
        response = item['model_response']
        assert response!=None

        full_prompt = create_test_prompt(demo_prompt, text, response)
        tasks_to_run.append((item, full_prompt))

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
    
        # 使用 executor.submit 提交任务，并建立 Future 到原始 Item 的映射
        future_to_item = {
            executor.submit(api_call_worker,  args.api_name, full_prompt): original_item
            for original_item, full_prompt in tasks_to_run
        }
        
        # 遍历已完成的任务并合并结果
        results_iterator = tqdm(
            concurrent.futures.as_completed(future_to_item), 
            total=len(future_to_item), 
            desc="Fetching Answers Except OCR"
        )
        
        for future in results_iterator:
            # 获取原始 Item 的引用
            original_item = future_to_item[future]
            
            try:
                # 获取线程结果
                extracted_answer = future.result()
                original_item['extracted_answer'] = extracted_answer
            except Exception as exc:
                # 记录错误
                print(f"\n[Error] Task failed for item: {exc}")
                original_item['extracted_answer'] = f"API_ERROR: {exc}"
            
            # 将更新后的 item 加入最终结果列表
            aa.append(original_item)

    # output_path = f'/mnt/afs/ningzhiwei/code/rl/TIR-Bench/sft/sft_result_{type}_output.json'
    aa = aa+ocr_items
    output_path = os.path.join(args.input_path, "Extract_Answer", f"{args.model_name}.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path , 'w') as f:
        json.dump(aa, f,ensure_ascii=False, indent=2)

    print(f"Extract Result save to {os.path.abspath(output_path)}")

if __name__ == "__main__":
    main()