import json
import requests
import os
import base64
import random
import math
import copy
import time
import argparse
from io import BytesIO
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
from openai import OpenAI

from desc import TOOLS, SYSTEM_ROLE_PROMPT
from utils import run_toolcall, toolcall_message_convert
from infer import *
from calculate_score import vsp, mira, realunify, unimmmu, tir

# ==========================================
# 配置区域
# ==========================================

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://duomotai.openai.azure.com/")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "YOUR_AZURE_OPENAI_API_KEY")
AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-2024-11-20")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")

# 路径配置
Benchmark_tool_data = {
    'TIR':'/mnt/afs/share_data/mashengnan/o3like/image_text_cot/benchmark/TIR-Bench/TIR-Bench_data.jsonl',
    'MIRA':'/mnt/afs/share_data/mashengnan/o3like/image_text_cot/benchmark/MIRA/processed_data.jsonl',
    'Unimmmu':'/mnt/afs/share_data/shangziyi/datasets/unimmmu/vqa/unimmmu_tool_call.jsonl',
    'Realunify':'/mnt/afs/share_data/mashengnan/o3like/image_text_cot/benchmark/RealUnify/GEU_step_processed.jsonl',
    'VSP':'/mnt/afs/share_data/mashengnan/o3like/image_text_cot/benchmark/VSP/maze_task_main.jsonl'
}
Benchmark_wotool_data = {
    'TIR':'/mnt/afs/share_data/mashengnan/o3like/image_text_cot/benchmark/TIR-Bench/TIR-Bench_data.jsonl',
    'MIRA':'/mnt/afs/share_data/mashengnan/o3like/image_text_cot/benchmark/MIRA/processed_data.jsonl',
    'Unimmmu':'/mnt/afs/share_data/shangziyi/datasets/unimmmu/vqa/unimmmu_direct.jsonl',
    'Realunify':'/mnt/afs/share_data/mashengnan/o3like/image_text_cot/benchmark/RealUnify/GEU_step_processed.jsonl',
    'VSP':'/mnt/afs/share_data/mashengnan/o3like/image_text_cot/benchmark/VSP/maze_task_main_other_models.jsonl'
}

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, help='测评的模型名字')
    parser.add_argument('--ip_list', type=str, nargs='+',help='vllm api 列表')
    parser.add_argument('--save_interim', action='store_true', default=False, help='是否需要保存过程中的中间结果文件。')
    parser.add_argument('--result_path', type=str, help='保存测评结果路径')
    parser.add_argument('--workers', type=int, help='并发数')
    parser.add_argument('--max_turns', type=int, help='最大toolcall轮数')
    parser.add_argument('--need_summary', action='store_true', default=False, help='是否需要强制给出一个选项。')
    parser.add_argument('--use_tools', action='store_true', default=False, help='是否需要调用工具。')
    parser.add_argument('--temperature',  type=float, default=0.8)
    parser.add_argument('--top_p', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=-1)
    parser.add_argument('--max_tokens',  type=int, default=1024*4)
    parser.add_argument('--tasks',  type=str, nargs='+',help='task 列表')
    parser.add_argument('--close_api',  action='store_true', default=False, help='是否使用闭源api')

    args = parser.parse_args()

    if args.close_api:
        args.use_tools = False
        print(f"args.use_tools: {args.use_tools}")
        print('==== toolcall would be closed when using close api... ====')

    # 1. 准备数据
    client_urls = []
    if args.ip_list:
        client_urls = [f"http://{host_ip}:8000/v1" for host_ip in args.ip_list]

    for task in args.tasks:
        if args.use_tools:
            data_path = Benchmark_tool_data[task]
        else:
            data_path = Benchmark_wotool_data[task]

        save_path = os.path.join(args.result_path, task, f'{args.model_name}.json')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)


        # save cot image path
        task_args = copy.copy(args)
        if task_args.save_interim:
            image_save_dir = os.path.join(task_args.result_path, task, "Images")  # 保存中间的cot image的位置
            task_args.image_save_dir = image_save_dir
            os.makedirs(image_save_dir, exist_ok=True)

        datas = read_jsonl_file(data_path)

        # 断点续传逻辑
        processed_ids = set()
        aa = []
        if os.path.exists(save_path):
            try:
                tested_data = read_json_file(save_path)
                filtered_data = [item for item in tested_data if 'ERROR' not in item['model_response']]
                with open(save_path, 'w') as f:
                    json.dump(filtered_data, f, ensure_ascii=False, indent=2)

                with open(save_path, 'r') as f:
                    aa = json.load(f)
                    processed_ids = {x["hash_uid"] for x in aa}
            except json.JSONDecodeError:
                print("JSON file corrupted, starting over or handle manually.")
                aa = []

        print(f"Loaded {len(aa)} processed samples.")
        print(f"Total samples: {len(datas)}")

        remaining = [x for x in datas if x["hash_uid"] not in processed_ids]
        print(f"Remaining to process: {len(remaining)}")

        if remaining:
            # 2. 准备 Worker 参数
            worker_args = [(item, client_urls, task_args) for item in remaining]
            
            with Pool(task_args.workers) as pool:
                processed_count = 0
                for result in tqdm(pool.imap_unordered(worker_process, worker_args), total=len(remaining)):
                    aa.append(result)
                    processed_count += 1

                    if processed_count % 10 == 0:
                        with open(save_path, 'w') as f:
                            json.dump(aa, f, ensure_ascii=False, indent=2)
            
            with open(save_path, 'w') as f:
                json.dump(aa, f, ensure_ascii=False, indent=2)

            print(f"Finished. Total saved: {len(aa)}")

        else:
            print("All done.")


        print('*'*30)
        print(f'{task} score:')
        
        if task == 'VSP':
            vsp.calculate_score(save_path)
        elif task == 'TIR':
            tir.calculate_score(save_path, task_args.workers)
        elif task == 'MIRA':
            mira.calculate_score(save_path)
        elif task == 'Realunify':
            realunify.calculate_score(save_path)
        elif task == 'Unimmmu':
            unimmmu.calculate_score(save_path,task_args.use_tools, os.path.join(task_args.result_path, task), task_args.workers)


if __name__ == "__main__":
    main()